# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-replay latency of FP16 QSA preparation, excluding projection/top-k.

Run with the serving runtime's Python and PYTHONPATH; do not overlap a model.
This is a kernel benchmark, not an end-to-end throughput or quality result.
"""

import json

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.qwen4_exp.amd.indexer_qsa import apply_qsa_rmsnorm, apply_qsa_rope
from vllm.models.qwen4_exp.amd.ops.qsa import (
    qsa_compress_groups_with_ratio,
    qsa_store_cache_rows,
)
from vllm.models.qwen4_exp.amd.ops.qsa_pre_indexer import qsa_pre_indexer
from vllm.models.qwen4_exp.common.qsa_cache import (
    canonical_qsa_rope_positions,
    circular_qsa_slot_mapping,
    compressed_qsa_slot_mapping,
)
from vllm.triton_utils import triton


@torch.inference_mode()
def benchmark(tokens: int, requests: int) -> dict:
    device, dtype = "cuda", torch.float16
    heads, dim, ratio, state_size, page = 4, 128, 4, 8, 256
    length = tokens // requests
    start = 65533
    ends = [start + length] * requests
    logical = torch.arange(start, start + length, device=device).repeat(requests)
    positions = torch.stack((logical, logical // 7 + 3, logical // 13 + 11))
    position_rows = canonical_qsa_rope_positions(positions)
    token_to_req = torch.arange(requests, device=device, dtype=torch.int32)
    token_to_req = token_to_req.repeat_interleave(length)
    query_start = torch.arange(requests + 1, device=device, dtype=torch.int32) * length
    state_table = torch.arange(requests, device=device, dtype=torch.int32).view(-1, 1)
    blocks_per_request = triton.cdiv(start + length, page * ratio)
    compressed_table = torch.arange(
        requests * blocks_per_request, device=device, dtype=torch.int32
    ).view(requests, -1)
    raw_slots = circular_qsa_slot_mapping(
        state_table, token_to_req, logical, state_size, query_start
    )
    compressed_slots = compressed_qsa_slot_mapping(
        compressed_table, token_to_req, logical, page, ratio
    )
    work = [
        (request, group)
        for request in range(requests)
        for group in range(max(1, ends[request] // ratio - start // ratio))
    ]
    max_work = (tokens + (ratio - 1) * requests) // ratio
    work += [(-1, -1)] * (max_work - len(work))
    k_work = torch.tensor(work, device=device, dtype=torch.int32)
    raw = torch.zeros(requests, state_size, 1, dim + 12, device=device, dtype=dtype)
    raw_positions = raw[..., dim:].view(torch.int64)
    history = torch.arange(start - state_size, start, device=device)
    raw[:, history % state_size, 0, :dim] = torch.randn(
        requests, state_size, dim, device=device, dtype=dtype
    )
    raw_positions[:, history % state_size, 0] = torch.stack(
        (history, history // 7 + 3, history // 13 + 11), dim=-1
    )
    compressed = torch.zeros(
        requests * blocks_per_request, page, 1, dim, device=device, dtype=dtype
    )
    qk = torch.randn(tokens, (heads + 1) * dim, device=device, dtype=dtype)
    q_out = torch.empty(tokens, heads, dim, device=device, dtype=dtype)
    q_norm = GemmaRMSNorm(dim, eps=1e-6).to(device=device, dtype=dtype)
    k_norm = GemmaRMSNorm(dim, eps=1e-6).to(device=device, dtype=dtype)
    q_norm.weight.copy_(torch.randn_like(q_norm.weight) * 0.2)
    k_norm.weight.copy_(torch.randn_like(k_norm.weight) * 0.2)
    with torch.device(device):
        rope = get_rope(
            head_size=256,
            max_position=131072,
            dtype=dtype,
            rope_parameters={
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        )

    def unfused():
        q = qk[:, : heads * dim].reshape(-1, dim)
        q = apply_qsa_rmsnorm(q_norm, q).view(tokens, heads, dim)
        apply_qsa_rope(rope, positions, q)
        pooled, first = qsa_compress_groups_with_ratio(
            qk[:, heads * dim :].view(-1, 1, dim),
            position_rows,
            raw[..., :dim],
            state_table,
            token_to_req,
            query_start,
            logical,
            compressed_slots,
            ratio,
            raw_positions,
        )
        keys = apply_qsa_rmsnorm(k_norm, pooled.view(-1, dim)).view(-1, 1, dim)
        keys = apply_qsa_rope(rope, first.transpose(0, 1), keys)
        qsa_store_cache_rows(compressed, compressed_slots, keys)
        qsa_store_cache_rows(raw[..., :dim], raw_slots, qk[:, heads * dim :])
        qsa_store_cache_rows(raw_positions, raw_slots, position_rows)

    # Initialize folded weights and all existing-path kernels before capture.
    unfused()

    def fused():
        qsa_pre_indexer(
            qk[:, : heads * dim],
            qk[:, heads * dim :],
            positions,
            rope.cos_sin_cache,
            q_norm._rdna_w1,
            k_norm._rdna_w1,
            1e-6,
            q_out,
            raw,
            raw_slots,
            state_table,
            query_start,
            logical,
            compressed,
            compressed_slots,
            k_work,
            compress_ratio=ratio,
            mrope_section=(11, 11, 10),
            rope_pos_offset=dim,
        )

    fused()
    torch.accelerator.synchronize()
    baseline_us = triton.testing.do_bench_cudagraph(unfused, rep=500) * 1000
    fused_us = triton.testing.do_bench_cudagraph(fused, rep=500) * 1000
    return {
        "tokens": tokens,
        "requests": requests,
        "unfused_us": baseline_us,
        "fused_us": fused_us,
        "speedup": baseline_us / fused_us,
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    with set_current_vllm_config(VllmConfig()):
        for tokens, requests in (
            (1, 1),
            (3, 1),
            (6, 2),
            (9, 3),
            (12, 4),
            (1024, 1),
            (4096, 1),
        ):
            print(json.dumps(benchmark(tokens, requests)), flush=True)
