# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.triton_utils import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    # This will be fixed by https://github.com/vllm-project/vllm/pull/51540.
    with gpu_sync_allowed():
        chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    # Counting local-index zeros renumbers sequences when an empty one has no
    # chunks. Preserve the original sequence IDs, including gaps and empty input.
    chunk_indices = torch.tensor(
        [
            (sequence_idx, chunk_idx)
            for sequence_idx, count in enumerate(chunk_counts)
            for chunk_idx in range(count)
        ],
        dtype=cu_seqlens.dtype,
    ).reshape(-1, 2)
    return chunk_indices.to(
        device=cu_seqlens.device, dtype=cu_seqlens.dtype, non_blocking=True
    )


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_zeros(1), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
