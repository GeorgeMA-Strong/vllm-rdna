#!/bin/bash
# Production serve: breakable FULL_AND_PIECEWISE HIP graphs (FA-RDNA2 + W4A16
# + HIP KV + HIP GDN). Greedy PASS 3/3 on Qwen3.8-27B-AWQ-INT4, TP=2, 2026-09-09.
# Shared setup lives in scripts/rdna_launcher_common.sh.
#
# Required env: MODEL (checkpoint), VENV (or an activated venv).
# Usage: MODEL=/path/to/model VENV=/path/to/venv PORT=18094 \
#          HIP_VISIBLE_DEVICES=0,1 ./scripts/serve_gfx1030_full.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rdna_launcher_common.sh"
rdna_require_model
rdna_init

PORT="${PORT:-18094}"
TP="${TP:-2}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-200000}"
if [ "$MAX_MODEL_LEN" -lt 32768 ]; then
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN is below the 32768 floor; use 200000 in production." >&2
  exit 1
fi

source "$VENV/bin/activate"
export HIP_VISIBLE_DEVICES

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RDNA2_FA="${VLLM_USE_RDNA2_FA:-1}"
# Isolation only: set to 1 to skip mixed decode+prefill steps.
# Production keeps mixed decode (prefix cache + align CoW/zeroing).
export VLLM_ROCM_NO_MIXED_BATCH="${VLLM_ROCM_NO_MIXED_BATCH:-0}"
# Do not hash the live last 784-token hybrid GDN+FA page while decode
# is still appending to it. Prefix cache still hashes completed pages.
export VLLM_ROCM_SKIP_LIVE_TAIL_HASH="${VLLM_ROCM_SKIP_LIVE_TAIL_HASH:-1}"
export VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

rdna_tunableop || exit 1

# Mixed 16k skip_compiled hits reserved-unallocated holes next to FULL
# keepalives. expandable_segments:True is required for that hole (serve26
# 1k c=8 8/8). False + a 128 MiB persist floor regressed 1k c=8 to 3/8.
if [ "${VLLM_PLE_CPU_OFFLOAD:-0}" = "1" ]; then
  # PLE offload exports CUDA tensors over IPC; VMM-backed (expandable segment)
  # memory cannot be shared that way on ROCm.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-pix}"
export RCCL_P2P_NET_DISABLE=1
export RCCL_P2P_BATCH_ENABLE=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
# Breakable cudagraphs (2026-09-12): the GDN + FA-RDNA2 attention run eager
# (live data) while the rest of the model executes the FULL_AND_PIECEWISE
# graphs. This is the only TP=4 config where the RDNA2 W4A16 + FA-RDNA2 HIP
# path is correct under cudagraphs (verified: 16k/1k c=8 = 31.3 tok/s,
# coherent; the non-breakable TP=4 replay NaNs at the GDN).
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
# Custom all-reduce is correct under breakable cudagraphs as of 2026-09-16;
# it beat PYNCCL in all six matrix cells (verified 0/18 with prefix caching).
# Stock custom all-reduce pulls over PCIe and mis-detects gfx1030 capture.
# rdna_ar one-shot reads the input in place and stays <= 64 KiB; above that
# RCCL is used. Two-shot is NOT boot-safe under PCIe load (wedges warmup).
export VLLM_FORCE_CUSTOM_ALL_REDUCE="${VLLM_FORCE_CUSTOM_ALL_REDUCE:-0}"
export VLLM_RDNA_AR=${VLLM_RDNA_AR:-1}
export VLLM_RDNA_AR_MAX_KB="${VLLM_RDNA_AR_MAX_KB:-64}"
export VLLM_RDNA_AR_ONESHOT_KB="${VLLM_RDNA_AR_ONESHOT_KB:-64}"
# GQA multi-head prefill attention (validated -15.9% cold 16k,
# -5.8% 16k/1k c=8). Set off to revert to varlen/splitk.
export VLLM_FA_RDNA2_GQA_MODE="${VLLM_FA_RDNA2_GQA_MODE:-subgroup}"

# Hybrid GDN page is 24.50 MiB (784-token block). One 200k request
# needs 6.27 GiB, so the 200k default pin is 7e9. That left 0 B free
# after FULL keepalives; mixed FA/GDN scratch OOMed or recycled graph
# pages. For 16k/1k benches set MAX_MODEL_LEN=32768 and
# KV_CACHE_MEMORY=6000000000 (233 blocks; 16k c=8 = 232).
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-7000000000}"

# FULL decode graphs at 1/2/4/8 (and piecewise 16). Prefill-chunk graphs
# at 256/512/1024/2048 are opt-in — capturing 2048 on 32GB TP=2 OOMs
# during warmup. Override with COMPILATION_CONFIG if you have headroom:
#   max_cudagraph_capture_size=2048,
#   cudagraph_capture_sizes=[1,2,4,8,16,256,512,1024,2048]
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"compile_ranges_endpoints\":[]}}"

if [ "${ENABLE_PREFIX_CACHING:-1}" = "0" ]; then
  PREFIX_CACHE_FLAG=""
else
  PREFIX_CACHE_FLAG="--enable-prefix-caching"
fi

rdna_kill_stale
rdna_run_cwd
# A rank JIT-compiling Triton during the V2 warmup blocks the others in the
# logits allgather past PyTorch's 600s NCCL timeout; give it room.
DIST_TIMEOUT="${DIST_TIMEOUT:-1800}"
BLOCK_SIZE="${BLOCK_SIZE-16}"
BLOCK_ARGS=()
[ -n "$BLOCK_SIZE" ] && BLOCK_ARGS=(--block-size "$BLOCK_SIZE")
# EXTRA_ARGS e.g. --enforce-eager for isolation cells.
exec python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "${MAX_NUM_SEQS:-6}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}" \
  --distributed-timeout-seconds "$DIST_TIMEOUT" \
  --dtype float16 \
  --gpu-memory-utilization "${GPU_MEM:-0.90}" \
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY" \
  "${BLOCK_ARGS[@]}" \
  ${PREFIX_CACHE_FLAG} \
  --language-model-only \
  --skip-mm-profiling \
  --trust-remote-code \
  --compilation-config "$COMPILATION_CONFIG" \
  ${EXTRA_ARGS:-}
