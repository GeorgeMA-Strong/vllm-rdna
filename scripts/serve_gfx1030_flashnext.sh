#!/bin/bash
# Flash-Next production serve on gfx1030 (TP=4, Qwen3.8-Flash-Next-AWQ-W4A16).
# Validated 2026-09-17: FULL_AND_PIECEWISE + prefix caching + max_num_seqs 6.
#
#   PP 3331 tok/s agg, TG 72.70 tok/s, TTFT 39.3 s at 16k/1k c=8 (PIECEWISE).
#   Correctness: 18/18 sequential, 6/6 c=8, 8/8 16k shared-prefix.
#   FULL_AND_PIECEWISE executes as PIECEWISE on ROCm (rocm_full_executes_as_piecewise)
#   — verified identical and correct 2026-09-18; the historical "FULL_AND_PIECEWISE
#   corrupts at c=8" report traced to probe artifacts (reasoning-parser field +
#   reasoning-budget exhaustion), not the graphs.
#
# Requires commit 388a61b6f (the GDN sanitizer fix) for fresh-server long-prompt
# correctness.
#
# Vision is ON by default with the validated pixel cap (2026-09-17). Without
# the cap the mm-profiling dummy image (~24.8M px) makes the vision encoder's
# SDPA math backend materialize a 64 GiB LxL fp32 score matrix and startup
# OOMs on 30 GiB GPUs. NOTE: --limit-mm-per-prompt alone is NOT sufficient
# (the image count was already 1; the size is the driver).
# max_pixels=1605632 keeps images up to ~1424x1424 full-resolution.
# No --max-model-len by default: the checkpoint's native 262144-token context
# is used (the pinned 5 GiB KV pool holds ~313k tokens). Set MAX_MODEL_LEN to
# opt into a cap.
#
# Shared setup lives in scripts/rdna_launcher_common.sh.
#
# Required env: MODEL (checkpoint), VENV (or an activated venv), and
#   VLLM_PLE_QUANT_DIR (the int4 PLE sidecar directory).
# Usage: MODEL=/path/to/flash-next VENV=/path/to/venv \
#          VLLM_PLE_QUANT_DIR=/path/to/ples_int4 bash scripts/serve_gfx1030_flashnext.sh
set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rdna_launcher_common.sh"
rdna_require_model
rdna_require VLLM_PLE_QUANT_DIR "Pass the int4 PLE sidecar directory (…/ples_int4)."
rdna_init

PORT="${PORT:-18094}"
TP="${TP:-4}"
SERVED_NAME="${SERVED_NAME:-flash-next}"
# In-flight cap 6: the Flash-Next corruption threshold is below 8; clients may
# still send 8/10/16 concurrent requests (they queue).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-7000000000}"
GPU_MEM="${GPU_MEM:-0.90}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
LOG="${LOG:-${VLLM_LOG_DIR:-$source_dir/cache/logs}/flashnext_server.log}"
mkdir -p "$(dirname "$LOG")"

# PLE (n-gram sidecar) CPU offload.
export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# Stock custom all-reduce pulls over PCIe and mis-detects gfx1030 capture.
# rdna_ar covers fp16/bf16/fp32 up to VLLM_RDNA_AR_MAX_KB (two-shot above
# VLLM_RDNA_AR_ONESHOT_KB). Override either var to A/B the old path.
export VLLM_FORCE_CUSTOM_ALL_REDUCE="${VLLM_FORCE_CUSTOM_ALL_REDUCE:-0}"
export VLLM_RDNA_AR="${VLLM_RDNA_AR:-1}"
export VLLM_RDNA_AR_MAX_KB="${VLLM_RDNA_AR_MAX_KB:-64}"
export VLLM_RDNA_AR_ONESHOT_KB="${VLLM_RDNA_AR_ONESHOT_KB:-64}"
# FA-RDNA2 = the fastest validated HIP attention path; set to 0 (or pass
# --attention-backend) to fall back to the Triton backend.
export VLLM_USE_RDNA2_FA="${VLLM_USE_RDNA2_FA:-1}"
export VLLM_FA_RDNA2_GQA_DECODE="${VLLM_FA_RDNA2_GQA_DECODE:-1}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_RDNA_FUSED_HC=0
export NCCL_P2P_LEVEL=pix
export RCCL_P2P_NET_DISABLE=1
export RCCL_P2P_BATCH_ENABLE=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1

rdna_tunableop || exit 1

source "$VENV/bin/activate"

# Run from a persistent, non-repo CWD (never /tmp): avoids /tmp per the storage
# policy and avoids shadowing the vllm package when CWD is the tree root.
rdna_kill_stale
rdna_run_cwd
nohup setsid bash -c "python -m vllm.entrypoints.cli.main serve \"$MODEL\" \
  --served-model-name \"$SERVED_NAME\" \
  --port $PORT --host 0.0.0.0 --tensor-parallel-size $TP \
  ${MAX_MODEL_LEN:+--max-model-len $MAX_MODEL_LEN} --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens ${MAXBAT:-2048} \
  --long-prefill-token-threshold ${LPTH:-0} \
  --kv-cache-memory-bytes $KV_CACHE_MEMORY --gpu-memory-utilization $GPU_MEM \
  --dtype float16 --trust-remote-code --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-expert-parallel \
  --limit-mm-per-prompt '{\"image\":1}' --mm-processor-kwargs '{\"max_pixels\":1605632}' \
  --distributed-timeout-seconds 1800 \
  ${BLOCK_SIZE:+--block-size $BLOCK_SIZE} \
  --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"compile_ranges_endpoints\":[]}' \
  ${EXTRA_ARGS:-}" > "$LOG" 2>&1 < /dev/null &
disown
echo "launched flash-next server pid $! log=$LOG (TP=$TP, PIECEWISE, max_num_seqs=$MAX_NUM_SEQS)"

for i in $(seq 1 40); do
  sleep 15
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
     && grep -q "init engine" "$LOG" 2>/dev/null; then
    echo "READY after ~$((i * 15))s"; exit 0
  fi
  if ! pgrep -f "entrypoints.cli.main serve" >/dev/null; then
    echo "SERVER DIED"; tail -15 "$LOG"; exit 1
  fi
done
echo "TIMEOUT"; tail -15 "$LOG"; exit 1
