#!/usr/bin/env bash
# Qwen3.8-27B AWQ-INT4 (compressed-tensors W4A16, dense hybrid GDN) on gfx1030.
# Sibling of serve_gfx1030_flashnext_mtp.sh, tuned for the dense 27B: no
# --max-model-len (the model's own context is used), the full HIP stack
# (FA-RDNA2 attention + RDNA2 W4A16/W4A8 GEMMs), prefix caching, F&P graphs.
#
# Shared setup lives in scripts/rdna_launcher_common.sh.
#
# Usage:
#   MODEL=/path/to/qwen3.8-27b-awq VENV=/path/to/venv \
#     MTP=0 bash scripts/serve_gfx1030_27b_dense.sh            # plain decode
#   MODEL=... VENV=... MTP=2 bash scripts/serve_gfx1030_27b_dense.sh
#
# Required env: MODEL (checkpoint), VENV (or an activated venv).
# Optional env: MTP(0|2), W4A8(0|1), RDNA_AR(0|1), ATTN(fa|triton), TP, PORT,
#   MAXBAT, KV, SEQS, MAXLEN (unset = model default), CG_MODE, CG_SIZES,
#   GPUIDS, VLLM_TREE.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rdna_launcher_common.sh"
rdna_require_model
rdna_init

port=${PORT:-18210}
tp=${TP:-4}
attn=${ATTN:-fa}
mtp=${MTP:-0}
w4a8=${W4A8:-0}
rdna_ar=${RDNA_AR:-1}

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_RDNA_FUSED_SE=1
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0 VLLM_RDNA_DENSE_GEMV=0
# W4A8 (int4 x int8 sdot4) opt-in dense prefill path.
export VLLM_RDNA2_W4A8_SDOT4=$w4a8
# One-shot up to 64 KiB, RCCL above; the stock custom-AR force flag stays off
# (it PCI-SERRs this chassis, AGENTS.md 2026-09-24).
export VLLM_FORCE_CUSTOM_ALL_REDUCE=0
export VLLM_RDNA_AR=$rdna_ar VLLM_RDNA_AR_MAX_KB=${VLLM_RDNA_AR_MAX_KB:-64} VLLM_RDNA_AR_ONESHOT_KB=${VLLM_RDNA_AR_ONESHOT_KB:-64} VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
export VLLM_ENABLE_STARTUP_PLAN=0
export VLLM_TUNED_CONFIG_FOLDER=$source_dir/tuned-moe

rdna_tunableop || exit 1
rdna_select_attention "$attn"

cg_mode=${CG_MODE:-FULL_AND_PIECEWISE}
rdna_mtp_args "$mtp"
maxlen_arg=()
[ -n "${MAXLEN:-}" ] && maxlen_arg=(--max-model-len "$MAXLEN")

# EAGER=1 skips torch.compile + cudagraphs. The W4A8 fast path lives in the
# eager model forward, so shapes/acceptance are unaffected; this only trades
# throughput for a boot that cannot be killed by a mid-compile chassis reset.
compile_args=(--compilation-config "{\"cudagraph_mode\":\"$cg_mode\",\"cudagraph_capture_sizes\":$capture_sizes,\"compile_ranges_endpoints\":[]}")
if [ "${EAGER:-0}" = "1" ]; then
  compile_args=(--enforce-eager)
fi

rdna_kill_stale
exec "$VENV/bin/python" -m vllm.entrypoints.cli.main serve \
  --model "$MODEL" --served-model-name q27d \
  --host 127.0.0.1 --port "$port" \
  --attention-backend "$attention_backend" \
  --tensor-parallel-size "$tp" \
  --dtype float16 --block-size 1024 \
  --max-num-seqs "${SEQS:-8}" --max-num-batched-tokens "${MAXBAT:-2048}" \
  --kv-cache-memory-bytes "${KV:-8000000000}" \
  --gpu-memory-utilization "${GMEM:-0.9}" \
  "${compile_args[@]}" \
  "${maxlen_arg[@]}" \
  "${spec_args[@]}" \
  --enable-prefix-caching --mamba-cache-mode align \
  --language-model-only \
  --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels":1605632}'
