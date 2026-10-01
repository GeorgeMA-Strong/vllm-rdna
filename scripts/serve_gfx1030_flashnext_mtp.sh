#!/usr/bin/env bash
# Flash-Next serve with MTP=0/2 on gfx1030 (TP4/EP4, Qwen3.8-Flash-Next-AWQ-W4A16).
# Sibling of scripts/serve_gfx1030_flashnext.sh (the non-MTP production launcher).
# Validated 2026-09-27 on rdna_extras (30632b2fa + shared TunableOp rows):
#   MTP=2 beats MTP=0 at 8x16k/1k c=8 (73.8 vs 67.3 tok/s); MTP=0 wins at
#   8x1k/512 c=8 (166.3 vs 135.5). 8/8 cells, coherent outputs.
#   Rows: tunableop/rocm7.14-rocblas5.5/ (lookup-only env via
#   tools/rdna2_028/tunableop_env.sh, selected by the rocBLAS build).
#
# Shared setup (tree/venv/ROCm/TunableOp/teardown) lives in
# scripts/rdna_launcher_common.sh.
#
# Usage:
#   MODEL=/path/to/flash-next VENV=/path/to/venv \
#     MTP=2 bash scripts/serve_gfx1030_flashnext_mtp.sh   # spec decode (16k+)
#   MODEL=... VENV=... MTP=0 bash scripts/serve_gfx1030_flashnext_mtp.sh
#
# Required env: MODEL (checkpoint), VENV (or an activated venv), and
#   VLLM_PLE_QUANT_DIR (the int4 PLE sidecar directory).
# Optional env: MTP, TUNABLEOP, ATTN (triton|fa), PORT, MAXLEN, MAXBAT, LPTH,
#   VLLM_TREE, HIP_VISIBLE_DEVICES, VLLM_RDNA_AR_MAX_KB, CG_SIZES.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rdna_launcher_common.sh"
rdna_require_model
rdna_require VLLM_PLE_QUANT_DIR "Pass the int4 PLE sidecar directory (…/ples_int4)."
rdna_init

port=${PORT:-18096}
attn=${ATTN:-fa}   # FA-RDNA2 = the fastest validated HIP attention path (ATTN=triton falls back)
mtp=${MTP:-2}

export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_MOE_PREFILL=0 VLLM_GDN_HIP_PREFILL=0
export VLLM_RDNA_FUSED_SE=1
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0 VLLM_RDNA_DENSE_GEMV=0
# Two-shot is racy under PCIe load (wedges warmup: 'peer flag never arrived');
# default to the pre-two-shot path: one-shot up to 64 KiB, RCCL above.
export VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=${VLLM_RDNA_AR_MAX_KB:-64} VLLM_RDNA_AR_ONESHOT_KB=${VLLM_RDNA_AR_ONESHOT_KB:-64} VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
export VLLM_ENABLE_STARTUP_PLAN=0
export VLLM_TUNED_CONFIG_FOLDER=$source_dir/tuned-moe

rdna_tunableop || exit 1
rdna_select_attention "$attn"

compile_mode=${COMPILE_MODE:-3}
cg_mode=${CG_MODE:-FULL_AND_PIECEWISE}
cg_extra=${CG_EXTRA:-,\"compile_ranges_endpoints\":[]}
rdna_mtp_args "$mtp"

rdna_kill_stale
exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name flash-next \
  --host 127.0.0.1 --port "$port" \
  --attention-backend "$attention_backend" \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --dtype float16 --max-model-len ${MAXLEN:-262144} --block-size 1024 \
  --max-num-seqs 8 --max-num-batched-tokens ${MAXBAT:-2048} \
  --long-prefill-token-threshold ${LPTH:-0} \
  --prefill-schedule-interval ${PREFILL_INTERVAL:-1} \
  --kv-cache-memory-bytes 4026531840 \
  --compilation-config "{\"mode\":$compile_mode,\"cudagraph_mode\":\"$cg_mode\",\"cudagraph_capture_sizes\":$capture_sizes$cg_extra}" \
  "${spec_args[@]}" \
  --enable-prefix-caching --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels":1605632}'
