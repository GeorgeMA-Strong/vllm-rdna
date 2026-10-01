#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# EXL3 27B (Qwen3.8-27B-exl3-3.00bpw, mul1, TP=4) production launcher for
# gfx1030. Derived from the validated recipe in
# bench_results/2026-10-01_exl3-27b-mul1-tp4/exl3_27b_tp4_serve.sh and given
# the same MTP / capture / cache-root knobs as the Flash-Next launcher so the
# TunableOp capture driver can drive it. Shared setup lives in
# scripts/rdna_launcher_common.sh.
#
# Required env: MODEL (checkpoint), VENV (or an activated venv).
# Optional env: MTP (0|2, default 0), EAGER (0|1, default 1), TUNABLEOP (0|1,
#   default 1), PORT, MAXLEN, MAXBAT, KVCAP, CG_MODE, CG_SIZES, VLLM_TREE,
#   HIP_VISIBLE_DEVICES, VLLM_CACHE_ROOT, and the usual TunableOp env vars.
#
# In capture mode the caller sets
#   PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 PYTORCH_TUNABLEOP_UNTUNED_FILENAME=<dir>/untuned.csv
# and configure_tunableop keeps ENABLED=1 so record mode works; record_untuned
# disables the results lookup for the life of the boot (capture boot !=
# validation boot).
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rdna_launcher_common.sh"
rdna_require_model
rdna_init

port=${PORT:-18105}
mtp=${MTP:-0}
eager=${EAGER:-1}
maxlen=${MAXLEN:-20480}
maxbat=${MAXBAT:-2048}
kvcap=${KVCAP:-6000000000}
cg_mode=${CG_MODE:-FULL_AND_PIECEWISE}

# Mandatory HIP stack for the EXL3 27B.
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_EXL3_DEBUG=${VLLM_EXL3_DEBUG:-1}
# TP=4 allreduce: RCCL PXB + Simple.
export NCCL_P2P_LEVEL=pxb RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1 NCCL_PROTO=Simple RCCL_MSCCL_ENABLE=0
export VLLM_FORCE_CUSTOM_ALL_REDUCE=0
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}

rdna_tunableop || exit 1
# FA-RDNA2 (never Triton).
rdna_select_attention fa
unset FLASH_ATTENTION_TRITON_AMD_ENABLE || true

rdna_mtp_args "$mtp"
if [ "$eager" = "1" ]; then
  cg_args=(--enforce-eager)
else
  cg_args=(--compilation-config "{\"cudagraph_mode\":\"$cg_mode\",\"cudagraph_capture_sizes\":$capture_sizes,\"compile_ranges_endpoints\":[]}")
fi

rdna_kill_stale
rdna_run_cwd
exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name exl3-27b-mul1 \
  --host 127.0.0.1 --port "$port" \
  --attention-backend RDNA_ATTN \
  --tensor-parallel-size 4 \
  --dtype float16 --max-model-len "$maxlen" \
  --max-num-seqs 8 --max-num-batched-tokens "$maxbat" \
  --gpu-memory-utilization 0.90 --kv-cache-memory-bytes "$kvcap" \
  --language-model-only --skip-mm-profiling --trust-remote-code \
  --enable-prefix-caching --mamba-cache-mode align \
  "${spec_args[@]}" \
  "${cg_args[@]}"
