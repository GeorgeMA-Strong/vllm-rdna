#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# EXL3 27B (Qwen3.8-27B-exl3-3.00bpw, mul1, TP=4) production launcher for
# gfx1030. Derived from the validated recipe in
# bench_results/2026-10-01_exl3-27b-mul1-tp4/exl3_27b_tp4_serve.sh and given
# the same MTP / capture / cache-root knobs as the Flash-Next launcher so the
# TunableOp capture driver can drive it.
#
# Env: MTP (0|2, default 0), EAGER (0|1, default 1), TUNABLEOP (0|1, default 1),
#      PORT, MODEL, MAXLEN, MAXBAT, KVCAP, CG_MODE, CG_SIZES, VLLM_TREE, VENV,
#      HIP_VISIBLE_DEVICES, VLLM_CACHE_ROOT, and the usual TunableOp env vars.
#
# In capture mode the caller sets
#   PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 PYTORCH_TUNABLEOP_UNTUNED_FILENAME=<dir>/untuned.csv
# and the launcher's configure_tunableop keeps ENABLED=1 so record mode works;
# record_untuned disables the results lookup for the life of the boot (capture
# boot != validation boot).
set -euo pipefail

source_dir=${VLLM_TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
runtime=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.28.0}
model=${MODEL:-$HOME/models/Qwen3.8-27B-exl3-3.00bpw}
port=${PORT:-18105}
mtp=${MTP:-0}
eager=${EAGER:-1}
tunableop=${TUNABLEOP:-1}
maxlen=${MAXLEN:-20480}
maxbat=${MAXBAT:-2048}
kvcap=${KVCAP:-6000000000}
cg_mode=${CG_MODE:-FULL_AND_PIECEWISE}

export PYTHONPATH=$source_dir
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export GPU_MAX_HW_QUEUES=2

ROCM_SDK_LIB="$runtime/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ROCM_SDK="$runtime/lib/python3.12/site-packages/_rocm_sdk_core/lib"
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$runtime/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Mandatory HIP stack for the EXL3 27B.
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MOE=0
export VLLM_RDNA_FORCE_FP16=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
export VLLM_BATCH_INVARIANT=0
# FA-RDNA2 (never Triton).
export VLLM_USE_RDNA2_FA=1
unset FLASH_ATTENTION_TRITON_AMD_ENABLE || true
# EXL3 diagnostics.
export VLLM_EXL3_DEBUG=${VLLM_EXL3_DEBUG:-1}
# TP=4 allreduce: RCCL PXB + Simple.
export NCCL_P2P_LEVEL=pxb RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1 NCCL_PROTO=Simple RCCL_MSCCL_ENABLE=0
export VLLM_FORCE_CUSTOM_ALL_REDUCE=0
# Caches: caller may pin a per-arm root; default inside the source tree.
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$source_dir/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$source_dir/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$source_dir/cache/inductor}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$source_dir/cache/extensions}
export ROCM_HOME=${ROCM_HOME:-/opt/rocm/core-7.14} ROCM_PATH=${ROCM_PATH:-/opt/rocm/core-7.14}
export HIP_PATH=${HIP_PATH:-/opt/rocm/core-7.14} HIP_ROOT_DIR=${HIP_ROOT_DIR:-/opt/rocm/core-7.14}
export PATH=/opt/rocm/core-7.14/bin:$PATH

# TunableOp rows live in the repo (keyed by the rocBLAS build hash).
source "$source_dir/tools/rdna2_028/tunableop_env.sh"
if [ "$tunableop" = "1" ]; then
  configure_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "$source_dir/tunableop"
else
  export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
  export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
fi

if [ "$mtp" = "0" ]; then
  spec_args=()
  capture_sizes=${CG_SIZES:-'[1,2,4,8]'}
else
  spec_args=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp,\"use_local_argmax_reduction\":true}")
  decode_width=$((mtp + 1))
  sizes=""
  for mult in 1 2 4 8; do
    sizes="$sizes$((decode_width * mult)),"
  done
  capture_sizes=${CG_SIZES:-"[${sizes%,}]"}
fi

if [ "$eager" = "1" ]; then
  cg_args=(--enforce-eager)
else
  cg_args=(--compilation-config "{\"cudagraph_mode\":\"$cg_mode\",\"cudagraph_capture_sizes\":$capture_sizes,\"compile_ranges_endpoints\":[]}")
fi

# Kill leftovers scoped to THIS tree's VLLM_CACHE_ROOT so a co-tenant server on
# other GPUs (and other cache roots) is never touched.
for _sig in TERM KILL; do
  for _p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
    [ -r "/proc/$_p/environ" ] || continue
    tr '\0' '\n' < "/proc/$_p/environ" 2>/dev/null | grep -q "^VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}$" && kill -"$_sig" "$_p" 2>/dev/null
  done
  [ "$_sig" = "TERM" ] && sleep 8
done
sleep 3

cd "$HOME"
exec "$runtime/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$model" \
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
