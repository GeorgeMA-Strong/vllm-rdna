#!/usr/bin/env bash
# Flash-Next serve with MTP=0/2 on gfx1030 (TP4/EP4, Qwen3.8-Flash-Next-AWQ-W4A16).
# Sibling of scripts/serve_gfx1030_flashnext.sh (the non-MTP production launcher).
# Validated 2026-09-27 on rdna_extras (30632b2fa + shared TunableOp rows):
#   MTP=2 beats MTP=0 at 8x16k/1k c=8 (73.8 vs 67.3 tok/s); MTP=0 wins at
#   8x1k/512 c=8 (166.3 vs 135.5). 8/8 cells, coherent outputs.
#   Rows: tunableop/rocblas-f30bb442e9b5/ (lookup-only env via
#   tools/rdna2_028/tunableop_env.sh, keyed by the rocBLAS build hash).
#
# Usage:
#   MTP=2 bash scripts/serve_gfx1030_flashnext_mtp.sh   # spec decode (16k+)
#   MTP=0 bash scripts/serve_gfx1030_flashnext_mtp.sh   # plain decode (short prompts)
# Env: MTP, TUNABLEOP (default 1), ATTN (triton|fa), PORT, MAXLEN, MODEL,
#      VLLM_TREE, VENV, VLLM_RDNA_AR_MAX_KB, CG_SIZES.
set -euo pipefail

source_dir=${VLLM_TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
runtime=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.28.0}
model=${MODEL:-$HOME/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
port=${PORT:-18096}
attn=${ATTN:-triton}
mtp=${MTP:-2}
tunableop=${TUNABLEOP:-1}

export PYTHONPATH=$source_dir
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export GPU_MAX_HW_QUEUES=2

ROCM_SDK_LIB="$runtime/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ROCM_SDK="$runtime/lib/python3.12/site-packages/_rocm_sdk_core/lib"
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$runtime/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_PLE_QUANT_DIR=${VLLM_PLE_QUANT_DIR:-$HOME/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_MOE_PREFILL=0 VLLM_GDN_HIP_PREFILL=0
export VLLM_RDNA_FUSED_SE=1
export VLLM_TUNED_CONFIG_FOLDER=$source_dir/tuned-moe
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0 VLLM_RDNA_DENSE_GEMV=0
# Two-shot is racy under PCIe load (wedges warmup: 'peer flag never arrived');
# default to the pre-two-shot path: one-shot up to 64 KiB, RCCL above.
export VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=${VLLM_RDNA_AR_MAX_KB:-64} VLLM_RDNA_AR_ONESHOT_KB=${VLLM_RDNA_AR_ONESHOT_KB:-64} VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
export VLLM_ENABLE_STARTUP_PLAN=0 VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$source_dir/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$source_dir/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$source_dir/cache/inductor}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$source_dir/cache/extensions}

# TunableOp rows live in the repo (tunableop/rocblas-<libsha>/); the helper wires
# a lookup-only env keyed by the rocBLAS build. Set TUNABLEOP=0 to disable.
source "$source_dir/tools/rdna2_028/tunableop_env.sh"
if [ "$tunableop" = "1" ]; then
  configure_mtp_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "$source_dir/tunableop"
else
  export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
  export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
fi

if [ "$attn" = "fa" ]; then
  export VLLM_USE_RDNA2_FA=1
  attention_backend=RDNA_ATTN
  unset FLASH_ATTENTION_TRITON_AMD_ENABLE || true
else
  export VLLM_USE_RDNA2_FA=0
  attention_backend=TRITON_ATTN
  export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
fi

compile_mode=${COMPILE_MODE:-3}
cg_mode=${CG_MODE:-FULL_AND_PIECEWISE}
cg_extra=${CG_EXTRA:-,\"compile_ranges_endpoints\":[]}

if [ "$mtp" = "0" ]; then
  spec_args=()
  capture_sizes=${CG_SIZES:-'[1,2,4,8]'}
else
  spec_args=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp,\"use_local_argmax_reduction\":true}")
  # Verify batches are num_seqs * (mtp+1). Capture decode_width x powers of two;
  # intermediate sizes (9/15/18/21) fault on replay in a host-VA copy kernel.
  decode_width=$((mtp + 1))
  sizes=""
  for mult in 1 2 4 8; do
    sizes="$sizes$((decode_width * mult)),"
  done
  capture_sizes=${CG_SIZES:-"[${sizes%,}]"}
fi

exec "$runtime/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$model" --served-model-name flash-next \
  --host 127.0.0.1 --port "$port" \
  --attention-backend "$attention_backend" \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --dtype float16 --max-model-len ${MAXLEN:-262144} --block-size 1024 \
  --max-num-seqs 8 --max-num-batched-tokens ${MAXBAT:-2048} \
  --long-prefill-token-threshold ${LPTH:-0} \
  --kv-cache-memory-bytes 4026531840 \
  --compilation-config "{\"mode\":$compile_mode,\"cudagraph_mode\":\"$cg_mode\",\"cudagraph_capture_sizes\":$capture_sizes$cg_extra}" \
  "${spec_args[@]}" \
  --enable-prefix-caching --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels":1605632}'
