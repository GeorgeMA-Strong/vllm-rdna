#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Latest performance recipe, preserving the actual production workload.
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
export VLLM_RDNA_MOE_RESIDENT=1 VLLM_RDNA_MOE_RESIDENT_SKINNY=1
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0
export VLLM_RDNA2_W4A8_SDOT4=0 VLLM_RDNA2_MOE_FP32_ACCUM=0
export VLLM_RDNA_DYNAMIC_PREFILL=0 VLLM_RDNA_DECODE_STALL_MS=0
export VLLM_RDNA_HC_PREFILL_FUSED=${VLLM_RDNA_HC_PREFILL_FUSED:-2}
export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MOE=0
export VLLM_USE_AOT_COMPILE=0
export OMP_NUM_THREADS=4 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 VLLM_RDNA_AR_ONESHOT_KB=64
unset VLLM_PLE_CPU_OFFLOAD VLLM_PLE_QUANT_DIR TUNABLEOP_ALLOW_MISMATCH
export EXTRA_ARGS='--pipeline-parallel-size 1 --enable-ep-weight-filter --kv-offloading-size 64 --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --default-chat-template-kwargs {"enable_thinking":false} --kernel-config {"moe_backend":"triton"}'

exec bash "$source_dir/tools/rdna/serve_rdna.sh" \
    RECIPE=flashnext-mtp2 \
    MODEL=/home/george/v620-vllm/models/intel-autoround \
    VENV="$source_dir/.venv" ROCM_SDK_ROOT=/opt/rocm/core-10.0 \
    HOST=127.0.0.1 PORT=8082 SERVED_NAME=active \
    TP=4 SEQS=4 MAXBAT=4096 MAXLEN=262144 BLOCK_SIZE=1024 \
    KV=4026531840 PLE=bf16 W4A8=0 TUNABLEOP=1 \
    TUNABLEOP_PROFILE=rocm10-rocblas5.6 \
    CG_SIZES='[3,6,12]' \
    LIMIT_MM_PER_PROMPT='{"image":255,"video":32}' \
    MM_PROCESSOR_KWARGS='{"max_pixels":602112}' "$@"
