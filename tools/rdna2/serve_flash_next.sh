#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Development launcher for the four-V620 Intel checkpoint baseline.
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
workspace=${V620_WORKSPACE:-$(dirname -- "$source_dir")}
venv=${V620_VENV:-$workspace/.venv}
model=${V620_MODEL:-$workspace/models/intel-autoround}
mtp_tokens=${V620_MTP_TOKENS:-0}
kv_cache_bytes=${V620_KV_CACHE_BYTES:-4294967296}
if [[ ! "$mtp_tokens" =~ ^[0-9]+$ ]]; then
    printf 'V620_MTP_TOKENS must be a nonnegative integer.\n' >&2
    exit 2
fi
if [[ ! "$kv_cache_bytes" =~ ^[1-9][0-9]*$ ]]; then
    printf 'V620_KV_CACHE_BYTES must be a positive integer.\n' >&2
    exit 2
fi
if [[ ! -x "$venv/bin/python" || ! -f "$model/config.json" ]]; then
    printf 'Set V620_VENV and V620_MODEL to the built environment and local checkpoint.\n' >&2
    exit 2
fi

site_packages=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
runtime_sdk="$site_packages/_rocm_sdk_core"
export PATH="$venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$runtime_sdk/lib:$runtime_sdk/lib/host-math/lib:/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The first successful run used this after a driver SDMA weight-copy fault.
# Override explicitly when comparing transfer performance and stability.
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}
export VLLM_USE_V2_MODEL_RUNNER=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1

speculative_args=()
if (( mtp_tokens > 0 )); then
    speculative_args=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp_tokens}")
fi
cd "$source_dir"
exec "$venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name qwen3.8-flash-next \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 4 --enable-expert-parallel --enable-ep-weight-filter \
    --dtype bfloat16 --max-model-len 262144 \
    --max-num-seqs 4 --max-num-batched-tokens 2048 \
    --kv-cache-memory-bytes "$kv_cache_bytes" --enforce-eager \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":1638400}' \
    --engram-config '{"cpu_offload":true}' \
    "${speculative_args[@]}" "$@"
