#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Isolated startup-scoped GPU tracing. Throughput must be measured without it.
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
runtime=${V620_RUNTIME:-/home/george/v620-experiments/upstream-20260915/v620-vllm-testing}
sdk=$runtime/.venv/lib/python3.12/site-packages/_rocm_sdk_core
export PYTHONPATH=$source_dir
export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib/host-math/lib:/opt/rocm/core-10.0/lib"
export VLLM_SERVER_DEV_MODE=1
export V620_WORKER_EXTENSION_CLS=tools.rdna2.v620_decode_trace.V620DecodeTrace
export V620_RAW_TRACE_OUTPUT=${V620_RAW_TRACE_OUTPUT:?Set an absolute output directory}
unset V620_PROFILE_DIR

exec "$runtime/.venv/bin/python" "$sdk/bin/rocprofv3" \
    --selected-regions --attach-sync-output --kernel-trace \
    --hip-runtime-trace --memory-copy-trace --output-format csv \
    --output-directory "$V620_RAW_TRACE_OUTPUT/trace" \
    -- /bin/bash "$source_dir/tools/rdna2/serve_v620_baseline.sh"
