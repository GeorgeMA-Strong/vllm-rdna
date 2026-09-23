#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prepared launcher for CUDA graph piecewise qualification. It never stops an
# existing service; serve_v620_baseline.sh refuses to overlap any vLLM process.
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
profile=${V620_PIECEWISE_PROFILE:-full-and-piecewise-breakable}
export V620_PIECEWISE_PROFILE=$profile

case "$profile" in
    piecewise-breakable)
        export VLLM_USE_BREAKABLE_CUDAGRAPH=1
        export V620_COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","compile_ranges_endpoints":[]}'
        ;;
    full-and-piecewise-breakable)
        export VLLM_USE_BREAKABLE_CUDAGRAPH=1
        export V620_COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","compile_ranges_endpoints":[],"cudagraph_capture_sizes":[3,6,12]}'
        ;;
    full-and-piecewise-compiled)
        export VLLM_USE_BREAKABLE_CUDAGRAPH=0
        export V620_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","compile_ranges_endpoints":[4096],"cudagraph_capture_sizes":[3,6,12]}'
        ;;
    *)
        printf 'Unknown V620_PIECEWISE_PROFILE=%s\n' "$profile" >&2
        printf 'Expected piecewise-breakable, full-and-piecewise-breakable, or full-and-piecewise-compiled.\n' >&2
        exit 2
        ;;
esac

export V620_ENABLE_RESIDENT=${V620_ENABLE_RESIDENT:-1}
export V620_ENABLE_SKINNY=${V620_ENABLE_SKINNY:-1}
export VLLM_RDNA_FUSED_SE=${VLLM_RDNA_FUSED_SE:-1}

printf 'Prepared V620 CUDA graph test profile: %s\n' "$profile" >&2
printf 'Compilation config: %s\n' "$V620_COMPILATION_CONFIG" >&2
exec /bin/bash "$source_dir/tools/rdna2/serve_v620_baseline.sh" "$@"
