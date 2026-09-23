#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Validation only: this script never starts, stops, or restarts a service.
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
runtime=${V620_RUNTIME:-/home/george/v620-experiments/upstream-20260915/v620-vllm-testing}
bench_root=${V620_CONTEXT_BENCH:-/home/george/v620-vllm-testing/context-bench-92286b2}
base_url=${V620_BASE_URL:-http://127.0.0.1:8080}
model=${V620_MODEL_NAME:-active}
profile=${V620_PIECEWISE_PROFILE:-full-and-piecewise-breakable}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
result_root=${V620_RESULT_ROOT:-/home/george/v620-experiments/piecewise-$profile-$stamp}
python=$runtime/.venv/bin/python

mkdir -p "$result_root"
curl -fsS --max-time 5 "$base_url/v1/models" >"$result_root/models.json"

api_pid=$(pgrep -u "$(id -u)" -f 'vllm.entrypoints.openai.api_server' | head -1 || true)
if [[ -z $api_pid ]]; then
    printf 'No vLLM API process found. This validator does not start one.\n' >&2
    exit 2
fi
tr '\0' '\n' <"/proc/$api_pid/environ" >"$result_root/process-environment.txt"
if ! grep -q '^V620_PIECEWISE_PROFILE='"$profile"'$' \
    "$result_root/process-environment.txt"; then
    printf 'Running API process is not the requested profile %s.\n' "$profile" >&2
    exit 2
fi

run_greedy_set() {
    local phase=$1 repetition
    : >"$result_root/greedy-$phase.txt"
    for repetition in 1 2 3 4 5 6; do
        printf 'repetition=%s\n' "$repetition" | tee -a \
            "$result_root/greedy-$phase.txt"
        "$python" "$source_dir/tools/probe_greedy_correctness.py" \
            --url "$base_url/v1/completions" --model "$model" \
            2>&1 | tee -a "$result_root/greedy-$phase.txt"
    done
}

run_greedy_set before-load

PYTHONPATH="$bench_root/src" "$python" -m llm_context_bench \
    --base-url "$base_url" --model "$model" --engine vllm \
    --profile "piecewise-$profile" --suite all --lane performance \
    --sizes 16k 32k 64k 128k --input-size-tolerance-percent 11 \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --timeout 240 --max-retries 0 \
    --output "$result_root/context-bench.json" \
    --command "$source_dir/tools/rdna2/serve_v620_piecewise_test.sh" \
    --system "4x V620; $profile; 3.75GiB KV; MTP2"

run_greedy_set after-load
"$python" "$source_dir/tools/probe_concurrent_greedy.py" \
    --url "$base_url/v1/completions" --model "$model" --concurrency 4 \
    2>&1 | tee "$result_root/greedy-concurrent-4.txt"

"$python" "$source_dir/tools/rdna2/check_prefix_reuse.py" \
    --base-url "$base_url" --model "$model" --lines 1600 --skip-identical \
    --output "$result_root/prefix-32k"
"$python" "$source_dir/tools/rdna2/check_prefix_reuse.py" \
    --base-url "$base_url" --model "$model" --lines 6000 --skip-identical \
    --output "$result_root/prefix-128k"
"$python" "$source_dir/tools/rdna2/check_vision_prefix_reuse.py" \
    --base-url "$base_url" --model "$model" --image-size 1260 --lines 1600 \
    --output "$result_root/vision-prefix-35k.json"

printf 'PASS: piecewise validation completed: %s\n' "$result_root"
