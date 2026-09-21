#!/usr/bin/env bash
set -euo pipefail
root=/home/george/v620-experiments/decode-qsa-20260920
runtime=/home/george/v620-experiments/upstream-20260915/v620-vllm-testing
export PYTHONPATH=$root/source
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0 VLLM_RDNA_DENSE_GEMV=0
export VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
export VLLM_ENABLE_STARTUP_PLAN=0 VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export VLLM_CACHE_ROOT=$root/cache/vllm TRITON_CACHE_DIR=$root/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$root/cache/inductor TORCH_EXTENSIONS_DIR=$root/cache/extensions
export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
sdk=$runtime/.venv/lib/python3.12/site-packages/_rocm_sdk_core
export PATH="$runtime/.venv/bin:/opt/rocm/core-10.0/bin:/opt/rocm/core-10.0/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib/host-math/lib:/opt/rocm/core-10.0/lib"
source "$root/source/tools/rdna2/tunableop_env.sh"
configure_v620_tunableop "$runtime/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librocblas.so.5" "$runtime/tunableop"
command=("$runtime/.venv/bin/python" -m vllm.entrypoints.openai.api_server
 --model /home/george/v620-vllm/models/intel-autoround --served-model-name active qwen3.8-flash-next
 --host 0.0.0.0 --port 8080 --tensor-parallel-size 4 --pipeline-parallel-size 1 --enable-expert-parallel --enable-ep-weight-filter
 --dtype float16 --max-model-len 262144 --block-size 1024 --max-num-seqs 4
 --max-num-batched-tokens 4096 --kv-cache-memory-bytes 4026531840
 --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,12]}'
 --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
 --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
 --default-chat-template-kwargs '{"enable_thinking":false}'
 --limit-mm-per-prompt '{"image":255,"video":32}' --mm-processor-kwargs '{"max_pixels":1048576}'
 --enable-prefix-caching --mamba-cache-mode align --kernel-config '{"moe_backend":"triton"}')
if [[ ${1:-} == --dry-run ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
if pgrep -u "$(id -u)" -f 'vllm.entrypoints|VLLM::EngineCore|VLLM::Worker' >/dev/null; then
 echo 'Existing vLLM process present; refusing overlap.' >&2; exit 2
fi
"$runtime/.venv/bin/python" "$root/source/tools/rdna2/check_v620_tuning.py" --rows-template "$PYTORCH_TUNABLEOP_FILENAME" "${command[@]:1}"
"$runtime/.venv/bin/python" -c 'import vllm; assert "/decode-qsa-20260920/" in vllm.__file__,vllm.__file__'
cd "$root/source"
exec "${command[@]}" "$@"
