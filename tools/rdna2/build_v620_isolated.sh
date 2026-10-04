#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build in this checkout; never install into the reference runtime.
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
reference=${V620_REFERENCE:-/home/george/v620-experiments/rdna-extras-fresh-20260929}
dependencies=${V620_DEPENDENCIES:-/home/george/v620-experiments/upstream-20260915/v620-vllm-testing/.venv}
rocm=${V620_ROCM:-/opt/rocm/core-10.0}
jobs=${V620_BUILD_JOBS:-4}

[[ $source_dir != "$reference" ]] || { printf 'Refusing a reference-tree build.\n' >&2; exit 2; }
[[ -x $dependencies/bin/python ]] || { printf 'Dependency Python missing.\n' >&2; exit 2; }
[[ -f $reference/vllm/third_party/triton_kernels/__init__.py ]] || {
    printf 'Reference Triton kernel source missing.\n' >&2; exit 2;
}

if [[ ! -x $source_dir/.venv/bin/python ]]; then
    "$dependencies/bin/python" -m venv --without-pip "$source_dir/.venv"
fi
reference_site=$dependencies/lib/python3.12/site-packages
isolated_site=$source_dir/.venv/lib/python3.12/site-packages
# Reuse immutable dependency packages without modifying or reinstalling them.
# Exclude the reference vLLM package and its editable-import hooks.
for entry in "$reference_site"/*; do
    name=${entry##*/}
    case $name in
        vllm | __editable__*vllm* | *vllm*.pth | easy-install.pth) continue ;;
    esac
    if [[ ! -e $isolated_site/$name && ! -L $isolated_site/$name ]]; then
        ln -s -- "$entry" "$isolated_site/$name"
    fi
done
for executable in rocm-sdk cmake ninja; do
    if [[ -x $dependencies/bin/$executable && ! -e $source_dir/.venv/bin/$executable ]]; then
        ln -s -- "$dependencies/bin/$executable" "$source_dir/.venv/bin/$executable"
    fi
done

export PATH="$source_dir/.venv/bin:$dependencies/bin:$rocm/bin:$rocm/llvm/bin:$PATH"
export PYTHONPATH=$source_dir
sdk=$isolated_site/_rocm_sdk_core
export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib/host-math/lib:$rocm/lib"
TRITON_KERNELS_SRC_DIR=$(cd -- "$reference/vllm/third_party/triton_kernels" && pwd -P)
export TRITON_KERNELS_SRC_DIR
export CMAKE_HIP_COMPILER=$rocm/lib/llvm/bin/clang++
export HIP_DEVICE_LIB_PATH=$rocm/lib/llvm/amdgcn/bitcode
"$source_dir/.venv/bin/python" -c 'import setuptools_scm, sys; setuptools_scm.get_version(root=sys.argv[1], write_to="vllm/_version.py", fallback_version="0.28.0")' "$source_dir"

if [[ ${V620_BUILD_SKIP_NATIVE:-0} != 1 ]]; then
"$dependencies/bin/cmake" -S "$source_dir" -B "$source_dir/build-native" -G Ninja \
    -DVLLM_TARGET_DEVICE=rocm \
    -DVLLM_PYTHON_EXECUTABLE="$source_dir/.venv/bin/python" \
    -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$source_dir"
"$dependencies/bin/cmake" --build "$source_dir/build-native" --parallel "$jobs"
"$dependencies/bin/cmake" --install "$source_dir/build-native" --prefix "$source_dir"
fi

"$source_dir/.venv/bin/python" -c 'import torch, vllm, vllm._rocm_C; print("torch:", torch.__version__); print("HIP:", torch.version.hip); print("vLLM:", vllm.__file__); print("native:", vllm._rocm_C.__file__)'
sha256sum "$source_dir/vllm/_rocm_C.abi3.so"
