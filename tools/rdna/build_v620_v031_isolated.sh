#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Reuse ROCm10 without installing into or changing the reference environment.
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
reference=/home/george/v620-experiments/rdna-extras-fresh-20260929
dependencies=/home/george/v620-experiments/upstream-20260915/v620-vllm-testing/.venv
rocm=/opt/rocm/core-10.0
jobs=${V620_BUILD_JOBS:-4}
[[ $source_dir != "$reference" && -x $dependencies/bin/python ]]
[[ -f $source_dir/CMakeLists.txt && -d $rocm/include ]]
reference_site=$dependencies/lib/python3.12/site-packages
isolated_site=$source_dir/.venv/lib/python3.12/site-packages

if [[ ! -x $source_dir/.venv/bin/python ]]; then
    "$dependencies/bin/python" -m venv --without-pip "$source_dir/.venv"
    # Physical top-level packages, not symlinks: dependency upgrades cannot
    # follow links into the reference. The original runtime was audited first.
    [[ -z $(find "$reference_site" -maxdepth 1 -type l -print -quit) ]]
    cp -a --reflink=auto "$reference_site/." "$isolated_site/"
    mkdir -p "$source_dir/dependency-stash"
    for entry in "$isolated_site"/*; do
        name=${entry##*/}
        case $name in
            vllm | vllm-*.dist-info | __editable__*vllm* | *vllm*.pth | easy-install.pth)
                mv -- "$entry" "$source_dir/dependency-stash/" ;;
        esac
    done
fi

# Install a pinned package-manager binary only inside this experiment.
uv_dir=$source_dir/.tools/uv-x86_64-unknown-linux-gnu
if [[ ! -x $uv_dir/uv ]]; then
    mkdir -p "$source_dir/.tools"
    curl --fail --location --retry 2 \
        https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz \
        --output "$source_dir/.tools/uv.tar.gz"
    tar -xzf "$source_dir/.tools/uv.tar.gz" -C "$source_dir/.tools"
fi
"$uv_dir/uv" pip install --python "$source_dir/.venv/bin/python" --no-deps \
    'huggingface_hub>=1.31.0,<2.0' 'xgrammar==0.2.7' 'oss-harmony>=0.0.11'

export PATH="$source_dir/.venv/bin:$dependencies/bin:$rocm/bin:$rocm/llvm/bin:$PATH"
export PYTHONPATH=$source_dir
sdk=$isolated_site/_rocm_sdk_core
export LD_LIBRARY_PATH="$isolated_site/_rocm_sdk_libraries/lib:$sdk/lib:$sdk/lib/host-math/lib:$isolated_site/torch/lib:$rocm/lib"
export CMAKE_HIP_COMPILER=$rocm/lib/llvm/bin/clang++
export HIP_DEVICE_LIB_PATH=$rocm/lib/llvm/amdgcn/bitcode
export SETUPTOOLS_SCM_PRETEND_VERSION=0.31.0+rdna.validation
"$source_dir/.venv/bin/python" -c 'import setuptools_scm,sys; setuptools_scm.get_version(root=sys.argv[1],write_to="vllm/_version.py",fallback_version="0.31.0+rdna.validation")' "$source_dir"

# Let this branch fetch its own pinned Triton kernel sources; do not reuse an
# older installed source directory across the v0.31 rebase.
if [[ ${V620_BUILD_SKIP_NATIVE:-0} != 1 ]]; then
    "$dependencies/bin/cmake" -S "$source_dir" -B "$source_dir/build-native" -G Ninja \
        -DVLLM_TARGET_DEVICE=rocm \
        -DVLLM_PYTHON_EXECUTABLE="$source_dir/.venv/bin/python" \
        -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$source_dir"
    "$dependencies/bin/cmake" --build "$source_dir/build-native" --parallel "$jobs"
    "$dependencies/bin/cmake" --install "$source_dir/build-native" --prefix "$source_dir"
fi
[[ -f $source_dir/vllm/_rocm_C.abi3.so ]]
# CMake installs native libraries but not distribution metadata. Generate the
# normal source-tree egg-info without downloading replacement native wheels
# or invoking a second C++/Rust build. PYTHONPATH supplies this editable tree.
env VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.31.0+rdna.rocm100.validation \
    "$source_dir/.venv/bin/python" "$source_dir/setup.py" egg_info
"$source_dir/.venv/bin/python" -c 'import importlib.metadata as m; import torch,vllm,vllm._custom_ops; from vllm.platforms import current_platform; assert current_platform.is_rocm(); print("package:",m.version("vllm"),"torch:",torch.__version__,"HIP:",torch.version.hip,"vLLM:",vllm.__file__)'
sha256sum "$source_dir/vllm/_rocm_C.abi3.so"
