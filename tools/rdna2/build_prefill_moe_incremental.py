# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebuild only the expert kernel, reusing immutable qualified native objects."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    reference = args.reference.resolve()
    if reference == root:
        raise RuntimeError("Reference must not be the experiment checkout")
    changed = "csrc/rocm/moe_q_gemm_rdna2.cu"
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "csrc"], cwd=root)
        .decode()
        .split("\0")
    )
    for name in filter(None, tracked):
        source = root / name
        if source.suffix in (".cu", ".cpp", ".cuh", ".h"):
            relative = Path(name)
            if str(relative) != changed and sha(source) != sha(reference / relative):
                raise RuntimeError(f"Unqualified additional native change: {relative}")
    protected = reference / "vllm/_rocm_C.abi3.so"
    before = sha(protected)
    build = root / "build-prefill-moe"
    build.mkdir(exist_ok=True)
    base_build = reference / "build-native"
    ninja = reference / ".venv/bin/ninja"
    env = os.environ.copy()
    sdk = root / ".venv/lib/python3.12/site-packages/_rocm_sdk_core"
    env["PATH"] = f"{root}/.venv/bin:/opt/rocm/core-10.0/bin:" + env["PATH"]
    env["LD_LIBRARY_PATH"] = (
        f"{sdk}/lib:{sdk}/lib/host-math/lib:/opt/rocm/core-10.0/lib"
    )

    def call(values):
        subprocess.run(list(map(str, values)), cwd=base_build, env=env, check=True)

    hipify = subprocess.check_output(
        [
            sys.executable,
            root / "cmake/hipify.py",
            "-p",
            root / "csrc",
            "-o",
            build / "csrc",
            root / "csrc/rocm/moe_accum_rdna2.cuh",
            root / changed,
        ],
        cwd=base_build,
        env=env,
        text=True,
    )
    print(hipify, flush=True)
    hip_source = Path(hipify.strip().splitlines()[-1]).resolve()
    if not hip_source.is_relative_to(root) or not hip_source.is_file():
        raise RuntimeError("Hipify did not return an isolated generated source")
    # Quoted includes next to the source must resolve to the qualified,
    # already-hipified common headers, not the original CUDA headers.
    generated = build / "generated"
    generated.mkdir(exist_ok=True)
    generated_source = generated / hip_source.name
    shutil.copy2(hip_source, generated_source)
    obj = "CMakeFiles/_rocm_C.dir/csrc/rocm/moe_q_gemm_rdna2.hip.o"
    output_obj = build / obj
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    commands = subprocess.check_output(
        [str(ninja), "-t", "commands", obj], cwd=base_build, env=env, text=True
    ).splitlines()
    command = next(line for line in commands if obj in line and " -c " in line)
    values = shlex.split(command)
    values.insert(1, "-I" + str(base_build / "csrc/rocm"))
    for flag, value in (
        ("-MT", output_obj),
        ("-MF", str(output_obj) + ".d"),
        ("-o", output_obj),
        ("-c", generated_source),
    ):
        values[values.index(flag) + 1] = str(value)
    (build / "compile-command.json").write_text(json.dumps(values, indent=2))
    call(values)
    commands = subprocess.check_output(
        [str(ninja), "-t", "commands", "_rocm_C"], cwd=base_build, env=env, text=True
    ).splitlines()
    command = next(
        line
        for line in reversed(commands)
        if line.startswith(": &&") and "-o _rocm_C.abi3.so" in line
    )
    values = shlex.split(command)[2:-2]
    values[values.index(obj)] = str(output_obj)
    values[values.index("-o") + 1] = str(root / "vllm/_rocm_C.abi3.so")
    (build / "link-command.json").write_text(json.dumps(values, indent=2))
    call(values)
    if sha(protected) != before:
        raise RuntimeError("Reference native artifact changed")
    print(
        json.dumps(
            {
                "reference_sha256": before,
                "candidate_sha256": sha(root / "vllm/_rocm_C.abi3.so"),
            }
        )
    )


if __name__ == "__main__":
    main()
