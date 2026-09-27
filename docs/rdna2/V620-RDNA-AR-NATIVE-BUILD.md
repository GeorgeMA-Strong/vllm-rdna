# V620 RDNA all-reduce: matching native build

## Status

Local implementation only. The live service has not been modified or restarted.
The failure observed on September 27 was a missing
`torch.ops._rocm_C.rdna_ar_timeout_info`, not a proven peer-to-peer hardware
fault. Its definition and Torch registration already exist in this checkout.
The process loaded `_rocm_C` from a different runtime source directory.

The change checks the six required native operators before peer-buffer
allocation, with all ranks agreeing on fallback. General vLLM serving still
falls back to RCCL for an incomplete native build. The V620 baseline launcher,
which explicitly requests RDNA AR, instead fails preflight before loading the
model if an operator is missing or Python/native imports come from another
checkout. A valid binary must reside in the selected checkout's `vllm/` directory;
a symlink resolving to a foreign runtime does not satisfy this check.

The preflight verifies operator availability and import location, not GPU
correctness or exact binary reproducibility. It does not bypass timeout
protection, delete wedge markers, change the 64 KiB cutoff, or enable native
collectives after a failed self-test.

## Required later build (not executed)

Keep the current service untouched until a server build/test window is approved.
Use a separate Git checkout and the matching ROCm/PyTorch toolchain. Do not
overwrite the shared runtime's loaded `.so`, copy a binary from another revision,
or change vLLM source on the server. Commit/push locally, then pull the exact
branch on the server before building.

With `source_root` pointing to that isolated Git checkout, `runtime_python` to
the matching `.venv/bin/python`, and `rocm_root` to the matching compiler/SDK:

```bash
build_root=$(mktemp -d /tmp/v620-rdna-ar-build.XXXXXX)
export PYTORCH_ROCM_ARCH=gfx1030
export ROCM_PATH="$rocm_root"
cmake -S "$source_root" -B "$build_root" -G Ninja \
  -DVLLM_TARGET_DEVICE=rocm \
  -DVLLM_PYTHON_EXECUTABLE="$runtime_python" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
  -DROCM_PATH="$rocm_root" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$source_root"
cmake --build "$build_root" --target _rocm_C --parallel 4
cmake --install "$build_root" --component _rocm_C
PYTHONPATH="$source_root" "$runtime_python" \
  "$source_root/tools/rdna2/check_rdna_ar_native.py" \
  --source-root "$source_root"
```

This is the repository's documented incremental CMake workflow scoped to
`_rocm_C`; it has not been qualified against the server's ROCm 10 SDK layout.
Compiler, library search paths and existing toolchain workarounds must match
the installed PyTorch runtime. Do not resolve build failures by modifying the
active service's environment or source files.

## Qualification gates before activation

1. Preflight reports the intended checkout's Python and native extension,
   including `rdna_ar_timeout_info`. Record Git revision, binary hash and
   compiler/runtime versions.
2. All four ranks pass the native startup self-test; logs select RDNA_ONESHOT
   rather than only PYNCCL. Never make a missing timeout operator return zero.
3. Test FP16/FP32 exact sums with changing inputs, small/decode-shaped messages
   through 64 KiB, and fallback above that limit.
4. Verify FULL graph capture and repeated replay with changing inputs,
   consecutive collectives, retained outputs, and interleaved eager work.
   Confirm timeout checks remain active. CPU mocks do not establish these.
5. Compare otherwise-identical RCCL and RDNA AR runs at concurrency 1/2/3/4,
   including MTP2, tool returns, mixed prefills, and long-context RAM reload.
   Report per-session latency and aggregate throughput, not a single token/s
   sample. Only then consider activation.

Local regression command (no HIP required):

```bash
.venv/bin/python -m pytest --noconftest tests/distributed/test_rdna_ar.py -q
```

The focused suite isolates the communicator with lightweight vLLM stubs while
using real PyTorch namespaces. It covers abort handling, markers, per-rank
missing operators, and strict preflight import provenance. Heavy repository
conftest fixtures are intentionally excluded. GPU qualification remains pending.
