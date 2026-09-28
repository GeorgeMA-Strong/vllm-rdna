# W4A8 sdot4 wiring — opt-in drop-in for the dense W4A16 prefill on gfx1030

Branch: `w4a8-wiring` (= `rdna_extras` + the production wrapper around the
explore kernel from PR #9). Default OFF — the dispatcher takes the existing
`gptq_gemm_rdna2_prefill` path unless `VLLM_RDNA2_W4A8_SDOT4=1`.

The kernel body (`w4a8_gemm_kernel`, `w4a8_act_quant_kernel`, `Cfg`,
`W4A8_EXPLORE_CONFIGS`, etc.) is copied verbatim from
`/tmp/pr9_branch/csrc/rocm/explore/w4a8_sdot4.cuh` into the sibling
`csrc/rocm/w4a8_sdot4_rdna2.cuh`. The explore source of truth stays
untouched per the "do not modify `csrc/rocm/explore/*` in place" rule —
the fork never had an `explore/` directory and this branch does not create
one. The rename to `w4a8_sdot4_rdna2.cuh` makes provenance clear (the
explore kernel is the explore kernel; the production copy is the
production copy).

## What changed

| File | Purpose |
|---|---|
| `csrc/rocm/w4a8_sdot4_rdna2.cu` | Production TU. Wraps the explore `w4a8_act_quant_kernel` and `w4a8_gemm_kernel`; reuses `W4A8_EXPLORE_CONFIGS` to instantiate one launch per group size. Two entry points: `w4a8_act_quant_rdna2` and `w4a8_gemm_rdna2`. |
| `csrc/rocm/w4a8_sdot4_rdna2.cuh` | Verbatim copy of the explore header (renamed). `W4A8_EXPLORE_CONFIGS` and the kernel templates are unchanged. |
| `csrc/rocm/ops.h` | Declarations for the two ops. |
| `csrc/rocm/torch_bindings.cpp` | Registers both ops inside `#ifdef VLLM_ROCM_GFX1030`, next to `gptq_gemm_rdna2_prefill`. |
| `CMakeLists.txt` | Adds `csrc/rocm/w4a8_sdot4_rdna2.cu` to the gfx1030 source list (line 1541). |
| `vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py` | Opt-in dispatcher branch. Reads `VLLM_RDNA2_W4A8_SDOT4` and the op schema list, runs the W4A8 path on the prefill kernel, falls back to `gptq_gemm_rdna2_prefill` on nonzero status. |
| `tests/kernels/quantization/test_rdna2_w4a8.py` | G2-mirror correctness tests: act_quant vs NumPy reference, gemm vs `gptq_gemm_rdna2_prefill` (rel-L2 < 0.1), invalid group-size error, clean skip when ops are absent. |
| `docs/rdna2/w4a8-wiring.md` | This file. |

The kernel body (`csrc/rocm/explore/w4a8_sdot4.cuh`), the W4A16/MoE/attention
TUs, `rdna_extras`, and the launchers are not touched.

## Op contract

```
w4a8_act_quant_rdna2(x, group_size, a_i8, a_scale, a_asum) -> int
  x       [M, K]            fp16, contiguous on dim 1
  a_i8    [T, K/8, 8, 8]    int8   (T = ceil(M / 8))
  a_scale [T, K/G, 8]       fp32   (per-(token, group), A_GROUP variant)
  a_asum  [T, K/G, 8]       int32
  Returns 0 on success, a positive hipError_t, or a negative "not eligible"
  code (-1 .. -6, see w4a8_sdot4_rdna2.cu).

w4a8_gemm_rdna2(a_i8, w_packed, qzeros, scales, a_scale, asum, out,
                k, group, zero_offset, config_id, split_k) -> int
  w_packed [K/8, N]         uint32  (the same shuffled buffer
                                      RDNA2W4A16LinearKernel produces:
                                      zero-extended nibbles + gptq_shuffle)
  qzeros   [K/G, N/8]       uint32  (packed along dim 1)
  scales   [K/G, N]         fp16
  out      [M, N]           fp16    (split_k=1: plain stores;
                                    split_k>1: pk4 CAS atomic add,
                                    the caller must zero-init.)
  zero_offset 0 for AWQ (uint4), 1 for GPTQv1 (uint4b8).
  config_id    8 = a8_lds_k32_ag (recommended); 0..7 from the explore sweep.
  split_k      0/1 = pick with pick_split_k; >1 = pinned group-aligned split.
  Returns 0, a positive hipError_t, or a negative "not eligible" code.
```

## Build

```bash
# Local: rebuild only _rocm_C against the existing venv on the build server.
# See AGENTS.md "Cold-start install on a fresh remote" for the full env
# setup; the W4A8 TU needs the same toolchain (hipcc 7.14, PYTORCH_ROCM_ARCH
# includes gfx1030, VLLM_ROCM_GFX1030 defined).

cd /Users/kletorch/Projects/infrastructure/gfx1030_optimized/vllm-rdna-0.28.0
git switch w4a8-wiring
# rsync to the build server (par1-cs25 = 192.168.1.84) before building.
rsync -avz --exclude='.git/' -e "ssh -i ~/.ssh/id_ed25519_ansible" \
    ./ chenco_adm@par1-cs25:~/vllm_humanwork/

ssh chenco_adm@par1-cs25
source /home/chenco_adm/Apps/vllm/venv-7.14.0/bin/activate
cd /home/chenco_adm/vllm_humanwork
export SETUPTOOLS_SCM_PRETEND_VERSION=0.20.1.dev99
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH='gfx1030'
export VLLM_PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python
export MAX_JOBS=16
export CMAKE_BUILD_TYPE=RelWithDebInfo
export CMAKE_HIP_COMPILER=/opt/rocm/core-7.14/bin/hipcc
export ROCM_HOME=/opt/rocm/core-7.14
export ROCM_PATH=/opt/rocm/core-7.14
export HIP_PATH=/opt/rocm/core-7.14
export HIP_ROOT_DIR=/opt/rocm/core-7.14
export CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$PATH
rm -rf build/ vllm/*.abi3.so
pip install -e . --no-build-isolation --no-deps
```

Verify the ops are registered (this is the same gate the test file uses):

```bash
python -c "
import torch
schemas = torch._C._jit_get_all_schemas()
names = sorted({str(s) for s in schemas if 'w4a8' in str(s)})
for n in names: print(n)
"
# Expect:
#   _rocm_C::w4a8_act_quant_rdna2(Tensor x, int group_size, Tensor(a!) a_i8,
#                                  Tensor(a!) a_scale, Tensor(a!) a_asum) -> int
#   _rocm_C::w4a8_gemm_rdna2(Tensor a_i8, Tensor w_packed, Tensor qzeros,
#                             Tensor scales, Tensor a_scale, Tensor asum,
#                             Tensor(a!) out, int k, int group,
#                             int zero_offset, int config_id, int split_k) -> int
```

## Validate

### Pytest (unit, no model)

```bash
cd /home/chenco_adm/vllm_humanwork
.venv/bin/python -m pytest tests/kernels/quantization/test_rdna2_w4a8.py -v
```

The suite covers:
- `test_w4a8_act_quant_matches_numpy[M/g]` — act_quant vs the NumPy
  reference (per-(token, group) quant + tiled layout, bit-exact).
- `test_w4a8_act_quant_rejects_bad_group_size` — invalid group_size (16)
  returns negative status (kBadGroup = -6).
- `test_w4a8_gemm_matches_w4a16_prefill[g]` — gemm vs
  `gptq_gemm_rdna2_prefill` on the same packed weight buffer; rel-L2 < 0.1.
- `test_w4a8_dispatcher_uses_w4a16_when_env_var_unset` — with the env var
  absent, `apply_weights` matches `gptq_gemm_rdna2_prefill` byte-for-byte.

All four skip cleanly when the ops are absent (no GPU, partial build, or
non-gfx1030).

### In-model A/B

Pick a dense W4A16 checkpoint (AWQ or GPTQv1). The W4A8 path only fires
when the kernel name is `"prefill"` (M <= 256 with K >= 4096, or AWQ with
M > 256; see `_rdna2_w4a16_select_kernel` in
`vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py`).

```bash
# Baseline (env var unset = W4A16 path, default).
scripts/serve_gfx1030_flashnext.sh   # whatever launcher you use

# A/B: same launcher, with W4A8 prefill enabled.
VLLM_RDNA2_W4A8_SDOT4=1 scripts/serve_gfx1030_flashnext.sh
```

A meaningful A/B compares the prefill throughput and TTFT at M ~ 256,
K ~ 4096, N ~ 4096 (the W4A16 prefill bucket). Decode (M <= 32, K >= 4096)
routes to `gptq_gemm_rdna2` and is unaffected.

Sanity check: greedy outputs must match the W4A16 baseline. The W4A8
kernel's per-(token, group) scales are the A_GROUP variant
(`a8_lds_k32_ag`, config_id 8), so accuracy should be close to or better
than the per-token variant. The explore G1/G2 numbers measured per-(token,
G=32/64/128) PPL deltas against the fp16 W4A16 reference; the G2 (G=32 and
G=64) cells passed. See
`benchmarks/kernels/w4a8_sdot4_explore/test_reference.py` for the exact
tolerance.

## Env gates

| Env | Effect |
|---|---|
| `VLLM_RDNA2_W4A8_SDOT4=1` | Opt in to the W4A8 path. Default OFF (unset = W4A16). |
| (anything else) | No change vs `rdna_extras`. |

No new env vars in the platform selector (per AGENTS.md — config-propagated
only). No default behavior change. No changes to existing TUs, the
dispatcher for other kernel names, or the launchers.

## Known limits

- **split_k > 1 is OFF in the wired path.** The pk4 fp16 CAS epilogue is
  order-dependent (see
  `csrc/rocm/explore/w4a8_sdot4.cuh:atomic_add_f16x2/f16x4`); `W4A8_DEFAULT_SPLIT_K`
  is pinned to 1 in the dispatcher and the output buffer is allocated
  uninitialised (`torch.empty`). If you bump the default, pre-zero `out`
  first.
- **Act-order (`g_idx`) is rejected.** The W4A8 GEMM reads a contiguous
  A; an act-order pack would read out of permutation. `_rdna2_w4a8_eligible`
  returns False when `c.has_g_idx` so the dispatcher falls through to
  `gptq_gemm_rdna2_prefill`.
- **MT=8 hardcoded.** The wired path uses the `a8_lds_k32_ag` config
  (MT=8). Changing the recommended config requires updating
  `W4A8_DEFAULT_CONFIG_ID` and `W4A8_DEFAULT_M_TILE` in `rdna2_w4a16.py`
  together (they must match the Cfg template's M_TILE).
- **M threshold lives in the W4A16 selector.** The W4A8 branch only
  fires when `_rdna2_w4a16_select_kernel` already returns `"prefill"`. No
  additional M-vs-N branching in Python — the C++ side returns
  `kLdsTooBig` / `kBadShape` if the split/LDS budget doesn't fit, and the
  dispatcher falls back to the W4A16 path.

## Pointer back to the explore work

The explore-only tree lives at `/tmp/pr9_branch/` (extracted copy of upstream
branch `explore/w4a8-sdot4`):

- Explore kernel body (source of truth, unchanged):
  `/tmp/pr9_branch/csrc/rocm/explore/w4a8_sdot4.cuh`
  (copied verbatim into `csrc/rocm/w4a8_sdot4_rdna2.cuh` here).
- Explore C ABI (replaced by this branch's `.cu`):
  `/tmp/pr9_branch/csrc/rocm/explore/w4a8_sdot4_capi.cu`
- Explore ISA shim (CPU compilation without ROCm):
  `/tmp/pr9_branch/csrc/rocm/explore/w4a8_sdot4_isa_shim.h`
- Explore NumPy reference (CPU checks, no torch/GPU):
  `/tmp/pr9_branch/benchmarks/kernels/w4a8_sdot4_explore/reference.py`
- Explore G2 correctness tests:
  `/tmp/pr9_branch/benchmarks/kernels/w4a8_sdot4_explore/test_reference.py`
