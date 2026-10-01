# Archived — rocblas-c27e2252cc7a (foreign build, not the fork's serving set)

This directory is **not** part of the fork's canonical TunableOp set. It was
imported from PR #5 (commit `ca83d922a`) as a reference copy of the
Leapdragon/George V620 table and is kept here only so the data is not lost.

## Why it is archived, not merged

| | this archive | canonical `../rocblas-f30bb442e9b5/` |
|---|---|---|
| rocBLAS build | `5.6.0.8d1ae90e` | `5.5.0.cd957402` |
| librocblas sha256[:12] | `c27e2252cc7a` | `f30bb442e9b5` |
| torch | `2.13.0+rocm10.0.0` | `2.12.0+rocm7.14.0` |
| HIP | `7.15.26333` | `7.14.60850` |
| venv | `/home/george/v620-vllm-testing/.venv` | `venv-7.14.0_0.28.0` |
| rows / rank | 70 | 783 |

TunableOp solution IDs are **build-specific**: the solver numbers in these rows
are only valid for the exact `librocblas.so.5` that generated them. They cannot
be folded into the canonical set — the two builds do not share a hash key, and a
row from one build will not validate on the other.

## How to reactivate

The helper (`tools/rdna2_028/tunableop_env.sh`) is hash-keyed: it reads
`<rows root>/rocblas-<sha256(librocblas)[:12]>/`. If you actually serve the
`5.6.0.8d1ae90e` build, move this directory back into the rows root so the
lookup finds it:

```bash
git mv tunableop/archive/rocblas-c27e2252cc7a tunableop/rocblas-c27e2252cc7a
```

Until then it stays archived so `tunableop/` shows exactly one canonical folder.

Provenance is preserved verbatim in `provenance.json`.
