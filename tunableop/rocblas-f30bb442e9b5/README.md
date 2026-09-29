# FP16 TunableOp rows — rocblas-f30bb442e9b5

Build-specific rocBLAS solver rows for the serving stack on 4× V620 (gfx1030),
captured during Flash-Next serving with the 0.28.0 tree. They cover the dense
FP16 projections at the MTP verify batch sizes (M = 16/24/32), decode and
chunked-prefill shapes. Without them those shapes land on latency-bound small-K
rocBLAS kernels (`Cijk_MT32x32x8`, K-depth 8 → 320 K-iterations, measured
< 1 TFLOPS), the largest single item in the MTP step budget.

The directory hash is the first 12 hex chars of the `librocblas.so.5` sha256
(see `provenance.json`). 903 rows per rank; all four rank files share the same
shape keys (solver choices are per-card).

## Storage policy (mandatory)

- **Never** store rows in `/tmp` or the run CWD. Both are wiped / vary between
  runs, which silently drops c=1 and small-batch shapes back to rocBLAS
  heuristics.
- **This directory is the shared source of truth.** It ships with the fork
  (`opengfx1030/vllm-rdna`) so every user gets the same tuned rows.
- **Per-user fallback:** `$HOME/.cache/tunableop/tunableop_results.csv`. The
  helper uses it automatically when this build has no rows, and it is where
  online tuning writes.

## Use

Consumption is a one-liner — source the helper and point it at this build:

```bash
source tools/rdna2_028/tunableop_env.sh
configure_tunableop "$HOME/Apps/vllm/venv-7.14.0_0.28.0/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librocblas.so.5" \
                    "$PWD/tunableop"
```

It selects `tunableop/rocblas-<hash>/` from the loaded rocBLAS build and sets a
**lookup-only** environment (`PYTORCH_TUNABLEOP_ENABLED=1`, `TUNING=0`,
`PYTORCH_TUNABLEOP_FILENAME` at the rank files). The launchers
(`scripts/serve_gfx1030_flashnext_mtp.sh`, `scripts/serve_gfx1030_27b_dense.sh`,
`scripts/serve_gfx1030_flashnext.sh`, `scripts/serve_gfx1030_full.sh`) already
do this. A healthy start logs:

```
TunableOp lookup enabled for rocBLAS build f30bb442e9b5 (rows: .../tunableop/rocblas-f30bb442e9b5); tuning off ...
```

- `PYTORCH_TUNABLEOP_TUNING=1` re-tunes shapes missing from the table (writes
  into the selected rows path; harvest into this directory when validated).
- If the rocBLAS hash differs, the helper warns and falls back to
  `~/.cache/tunableop/`; serving continues with default FP16 algorithms.

## Measured (4× V620, TP4, Qwen3.8-Flash-Next-AWQ-W4A16, MTP=2, c=8, seed 12345)

| cell | MTP-0 | MTP-2 (no rows) | MTP-2 + these rows |
| --- | ---: | ---: | ---: |
| 8×1k/512 | 166.3 tok/s / 40.8 ms | 129.4 / 50.1 | 135.5 / 43.4 |
| 8×16k/1k | 67.3 / 75.6 | 58.7 / 84.7 | **73.8 / 64.9** |

In-protocol A/B (same driver, same protocol, rows the only change): 140.5 vs
114.8 tok/s (+22.4 %). These rows require the draft-decode cudagraph fix
(`30632b2fa`, branch `rdna_extras`) to be present.

## Caveats

- Solution IDs are build-specific: never reuse across a different rocBLAS build
  (compare the library sha256 in `provenance.json`).
- Tuned algorithms may change the floating-point reduction order. Outputs are not
  bit-identical and greedy acceptance shifts slightly (observed 1.81 vs 2.43 at
  8×1k/512). Validation here was 8/8 cells plus coherent probes, not bit equality.
- Cold tuning is impractical online (>20 min of inline tuning per workload);
  ship these rows instead.

## Regeneration

```bash
# harvest the per-user cache after a tuning-enabled serving run:
#   $HOME/.cache/tunableop/tunableop_results{0..3}.csv
# then verify lookup beats the heuristic and copy into this directory.
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 <launch>
```

The historical leak this directory fixes wrote to `/tmp/tunableop_results{0..3}.csv`
(the PyTorch default name) from drivers that enabled TunableOp without setting
`PYTORCH_TUNABLEOP_FILENAME`. All launch paths now source the helper, so no
script can leave rows in `/tmp` or the CWD.
