# V620 FP16 TunableOp rows

Build-specific, **lookup-only** TunableOp solver rows for the gfx1030 serving
stack on 4× Radeon PRO V620. They were generated with the matching ROCm wheel
SDK. They do not quantize weights or activations. Solution IDs must never be
reused across a different rocBLAS build, even when version strings match.

## Canonical set

`rocblas-f30bb442e9b5/` is **the** canonical set shipped with the fork. It holds
**783 FP16 rows per rank** (identical on all four ranks) for the fork's serving
build:

| | |
|---|---|
| venv | `venv-7.14.0_0.28.0` |
| torch / HIP | `2.12.0+rocm7.14.0` / `7.14.60850` |
| rocBLAS | `5.5.0.cd957402` |
| `librocblas.so.5` sha256[:12] | `f30bb442e9b5` |

It is the shared set for all three serving families:

* **Flash-Next** — `Qwen3.8-Flash-Next-AWQ-W4A16`, MTP=0 and MTP=2 (includes the
  M ≤ 8 decode shapes and the M = 16/24/32 MTP-verify shapes).
* **27B AWQ dense** — the dense projection shapes at the aligned prefill grid.
* **EXL3 27B** — `Qwen3.8-27B-exl3-3.00bpw` (K = 5120/1536/4352; 0 collisions
  with the Flash-Next K set).

`provenance.json` in that directory records the library hash, package versions,
the capture/tune/curate campaign, and the lookup-hit proof. See its `README.md`
for the per-campaign deltas and regeneration recipe.

## Storage policy (mandatory)

* **One canonical folder per rocBLAS build hash**, named
  `rocblas-<sha256(librocblas)[:12]>/`. The helper selects the folder that
  matches the *loaded* build; solver IDs from another hash never validate.
* **Never** write rows to `/tmp` or the run CWD — both are wiped or vary between
  runs, which silently drops small-batch shapes back to rocBLAS heuristics.
* **Per-user fallback:** `$HOME/.cache/tunableop/tunableop_results.csv`. The
  helper uses it automatically when the loaded build has no rows, and it is where
  online tuning writes.

Consumption is a one-liner; the helper is lookup-only (`TUNING=0`) and keyed by
the rocBLAS build:

```bash
source <tree>/tools/rdna2_028/tunableop_env.sh
configure_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "<tree>/tunableop"
```

A healthy start logs:

```
TunableOp lookup enabled for rocBLAS build f30bb442e9b5 (rows: <tree>/tunableop/rocblas-f30bb442e9b5); tuning off ...
```

All serve launchers under `scripts/` and the in-tree bench/capture drivers that
start an engine go through this helper (or the `tools/rdna2/` mirror); a launcher
with a `TUNABLEOP=0` opt-out sets `PYTORCH_TUNABLEOP_ENABLED=0` and writes
nothing. `PYTORCH_TUNABLEOP_TUNING=1` re-tunes shapes missing from the table
(writes into the selected rows path; harvest into the canonical directory when
validated).

## Archived sets

`archive/` holds foreign-build row sets that are **not** the fork's serving set.
They are preserved (not deleted) because their solution IDs are only valid for
the exact build that generated them. Currently:

* `archive/rocblas-c27e2252cc7a/` — 70 rows from
  `torch 2.13.0+rocm10.0.0` / rocBLAS `5.6.0.8d1ae90e` (George's V620 test venv).
  Not usable on `f30bb442e9b5`; see its `README.md` for how to reactivate it if
  that build is ever served.

## Cache-aligned prefill

With automatic block sizing, Flash-Next TP4 conversation caching aligns
intermediate chunk ends to an 800-token recurrent-state grid. With a 4,096-token
scheduling budget, ordinary chunks therefore contain 4,000 tokens. Exact-shape
lookup cannot use 4,096-token rows for these chunks; the table carries the five
multiples of 800 up to 4,000 (and 1,024/2,048/3,072 for the 1,024 grid).

Use `--block-size 1024` with the 4,096-token MTP2 scheduling budget to retain
4,096-token ordinary chunks. Keep cache-aligned scheduling enabled; disabling it
breaks reusable conversation state. An 8,192-token budget on the 1,024 grid
additionally needs 5,120/6,144/7,168 rows.

The coverage check covers the known dense projection dimensions and aligned
chunks. Arbitrary prompt tails, mixed batches, or model/runner changes can still
introduce other shapes. Re-run the cold-context performance and prefix-reuse
checks before promoting any later optimization, and retain the previous release's
source, environment, tuning files and launch command for rollback. A different
rocBLAS hash requires a separately qualified table.

## Regenerate for another rocBLAS build

Run only with the inference service stopped, using the isolated testing
environment and its matching SDK library path. The tuning scripts deliberately
remove the serving environment's TunableOp enable/tuning overrides before
importing Torch: those environment variables take precedence over the Python API.

```bash
.venv/bin/python tools/rdna2/tune_v620_fp16.py \
  --output-root /path/to/testing/tunableop \
  --batch-tokens 800 1024 1600 2048 2400 3072 3200 4000 4096 8192
.venv/bin/python tools/rdna2/qualify_v620_fp16.py /path/to/testing/tunableop/rocblas-HASH
```

The first command tunes GPU 0 and records solver rows after every measured shape.
The second replays them on all four cards against FP32 references before writing
the other three serving files. Run the full model's output checks and benchmark
after qualification. Tuning may choose a different floating-point reduction order;
retaining FP16 weights does not imply bit-identical outputs.

For the current fork pipeline (capture → tune → curate → freeze + lookup-hit
proof) see `tools/rdna2_028/tunableop_rows_pipeline.sh`,
`tools/rdna2_028/curate_tunableop_rows.py` and the per-campaign sections in
`bench_results/`.
