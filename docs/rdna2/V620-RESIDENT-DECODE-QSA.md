# V620 TP4 served candidate: resident decode and bounded QSA

Source lineage: `82926d228` adds the resident kernel, and `46a8ec210`
records the persistent service. TP4, pipeline size 1 only.
No model, dense precision, MTP2, or power-limit changes.

As checked on 2026-09-22, `v620-tp4-fast.service` is enabled, active, and
healthy on the V620 host. It loads source from
`/home/george/v620-experiments/decode-qsa-20260920/source`. The host source is
an export without Git metadata; its code is closest to `46a8ec210`, with
formatting and small build-source differences. The service launcher is saved
as `tools/rdna2/serve_v620_tp4_fast.sh`. This branch also contains newer
upstream `rdna_extras` changes and must be qualified before redeployment.

`VLLM_RDNA_MOE_RESIDENT_SKINNY=1` enables the new kernel for supported
contiguous FP16 SILU inputs with one to four rows. Default is off; other
cases retain the tiled path. It consumes the existing resident INT4 words
and scales. FP16 dequantization/activation storage and FP32 accumulation
remain; summation order differs, so bitwise equality is not promised.

The QSA bound applies only to prefill. Decode and missing metadata retain
the capacity-wide path, preserving graph dimensions.

## Isolated validation

Six native reference and graph replay cases pass, covering model dimensions,
expert-map holes, repeated routes, changed inputs/routing, and no local experts.
Four QSA bound cases and three metadata tests pass. Two unrelated CPU dispatch
tests fail identically against the untouched baseline.

EP4 microbenchmark with unique top-k routes and 256 MiB cache flush:

| Rows | Warm tiled/new (ms) | Cold tiled/new (ms) |
| --- | --- | --- |
| 1 | 0.05384 / 0.05036 | 0.09592 / 0.08524 |
| 3 | 0.06120 / 0.05768 | 0.11292 / 0.09440 |
| 4 | 0.06240 / 0.06052 | 0.10952 / 0.09536 |

This fixture has four local experts. These are kernel timings, not whole-model
token rates.

QSA tests use 256 FP16 query rows, 65,536 compressed capacity columns, and
32K/64K/128K original-token contexts. Bounded/unbounded times were 1.892/2.908,
0.908/2.939, and 1.633/3.142 ms. Selected-token sets match exactly; ordering
can differ.

## Reproduction and isolation

Candidate: `/home/george/v620-experiments/decode-qsa-20260920`.
Protected default: `/home/george/v620-experiments/prefill-20260919`.
The candidate has separate source, native library, caches, launch scripts,
logs, and results. `build-native.py` reuses existing compiler/link settings
and hashes the protected library before/after to verify it stays unchanged.

Host artifacts:

- `serve-tp4-checkpoints.sh`: TP4, pipeline 1, EP4, MTP2, FP16 dense, original
  CPU PLE, 4096-token batches and 4 GiB KV allocation per GPU.
- `start-combined.py`: isolated launch, overlap and thermal guards, startup
  timing, sanity prompt and prefix-reuse check.
- `bench.sh`: unchanged context benchmark 92286b2, prose/code 16K/32K with
  unique cache salts through the loopback proxy.
- `check-model.py`: six deterministic sanity prompts versus saved baseline.

Combined full-model validation and raw results are recorded alongside these
artifacts. The stable boot service remains preserved during qualification.

## Combined run status

The combined TP4 service started in 373.879 seconds. Six deterministic prompts
matched the saved baseline, and both ordinary and tool follow-ups reused
cached context. Valid 16K/32K prefill cases reached 2011–2036 tok/s versus
1529–1543 tok/s in the immediately collected baseline. End-to-end decode
results were mixed; a decode throughput gain is not established.

The full 64K/128K run completed. Prose prefill was 1974/1859 tok/s and decode
59.37/53.86 tok/s. Coding prefill was 1985/1832 tok/s and decode 70.63/72.49
tok/s. Coding 32K received the same repetition flag as the baseline.
Temporary SSH/ARP loss interrupted observation, but the host returned with
unchanged uptime and the benchmark completed remotely. Recovered logs show no
service error, preemption, or overlapping requests. The resident kernel remains
opt-in in code, while the current fast service enables it. These single-sample
results do not establish a consistent end-to-end decode gain.
