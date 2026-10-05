# Experimental FP16 QSA indexer fusion

`VLLM_RDNA_QSA_FUSED_INDEXER=1` opts into a Triton preparation kernel for the
AMD Qwen4Exp indexer. The default is **off**. Keep every other serving parameter
unchanged when evaluating it; do not treat the isolated kernel speedup as a
whole-model speedup or a production recommendation.

The kernel combines query normalization/RoPE, key-group compression,
compressed-key normalization/RoPE, and raw/compressed cache writes. Projection
and token selection are unchanged. It adapts
[upstream #57947](https://github.com/vllm-project/vllm/pull/57947) to the fork's
FP16 arithmetic and packed position-state cache, rather than adopting the
donor's BF16 policy.

## Eligibility and precision

The opt-in requires ROCm gfx10x, FP16, four 128-dimensional query heads, one
KV head, power-of-two compression greater than one, and interleaved NeoX
MRoPE with 64 rotary dimensions. Unsupported layouts retain the old path.
The current GPU qualification uses compression ratio four, including ring
sizes four/eight and MRoPE sections `[11,11,10]`. Other eligible compression
ratios are not claimed qualified by those tests.

The feature does not change model weights, expert quantization, activation
precision, CPU PLE, speculation, cache capacity, or image support. In particular,
it does not introduce INT8 shadows or W4A8.

The FP16 port retains the existing sequential FP32 compression sum and the
FP16 rounding boundary before key normalization. Folded normalization weights
are cached only after checkpoint loading. The variance reduction and OCML
reciprocal square root match the HIP norm. The compressed-key rotary expression
retains the existing single-head FP16 contraction order.

One compression CTA per request owns historical ring reads and suffix writes;
other compression CTAs read current-chunk inputs. Paged strides and the packed
three-int64 MRoPE position tail are respected. GPU work metadata and stable
cache addresses are retained for graph replay.

## Checks and measurement boundaries

Run the GPU correctness suites with no model process occupying the device:

```bash
python -m pytest tests/models/qwen4_exp/test_qsa_pre_indexer.py \
  tests/models/qwen4_exp/test_qsa_amd.py -vv
```

AMD comparisons require bitwise-identical query/cache values and raw packed
position state; selected sets are compared without requiring atomic arrival
order. Coverage includes mixed prefill/decode, 1/3/6/9/12 decode rows, strided
pages, fresh/tiled prefill, near-256K positions, and changed-input graph replays.

The unchanged native selector is not generally deterministic at tied cutoffs.
A zero-filled long historical cache produced different selected sets even
between repeated reference calls with identical logits. The long-context
fixture therefore initializes historical keys and preserves that snapshot
across replays. Its strict selected-set assertion is not relaxed. No selector
determinism fix is part of this feature.

`benchmarks/kernels/benchmark_rdna_qsa_pre_indexer.py` measures captured
preparation only: no projection, selection, or remaining model layers. Serving
qualification must separately establish activation on every rank, graph hits,
counter-free single/concurrent PP and TG, paired full-model quality, tool turns,
prefix reuse, RAM reload, and image restoration. Keep rejected benchmark rows
and use identical fresh request scopes across matched arms.

## V620 qualification outcome (2026-10-05)

The first matched full-model trial does **not** establish a consistent speed
benefit. Keep the opt-in off. Serving/native code was frozen at `d79f66186e`;
later commits strengthened tests and documentation only. The control used the
same serving code with this flag off, with source based on `rdna_extras`
`121af1b478`. This is not a fresh comparison against an older production build.

Both arms kept the normal Intel AutoRound checkpoint, FP16 body, original BF16
CPU PLE, TP4/EP4/PP1, MTP2, context 262144, batch 4096, 64 GiB RAM offload,
full image/video profiling, and FULL decode graphs `[3,6,12]`. Native all-reduce
was active on every rank; the candidate used all twelve indexer fusions per
rank. The launch manifests differed only in the fusion flag.

Unmodified llm-context-bench coding, normal EOS, 1024 output tokens, three
valid repetitions per displayed row, with matched input hashes, tags, sampling,
prompt/output counts, and harness hash:

```text
Context      Control PP/TG      Fusion PP/TG       PP delta    TG delta
16K C1       1983.15 / 69.82    1978.64 / 67.49      -0.23%      -3.33%
16K C3 agg   1606.99 / 53.92    1611.04 / 52.60      +0.25%      -2.46%
64K C1       1921.61 / 71.02    1922.15 / 71.53      +0.03%      +0.72%
128K C1      1804.67 / 68.66    1804.78 / 70.93      +0.01%      +3.31%
```

Rates are tokens/second. Nominal context tiers contained approximately
18K/72K/144K actual prompt tokens. C3 is group aggregate, including mixed
prefill stalls and uneven completion, not per-chat or pure decode throughput.
The original control C3 speed report and candidate C3 diagnostic response
failed the unchanged repeated-token guard; both were preserved and rejected.
The displayed C3 pair used a separate matched retake, not relabeled failures.

Captured preparation alone improved 5.27–11.69x. Separate rank-zero target
graph spans fell 2.04% for three rows and 0.53% for twelve rows, including GPU
dependency waits. Neither measurement implies a whole-model speedup.

The final FP16 bit-pattern/cache/graph suites passed 25 tests with 16
platform-specific skips. The paired full-model quality suite completed all
18 requests on each arm: control 11/18, candidate 12/18, with zero newly failed
case verdicts. Baseline formatting/arithmetic failures remain; this is neither
a blanket quality pass nor proof of improved quality from one paired trial.

Prefix reuse, three approximately 99K chats with tool continuations, text RAM
restoration, and image RAM restoration passed. Restoring the approximately
200K text session read 11.47 GB from RAM with 2.185 s time to first token;
restoring the approximately 119K two-image session after 219K text pressure
read 6.89 GB with 1.916 s time to first token. Those are functional turnaround
measurements, not decode benchmarks or guarantees for every workload.
