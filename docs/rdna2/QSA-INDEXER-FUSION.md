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
