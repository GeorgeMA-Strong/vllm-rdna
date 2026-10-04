# Experimental resident MoE prefill tile16

`VLLM_RDNA_MOE_PREFILL_TILE=16` or `32` reuses each dequantized W4 weight across
16 or32 token rows rather than8 (the default). These are opt-in native HIP variants, not a
Triton kernel or an activation-precision change. The existing FP16 activations,
W4 weights, scale/zero handling, per-row dot-product order, split-K and selected
FP16/FP32 accumulation mode remain unchanged. No dense expert-weight cache is
introduced.

Default remains8. Dispatch requires the existing qualified Intel geometry:
at least 4096 rows, FP16, hidden2560, intermediate640, topk10, local/global
experts128/512 and group128 for both projections. All other configurations and
decode retain the original route. This experiment does not change graphs,
batch size, scheduling, MTP, PLE or the 64GiB RAM KV cache.

The regression tests were run before implementation and failed with unsupported
block_size_m16/32. After the isolated native rebuild, all24 targeted comparisons
pass: both expert projection shapes, both accumulation modes, unweighted output
and weighted top-k reduction, and tile8/16/32 versus tile4. The existing tolerance
was not changed. These are kernel tests, not model-quality clearance or evidence
of a whole-model throughput improvement.

The first tile16 variant spilled116–120bytes per thread and lost5.57% whole-model
prefill against bracketing controls. Staging one packed weight word at a time
eliminated those spills. The current wide variant uses two output columns per
thread, retaining four for original tiles, and FP16 pair CAS rather than the
original quad CAS. Per-element rounding and selected accumulation dtype stay
the same; reduction order remains nondeterministic as in the original kernel.
At4096 synthetic rows the complete expert-only helper measures tile16=8.666ms
against tile8=9.093/9.119ms (about5% component gain). Actual loaded TP4 and normal
generation tests must establish whether there is a useful model-level gain.
Padding-aware tile32 measures20.501ms and is rejected, not a serving setting.

The measurement extension compares actual loaded resident weights/routing, then
performs serialized tile8/tile16/tile8 full-prefill stage trials. Only normal
llm-context-bench regular/coding generation, concurrency and model-quality checks
can qualify a production change. More weight reuse can lose to register pressure
or padding, so a larger tile is not inherently faster.

`tools/rdna2/build_prefill_moe_incremental.py` verifies that all other native
sources match the qualified reference, rebuilds the changed object and links
only the experiment's library. It checks that the reference library's SHA256
did not change. Source deployment must still use local commit/push and server
Git pull; never edit serving source directly on the server.
