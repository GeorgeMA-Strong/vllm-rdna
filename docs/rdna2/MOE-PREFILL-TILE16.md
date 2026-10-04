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

The regression test was run before implementation and failed with unsupported
block_size_m16. After the isolated native rebuild, all eight comparisons passed:
both expert projection shapes, both accumulation modes, and tile8/tile16 versus
tile4. The existing tolerance was not changed. These are kernel tests, not
model-quality clearance or evidence of a whole-model throughput improvement.

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
