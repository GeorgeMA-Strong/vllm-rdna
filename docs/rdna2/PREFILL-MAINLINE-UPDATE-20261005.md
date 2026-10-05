# Mainline Qwen4Exp update — 2026-10-05

Read-only upstream review for the V620 / gfx1030 FP16 AutoRound service. No
server changes or GPU measurements were made for this note. Target mainline
reference: [`4a30c4cad01401d778adb03b029d9da767c59fed`](https://github.com/vllm-project/vllm/commit/4a30c4cad01401d778adb03b029d9da767c59fed).

## #59533: merge QSA and indexer projections

The October 5 merged [PR #59533](https://github.com/vllm-project/vllm/pull/59533)
([commit `042ab0305`](https://github.com/vllm-project/vllm/commit/042ab0305cc4215e2c6fb215f1d6f2e65dd7d4ac))
packs the main QSA Q/gate, K and V projection together with the replicated
indexer Q/K projection. At TP4, its separate `[3584,2560]` and `[640,2560]`
BF16 weights become one `[4224,2560]` GEMM. The output is split into views;
the fused QSA prepare kernel is taught the resulting row strides and uses i64
offsets. This is **projection/GEMM fusion**, not fusion of the downstream sparse
attention, PLE, GDN, or expert layers. LoRA retains separate projections.
[Source diff](https://github.com/vllm-project/vllm/commit/042ab0305cc4215e2c6fb215f1d6f2e65dd7d4ac).

The PR changes only `vllm/models/qwen4_exp/nvidia/*` plus tests. Its
low-latency GEMM plans are for NVIDIA SM90, SM100, SM103 and SM121; none is a
gfx1030 plan. An upstream checkout/upgrade will therefore **not switch the
V620 AMD model to this fused projection**.
[Changed files](https://github.com/vllm-project/vllm/pull/59533/files),
[dispatch source](https://github.com/vllm-project/vllm/blob/042ab0305cc4215e2c6fb215f1d6f2e65dd7d4ac/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py).

The author reports on **one GB300, BF16, TP4**, projection-only speedups of
1.49× at M=1, 1.45× at M=4, 1.47× at M=256, 1.41× at M=1024, **1.00× at
M=4096**, and 1.03× at M=16384. This is a small-step/short-prefill win, not
evidence of a broad 4K-chunk prefill-compute win. On 4× GB300, BF16, TEP4,
MTP3, 8K-input/1K-output SPEED-Bench, output throughput changed −0.4%, +1.5%,
+2.7%, +5.1% at concurrency 1,4,16,64 respectively; the PR states one sweep
per arm and no run-to-run variability estimate. No AMD, AutoRound, FP16, or
200K-context result was reported. It compared projection outputs with FP32
reference at rtol=.02/atol=.01, **not bitwise/strict FP16 equivalence**.
[Author's measurements and test protocol](https://github.com/vllm-project/vllm/pull/59533).

### AMD feasibility and exact hazards

The current AMD path still has a sharded `QKVParallelLinear` in
[`amd/qsa.py`](../../vllm/models/qwen4_exp/amd/qsa.py) and a separate
`ReplicatedLinear` indexer in
[`amd/indexer_qsa.py`](../../vllm/models/qwen4_exp/amd/indexer_qsa.py).
There is no duplicate AMD projection fusion here. The upstream design is
conceptually portable, but the implementation is not a file-copy: it depends
on the NVIDIA-specific QSA prepare and LL-GEMM dispatch. For the local path,
it would require a merged quantized linear that handles *different TP policies*
within one output: Q/gate sharded by rank, K/V replicated where KV heads are
fewer than TP ranks, indexer Q/K replicated on all ranks. The upstream custom
loader remaps `indexer.index_qk_proj` to shard 3 and temporarily overrides TP
rank for shards 1/2/3. Both legacy `weight_loader` and v2 loader are handled.
This exact mapping, packed scales/zero-points, AutoRound/INC layer exclusions,
and multimodal/LoRA fallback must be tested, not inferred from shape match.
[Upstream loader](https://github.com/vllm-project/vllm/blob/042ab0305cc4215e2c6fb215f1d6f2e65dd7d4ac/vllm/models/qwen4_exp/nvidia/qsa.py),
[model packed mapping](https://github.com/vllm-project/vllm/blob/042ab0305cc4215e2c6fb215f1d6f2e65dd7d4ac/vllm/models/qwen4_exp/nvidia/model.py).

For this V620 service, the ordinary activation interface is FP16 and its
AutoRound QSA/indexer projections may use W4A16 methods, unlike the author's
BF16 low-latency GEMM test. Even if the arithmetic is mathematically the same,
a larger merged GEMM can select a different kernel/reduction schedule and
therefore round FP16 outputs differently. A strict-FP16 gate is required;
do **not** relax tolerances or lower precision to claim a speedup. An honest
candidate experiment would first compare separate vs merged on actual local
weights at TP4, M=1/4/64/256/1024/4096 and mixed short-tail lengths, then
evaluate 64K/128K code prompts, resumed prefix/tool turns, and 200K context
before any deployment. The author benchmark's M=4096 neutrality makes this
lower priority than a measured, repeated per-layer prefill hotspot.

## Other merged changes: relevance and non-duplication

[`4f52fa35e` / #59990](https://github.com/vllm-project/vllm/commit/4f52fa35efc7c56b77221460346e0e9eea28638b)
fixes AutoRound/INC model construction: if `INCConfig` declares PLE unquantized,
the *shared* Qwen4Exp PLE embedding picks the BF16 unquantized method instead
of rejecting the quant config. This is important compatibility for the exact
Swift AutoRound family, but **not a prefill performance optimization**. The PR
reports the real-weight startup not reaching readiness because weights could
not be downloaded; dummy-weight serving and a unit test passed. This fix
is newly available to both NVIDIA and AMD code paths *if the common embedding
module is imported*. This fork has a separate AMD PLE implementation and no
`qwen4_exp/common/ngram_embedding.py`, so check its existing PLE/INC routing
before considering a port. Avoid changing the existing BF16 CPU PLE table.
[PR and validation caveat](https://github.com/vllm-project/vllm/pull/59990).

[`0c16eee3f` / #57816](https://github.com/vllm-project/vllm/commit/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a)
resets **SimpleCPU eager-store** GPU block placement cursors when an MRV2
request is preempted, so a resumed request stores the confirmed tail. It does
not change PLE offload, cache lookup, lazy mode, or eviction, and has no direct
global prefill compute benefit. It is a correctness item for anyone adopting
mainline SimpleCPU offload; compare with this fork's current RAM offload
mechanism before backporting.
[PR scope](https://github.com/vllm-project/vllm/pull/57816).

Related work is already merged: [#57097](https://github.com/vllm-project/vllm/pull/57097)
fuses the **subsequent QSA prepare** operations, not the projection GEMM;
[#59214](https://github.com/vllm-project/vllm/pull/59214) tunes NVIDIA SM100
decode GEMM plans. [#59431](https://github.com/vllm-project/vllm/pull/59431)
handles unquantized PLE under compressed-tensors, analogous but distinct from
INC/AutoRound #59990. None is the missing V620 FP16 W4A16 merged-projection
implementation. The [prior local prefill research](PREFILL-MAINLINE-RESEARCH-20261004.md)
already discusses AMD PLE/QSA kernel opportunities; do not count those as a
new speedup from #59533.

**Assessment:** #59533 is worthwhile as an implementation pattern to revisit
after profiling. Its only directly demonstrated E2E speedups are on NVIDIA;
the projection-only M=4096 result is neutral. For the requested *global
prefill compute* gain on FP16 gfx1030, it is not yet a proven top candidate.

## Local follow-up experiments (not upstream claims)

RDNA base `121af1b4786feb60748afe2fd73c2a49d459dc8b` includes PR36's adaptive
scheduler. It is incorporated with `VLLM_RDNA_DYNAMIC_PREFILL=0` for these
compute comparisons. No precision, MTP, KV-capacity or MM-limit changes.

The staged tile8 HIP kernel (`9070cf3f2`) passes24 weighted/reduced GPU cases
at existing tolerances. Component medians8.424/8.457ms versus prior9.093/9.119ms
are approximately7.3% faster for that expert helper only. Compiler metadata
shows115/116VGPRs, zero scratch, versus125 previously. There is no new
full-model speed result. Native SHA256:
`204deb40c50a10ca1390d857ec8cf506cae62a581a85a42ab35a64a09bfc4ee9`.

PLE lifetime release, direct packing and single-history layout preserve the
tested FP16 arithmetic/state. The full14-case AMD PLE suite passed after
direct packing; the six changed-path CPU/GPU cases pass after single-history
commit `f83835a7f`, with exact output/state equality. Norm code is unchanged.

None of these changes yet makes an8192 scheduled batch safe under the fixed
production MTP2,3.75GiB KV/GPU, full MM profiling and64GiB RAM offload settings.
The16K cold-prefill reproduction fails successively at packing, history and
convolution output allocation; the last requests160MiB on some ranks120MiB.
The guarded normal coding benchmark did not run after the last failure.
**No heavy prefill gain,8192 qualification or production promotion is claimed.**
Raw local artifacts and detailed experiment history:
[status](/Users/georgezagraid/Projects/AI/v620-vllm/review-artifacts/prefill-compute-2026-10-04/STATUS.md).
