# Precision-preserving prefill compute candidates — 2026-10-04

Scope: four V620 gfx1030 GPUs, TP4/EP4, Intel AutoRound W4A16 body,
existing FP16 GPU activation interface, original BF16 CPU PLE table, 64GiB RAM
KV. No activation/PLE quantization, INT8 shadows, MTP tuning, or scheduling
fairness changes. Research only; no server actions or fresh measurements.

Pinned RDNA baseline: `a6ab43cf9661935e05f2a601b8fc1336d6c42491`.
Pinned vLLM mainline: `155488d853a0bc42df227dbfc74005b3fd488e94`.
The official [v0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)
was published 2026-09-22. Findings below distinguish merged work from open PRs;
release numbering alone does not imply the AMD model uses NVIDIA optimizations.

## Shortlist

| Rank | Candidate | Actual compute removed | Practicality / limit |
| --- | --- | --- | --- |
| 1 | Row-shard replicated HC prefill dense computations | TP ranks compute different token rows of both FP16 down/up GEMMs instead of repeating them at every attention/MLP HC boundary | Strongest global target; gather only final block inputs/injection, not wide HC gates. Requires collective-cost and rounding validation. |
| 2 | Profile and optimize large resident W4A16 routed MoE/GEMM | Improve large-batch expert math/data movement rather than metadata | Only if trace shows dominance. Existing tile8 is already present; no new mainline gfx1030 replacement demonstrated. |
| 3 | Port fused PLE gate to AMD | Three norms and dot/gate temporary tensors become one kernel | Secondary: previous port failed strict FP16 comparison; no substantial model gain promised. |
| 4 | Fuse PLE short-conv residual additions | Standalone gated/outer residual memory passes | Small, additive; preserve both FP16 rounding boundaries and state semantics. |

These are ranked by useful implementation order, not an invented performance
percentage. First capture a warmed 4096-token-step trace that attributes time
to PLE lookup/copy, projections, gate, short-conv, GR, MoE, QSA and collectives.
The prior matched runtime measured roughly 1900 PP tokens/s at 64K and 1800 at
128K; latest RDNA was essentially flat. Those are whole-model observations,
not a component profile. [Local validation](/Users/georgezagraid/Projects/AI/v620-vllm/review-artifacts/rdna-extras-updates-2026-10-04/VALIDATION-STATUS.md).

## Global target: replicated HC down/up GEMMs

The AMD HC merged down/injection linear uses `disable_tp=True`; the up linear
is `ReplicatedLinear`, both explicitly `quant_config=None`. Its existing RDNA
op accelerates small decode rows, but prefill still executes two ordinary
`F.linear` calls, SiLU and gate mixing. That is duplicate FP16 computation on
every TP rank at attention/MLP HC boundaries throughout the model.
[Weights and sites](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/models/qwen4_exp/amd/hyperconnection.py#L120-L153),
[prefill fallback](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/model_executor/layers/rdna_ops.py#L54-L87).

A contained prototype can keep residual/state replication intact: slice token
rows per rank, execute the same FP16 down/SiLU/up/gate math, then gather only
narrow block inputs/injection logits in original order. Avoid gathering the
wide HC gate. This need not port the entire open sequence-parallel model or
PLE caches. Different GEMM M-shapes can select different rocBLAS algorithms
and rounding despite unchanged precision; strict reference/model evals remain
necessary. Unequal tails, mixed decode/prefill, capture and rank eligibility
must never cause mismatched collectives. Existing ROCm10 tuning rows may not
cover new shard M-shapes.

This is a source-based experimental proposal, not measured performance. It is
the most plausible global target identified here, conditional on proving GEMM
savings exceed PCIe gather cost; full residual SP can be considered afterward.

## 1. Mainline PLE gate fusion is missing from the AMD serving path

Before choosing this path, note the earlier local gate port was removed because
it failed strict FP16 comparison. Grouped-normalization fusion gave only about
0.5–0.8% prefill in historical16K trials. Do not weaken tolerances or repeat a
rejected port while claiming a large new win.
[Gate rejection](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/docs/rdna2/V620-FP16-PERFORMANCE.md),
[norm results](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/docs/rdna2/V620-BASELINE-PORT-20260922.md#fused-ple-grouped-normalization).

[PR54517](https://github.com/vllm-project/vllm/pull/54517) merged on September 2
as `f870b9297685bd3c968063b52b769315cb08fd7f`. It adds fused n-gram IDs,
PLE gate and dilated convolution, and merges key/value projections, in the
**NVIDIA** implementation. The current pinned mainline gate accepts both FP16
and BF16; its intermediate casts are explicit to match eager tensor boundaries.
[Pinned gate source](https://github.com/vllm-project/vllm/blob/155488d853a0bc42df227dbfc74005b3fd488e94/vllm/models/qwen4_exp/nvidia/ops/ple.py#L184-L281).

RDNA AMD still performs separate key/value projections, key/query norms, the
product/reduction/scale/sign/sqrt/sigmoid chain, gated-value multiplication and
third norm. It then invokes short-conv and adds the gated residual. The fused
NVIDIA op is not used by that AMD forward.
[AMD PLE forward](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/models/qwen4_exp/amd/ple_layer.py#L1312-L1349),
[replicated projections](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/models/qwen4_exp/amd/ple_layer.py#L612-L626).

The author measured the gate at 4096 tokens as 76.9us versus compiled106.1us
and eager1567.9us on GB300. Its *combined* changes reduced TP4 TTFT about
6–8.5% on that NVIDIA setup; the large microkernel factors are not whole-model
speedups and are not V620 predictions. A port should disable NVIDIA PDL calls,
retain input strides and all FP16 materialization casts, and compare against
the actual AMD path with independent expected math. Do not blindly port the
whole PR: CPU BF16 n-gram offload is different, and RDNA short-conv already has
native HIP paths. Merging quantized projections additionally requires exact
checkpoint-name, scale, group and zero-point handling; gate fusion does not
need that loader change. [PR54517 validation](https://github.com/vllm-project/vllm/pull/54517).

## 2. Precision-preserving residual fusion is also NVIDIA-only upstream

[PR55309](https://github.com/vllm-project/vllm/pull/55309) merged September 14.
It fuses outer PLE residual addition into convolution and optional QSA output
gate into attention, explicitly retaining original BF16/FP16 rounding
boundaries. NVIDIA B200 prefill PLE microbenchmarks improve 1.15–1.29x for that
component; QSA's large-row benefit is about 1%, not a prefill breakthrough.

AMD still computes the outer `hidden_states + self.ple(...)` separately,
following PLE's `gated_value + conv_output`. Both add boundaries must survive
a fused kernel: first round gated+conv to the existing activation dtype,
then add outer residual and round again. Preserve short-conv state updates,
accepted-token rollback, ragged requests and cache restoration.
[AMD model forward](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/models/qwen4_exp/amd/model.py#L299-L323).

## 3. GR/PLE sequence parallelism is a real structural idea, not an available flag

[PR56322](https://github.com/vllm-project/vllm/pull/56322) remains **OPEN** at
head `1a89be3a34165b28af4b0c833255dd6285df6581`. It shards token rows for
replicated GR/PLE while keeping projection weights replicated. Its PLE path
all-gathers normalized rows before convolution because convolution state is
request-wide. It exposes `--enable-hc-sp`; requirements include PP1 and MoE,
with no external LoRA on sharded GR/PLE projections. It does not lower precision.

Author-reported BF16 CPU-PLE TP2 tests show about 8% prefill TTFT improvement,
not a 2x/4x model speedup. Decode does not improve consistently across the
reported parallel configurations. These results are not our Intel W4A16 V620
profile. The proposal touches NVIDIA/common code, not the AMD model, which
currently explicitly rejects sequence-parallel MoE. Enabling an env variable
alone cannot make it work.
[AMD rejection](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/models/qwen4_exp/amd/model.py#L162-L198),
[SP proposal diff](https://github.com/vllm-project/vllm/pull/56322/files).

Our PP1/TP4 satisfies only some topology requirements. A port must account for
CPU offload token ordering, n-gram context, per-request convolution caches,
TP-replicated states, global token offsets, unequal tails, RAM-prefix reloads
and additional PCIe collectives. A smaller row-sharded PLE projection/gate
prototype, followed by gather before existing conv, may isolate the opportunity
without immediately changing all GR/MoE machinery. This is a proposed experiment,
not a validated implementation or performance claim.

## 4. Do not rediscover existing GEMM/MoE paths

RDNA already selects resident MoE tile8 at M>=4096 for the exact FP16 Intel
geometry: hidden2560, intermediate640, local/global experts128/512, topk10,
matching group128 scales. Small rows retain skinny decode / tile4 fallback.
[Current resident dispatch](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py#L178-L211).

The hybrid dense W4A16 linear implementation already has a gfx10 M>=256
dequant-to-FP16 plus rocBLAS path; the default RDNA2 implementation has a
different shape-dependent dispatcher, including ExLlama for large GPTQ GEMMs.
These are **linear** dispatchers, not replacements automatically covering
routed resident MoE. Existing model dense parameters may be unquantized FP16;
first prove which operations the served checkpoint actually uses before
promising gains from changing a backend.
[Hybrid source](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py),
[RDNA2 dispatcher](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py).

Large new expert GEMM work should preserve the existing int4 weights, scale /
zero semantics and FP16 activations. Full dense copies of every expert are not
a free solution on almost-full 30GiB GPUs. No W4A8/INT8 alternative is in scope.

## Already-landed / lower-priority work

Exact mainline changes, distinct from a portable large prefill speedup:

- `68363d64ec476bbbf5025b30ced0f6a2425b1291`,
  [PR47842](https://github.com/vllm-project/vllm/pull/47842): GDN output norm
  consumes original3D tensors rather than reshape/copy to2D and back. RDNA
  retains old reshapes in `_output_projection` and `forward_hip`. Every GDN
  layer benefits if its strided output gate required a copy, but this is
  memory/overhead, not faster recurrent math. AITER compile fusion is not
  applicable when AITER and compile are off.
- `b538d807ffe0976af7e5d107ccfe77e676b83b70`,
  [PR59536](https://github.com/vllm-project/vllm/pull/59536): group-local GDN
  checkpoint metadata is correctness, not an advertised math speedup.
- `31a8a266622781917cef482d01db415cfa3cbd92`,
  [PR54873](https://github.com/vllm-project/vllm/pull/54873): sparse GQA QSA
  prefill tuning targets NVIDIA, not a gfx1030 implementation.
- `a5c9179e731e8d2ec7e03485a3ab0754a5a27182`,
  [PR54915](https://github.com/vllm-project/vllm/pull/54915): compact NVIDIA
  indexer workspace reduces temporary traffic. RDNA already has live-context
  bounding; map actual workspaces/strides, do not blindly copy the change.

Pinned mainline FLA changes contain no new gfx1030 prefill math replacement.
RDNA's own dispatch commentary records native GDN HIP prefill8.7% slower than
Triton/FLA at16K and6.4% lower TP4 PP, hence default-off. Turning that flag on
is not a justified gain.
[RDNA GDN source](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py).

Mainline [PR58114](https://github.com/vllm-project/vllm/pull/58114) reduces PLE
metadata construction and synchronizations, with AMD consumer changes. It is
metadata overhead rather than the substantial prefill compute requested here.
Its documented NVIDIA C1 result did not demonstrate a reliable throughput
gain, while C8 improved around5%. Do not sell it as a large long-prefill fix.
RDNA HC fusion/HIP and QSA live-context bounding already exist in this baseline;
switching their flags is a configuration experiment, not inventing new kernels.
[HC implementation](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/vllm/models/qwen4_exp/amd/ops/hc.py),
[existing live-context result](https://github.com/opengfx1030/vllm-rdna/blob/a6ab43cf9661935e05f2a601b8fc1336d6c42491/docs/rdna2/V620-QSA-LIVE-PREFILL.md).

## Implementation gate

Start with measured component attribution, then one contained port and immutable
Git deployment. Keep unchanged baseline precision/configuration, compare warmed
16K/64K/128K actual PP, normal1024-token TG and three-chat aggregate numbers.
Use existing PLE tests for independent FP16/BF16 gate math and short-conv state;
test ragged/tail rows, extreme norms, mixed decode/prefill, graphs, RAM reload
and image/tool content. Quality checks and repetition/error rejection remain
mandatory. Reject a microkernel win that does not improve whole-model prefill.
