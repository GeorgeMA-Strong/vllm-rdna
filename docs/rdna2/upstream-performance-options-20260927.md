# Upstream performance options for V620 — 2026-09-27

Research only; no server access, restart, workload, source patch, or deployment
performed for this note. Target: Swift 1.5 Qwen3.8-Flash-Next AutoRound,
TP4 Radeon PRO V620 / gfx1030. Local/runtime findings and the action ranking
are recorded separately in `V620-DECODE-PRIORITIES-20260927.md`.

## Baseline and important architecture distinction

The official latest release is **v0.30.0**, published September 22, 2026,
tag commit `ced6857`. Main inspected through GitHub's source API was
`924707f1bf94ff583d89bff7522ee12ff032c286` (September 27). These are not
interchangeable: merged-after-release or release-branch-divergent work must
be checked before assuming it ships in the wheel.
[Release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0),
[inspected main](https://github.com/vllm-project/vllm/commit/924707f1bf94ff583d89bff7522ee12ff032c286).

Main's CMake accepts gfx1030, but its `on_gfx1x()` / RDNA platform predicate
still matches gfx11/gfx12, not gfx10. Therefore “ROCm supported” or “RDNA
kernel merged” does not imply that kernel dispatches on V620.
[Build configuration](https://github.com/vllm-project/vllm/blob/924707f1bf94ff583d89bff7522ee12ff032c286/CMakeLists.txt),
[platform predicates](https://github.com/vllm-project/vllm/blob/924707f1bf94ff583d89bff7522ee12ff032c286/vllm/platforms/rocm.py).

## Requested improvements

| Area | Upstream finding | Implication for this machine |
| --- | --- | --- |
| Native W4A16 dense | [#41394](https://github.com/vllm-project/vllm/pull/41394), merged May 29, implements FP16/BF16 dense HIP; current `RDNA3W4A16LinearKernel` requires **gfx1100**. | Not a gfx1030 drop-in; advertised RX7900 gains cannot be used as V620 predictions. |
| Native W4A16 MoE | [#44075](https://github.com/vllm-project/vllm/pull/44075), merged June 6, adds routed-expert HIP and fused output reduction; compiled/registered only for gfx1100. | Useful design reference, not an unimplemented free upgrade for RDNA2. Local decode dispatch coverage matters more than format support alone. |
| Hybrid W4A16 | Current main has HIP skinny decode for `M<=5`, with Triton prefill, sharing one packed weight buffer. Python dispatch gates gfx11/gfx12. | Does not replace the local gfx1030 kernel; compare actual small-batch/MTP shapes before transplanting. [Source](https://github.com/vllm-project/vllm/blob/924707f1bf94ff583d89bff7522ee12ff032c286/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py). |
| GDN prefill | Non-CUDA selects Triton/FLA. FlashInfer and CuTeDSL prefill paths target NVIDIA architectures. | No ready native gfx1030 HIP prefill path found in main. [Resolver](https://github.com/vllm-project/vllm/blob/924707f1bf94ff583d89bff7522ee12ff032c286/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py). |
| GDN decode | ROCm optional path is AITER **Triton**, not native HIP. Native fused MTP decode requires its compiled CUDA op and supported shapes/dtypes. | A gfx1030 HIP MTP implementation remains integration/kernel work, not just a backend flag. GDN is relevant: local investigator verified this model has 36 linear-attention layers. |
| Prefix caching | Existing upstream feature, with hybrid/MTP fixes in v0.30.0. | Audit hit rates and boundary retention first. Hits improve repeated-turn prefill/TTFT, not intrinsic token decode speed. |
| PLE | v0.30.0 includes NVIDIA PLE fusion, residual fusion, and UVA offload. | Read path scope carefully: not a transparent replacement for local AMD worker/IPC PLE. |
| Full / piecewise graphs | Upstream supports `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE`. Piecewise requires compilation support; full graphs require backend support. | Full decode already captured locally, so “enable full graphs” is not a new gain. Piecewise affects remaining mixed/prefill execution, not already-captured pure decode. [Design](https://docs.vllm.ai/en/latest/design/cuda_graphs/). |
| All-reduce | Upstream custom AR rejects more than two PCIe-only GPUs without full connectivity. QuickReduce's actual architecture guard accepts gfx94/gfx95. | Neither is a straightforward V620 TP4 substitute for the fork's `rdna_ar`. Restoring a matching native extension is a narrower experiment than rewriting upstream collectives. |
| Attention | Main has QSA-specific indexer/attention improvements. | Generic dense FlashAttention is not interchangeable with Flash-Next sparse QSA; prioritize the actual selected path. |

W4A16 dispatch guard:
[dense source](https://github.com/vllm-project/vllm/blob/924707f1bf94ff583d89bff7522ee12ff032c286/vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py).
All-reduce guards:
[custom AR](https://github.com/vllm-project/vllm/blob/924707f1bf94ff583d89bff7522ee12ff032c286/vllm/distributed/device_communicators/custom_all_reduce.py),
[QuickReduce](https://github.com/vllm-project/vllm/blob/924707f1bf94ff583d89bff7522ee12ff032c286/vllm/distributed/device_communicators/quick_all_reduce.py).

## Upstream changes worth inspecting selectively

- **GDN flat-layout fast-path #53623**, merged September 19 on main, passes
  `qkvz_layout` into AITER and removes redundant output initialization/copy.
  The author reports four GDN launches becoming two and roughly 2–3% serving
  throughput improvement on **8× MI355X**, not V620. API comparison shows its
  merge commit and v0.30.0 diverge; do not label this release-contained without
  checking the tagged source/cherry-picks. The RDNA AITER availability helper
  is explicitly the gfx12 path. [PR](https://github.com/vllm-project/vllm/pull/53623),
  [AITER guards](https://github.com/vllm-project/vllm/blob/924707f1bf94ff583d89bff7522ee12ff032c286/vllm/_aiter_ops.py).
- **PLE fusion #54517**, merged September 2, combines n-gram generation,
  short-convolution and gate work on the NVIDIA model path. Its GB300 TP4
  MTP3 end-to-end decode gains are around 3–5%, despite much larger isolated
  kernel multipliers. This is a useful fusion reference, not evidence for a
  many-fold V620 gain. [PR](https://github.com/vllm-project/vllm/pull/54517).
- **PLE residual/QSA gate fusion #55309**, merged September 14, removes
  separate pointwise work in NVIDIA code while preserving BF16/FP16 rounding.
  It reports 1.44× single-row PLE-kernel and 1.105× single-row QSA-kernel
  improvements on B200, not whole-model speedups. Consider porting the idea
  only after profiling the local corresponding kernels.
  [PR](https://github.com/vllm-project/vllm/pull/55309).
- **UVA PLE offload #54371**, merged September 9, keeps TP-sharded PLE weights
  pinned in RAM and lets the GPU retrieve rows on a side stream. Its changed
  model files are under `qwen4_exp/nvidia`, not AMD. Supporting this on the
  current AMD worker-based path requires porting and validation; do not simply
  replace the serving flag. [PR](https://github.com/vllm-project/vllm/pull/54371).
- **QSA prefill/decode split #54513**, merged September 2, uses separately
  tuned Triton paths. GB300 measurements show large prefill-kernel gains,
  but low-concurrency end-to-end decode gains are small or even negative
  in one MTP case. **FP8 QSA indexer #54890**, merged September 7, reduces
  indexer traffic but has little 8K end-to-end benefit in its author tests.
  Neither benchmark establishes V620 benefit.
  [Split PR](https://github.com/vllm-project/vllm/pull/54513),
  [FP8 PR](https://github.com/vllm-project/vllm/pull/54890).

## Prefix/cache correctness before performance claims

v0.30.0 lists three relevant fixes: stop zero-progress preemption cascades
when frees are deferred; retain both MTP replay boundaries for aligned prompt
resends; retire align-mode Mamba states across null gaps. These are
correctness/efficiency changes, not proof that this service currently hits
those bugs. Check existing local patches before backporting.
[Scheduler #49675](https://github.com/vllm-project/vllm/pull/49675),
[replay boundaries #54713](https://github.com/vllm-project/vllm/pull/54713),
[state retirement #55450](https://github.com/vllm-project/vllm/pull/55450).

Prefix caching avoids recomputing shared prompt history. It does not itself
shorten the token-generation computation, although avoiding competing
prefills can indirectly improve concurrent chat responsiveness.
[Official APC limits](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/#limits).

## Decision boundary

No defensible percentage gain can be predicted for V620 from these upstream
benchmarks. Most headline kernels target RDNA3/4, CDNA, or NVIDIA. For
decode-first work, prefer a small experiment on an existing local fast path
or a verified fallback over a broad mainline upgrade. Measure one and three
concurrent chats, with and without MTP and with the real graph-padded batch
sizes, then validate long-context resumed state and tool-turn prefix reuse.
Keep mainline integration separate from live service and preserve the working
KV offload fixes.
