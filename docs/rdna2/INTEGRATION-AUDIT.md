# Qwen3.8-Flash-Next on four V620s: integration audit

Reviewed 2026-09-09 (Europe/Warsaw). Target: fast Qwen3.8-Flash-Next with
**vision and MTP**, preserving useful model quality. No inference host is
available yet. No model weights were downloaded. Model metadata was read only.

## Result

Use this fork of `opengfx1030/vllm-rdna` as the integration repository, but
do not merge Leapdragon's whole branch or PR #3 wholesale. Both forks descend
from an August 23 mainline base. The target has no native `qwen4_exp` model
package; current mainline does. Start the model integration from mainline's
implementation, retain the target's RDNA kernel work, and port the donor's
ROCm PLE offload as a separate component. Resolve the weight format before
selecting or tuning the MoE kernels.

This branch currently contains tested startup-plan fixes, the existing
compile-thread fix from upstream PR #53892, and this audit. **The model/PLE
integration is not implemented or validated by these changes.**

## Pinned sources

| Source | Branch / revision | Role |
| --- | --- | --- |
| [Our fork](https://github.com/GeorgeMA-Strong/vllm-rdna) | `codex/qwen38-v620-integration` | Working branch |
| [Target](https://github.com/opengfx1030/vllm-rdna/tree/a4060647cfbb32d3b907eba6018501bb8b251d22) | `rdna_extras`, `a4060647cfbb32d3b907eba6018501bb8b251d22` | RDNA kernel base |
| [Leapdragon](https://github.com/leapdragon/vllm-rdna2-qwen/tree/35b351f5b79f072b9159aba39acd27bbaba25449) | `rdna2/qwen38-flash-next`, `35b351f5b79f072b9159aba39acd27bbaba25449` | ROCm offload and V620 tuning reference |
| [vLLM mainline](https://github.com/vllm-project/vllm/tree/60ad959b6f1a5c8f602edbd608c8decbc0788c50) | `60ad959b6f1a5c8f602edbd608c8decbc0788c50` | Current native model and shared fixes |
| [Target PR #3](https://github.com/opengfx1030/vllm-rdna/pull/3) | `4b9badd0a97ddd406342f296d041a32527e3b8d7` | Alternative model/ROCm implementation |

Common ancestor: `e25c586b9030a10702d78856b43ccae9481cc28c`.
At these pins, target/mainline have 137/703 unique commits respectively;
target/Leapdragon have 137/406. These counts are ancestry differences, not
counts of independently portable patches. PR #3 changes 85 files relative to
the target tip (13,113 insertions, 1,879 deletions).

## All PRs in the two requested repositories

The GitHub API was paginated over **all states**, not just open PRs.
Leapdragon's repository had **zero PRs**. The target had exactly these three:

| PR | State | Relevance / decision |
| --- | --- | --- |
| [#1](https://github.com/opengfx1030/vllm-rdna/pull/1) | Merged September 8 | Five donor all-reduce commits are already incorporated. Despite the title saying review-only, this is in the target tip. Do not duplicate it. It deliberately excludes donor commit `3cfe000` with the newer VRAM flag protocol. |
| [#2](https://github.com/opengfx1030/vllm-rdna/pull/2) | Open | GLM-5.3, targeting a separate `later/glm53-flash-awq` branch. Its author explicitly reports no GPU testing. Not part of the Qwen integration. |
| [#3](https://github.com/opengfx1030/vllm-rdna/pull/3) | Open, WIP | Adds Qwen4Exp, GPTQ/AWQ work, MTP and host PLE lookup. Its recipe reports vision+MTP on four V620s with a different GPTQ checkpoint. Useful comparison, but too broad to accept without component tests. |

## Mainline and pending upstream work

Searched current mainline history and PRs for Qwen3.8/Qwen4Exp, PLE offload,
RDNA/gfx1030, compilation caching and startup plans. This is a relevance review,
not a claim to have reviewed every PR in the main vLLM repository. Broad searches
were capped at 100 results; exact-symbol searches and model-path history were
also used. Status is a snapshot at the pinned revision.

| PR | State | What to reuse or watch |
| --- | --- | --- |
| [#53896](https://github.com/vllm-project/vllm/pull/53896) | Merged | Native Qwen3.8-Flash-Next/Qwen4Exp package, including AMD and NVIDIA implementations. Prefer this model contract over a donor snapshot. |
| [#55375](https://github.com/vllm-project/vllm/pull/55375) | Merged | Fixes strided state indices in NVIDIA fused PLE convolution, including MTP-shaped indices. Keep its regression scenario; this is not a demonstrated fix for the target's AMD GDN problem. |
| [#54890](https://github.com/vllm-project/vllm/pull/54890) | Merged | FP8 QSA indexer cache. Evaluate exact backend applicability; do not infer gfx1030 support from the title. |
| [#55272](https://github.com/vllm-project/vllm/pull/55272) | Merged | Removes compilation from the NVIDIA model path. Does not solve AMD startup time. |
| [#55513](https://github.com/vllm-project/vllm/pull/55513) | Merged at pinned HEAD | Block-FP8 MTP loading in mixed ModelOpt checkpoints. Relevant if checkpoint format changes; not an INT4 AutoRound fix. |
| [#53899](https://github.com/vllm-project/vllm/pull/53899) | Open | Base PLE CPU-offload infrastructure. Its documented offload validation is on GB200; ROCm needs the donor's HIP adaptation. |
| [#54070](https://github.com/vllm-project/vllm/pull/54070) | Open, stacked on #53899 | Disk-backed PLE table and warm reuse. Preserve model identity in cache keys before relying on it. |
| [#54129](https://github.com/vllm-project/vllm/pull/54129) | Open | Alternative mmap PLE design. Compare ownership and graph-replay semantics against the donor worker, not just boot time. |
| [#54371](https://github.com/vllm-project/vllm/pull/54371) | Open | UVA PLE offload and embedding TP. Its memory/coherence assumptions require separate ROCm review. |
| [#55040](https://github.com/vllm-project/vllm/pull/55040) | Open | AMD FP8 PLE loading, handling single-use generators and scales across split loader calls. Required if an FP8 PLE alternative is chosen. |
| [#55292](https://github.com/vllm-project/vllm/pull/55292) | Open | ROCm multi-step draft capability, including QSA. Reuse the capability design rather than another hardcoded metadata allowlist. |
| [#54642](https://github.com/vllm-project/vllm/pull/54642) | Open | Multimodal limits missing from compile-cache hashing. Relevant when switching vision modes. |
| [#52391](https://github.com/vllm-project/vllm/pull/52391) | Open | gfx1030 recognition; does not by itself provide this serving stack. Target already has broader RDNA changes. |
| [#53892](https://github.com/vllm-project/vllm/pull/53892) | Open | Preserve explicit `TORCHINDUCTOR_COMPILE_THREADS`; exact patch reused on this branch. Original commit `286626d4991200ff30ab23b8e578496b25b5b79f`, author Yao Xu. |
| [#55506](https://github.com/vllm-project/vllm/pull/55506) | Open | Mamba speculative block tables indexed by request slot. Add request-reordering coverage before MTP tuning. |
| [#55149](https://github.com/vllm-project/vllm/pull/55149) | Open | Bound GDN prefill workspace during KV sizing. Relevant to avoiding profile OOMs. |

## Findings, ordered by impact

### P1: target is missing the native target model

At `a4060647c`, `vllm/models/qwen4_exp/` and its native registry entry are absent.
The target's Qwen3.5/27B GDN kernels are not a substitute for Flash-Next's
QSA, PLE, hyperconnections, vision and dedicated MTP implementation.
PR #3 and mainline both contain model work; neither is incorporated here yet.

### P1: existing all-reduce port omits a donor ordering correction

The target's `csrc/rocm/rdna_allreduce.cuh` signals via host-coherent flags
after writing payloads into peer VRAM. Donor commit
`3cfe00035559a1bff0b05b6ac82cac5ad08fd5bc` moves flags beside each receiving
GPU's uncached staging area. The donor documents cross-destination PCIe
ordering and polling traffic as the reasons. PR #1 explicitly excluded this
commit because it mixes PLE work into the change.

Port the all-reduce part separately, with its buffers, IPC layout, pacing,
timeouts and boot self-test. Keep the target's opt-in policy until four-card
eager and graph-replay correctness are measured. No all-reduce changes were
made in this audit. The coherence risk is source/history evidence, not a fault
reproduced on our hardware.

### P1: memory-plan cache can override changed memory requirements — fixed

`vllm/v1/worker/startup_plan.py` used a graph hash for a memory allocation.
`CacheConfig.compute_hash()` excludes `gpu_memory_utilization`, and
`SchedulerConfig.compute_hash()` does not include `max_num_seqs`.
An old KV allocation could be reused after either changes. The fingerprint
also omitted `torch.version.hip` and queried device zero for every worker.

This branch adds those dependencies and uses the worker's actual device.
Schema version 2 invalidates old records. Non-object JSON, invalid UTF-8,
boolean memory sizes, and impossible size/baseline pairs now fall back to
profiling. Regression tests reproduce the failures on the original code.

This remains opt-in. The patch does not prove complete cache identity for
in-place weight edits, opaque op changes, driver changes or all multimodal
memory settings. Keep startup-plan reuse disabled during initial hardware
qualification and when changing checkpoints or runtime builds.

### P1: donor disk PLE reuse is not bound to a checkpoint

In the donor's `vllm/v1/ple_offload/worker.py`, `_ple_disk_attach` accepts an
existing `.bin` plus `.done.json` based on file length, shape and dtype only.
The filename is based on the layer/parameter name. Reusing a disk directory
for a different checkpoint with the same table shape silently reuses the old
table and skips the new checkpoint shards.

Before porting: key the table by pinned model revision, tensor identity,
shape, dtype, quantization/scale metadata and conversion version. Write its
completion record atomically and test interrupted conversion and concurrent
startup. For now this is an identified donor issue, not an implemented fix.

### P1: graph correctness and cache identity are unresolved

Target docs in `docs/profiling/2026-09-08-cudagraph-gdn-root-cause.md` report
incorrect output under piecewise graphs for **Qwen3.8-27B**, with an eager
control producing correct output. That is a related hybrid path, not proof
that Flash-Next fails in the same way. The investigation contains disproven
hypotheses; do not copy its speculative fixes as established facts.

Leapdragon disables compile-cache read/write by default. Its
`docs/rdna2/CHANGES.md` section 8 explains why traced-file hashing misses opaque
custom-op bodies/schemas and native extensions. `VLLM_RDNA_DENSE_INT8_ONLY`
also changes weight residency but is absent from its env registry, so it is
not directly covered by `envs.compile_factors()`. Target RDNA environment
knobs likewise need a registration/cache-key audit.

Use a build-and-configuration-specific cache namespace and test warm restarts
with vision and MTP changes. Register custom ops before cache loading. Reuse
upstream #54642 for multimodal hash coverage instead of duplicating it.

### P2: explicit parallel compilation was ignored — existing fix reused

Both original forks assign `TORCHINDUCTOR_COMPILE_THREADS=1` at import,
discarding an explicit user setting. This branch applies #53892's `setdefault`
change, retaining the default of one thread. Thread count should be tuned
against CPU RAM and the number of TP workers, not set to all cores per worker.

### P2: donor defaults and measurements do not match the goal

The donor launcher defaults to `GPUS=1,2,3,4` and vision disabled. A machine
with exactly four normally enumerated cards needs `0,1,2,3`; enumerate first.
Its launch recipe also adds INT8 dense shadows to an INT4-expert checkpoint
and uses a separate quantized PLE table. Those are additional approximation
choices; do not attribute their effects solely to the original INT4 weights.

PR #3's recipe uses a hardcoded executable path, a different GPTQ checkpoint,
and 0.98 GPU-memory utilization. Its reported TP-only result and the donor's
EP result use different kernels/checkpoints/topologies, so they do not decide
TP versus EP for our host.

## Checkpoint candidate

No matching Microsoft Qwen3.8 model was returned by the Hugging Face model API
search. The plausible candidate is
[Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound),
revision `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`.

Its card reports a four-task average of 0.8332 versus BF16 0.8362 (99.64%
relative). This is a publisher result, not our evaluation or a general quality
guarantee. Its configuration uses 4-bit symmetric groups of 128,
`quant_method=auto-round`, `packing_format=auto_round:auto_gptq`, with
16-bit exclusions for vision, MTP and PLE. MTP is declared in `text_config`.
We have not inspected the weight tensors or proved that every required tensor
is present. Absence of a separately named MTP file does not imply missing MTP.

The target already maps AutoRound through `INCConfig`, with GPTQ-compatible
schemes. Preserve this metadata and validate expert zero-point/layout handling.
Do not relabel the config as AWQ or compressed-tensors to force a kernel.
Leapdragon's selected
[wtdcode checkpoint](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
uses a different packed representation. No side-by-side quality evaluation
between these checkpoints was performed here.

PLE precision is a separate decision: the BF16 table is roughly 95 GiB of
host storage, FP8 roughly 48 GiB, and the donor INT4 sidecar roughly 30 GiB
including its scales. Include host RAM and page-cache pressure in the build
specification; four cards' VRAM does not replace this host-memory requirement.

## Integration sequence

1. **Refresh the common model baseline.** Keep the target as our fork, but
   reconcile its RDNA changes against pinned mainline in a separate integration
   branch. Take native model interfaces, config, registry, vision and MTP from
   mainline together. Review conflicts rather than replacing shared files with
   donor versions. Preserve mainline's post-model-merge fixes.
2. **Bring in PLE as one coherent subsystem.** Compare current #53899/#54070
   with the donor's worker/protocol, HIP driver shim, AMD layer, executor and
   runner integration. Preserve the donor's graph-safe CPU handshake; a plain
   CPU gather is not captured GPU work. Add table provenance and failure tests.
3. **Validate the selected AutoRound/GPTQ loading path.** Check index/tensor
   coverage for vision, PLE and MTP, exact group/zero-point conventions, peak
   repack memory, and per-rank experts. Use small synthetic tensors first.
   Compare PR #3's loading changes; its recipe explicitly warns about a 95 GiB
   temporary allocation in an alternate kernel path.
4. **Port communication independently.** Extract the newer same-destination
   flag protocol without replacing the target's GEMMs. Test all supported
   dtypes/message sizes, repeated graph replay and failure handling on TP4.
5. **Establish quality with vision and MTP.** Start with dense shadows disabled
   and a declared PLE precision. Compare eager versus compiled and MTP 0 versus
   1/2/3. Match greedy target tokens; use statistical checks for sampled output.
   Only then evaluate extra INT8/shadow-only compression and larger captures.
6. **Optimize measured bottlenecks.** Benchmark cold filesystem cache, warm
   filesystem cache, and warm compilation cache separately. Sweep TP versus
   TP+EP, prefill chunks, graph sizes and MTP depth on identical weights/prompts.

## Validation performed here

Contract for the changed module: persist a profiled allocation, reuse it only
for compatible inputs, and fall back to profiling for unusable records.
The cheapest test level is the existing startup-plan unit suite with synthetic
worker/platform objects and temporary JSON files; no model or GPU is needed.

```bash
.venv/bin/python tools/rdna2/test_startup_plan_cpu.py
```

- Original startup-plan source with the extended suite: **13 failed, 3 passed**.
- Patched source: **16 passed**.
- CPU runner loads the actual module and mocks only its import dependencies;
  this is not a GPU engine-startup test.
- Compile-thread patch: verified the changed source statement retains an
  explicit `8` and defaults to `1`. Full torch import behavior was not tested.
- Syntax: all 2,288 target and 2,318 donor `vllm/**/*.py` files compiled as
  Python source; no syntax errors. No inference or imports implied.
- Both donor and PR #3 launch scripts passed `bash -n`.
- All applicable repository pre-commit checks passed, including pinned Ruff,
  formatting, spelling, Markdown, mypy and import/header checks; whitespace
  checks passed as well.

## Hardware acceptance matrix

Record exact source SHA, ROCm/PyTorch/Triton build, checkpoint revision,
PLE representation, card IDs, PCIe topology, host RAM, NUMA layout and power
settings with every run. Use local model paths and offline mode after the user
authorizes downloads on the inference machine.

| Gate | Cases / required evidence |
| --- | --- |
| Loader completeness | Real index and tensor coverage for target, MTP, vision and PLE; no skipped/uninitialized parameters; per-rank peak host/device memory |
| Kernel correctness | GPTQ group-128 experts and dense paths against dequantized references; strided/non-aligned shapes, prefill/decode/MTP widths |
| PLE correctness | Random rows versus source table, batch/request reorder, delayed worker, restart, missing/corrupt sidecar, concurrent startup |
| Graph correctness | Eager/compiled tokens on the same prompts; fresh/warm cache; single and mixed prefill/decode batches; padded and non-contiguous state indices |
| Vision | Caption/OCR/diagram prompts, image-size and count limits, multi-image requests, chunked prefill, vision+MTP, complete warm restart |
| MTP | Depth 0/1/2/3, greedy parity, acceptance and tokens/verification, concurrency, prompt-boundary transitions and request reordering |
| Quality | Frozen text/reasoning/tool/vision cases; reference checkpoint comparison; isolate expert quantization, PLE quantization and dense shadows |
| Startup | Time weights, PLE preparation, repacking, compilation, profiling, graph capture and first request separately; cold and warm runs |
| Performance | TTFT, prefill throughput, per-user decode and aggregate throughput at c1/c4/c8 and 1K/8K/32K/128K context; only after correctness passes |
| Sustained operation | Multiple warm restarts and long mixed traffic; no PLE/all-reduce timeouts, GPU resets, corrupt replies or memory growth |

## Existing local work

An older sibling checkout of Leapdragon has 13 modified/added files (prefill
tiling and FLA tuning, 616 insertions / 16 deletions). It was read but not
modified. Those changes are not part of either upstream pin or this branch.
Revisit them as tuning candidates after correctness; the tile override needs
shape/group validation and an RDNA-only import guard before general reuse.
