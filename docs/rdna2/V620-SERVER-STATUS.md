# V620 Flash-Next integration status — 2026-09-10

The requested final configuration is Qwen3.8-Flash-Next with vision, MTP and
full context on four V620 cards. The model body belongs on the GPUs; the
n-gram / PLE table belongs in system RAM. The user permits a quantized table
in RAM if measurements support the change. BF16 is the reference.

This document records development state, not a validated serving recipe.

Text, basic vision and one-token MTP serve on all four GPUs with the original
BF16 PLE table in RAM and decode-only full graphs. The latest matched short
benchmark measures **21.59 output tokens/s**, about **32.7 tokens/s after the
first token**, with 1.037-second mean TTFT. Uncached 1,024-token requests
measure **253.41 prompt tokens/s**. Twenty-four elementary text checks pass
at concurrency one/two/four, as do two synthetic vision checks. Six fixed
greedy responses match the previous version's text, with different token
probabilities. The user's 60–90 tokens/s target, broad model quality and
actual native-context generation remain unverified. Large-context tests
remain deferred at the user's direction.

## Sources

- Target: `opengfx1030/vllm-rdna`, `a4060647cfbb32d3b907eba6018501bb8b251d22`.
- Donor: `leapdragon/vllm-rdna2-qwen`,
  `35b351f5b79f072b9159aba39acd27bbaba25449`.
- Mainline: `vllm-project/vllm`, `b28c3e1568bfae930f61d4b24940e47528c85d4a`.
- Our published starting point: `b60e6cca2`, including the startup-plan
  validation fixes. The current worktree merges mainline and ports the donor's
  CPU-worker protocol. The merge is recorded as `d0c04657b` and the RAM PLE/runtime port as
  `640c022c4`. The performance changes are recorded separately.

Mainline PR #54371 added NVIDIA UVA PLE offload and Engram configuration.
The AMD implementation did not use it. The integration keeps NVIDIA's path
and connects the donor CPU-worker path to the current Engram option on ROCm.

## Host and downloads

The host has four gfx1030 V620 GPUs, about 29.985 GiB usable VRAM per GPU,
247 GiB system RAM, an EPYC 7452 and ROCm 10.0.0. The build uses a separate
Python 3.12 environment and AMD PyTorch 2.13 for gfx1030. The existing
`llama-server.service` was temporarily stopped for the four-GPU vLLM startup
test; its files and configuration are unchanged. Restore it if abandoning the
vLLM test before a working replacement is available.

The selected first checkpoint is
[Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
at `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`.
Its publisher reports a four-task average of 0.8332 versus 0.8362 for BF16;
this is publisher evidence, not our evaluation or a V620 speed comparison.

The checkpoint contains approximately 73.38 GiB of non-PLE tensors including
vision and BF16 MTP, plus 95.37 GiB of BF16 PLE. The PLE tensors reside in
`model-00016-of-00017.safetensors`. All actual safetensor header names were
checked against the index; its total-size metadata is stale. The absent
shard number 00002 is not evidence of missing tensors.

The mixed precision comes from Intel's published conversion settings. Routed
expert projection matrices use INT4. Attention, routing, shared experts,
vision, embeddings, hyper-connections, PLE and MTP are excluded from that
quantization. A decoder block therefore contains both quantized and
unquantized weight matrices. W4A16 describes four-bit weights in the selected
matrices and 16-bit activations, not four-bit values in every tensor or every
intermediate calculation. RAM-table quantization is a separate experiment;
the serving baseline keeps the original BF16 table in system RAM.

A completed Intel shard was inspected on the host: the first expert's
down-projection is GPTQ int32 `[80, 2560]`, qzeros `[5, 320]` contain
`0x77777777`, and scales `[5, 2560]` are **FP16**. Preserve these packing and
scale semantics when evaluating a native RDNA2 adapter.

Remote task-owned files are under `/home/george/v620-vllm`:

- `models/intel-autoround`: pinned model download, including original BF16 PLE.
- `source`: direct copy of this worktree used for building.
- `.venv`: isolated build/runtime environment.
- `tools/v620-build-vllm.sh`: waits for the existing PyTorch installation,
  installs constrained ROCm dependencies, then builds for gfx1030.
- `logs/`: dependency, model-download and native-build logs.

Both model downloads completed successfully. The 17 safetensor files contain
224,280 tensors and 181,165,326,328 payload bytes. All index mappings, tensor
shape/dtype sizes, contiguous offsets and complete file sizes were verified;
this check did not recompute cryptographic hashes. The verification record is
`logs/checkpoint-verification.json` on the host.

The complete native runtime built successfully for ROCm 10/gfx1030 with eight
jobs, using verified Triton source commit
`0f380657dbf3ee86eb57558ff71df24f03b5d4e7`. All three native extensions import.
AMD SMI initially loaded two copies of its shared library and reported zero
GPUs despite HIP seeing four. Installing editable Python bindings from the
same wheel-bundled ROCm SDK fixed discovery: SMI and PyTorch both see four GPUs.
The build script now installs those matching bindings. This matches the
duplicate-library failure described in
[ROCm issue #6340](https://github.com/ROCm/ROCm/issues/6340).
The INT4 PLE sidecar download is stopped and its task-owned partial data removed.

The GitHub clone initially failed during checkout while fetching the empty
blob. Supplying that known empty object locally completed recovery. The
runtime still uses the separately transferred integration worktree.

## Implemented and checked

- Resolved the four mainline merge conflicts, retaining gfx1030 support,
  the target's Qwen3.5 graph policy and both CPU and RDNA2 MXFP4 branches.
- Kept startup-plan tests separate from GPU-worker imports: 16 tests pass.
- Ported donor PLE lifecycle, shared host buffers, HIP bindings, per-request
  completion protocol and model-runner V2 hooks. One CPU worker owns the table
  and supplies all local TP ranks. No GPU table is allocated by placeholders.
- Adapted configuration to current Engram settings, including explicit
  overrides of the legacy environment variable. ROCm V1, PP, context
  parallelism and microbatching remain unsupported by this port.
- Fixed plain RAM allocation. In the donor, constructing the outer model on
  `meta` also left the BF16 PLE table on `meta`, so shard copies could silently
  discard their data. Plain RAM mode now constructs only the PLE subtree on CPU.
- A small probe using the actual base-class source and real server PyTorch
  failed before this fix with a meta table. After the fix, CPU allocation,
  BF16 shard copy and lookup passed. Only import-time vLLM dependencies were
  stubbed; this probe does not validate the complete worker or HIP protocol.
- Added regression coverage for CPU ownership, allocation-free GPU placeholders
  and explicit Engram settings. All 22 initial worker tests passed.
- Reproduced a donor FP8 RAM-table loader failure with actual server PyTorch:
  NumPy cannot represent `Float8_e4m3fn`. Prefault views now expose storage
  bytes for non-INT4 layouts. A tiny FP8 table passes loading, repeated and
  reordered cross-shard lookups, and exact BF16 output against the reference.
  This validates compatibility, not model quality or full-table performance.
- Reproduced an RDNA2 W4A16 dequantization error on GPU: all-zero quantized
  weights produced output magnitudes 0.06640625 and 0.234375 for scales 0.003
  and 0.01. The shared kernel now subtracts the integer zero point exactly
  before scaling, avoiding the rounded scale-baked offset. All four scale
  cases now return exact zero after recompilation.
- The corrected actual MoE kernel passed 12 independent reference cases
  across all four V620 GPUs, including `[640, 2560]` and `[2560, 1280]`
  group-128 expert matrices. Worst relative L2 error was 0.0005791 against
  FP32 matmul using independently dequantized FP16 weights. This is kernel
  arithmetic validation, not a BF16-model quality benchmark. Regression tests
  were added to both existing dense and MoE kernel suites.
- A broader dense-kernel probe exposed an additional prefill split bug:
  K=640 could select 16 splits of 40 values, while the kernel reads whole
  32-value tiles. Split selection now requires equal, aligned tiles within
  the LDS budget. After the fix, 12 dense decode/prefill cases passed,
  including exact-zero invariants and independent reference matmuls. The
  worst relative L2 error was 0.0007505. There is no established causal link
  between this failure and the maintainer's full-graph concurrency report.
- Updated the inherited request-routing fixture for the donor's current
  host-buffer protocol: requests complete in arrival order, CPU results are
  replicated across TP workers, and completion counters publish each result.
  The combined PLE, dense and MoE suites passed: 115 passed, 49 expected
  failures. Their first run exposed mainline API changes in the inherited
  adapters: removed `has_g_idx` configuration and the changed `gptq_shuffle`
  signature. The adapters now match current APIs while retaining the empty
  g_idx argument required by the custom RDNA2 native ABI.
- Full-context TP4/EP4 configurations validate both without MTP and with one
  MTP draft token, using BF16 activations and Engram CPU offload. This checks
  configuration only, not serving or context capacity.
- Incorporated the donor ROCm QSA constructor fix: its Triton implementation
  does not need the FlashAttention binary used by its metadata base classes.
- The complete original BF16 PLE table loaded in 32.21 seconds. All 384 sampled
  rows (first, middle and last in each of 128 checkpoint shards) matched
  exactly. The CPU-owned table is `[320001536, 160]`, 95.37 GiB. Reported peak
  process RSS during mmap-and-copy loading was 187.52 GiB; this is not a cold
  disk benchmark. See remote `logs/ple-full-check.log`.
- Fixed dummy PLE loading to randomize parameters only, preserving hash
  constants and workspace initialized by CPU constructors. Previously
  `to_empty` discarded those buffers and generic ROCm dummy initialization
  zeroed persistent integer constants. The updated worker suite passes all
  24 cases, including FP8 and the optional new INT8 per-row RAM layout.
- The AMD attention suite and worker suite passed 33 tests before the extra
  INT8 parameterization; sparse-attention cases cover TP1/TP2/TP4 layouts.
- A uniform sample of 131,072 original BF16 rows gave relative L2 reconstruction
  errors of 0.025847 for FP8 E4M3 with FP32 row scales, 0.006766 for INT8 with
  FP16 row scales, and 0.085157 for symmetric INT4 group 16 with FP16 scales.
  Estimated complete table sizes are 48.88, 48.28 and 29.80 GiB respectively.
  These are local reconstructions, not measurements of published sidecars or
  model-level quality. See remote `logs/ple-quant-sample.log`.
- First four-GPU startup loaded the original Intel body at 18.49 GiB per GPU;
  rank-zero weight loading took 169.91 seconds. PLE loaded on CPU concurrently.
  Startup then failed because `torch_shm_manager` could not locate
  `librocm-openblas.so.0`. The launch environment now includes the matching
  SDK's `lib` and `lib/host-math/lib` directories. A spawned-process check
  passes shared CPU updates and GPU IPC updates on all four cards; a shutdown
  IPC-lifetime warning remains to investigate.
- Fixed the HIP shim to resolve through the HIP runtime already loaded by
  PyTorch. Loading the system's unversioned HIP library separately created a
  second runtime beside the wheel SDK. The third full startup registered all
  four shared host buffers, received all four PLE READY signals and entered
  the CPU lookup loop successfully. GPU-body loading took 188.62 seconds on
  rank zero and about 199 seconds on the other ranks, still 18.49 GiB/card.
- Fixed the PLE prefill wait deadline: the sleep path previously continued
  before testing the timeout, allowing a stalled worker to wait indefinitely.
  Two mock-clock regression tests pass. Batched CPU hashing also passes an
  independent integer reference with EOS boundaries and padded tokens.
- The third startup reached vision warmup, then LLVM aborted while compiling
  BF16 position interpolation (`llvm.amdgcn.fdot2.bf16.bf16`). A small actual
  vision-module probe reproduced the same failure. Disabling FP fusion did
  not fix it. RDNA2 BF16 interpolation now uses the existing PyTorch reference;
  all 18 interpolation tests pass on V620. A one-block vision encoder with
  dummy weights also completes a 16-by-16 patch grid with finite output.
  Other architectures and dtypes retain the Triton path. This has not yet
  established full vision inference.
- The next serving attempt limits images to 1,638,400 pixels, with up to four
  images per request and video disabled, following the donor's bounded vision
  profile. The requested native text context remains 262,144 tokens.
- The fourth startup completed vision warmup and reached language-model
  profiling. It then exposed an inherited EXL3 workaround in `Qwen2MoeMLP`
  that unconditionally clipped every model's activations to 65,000 and cast
  them to FP16. This breaks the Intel checkpoint's BF16 shared-expert
  down-projection and can unnecessarily change valid BF16 values. The guard
  now applies only to EXL3 FP16 activations; other formats use the regular
  activation. Two hardware regressions pass through the actual shared-expert
  module, covering BF16 values above the FP16 range and ordinary FP16 values.
  A small zero-weight probe of the selected Triton WNA16 MoE backend also
  executes in BF16 and returns exact zeros.
- The fifth startup advanced to QSA rotary encoding, then hit the same
  unsupported BF16 LLVM instruction inside `triton_mrope`. Promoting the
  rotation arithmetic to FP32 only on RDNA2 BF16 avoids this compiler path
  while preserving BF16 tensor storage. Twelve hardware cases pass against
  native PyTorch, covering both NeoX and adjacent pairing, BF16/FP16, batches
  1/32/2048 and random positions throughout the native 262,144-token range.
  Unrotated dimensions remain bit-exact. This is kernel validation, not a
  successful full-context generation test.
- Mainline's opt-in EP weight filter did not recognize GPTQ/AWQ `.qweight`
  names. Added that heavy-weight suffix while retaining all scale/metadata
  tensors. The loader suite passes 40 tests, including synthetic integer
  packed checkpoints; two unrelated GPT-2 download tests were excluded.
  The sixth startup enables this filter and includes the rotary fix.
- Sixth-run rank-zero checkpoint loading took 90.86 seconds versus 186.31
  seconds in the preceding run. Complete per-rank model loading took
  93.80–95.59 seconds, still 18.49 GiB/card. These are successive local runs,
  not controlled cold-storage benchmarks. Profiling completed and reported
  7.0 GiB available KV memory and 546,387 cache tokens (2.08 native contexts).
  Actual long-context requests remain untested.
- The sixth run then hit an NVIDIA-only router warmup import. ROCm gfx10's
  capability-family number also satisfies the generic `100` check, but
  CuTeDSL router kernels require NVIDIA CUDA. Both NVIDIA router warmup
  helpers now check the platform before importing their dependencies.
  All four warmup tests pass, including a regression with those optional
  modules explicitly unavailable.
- The seventh startup reached real language-model warmup and exposed the
  unsupported BF16 LLVM dot instruction in causal convolution. The existing
  variable-length convolution test reproduces that exact compiler abort
  without loading any model. RDNA2 BF16 forward and update kernels now
  promote only the multiply arithmetic to FP32; weight, activation and saved
  state storage remain unchanged. All 164 existing convolution reference
  tests pass on V620 (73.64 seconds), including variable-length batches,
  padding, gathered state, bias/activation combinations, and BF16/FP32.
  The eighth startup passed convolution and failed in the inherited GDN
  prefill dispatch: `_gdn_prefill_dispatch_available()` checks the GPU and
  registered symbols but not dtype. It selected `gdn_prefill_prep_rdna2` for
  BF16 input, which the native kernel correctly rejected with `mixed_qkv
  must be fp16`. Both prefill preparation and the subsequent native chain
  require matching dtype guards. The decode dispatch already checks FP16.
  The guard now also checks dtype in profiling, preparation and the native
  chain. The existing full-core split/unified attention test has been enabled
  on ROCm with four key heads and twelve value heads (the target TP4 geometry).
  It reproduces the original FP16 rejection before the fix.
- The BF16 fallback then spent over nine minutes in LLVM compilation inside
  WY recomputation's autotuner; a live process stack confirmed the location.
  Direct probes of the real kernel matched a PyTorch matrix reference.
  The selected 64-wide tile with eight warps/two stages compiled in 1.87
  seconds from an empty Triton cache. Relative L2 errors were 4.18e-8 for U
  and 2.87e-5 for W; short-run kernel times varied from 0.038 to 0.080 ms.
  These are small component measurements, not model PP/TG results.
  RDNA2 BF16 WY tuning now retains that configuration; other platforms and
  dtypes keep the original candidates. The original tuning run was stopped
  intentionally to test this measured change.
- Timed stacks then identified similarly expensive output-kernel tuning.
  A 64-by-64 tile with eight warps/two stages matched the independent matrix
  reference (relative L2 2.29e-9), compiled in 3.54 seconds in the probe, and
  took about 0.284 ms for its small 64-token workload. After restricting WY
  and output-kernel tuning, all eight full-core split/unified tests passed,
  including BF16/FP32 recurrent state, fresh/continuing prompts and mixed
  decode/prefill batches. This run still spent 291.07 seconds, mostly compiling
  the generic FP32-state candidates in `chunk_delta_h`.
- A three-chunk state-kernel probe (130 tokens, with an incomplete final
  chunk) verified saved intermediate state, new values and final FP32 state
  against independent matrix operations. BV=32 with eight warps/two stages
  compiled in 6.16 seconds and took 0.701 ms; the final-state relative L2 error
  was 4.67e-8. That configuration is now selected for RDNA2 BF16. All three
  tuning restrictions apply only to the tested K=V=128, BT=64 geometry.
  Other dtypes, platforms and head dimensions retain their original tuning.
  All eight full-core attention cases pass with the final configuration
  (29.40 seconds). This elapsed time includes existing compilation caches;
  it is not a controlled comparison of complete cold startup times.
  The ninth full model startup passed this path, then failed in QSA cache writes.
- AMD QSA inherited a FlashAttention cache method whose writer import is gated
  by FlashAttention availability. Without that optional package, real warmup
  raised `NameError: reshape_and_cache_flash is not defined`. The small hardware
  regression reproduced it. AMD QSA now calls the native ROCm cache writer
  directly. All 20 AMD QSA tests pass in 11.04 seconds, including eight exact
  cache-write cases with strided projections, skipped slots, trailing padding,
  one/two KV heads and 128/256-wide heads. Other cache locations remain exact.
  The ninth run loaded weights in 93.86 seconds and reported 547,911 cache
  tokens (2.09 times the native context); actual full-context generation is
  still untested.
- The tenth startup failed before warmup, while rank 3 copied vocabulary
  weights. The GPU driver reported SDMA0 read permission faults; the Python
  stack ended in `VocabParallelEmbedding.weight_loader`. This run had not
  reached the new QSA cache writer. A process-scoped `HSA_ENABLE_SDMA=0`
  diagnostic uses ROCm's alternative copy path, documented in
  [AMD's environment-variable reference](https://rocm.docs.amd.com/en/latest/reference/environment-variables/).
  Five copies each of the real embedding and output-head shards passed exact
  full-tensor comparisons on all four cards with that flag (40 copies total).
  This is not proof of the fault's underlying cause or a measured speed fix.
  The eleventh startup succeeded with this flag: service start 13:02:46 UTC,
  API ready 13:06:49 UTC (243 seconds), including 93.09 seconds loading weights.
  The first real text request returned `42` for `19 + 23`.
- `tools/rdna2/check_flash_next.py` records full responses and checks elementary
  arithmetic, repeated requests and synthetic image recognition. Concurrency
  is selectable as one, two or four. It exits unsuccessfully on mismatched
  answers or HTTP failures. These are smoke checks, not a quantization eval.
- Both synthetic vision requests returned `Blue` for the circle. The first
  took 15.64 seconds while new shapes compiled; the repeated request took
  2.50 seconds. All 24 arithmetic requests passed at concurrency one/two/four.
  This validates eager mode only; concurrent full graphs are not tested.
- A warmed, two-request `vllm bench serve` run with 128 input and 64 output
  tokens measured 2.13 output tokens/s, 1.054 seconds mean TTFT and 459.17 ms
  mean time per subsequent output token. It used the server's default sampling
  temperature, seed-zero random inputs, eager BF16 and SDMA disabled.
  A 1,024-input/one-output benchmark included prefix-cache reuse, so its
  359.11 total tokens/s is not an uncached prompt-processing measurement.
  A separate probe with unique cache salts, temperature zero and exact token-ID
  prompts measured 219.64 prompt tokens/s end to end across three 1,024-token
  requests. Each produced one token; elapsed time was 4.65–4.67 seconds.
  These are initial small-sample baselines, not optimized throughput claims.
- `tools/rdna2/serve_flash_next.sh` preserves the successful runtime settings.
  `V620_MODEL`, `V620_VENV` and `V620_WORKSPACE` select local paths. PLE CPU
  offload remains enabled. `V620_MTP_TOKENS=1` selects the next MTP experiment;
  the default is the tested non-speculative baseline.
- The first MTP startup began at 13:20:14 UTC. The preceding baseline shutdown
  was not clean: broken-pipe/IPC warnings were followed by systemd's 90-second
  stop timeout, which killed the remaining API/worker processes. The service
  currently uses `KillMode=control-group`; shutdown ordering and CPU-worker
  lifetime need investigation before claiming reliable restart behavior.
- MTP became ready at 13:25:10 UTC. Both synthetic vision requests and all
  24 arithmetic requests at concurrency one/two/four pass. The matched
  128-input/64-output, two-request benchmark measured 3.17 output tokens/s,
  1.285 seconds mean TTFT and 299.88 ms mean subsequent-token time. Draft
  acceptance was 64.10% (50/78 tokens), with mean acceptance length 1.64.
  This approximately 49% improvement is a small-sample measurement and still
  falls far short of the required speed. The first actual full-context test
  uses 262,112 prompt tokens plus 32 forced output tokens, with three synthetic
  retrieval records at distant positions. It was cancelled before completion
  at the user's direction; this was the wrong test priority at current speed.
  Resume large-context tests only after performance is reasonable.
- Applied Windless84's open [upstream PR #56026](https://github.com/vllm-project/vllm/pull/56026),
  head `4a365dee4aed8ae3135213b4cf6ce04f6d82e756`, to identify all separately
  prefixed MTP cache groups. Extended its existing test for the target's two
  draft groups. The regression failed before the fix; all 107 cache utility
  tests now pass, as do scoped pre-commit and manual mypy 3.12 checks.
  The current engine predates this change; live verification requires restart.
  Do not infer actual GPU prefix reuse solely from the startup warning.
- A 13:40 UTC live allocation audit found no stale model instance: each GPU
  has exactly one current model worker, all in the serving unit's cgroup.
  The RAM-table worker owns an additional approximately 152 MB on one GPU.
  Model plus MTP is about 19.8 GiB/card, and KV allocation is 5.54 GiB/card.
  During full-context processing only 159–376 MB VRAM remained free, so the
  next launch needs a smaller explicit KV budget and a fresh capacity check.
- Cancelled the large prompt at 13:48:42 UTC and signalled the API first.
  The API still force-killed EngineCore after its 15-second ROCm cleanup
  grace, and the resource tracker reported leaked semaphore/shared-memory
  objects; do not call this a clean-shutdown fix. By 13:49:23 UTC every old
  inference PID was gone, AMD SMI reported no GPU processes and all four GPUs
  used only 16 MB. The next unit started at 13:49:47 with `KillMode=mixed`,
  a 4 GiB explicit per-GPU KV budget and a bounded torch profiler. The next
  request uses 128 input/32 output tokens, profiling six decoder iterations.
  Native context capacity remains configured, but large tests are deferred
  until the speed bottleneck is fixed, per the user's direction.
- All scoped pre-commit hooks, including mypy 3.10, pass with the vision
  interpolation change. The manual mypy 3.12 hook also passes. All 4,303
  Python files under `vllm` and `tests` passed syntax compilation. The imported
  mainline merge contains existing whitespace warnings outside these fixes;
  the unstaged integration diff passes `git diff --check`.

## Latest performance measurements

All generation rows below use 128 input tokens, 64 output tokens, two
measured requests after one warmup, concurrency one, seed-zero random inputs
and the server's default sampling temperature. These small samples do not
establish broad model quality or a guaranteed production throughput.

| Runtime | Output tokens/s | Mean TTFT | Mean TPOT | Draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| Eager, MTP off | 2.13 | 1.054 s | 459.17 ms | — |
| Eager, MTP one | 3.17 | 1.285 s | 299.88 ms | 64.10% |
| Decode graphs, MTP one | 3.20 | 1.319 s | 296.41 ms | 64.10% |
| BF16 skinny GEMM, graphs, MTP one | 16.48 | 1.076 s | 44.54 ms | 50.00% |
| BF16 expert GEMV plus skinny GEMM | 21.59 | 1.037 s | 30.58 ms | 53.01% |

The last row corresponds to about 32.7 tokens/s after the first token and
46.43 ms mean inter-response interval; MTP can return multiple tokens together.
All 24 elementary text checks at concurrency one/two/four and both synthetic
vision checks pass. Six fixed greedy responses match the preceding version's
text, but token log probabilities differ by up to 0.1245. No broad evaluation
against the BF16 checkpoint has run on this host.

The narrow gfx1030 skinny GEMM port keeps BF16 operands and FP32 accumulation.
It passed 132 isolated reference comparisons across four GPUs and 138 native
kernel/dispatch tests. The old HC projection `[2,10240] @ [10240,336]` measured
2.305 ms in isolation versus 0.0091 ms with the port. The integrated BF16
expert GEMV preserves the WNA16 INT4 layout, packed zero points, group-128
scales and BF16 dequantization. All 44 focused hardware cases pass, including
strides, routing weights, EP zeros, changing graph inputs and full MoE output
references. Its M=2 gate/up and down projections improved about 7.65x and
6.08x in isolated graph benchmarks. Applicable hooks and mypy 3.12 pass.

Uncached 1,024-token prompts with one output token and unique cache salts
measure 253.41 prompt tokens/s across three warmed requests. This remains
well below an optimized prefill target. Actual PyNccl/RCCL reductions measure
about 73–79 microseconds for small BF16 tensors on this topology. Exact dyadic
results and changing graph inputs pass on all four ranks. Explicit Tree/LL
and Ring/LL settings have mixed performance; serving keeps automatic selection.
The donor's custom RDNA all-reduce rejects BF16 and remains disabled.

The service uses a 4 GiB KV allocation per GPU and reports 291,356 KV tokens,
1.11 times the configured native context. This is a capacity estimate, not
successful native-context generation. After the short benchmark, each GPU
held about 19.80 GiB of model tensors, 4 GiB of KV, 0.36 GiB of other live
Torch tensors, 0.16 GiB of unused Torch reservation and 1.47–1.64 GiB outside
Torch reservation, leaving about 4.0–4.2 GiB free. These are live snapshots.
The original BF16 PLE table stays in RAM. `MemorySwapMax=0` prevents serving
swap, and measured swap remains zero.

The 15:19:57 and 15:32:48 UTC restarts verified all previous serving PIDs gone
and every GPU at 16 MB before replacing the process. This rules out overlapping
model instances at those restarts. Forced worker teardown and IPC cleanup
warnings remain; it is not a fully clean shutdown fix. The latest temporary
profiler measured four eager decode steps and restored graph dispatch on all
four ranks. The following text smoke test passed. Eager profiling overhead
is substantial, so its module intervals are not graph-mode throughput.

## Remaining work

1. Reach the requested 60–90 tokens/s and improve prefill using measured
   bottlenecks. A remaining scalar BF16 gate projection is under investigation.
2. Run broader model and vision evaluation, including MTP behavior and
   sustained concurrent requests. Elementary smoke checks are insufficient.
3. Diagnose the original SDMA copy fault and complete clean PLE/worker shutdown.
4. Validate actual 262,144-token generation after speed is reasonable, as the
   user directed. Keep model-body and KV tensors on GPUs, with PLE in RAM.
5. Qualify cold/warm loading and publish the tested fork with reproducible
   configuration and source attribution. Any upstream PR requires human review.
