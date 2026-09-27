# Swift / V620 decode priorities — 2026-09-27

Investigation only, on `codex/rdna-decode-performance-investigation`, based
on `28046ee3a`. No server files, settings, service state, or workloads were
changed during this investigation. Read-only startup logs, metrics, checkpoint
metadata, process mappings, and ELF symbols were inspected. No new performance
measurements were made. AI assistance was used.

## Current service evidence

Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound, TP4/EP4, MTP2,
262144 model limit, 4096 scheduled-token budget, FULL_DECODE_ONLY,
capture sizes 3/6/12, prefix caching and Mamba align, 64 GiB RAM KV.
The model has 36 GDN and 12 QSA target layers, plus MTP.

- Startup logs at 12:20:07 confirm `rocm_moe_skinny: using
  moe_skinny_int4_decode (M=6, topk=10, N=640, sequential Triton WNA16
  layout)` on all four ranks. Thus the TRITON backend label does **not** mean
  every MoE decode executes Triton. The disabled resident-skinny setting is a
  different path.
- FULL target and draft graphs were captured. Graph capture is not a
  measurement of replay coverage under every mixed workload.
- PLE weight loading completed; its 128 BF16 checkpoint table shards contain
  51,200,245,760 values / 102,400,491,520 bytes / 95.3679 GiB. The CPU
  embedding explicitly uses BF16 in `vllm/models/qwen4_exp/amd/ple_layer.py`.
- Current cache counters contained only the 26-token smoke prompt: 26 queries,
  zero hits. That sample cannot assess real conversation cache effectiveness.
- `rdna_ar` failed its startup self-test on every rank with missing
  `_rocm_C.rdna_ar_timeout_info`. Both TP and EP selected only PYNCCL/RCCL.
  This is an operator-availability failure, not evidence of bad physical P2P.
- `/proc/12698/maps` identified the loaded extension as
  `/home/george/v620-experiments/upstream-20260915/v620-vllm-testing/source/vllm/_rocm_C.abi3.so`.
  The current Git checkout contains no `_rocm_C` binary. Read-only `nm -D`
  inspection found `rdna_ar_init`, `rdna_ar_all_reduce`,
  `moe_skinny_int4_decode`, `moe_gptq_gemm_rdna2`, and `gdn_decode_rdna2`,
  but not `rdna_ar_timeout_info`; current source declares and registers it.
  This supports a Python/native build mismatch. No rebuild was attempted.

## Priority order

1. **Restore matched native build and qualify rdna_ar.** Existing implementation,
   concrete missing operator, relatively low implementation effort. Startup
   self-test plus changing-input graph all-reduce tests must pass before A/B.
   The configured 64 KiB cutoff targets decode-sized messages, not large
   prefills. Keep RCCL fallback and timeout protection; do not delete the check.
   Performance benefit remains unmeasured on Swift.
2. **Qualify existing HIP skinny MoE at M=9–12.** Python dispatch caps M at 8
   (`rocm_moe_skinny.py`); native `skinny_gemms_int4.cu` accepts 1–16.
   MTP2 target verification has three token rows per request: three concurrent
   requests need nine rows and the current graph ladder can pad to twelve.
   This exceeds the skinny cap, selecting the fallback. Test M=1/3/6/9/12/16,
   EP mapping, padding, changing-input graph replay, and performance crossover.
   Increasing a cap is not sufficient proof of correctness or speed. Sequential
   symmetric packing and shuffled GPTQ/AWQ layouts must remain distinct.
3. **GDN decode/MTP fusion after profiling.** Native non-spec decode already
   exists, but it is not equivalent to the multi-token speculative path.
   Do not transfer older non-spec microbenchmarks to current Swift MTP2.
4. **PLE fusion/CPU transfer overlap only if on the critical path.** Working
   CPU BF16 PLE is already present. Table size does not imply the whole table
   crosses PCIe per token; selected rows are gathered. Do not change precision
   or replace CPU offload with upstream UVA without separate validation.
5. **Piecewise prefill graphs later.** Requires prefill-sized capture coverage,
   additional memory, and preserving genuine FULL target/draft replay. Existing
   historical piecewise results regressed decode under an older FULL-to-piecewise
   redirect; the branch now disables that redirect. Neither old regression nor
   the fix establishes present performance. See `V620-FULL-AND-PIECEWISE.md`.
6. **Generic fa_rdna2 last for this model.** Target attention uses custom sparse
   QSA, not generic FlashAttention; `Qwen4ExpQSAFlashAttentionImpl` explicitly
   bypasses `flash_attn_varlen_func`. Optimize QSA/indexer if profiles justify it.
   A generic FA kernel replacement does not automatically accelerate QSA.

Prefix caching remains a correctness and chat-turn latency priority, separate
from steady-state tokens/s. Validate unchanged token prefixes, growing tool
transcripts, distinct sessions, image reuse, and RAM reload separately once
workload testing is authorized. Compare metric deltas rather than lifetime hit
ratios or fresh-prompt benchmarks. Prefix caching can indirectly help concurrent
decode by avoiding unnecessary prefill work.

## Historical evidence, not Swift guarantees

`V620-BASELINE-PORT-20260922.md` records a bundled resident-MoE change with
33–38% prefill improvement on the prior Intel checkpoint. The later skinny
toggle did not establish a consistent decode benefit. These results do not
justify a numerical Swift speedup prediction.

`qwen_gdn_linear_attn.py::_gdn_prefill_dispatch_available` keeps native HIP
prefill opt-in and records an older 8.7% elapsed-time regression versus Triton.
HIP is not inherently faster than Triton. Avoid changing this setting solely
to eliminate JIT: warm-cache steady-state execution is a separate measurement.

## Later experiment contract

Only with authorization for server testing: local edit, focused tests, Git
commit/push, server Git pull, exact matching native build, isolated test service
or agreed restart, then A/B. Measure per-session inter-token latency and aggregate
tokens/s at concurrency 1/2/3/4 with MTP acceptance, identical workload distribution,
cache-controlled prompts, and warm kernels. Include tool-return/mixed-prefill
batches and long-context RAM eviction/resume. No projected percentage is treated
as measured improvement.
