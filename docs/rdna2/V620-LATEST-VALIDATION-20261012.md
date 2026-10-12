# V620 latest-runtime validation — 2026-10-12

Production reference: `7462596a5f67ea87aaa0d67fa2696b4efe825212`.
Candidate upstream pin: `64d59b8942d3b982f1d91f0f3e625dfda4418106` (v0.31).
Branch: `codex/rdna-latest-validation-20261012`.

All source and harness edits are local → signed commit → push → server Git
clone/pull. Production source, native artifacts and dependency environment
must remain unchanged. Use separate loopback-only temporary services.

Normal Intel AutoRound weights, FP16 body, original BF16 CPU PLE, TP4/EP4/PP1,
context262144, block1024, seq4, batch4096, KV4026531840/rank, RAM64GiB, image255
and video32, full multimodal profiling, and MTP2 are retained. No W4A8, INT8
shadows, quantized PLE, SSD KV, clock or power changes. The new runtime uses
upstream BF16 PLE offload, FULL_AND_PIECEWISE graphs/compile3, native attention,
new HC fusion and required TP MTP local-argmax configuration. These are a stack
comparison, not an isolated one-variable HC benchmark. The existing ROCm10
library and matching 70-row TunableOp profile are retained.

## Current server state and restoration

Initial state: no model running; `v620-tp4-git.service` is a broken symlink to
the removed `kv-offload-ram-20260926` checkout. Source, checkpoint and original
Python environment still exist. No persistent unit/link has been changed.
The user authorized repairing/restoring the original default after testing.
Both links were repaired to the preserved production base unit at16:31:41UTC;
the existing Intel-model/64GiB drop-ins remain unchanged. At16:37:57UTC the
restored default returned health200 and exact READY in0.384s, source7462596a5,
PID67526, enabled for boot, restart count0. The candidate is stopped, not promoted.

## Planned qualification

1. Fresh production baseline using its unchanged launcher and source.
2. Isolated full native build, import/provenance and relevant HC/PLE tests.
3. Warm matched llm-context-bench coding16K/64K/128K C1 and16K C3, three
   valid repetitions, normal EOS/output budgets/sampling, identical fresh
   fixture/tag scopes. Preserve failed reports; never loosen output guards.
4. Canonical paired regular+coding quality, all sizes. Compare case verdicts,
   not execution-success alone; no blanket quality pass from unit tests.
5. Prefix reuse, three100K chats with tool continuations, 200K→100K→200K RAM
   reload and image restoration, with positive measured transfer traffic.
6. Report exact environment/commands, PP and decode separately, aggregate vs
   per-chat clearly, memory/errors, and restoration status. No auto-promotion.

## Qualification outcome so far

The isolated native build completed in11m12s; native SHA256 is
`441180e95b2523e99e426bc2316949082fa5df156d4c9b66de905087a3ae2bb9`.
Original native artifacts and dependency environment are unchanged.
Candidate Hub2.2 initially prevented test collection; the builder now limits
Hub to<2, matching preserved Transformers5.17.0. Import succeeds with Hub1.33.
All34 HC and8 AMD PLE tests pass, including PLE graph replay and FP16/BF16 checks.

The unmodified64d candidate nevertheless fails model initialization on all four
ranks with `Qwen4Exp PLE embedding does not support quantization config INCConfig`.
The same exception reproduces with only the actual checkpoint configuration
and `Qwen4ExpPLEEmbeddingMethod.from_quant_config`, without loading weights.
INC4-bit/group128 is correctly detected; the PLE selector lacks INC support.
No source, quantization, precision or model workaround was applied.

Candidate benchmarks, quality, tool/prefix and RAM/image gates therefore could
not run. **No latest-versus-production speedup is established.**

Fresh production-only normal coding measurements (tok/s):

| Workload | PP | TG | Valid groups |
| --- | ---: | ---: | ---: |
| 16K C1 | 2022.11 | 73.71 | 3/3 |
| 64K C1 | invalid | invalid | 0/3 |
| 128K C1 | 1819.58 | 70.65 | 3/3 |
| 16K C3 aggregate | 1605.97 | 56.13 | 3/3 |

Initial64K and one separate locked retake both fail the repeated-output guard.
Canonical production quality: regular8/9, coding3/9, total11/18. This is not a
quality pass; preserve raw failures and do not relax guards or retry to green.
Full reports and exact environment/command manifests are preserved under the
workspace's `review-artifacts/rdna-latest-validation-2026-10-12/` directory.

## Active collective backend

The initial production control logs show RCCL fallback, despite its exported
AR flag1: an existing October8 native-AR timeout marker disables the native
path. Preserve that original marker and include it in the audit. The first
candidate run must pass `VLLM_RDNA_AR=0` to the launch wrapper so the backend is
matched. Any later latest-native-AR trial is a separate arm, not an unlabeled
source-only gain or a silent reset of production's safety state.
