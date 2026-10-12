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
Asked the user whether to repair/restore the old default after testing or leave
the original inactive state. Until answered, do not modify persistent units.

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

No benchmark result or speedup is claimed yet.
