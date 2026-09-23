# V620 CUDA Graph Piecewise Test

Status: prepared only. No launcher or validator in this branch has been run on
the serving host, and the active service has not been restarted.

## Isolation

Test branch: `codex/v620-piecewise-test-20260923`

Base commit: `532ed3f930f475e731c82726e079f3374ab5b33f`, the exact head of
PR #17 and the active service at preparation time. The test branch changes
launch and validation files only. It leaves the PR branch and its measured
defaults unchanged.

The prepared launcher refuses to overlap any existing vLLM process. The
validator contains no start, stop, restart, kill, or service-manager command.

## Profiles

`V620_PIECEWISE_PROFILE` selects one of three explicit profiles:

| Profile | Compilation | CUDA graph mode | Purpose |
| --- | --- | --- | --- |
| `piecewise-breakable` | mode 0 | `PIECEWISE` | First test; isolates piecewise graphs without the full-decode dispatcher. |
| `full-and-piecewise-breakable` | mode 0 | `FULL_AND_PIECEWISE` | Follow-up test of the ROCm breakable graph path without Inductor compilation. |
| `full-and-piecewise-compiled` | mode 3 | `FULL_AND_PIECEWISE` | Separate Inductor experiment after the breakable profile passes. |

All profiles preserve the measured model, TP4/EP4, MTP2, 3.75 GiB KV cache,
1,638,400-pixel vision cap, prefix checkpoints, resident INT4 experts, fused
shared expert, PLE offload, TunableOp table, and maximum four requests.

## Prepared launch

The first test profile is selected by the checked-in test unit. When a test
window is approved, the operator can switch the server checkout through Git,
install or link the checked-in unit, and start it. Those lifecycle actions are
intentionally not included in the validation script.

Foreground command for the first profile:

```bash
cd /home/george/v620-experiments/baseline-git-20260922
export V620_PIECEWISE_PROFILE=piecewise-breakable
export V620_HOST=0.0.0.0 V620_PORT=8080
exec tools/rdna2/serve_v620_piecewise_test.sh
```

Use `--dry-run` to print the resolved Python command without starting a model:

```bash
V620_PIECEWISE_PROFILE=piecewise-breakable \
  tools/rdna2/serve_v620_piecewise_test.sh --dry-run
```

## Validation

Run the validator only after the test service is healthy:

```bash
V620_PIECEWISE_PROFILE=piecewise-breakable \
  tools/rdna2/validate_v620_piecewise_test.sh
```

It performs these gates in order:

1. Confirms the API process advertises the requested test profile.
2. Runs 18 sequential full-completion correctness probes before load.
3. Runs `llm-context-bench` regular and coding cases at 16K, 32K, 64K, and
   128K with retries disabled.
4. Runs another 18 correctness probes after load to detect `duct`, `!`, NaN,
   repeated-character, and first-token-only corruption.
5. Runs four distinct prompts concurrently.
6. Verifies direct appended-turn checkpoint reuse at approximately 32K and
   128K.
7. Verifies a 1,260 × 1,260 image inside an approximately 35K conversation and
   requires at least 70% cached context plus a follow-up TTFT below half of the
   initial turn.

The speed measurements come only from `llm-context-bench`. Other probes are
correctness and cache-reuse gates.

## Qualification criteria

The profile is eligible for comparison only if:

- Startup logs show the requested piecewise or breakable graph captures.
- Every pre-load, post-load, and concurrent correctness probe passes.
- Regular and coding 64K/128K cases finish without timeout or preemption.
- The 32K and 128K appended turns pass the cache and TTFT thresholds.
- The multimodal long-chat probe passes at the configured vision cap.
- No worker crash, GPU page fault, NaN, `duct`, or bang-storm output appears.

Compare valid benchmark rows against the saved PR #17 baseline:

| Case | Baseline prefill tok/s | Baseline decode tok/s |
| --- | ---: | ---: |
| Regular 16K | 1,957.8 | 69.7 |
| Coding 16K | 1,982.8 | 68.5 |
| Regular 32K | 1,977.2 | 67.5 |
| Regular 64K | 1,932.5 | 56.6 |
| Coding 64K | 1,928.0 | 69.8 |
| Regular 128K | 1,823.5 | 54.9 |
| Coding 128K | 1,813.9 | 78.2 |

Record startup time and per-GPU memory along with throughput. Retain the
decode-only service unless the piecewise profile passes every correctness gate
and produces a repeatable performance or latency gain within the same 3.75 GiB
KV and vision budget.
