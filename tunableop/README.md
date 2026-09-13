# V620 FP16 TunableOp rows

These rows reuse Leapdragon's build-specific, lookup-only TunableOp approach.
They were generated with the matching ROCm wheel SDK on four V620s. They do not
quantize weights or activations. Solution IDs must never be reused across a
different rocBLAS build, even when version strings match.

## Qualified build

`rocblas-c27e2252cc7a` contains 21 FP16 dense matrix shapes for 1,024, 2,048, and
4,096 input rows. All shapes passed independent FP32 comparisons on all four
GPUs. `provenance.json` records the full library hash and package versions.
Each rank uses an identical copy of the qualified rows.

The launcher checks the library hash and requires all four rank files. It keeps
online tuning disabled. Runtime-generated rows under the testing directory take
precedence over these bundled rows; `V620_TUNABLEOP_ROOT` selects an explicit root.

```bash
V620_MM_LIMIT='{"image":4,"video":1}' \
V620_MTP_TOKENS=0 \
VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 HSA_FORCE_FINE_GRAIN_PCIE=1 \
V620_TUNABLEOP=1 \
V620_ROCBLAS_LIBRARY=/path/to/site-packages/_rocm_sdk_libraries/lib/librocblas.so.5 \
bash tools/rdna2/serve_v620_candidate.sh --max-num-batched-tokens 4096
```

## Full-model validation

Tested with the existing Intel AutoRound INT4 expert checkpoint, FP16 dense
weights, original BF16 PLE in CPU RAM, TP4/EP4, and MTP disabled. Dense INT8 shadows
were disabled. The configured context was 262,144 tokens, with 4 GiB of KV memory
per GPU. This validation covered 16k and 32k benchmark tiers.

The unmodified `llm-context-bench` runner at
`92286b24065565f4929e78c45f776029480e9939` used its locked sampling and requested
1,024 output tokens. An explicit 11% input-size tolerance accommodates this
tokenizer; actual prompt counts are shown below.

| Workload | Actual prompt tokens | Before prefill tok/s | Tuned prefill tok/s | Tuned decode tok/s | Trial |
| --- | ---: | ---: | ---: | ---: | --- |
| Code 16k | 18,063 | 1,194.77 | 1,441.48 | 41.21 | Valid |
| Code 32k | 36,135 | 1,194.31 | 1,441.21 | 41.10 | Valid |
| Prose 16k | 16,750 | 1,031.65 | 1,403.52 | 41.32 | Invalid: stopped at 731 output tokens |
| Prose 32k | 33,455 | — | 1,444.13 | 41.20 | Invalid: stopped at 776 output tokens |

The prose timings are retained observations and are excluded from valid trial
comparisons. The untuned prose 32k trial also stopped early. Both configurations
passed all four long-context quality cases and the single/four-request smoke
checks. This is a bounded regression evaluation, not a broad accuracy benchmark.

The tuned run reached a healthy API in 243.25 seconds, versus 255.26 seconds for
the untuned MTP0 run. These were successive boots with existing filesystem caches,
not a controlled cold-storage startup comparison.

## Regenerate for another rocBLAS build

Run only with the inference service stopped, using the isolated testing
environment and its matching SDK library path. The tuning script deliberately
removes the serving environment's TunableOp enable/tuning overrides before
importing Torch: those environment variables take precedence over the Python API.

```bash
.venv/bin/python tools/rdna2/tune_v620_fp16.py --output-root /path/to/testing/tunableop
.venv/bin/python tools/rdna2/qualify_v620_fp16.py /path/to/testing/tunableop/rocblas-HASH
```

The first command tunes GPU 0 and records solver rows after every measured shape.
The second replays them on all four cards against FP32 references before writing
the other three serving files. Run the full model's output checks and benchmark
after qualification. Tuning may choose a different floating-point reduction
order; retaining FP16 weights does not imply bit-identical outputs.
