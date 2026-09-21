# V620 baseline before porting the fast service

The baseline source is upstream `rdna_extras` at `f3dd65fa70636566c11b249209d281c0b63c819e`.
Commit `9daf24d68949be29b9506ce640fd46367ca0b9cc` adds only the
benchmark launcher. No runtime source edits were present during this run.
The server cloned and pulled the branch from GitHub and built its own gfx1030
native modules from that checkout.

The server ran four V620 GPUs with TP4, EP4, MTP2, FP16, CPU PLE, a 4096-token
batch limit, and the launch options in `tools/rdna2/serve_v620_baseline.sh`.
The model was `/home/george/v620-vllm/models/intel-autoround`.
The active fast service was stopped and had zero running or waiting requests
before the baseline started.

Only `llm-context-bench` was used. The locked harness checkout was `92286b2`,
with SHA-256 `ec553141cd5f63d962ec1bbe6fbaf65effc44d032e5d055d4e6376824138e5ae`
reported by the result file. The command used both suites, the performance
lane, 16k inputs, one measured repetition per case, streaming, a 240-second
timeout, and disabled thinking. The harness performed its normal warmup.

| 16k suite | Prompt tokens | Prefill tokens/s | Generation tokens/s | TTFT s | Valid |
| --- | ---: | ---: | ---: | ---: | --- |
| Regular prose | 16,750 | 1,029.1 | 57.1 | 16.28 | Yes |
| Coding | 18,063 | 1,485.4 | 66.8 | 12.16 | Yes |

These are single-run observations, not confidence intervals. The full JSON
result remains on the V620 server at
`/home/george/v620-experiments/base-16k-f3dd65fa7.json`.
