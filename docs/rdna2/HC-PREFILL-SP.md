# Experimental HC prefill compute sharding

`VLLM_RDNA_HC_PREFILL_SP=1` partitions the two replicated hyperconnection
FP16 matrix multiplications and gate mixing across tensor-parallel ranks.
Each rank computes contiguous token rows, then all-gathers the narrow block
input and low-rank down/injection output in original token order. The wide
gate stays local. Weights, activation formats and FP32 gate arithmetic remain
unchanged. There is no new quantization or cached shadow weight.

Default is off. The route requires FP16 contiguous input, at least 1024 rows,
more than one TP rank, a row count divisible by TP size and no INT8 shadows.
Small decode batches, mixed-prefill tails that fail the divisibility check,
TP1 and other formats retain their original implementation. This does not
enable sequence-parallel MoE or change PLE, convolution state, prefix/KV
offload, model hidden-state layout, MTP or scheduling policy.

This is a contained implementation of the replicated-compute opportunity
described by [mainline PR56322](https://github.com/vllm-project/vllm/pull/56322),
not a port of its full model sequence-parallel machinery. Different GEMM row
shapes can choose different FP16 accumulation algorithms, so matching precision
alone does not establish output equivalence. Independent numerical tests,
multi-rank ordering tests and model-quality checks are required. Additional
PCIe collectives can cost more than the saved computation: whole-model PP,
decode and three-chat results determine whether the route is useful.

## Measurement

Use an isolated server and original model/launch parameters. The optional
`V620_PROFILE_PREFILL=1` launcher switch adds a diagnostic worker extension.
GPU event spans cover execution and host-launch gaps; they are not summed
kernel-only times. They do not synchronize each measured call. Profile only
after warmup, and time normal uninstrumented requests separately.

The extension provides `start_prefill_stage_profile`,
`stop_prefill_stage_profile` and `set_hc_prefill_sp` through the loopback-only
development RPC endpoint. Never expose development RPC on a public interface.
The toggle allows controlled off/on/off trials in one loaded process; it is
not a production configuration API. No speedup is established yet.
