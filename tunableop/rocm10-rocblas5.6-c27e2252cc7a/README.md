# Preserved production reference: c27e2252cc7a

These four rank files are byte-identical to `ab332164d`'s
`tunableop/rocm10-rocblas5.6/`. This preserves the existing production solver
selection during a source-only A/B. This is not newly generated or newly
qualified tuning data.

The library SHA256 is
`c27e2252cc7a4b6e174601a180a169822dcf15a29e634046daecb61498e3bd2a`.
The newer `rocm10-rocblas5.6` profile targets `d05939967b5f`, a different
rocBLAS binary. Never apply those solver IDs to this build or disable the hash
guard. Keep the existing runtime for the first comparison.
