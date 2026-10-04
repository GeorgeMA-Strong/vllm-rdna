# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict real-checkpoint HC comparison with matched full/shard GEMM solvers."""

import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-solvers", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "PYTORCH_TUNABLEOP_ENABLED",
        "PYTORCH_TUNABLEOP_TUNING",
        "PYTORCH_TUNABLEOP_FILENAME",
    ):
        os.environ.pop(name, None)
    os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    os.environ["VLLM_RDNA_HC_PREFILL_HIP"] = "1"
    import torch
    import torch.cuda.tunable as tunable
    from safetensors import safe_open

    from vllm.models.qwen4_exp.amd.ops.hc import hc_gate_mix, hc_silu

    rows = list(csv.reader(args.profile.open()))
    solver_map = {}
    for n, k in ((336, 10240), (10240, 320)):
        full = f"tn_{n}_4096_{k}_ld_{k}_{k}_{n}"
        shard = f"tn_{n}_1024_{k}_ld_{k}_{k}_{n}"
        selected = next(
            row[2]
            for row in rows
            if row[0] == "GemmTunableOp_Half_TN" and row[1] == full
        )
        solver_map[shard] = selected
        if args.match_solvers:
            for row in rows:
                if row[0] == "GemmTunableOp_Half_TN" and row[1] == shard:
                    row[2] = selected
    profile = args.output_dir / "profile.csv"
    with profile.open("w") as output:
        csv.writer(output).writerows(rows)
    torch.accelerator.set_device_index(0)
    tunable.set_filename(str(args.output_dir / "used.csv"), insert_device_ordinal=False)
    tunable.enable(True)
    tunable.tuning_enable(False)
    assert tunable.read_file(str(profile))
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = "model.language_model.layers.0.mlp_hyper_connection."

    def tensor(suffix):
        name = prefix + suffix + ".weight"
        with safe_open(args.model / index[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name).to(device="cuda", dtype=torch.float16)

    down = torch.cat(
        (
            tensor("input_mix_weight_down"),
            tensor("block_inject_weight"),
            torch.zeros(12, 10240, device="cuda", dtype=torch.float16),
        ),
        dim=0,
    )
    up = tensor("input_mix_weight_up")
    generator = torch.Generator(device=down.device).manual_seed(620)
    x = torch.randn(
        4096, 10240, device="cuda", dtype=torch.float16, generator=generator
    )

    def compute(value):
        dai = torch.nn.functional.linear(value, down)
        gate = torch.nn.functional.linear(hc_silu(dai[:, :320].contiguous(), 4), up)
        return hc_gate_mix(value, gate, 4), dai, gate

    expected = compute(x)
    parts = [compute(value) for value in x.split(1024)]
    actual = [torch.cat([part[i] for part in parts], dim=0) for i in range(3)]
    result = {
        "match_solvers": args.match_solvers,
        "requested_solvers": solver_map,
        "comparisons": [],
        "used_results": tunable.get_results(),
    }
    for name, value, reference in zip(
        ("block_input", "down", "gate"), actual, expected
    ):
        errors = []
        try:
            torch.testing.assert_close(value, reference)
        except AssertionError as error:
            errors.append(str(error))
        result["comparisons"].append(
            {
                "name": name,
                "errors": errors,
                "max_abs": (value - reference).abs().max().item(),
            }
        )
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "used_results"}
        ),
        flush=True,
    )
    if any(value["errors"] for value in result["comparisons"]):
        raise RuntimeError("Strict HC output comparison failed")


if __name__ == "__main__":
    main()
