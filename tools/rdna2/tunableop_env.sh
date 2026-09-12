#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Sourced by the candidate launcher. Adapted from leapdragon c41f8f4c8.
# Solution IDs belong to a rocBLAS build, not just a version number.
configure_v620_tunableop() {
    local library=$1 rows_root=$2 library_id rows_dir rank
    export PYTORCH_TUNABLEOP_ENABLED=0
    export PYTORCH_TUNABLEOP_TUNING=0
    export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
    if [[ ! -f $library ]]; then
        printf 'TunableOp lookup disabled: rocBLAS library is unavailable.\n' >&2
        return 1
    fi
    library_id=$(sha256sum -- "$library")
    library_id=${library_id:0:12}
    rows_dir=$rows_root/rocblas-$library_id
    for rank in 0 1 2 3; do
        if [[ ! -s $rows_dir/tunableop_results$rank.csv ]]; then
            printf 'TunableOp lookup disabled: missing rank %s rows for %s.\n' "$rank" "$library_id" >&2
            return 1
        fi
    done
    export PYTORCH_TUNABLEOP_FILENAME=$rows_dir/tunableop_results.csv
    export PYTORCH_TUNABLEOP_ENABLED=1
    printf 'TunableOp lookup enabled for rocBLAS build %s; tuning stays off.\n' "$library_id" >&2
}
