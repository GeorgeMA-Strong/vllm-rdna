#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Sourced by the MTP launcher. Mirror of tools/rdna2/tunableop_env.sh for the
# MTP rows in tunableop/rocblas-f30bb442e9b5/. Solution IDs belong to a rocBLAS
# build, not just a version number.
configure_mtp_tunableop() {
    local library=$1 rows_root=$2 library_id rows_dir rank
    export PYTORCH_TUNABLEOP_ENABLED=0
    unset PYTORCH_TUNABLEOP_FILENAME
    export PYTORCH_TUNABLEOP_TUNING=${PYTORCH_TUNABLEOP_TUNING:-0}
    export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
    if [[ ! -f $library ]]; then
        printf 'WARNING: TunableOp disabled: rocBLAS library is unavailable; using default FP16 algorithms.\n' >&2
        return 0
    fi
    library_id=$(sha256sum -- "$library")
    library_id=${library_id:0:12}
    rows_dir=$rows_root/rocblas-$library_id
    for rank in 0 1 2 3; do
        if [[ ! -s $rows_dir/tunableop_results$rank.csv ]]; then
            printf 'WARNING: TunableOp disabled: missing rank %s rows for rocBLAS %s; using default FP16 algorithms.\n' "$rank" "$library_id" >&2
            return 0
        fi
    done
    export PYTORCH_TUNABLEOP_FILENAME=$rows_dir/tunableop_results.csv
    export PYTORCH_TUNABLEOP_ENABLED=1
    printf 'TunableOp lookup enabled for rocBLAS build %s; tuning stays off (set PYTORCH_TUNABLEOP_TUNING=1 to re-tune missing shapes).\n' "$library_id" >&2
}
