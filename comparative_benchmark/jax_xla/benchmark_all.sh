#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

VENV_DIR="${OOBI_VENV_DIR:-jax-benchmarks.venv}"
PYTHON="${PYTHON:-/usr/bin/python3}"
TARGET_DEVICE="${1:-${OOBI_TARGET_DEVICE}}"
OUTPUT_PATH="${2:-${OOBI_OUTPUT}}"

TD="$(cd $(dirname $0) && pwd)"

if [ "${TARGET_DEVICE}" = "a2-highgpu-1g" ]; then
  export WITH_CUDA=1
fi

VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"
unset WITH_CUDA

declare -a GPU_BENCHMARK_NAMES=(
  "models/T5_LARGE_FP32_JAX_512XI32_BATCH\d+/inputs/.+/expected_outputs/.+/target_devices/a2-highgpu-1g"
  "models/T5_LARGE_4CG_FP32_JAX_512XI32_BATCH\d+/inputs/.+/expected_outputs/.+/target_devices/a2-highgpu-1g"
)

declare -a CPU_BENCHMARK_NAMES=(
  "models/T5_LARGE_FP32_JAX_512XI32_BATCH1/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
  "models/T5_LARGE_FP32_JAX_512XI32_BATCH16/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
  "models/T5_LARGE_FP32_JAX_512XI32_BATCH32/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
  "models/T5_LARGE_4CG_FP32_JAX_512XI32_BATCH1/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
  "models/T5_LARGE_4CG_FP32_JAX_512XI32_BATCH16/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
  "models/T5_LARGE_4CG_FP32_JAX_512XI32_BATCH32/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
)

if [ "${TARGET_DEVICE}" = "a2-highgpu-1g" ]; then
  BENCHMARK_NAMES=("${GPU_BENCHMARK_NAMES[@]}")
  ITERATIONS=50
elif [ "${TARGET_DEVICE}" = "c2-standard-16" ]; then
  BENCHMARK_NAMES=("${CPU_BENCHMARK_NAMES[@]}")
  ITERATIONS=20
else
  echo "Unsupported target device ${TARGET_DEVICE}."
  exit 1
fi

"${TD}/../scripts/create_results_json.sh" "${OUTPUT_PATH}"

for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
  "${TD}/run_benchmarks.py" \
    --benchmark_name="${benchmark_name}" \
    --output="${OUTPUT_PATH}" \
    --iterations="${ITERATIONS}" \
    --verbose
done
