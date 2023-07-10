#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
# PYTHON: Python interpreter, default: /usr/bin/python3
# OOBI_VENV_DIR: path to create Python virtualenv, default: pt-benchmarks.venv
# OOBI_TARGET_DEVICE: target benchmark device, can also be specified the first
#   argument.
# OOBI_OUTPUT: path to output benchmark results, can also be specified the
#   second argument.
#
# Example usage:
# ./benchmark_all.sh a2-highgpu-1g /tmp/results.json

set -xeuo pipefail

PYTHON="${PYTHON:-/usr/bin/python3}"
VENV_DIR="${OOBI_VENV_DIR:-pt-benchmarks.venv}"
TARGET_DEVICE="${1:-${OOBI_TARGET_DEVICE}}"
OUTPUT_PATH="${2:-${OOBI_OUTPUT}}"

TD="$(cd $(dirname $0) && pwd)"

VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

declare -a GPU_BENCHMARK_NAMES=(
  "models/RESNET50_FP32_PT_.+/inputs/.+/expected_outputs/.+/target_devices/a2-highgpu-1g"
  "models/RESNET50_FP16_PT_.+/inputs/.+/expected_outputs/.+/target_devices/a2-highgpu-1g"
  "models/BERT_LARGE_FP32_PT_.+/inputs/.+/expected_outputs/.+/target_devices/a2-highgpu-1g"
  "models/BERT_LARGE_FP16_PT_.+/inputs/.+/expected_outputs/.+/target_devices/a2-highgpu-1g"
)

declare -a CPU_BENCHMARK_NAMES=(
  "models/RESNET50_FP32_PT_.+_BATCH1/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
  "models/RESNET50_FP32_PT_.+_BATCH32/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
  "models/RESNET50_FP32_PT_.+_BATCH64/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
  "models/BERT_LARGE_FP32_PT_.+_BATCH1/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
  "models/BERT_LARGE_FP32_PT_.+_BATCH32/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
  "models/BERT_LARGE_FP32_PT_.+_BATCH64/inputs/.+/expected_outputs/.+/target_devices/c2-standard-16"
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
