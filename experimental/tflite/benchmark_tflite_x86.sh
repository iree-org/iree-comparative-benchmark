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
# OOBI_TARGET_DEVICE: target benchmark device, can also be specified the first
#   argument.
# OOBI_VENV_DIR: name of the virtual environment.
# OOBI_TEMP_DIR: the path to store temporary files to.
# TFLITE_BENCHMARK_BINARY: the path to a custom built TFLite benchmark binary.
#
# Example usage:
# ./benchmark_tflite_x86.sh <target-device> <results-path>

set -xeuo pipefail

VENV_DIR="${OOBI_VENV_DIR:-tflite-benchmarks.venv}"
PYTHON="${PYTHON:-/usr/bin/python3}"
ROOT_DIR="${OOBI_TEMP_DIR:-/tmp/openxla-benchmark}"
TARGET_DEVICE="${1:-"${OOBI_TARGET_DEVICE}"}"
OUTPUT_PATH="${2:-"${OOBI_OUTPUT}"}"

# Download benchmark tool if not set.
if [[ -z "${TFLITE_BENCHMARK_BINARY}" ]]; then
  wget -P "${ROOT_DIR}" "https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model"
  TFLITE_BENCHMARK_BINARY="${ROOT_DIR}/linux_x86-64_benchmark_model"
  chmod +x "${BENCHMARK_BINARY_PATH}"
fi

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

"${TD}/../../comparative_benchmark/scripts/create_results_json.sh" "${OUTPUT_PATH}"

# We don't benchmark FP16 or BF16 on CascadeLake since they do not have CPU support.
declare -a BENCHMARK_NAMES=(
  "models/BERT_BASE_FP32_TFLITE_I32_SEQLEN.+/.+"
  "models/BERT_BASE_DYN_QUANT_TFLITE_I32_SEQLEN.+/.+"
  "models/BERT_BASE_INT8_TFLITE_I32_SEQLEN.+/.+"
  "models/VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32/.+"
  "models/VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32/.+"
  "models/VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8/.+"
)

# Thread-to-taskset config. If the taskset is blank, the benchmark is not pinned to a specific cpu id.
# Here we run the benchmarks on 1, 8, 15 and 30 threads that are not pinned to any cores.
THREAD_CONFIG="{1: '-c 0', 8: '-c 0-7', 15: '-c 0-14', 30: '-c 0-29'}"
ITERATIONS=5

for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
  "${TD}/run_benchmarks.py" \
    --benchmark_name="${benchmark_name}" \
    --target_device="${TARGET_DEVICE}" \
    --output="${OUTPUT_PATH}" \
    --tflite_benchmark_binary="${TFLITE_BENCHMARK_BINARY}" \
    --thread_config="${THREAD_CONFIG}" \
    --iterations="${ITERATIONS}" \
    --root_dir="${ROOT_DIR}" \
    --verbose
done
