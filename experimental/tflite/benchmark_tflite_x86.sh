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
#
# Example usage:
# ./benchmark_tflite_x86.sh <target-device> <results-path>

set -xeuo pipefail

VENV_DIR="${OOBI_VENV_DIR:-tflite-benchmarks.venv}"
PYTHON="${PYTHON:-/usr/bin/python3}"
ROOT_DIR="${OOBI_TEMP_DIR:-/tmp/openxla-benchmark}"
TARGET_DEVICE="${1:-"${OOBI_TARGET_DEVICE}"}"
OUTPUT_PATH="${2:-"${OOBI_OUTPUT}"}"

# Download benchmark tool.
wget -P "${ROOT_DIR}" "https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model"
BENCHMARK_BINARY_PATH="${ROOT_DIR}/linux_x86-64_benchmark_model"
chmod +x "${BENCHMARK_BINARY_PATH}"

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

"${TD}/../../comparative_benchmark/scripts/create_results_json.sh" "${OUTPUT_PATH}"

declare -a BENCHMARK_NAMES=(
  "models/BERT_BASE_FP32_TFLITE_I32_SEQLEN.+/.+"
  "models/BERT_BASE_FP16_TFLITE_I32_SEQLEN.+/.+"
  "models/BERT_BASE_DYN_QUANT_TFLITE_I32_SEQLEN.+/.+"
  "models/BERT_BASE_INT8_TFLITE_I32_SEQLEN.+/.+"
  "models/VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32/.+"
  "models/VIT_CLASSIFICATION_FP16_TFLITE_3X224X224XF32/.+"
  "models/VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32/.+"
  "models/VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8/.+"
)

# Thread-to-taskset config. If the taskset is blank, the benchmark is not pinned to a specific cpu id.
# Here we run the benchmarks on 1, 8, 15 and 30 threads that are not pinned to any cores.
THREAD_CONFIG="{1: '', 8: '', 15: '', 30: ''}"
ITERATIONS=5

for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
  "${TD}/run_benchmarks.py" \
    --benchmark_name="${benchmark_name}" \
    --target_device="${TARGET_DEVICE}" \
    --output="${OUTPUT_PATH}" \
    --tflite_benchmark_binary="${BENCHMARK_BINARY_PATH}" \
    --thread_config="${THREAD_CONFIG}" \
    --iterations="${ITERATIONS}" \
    --root_dir="${ROOT_DIR}" \
    --verbose
done
