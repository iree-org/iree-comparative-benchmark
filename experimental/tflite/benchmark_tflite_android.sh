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
# OOBI_ANDROID_BENCHMARK_DIR: the on-device directory where benchmark artifacts are copied to.
# TFLITE_BENCHMARK_BINARY: the path to a custom built TFLite benchmark binary.
#
# Example usage:
# ./benchmark_tflite_x86.sh <target-device> <results-path>

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${OOBI_VENV_DIR:-tflite-benchmarks.venv}"
PYTHON="${PYTHON:-/usr/bin/python3}"
ROOT_DIR="${OOBI_ANDROID_BENCHMARK_DIR:-/data/local/tmp/tflite_benchmarks}"
TARGET_DEVICE="${1:-"${OOBI_TARGET_DEVICE}"}"
OUTPUT_PATH="${2:-"${OOBI_OUTPUT}"}"

# Setup benchmarking environment.
adb shell "rm -rf ${ROOT_DIR}"
adb shell "mkdir ${ROOT_DIR}"

adb push "${TD}/../../comparative_benchmark/scripts/set_android_scaling_governor.sh" "${ROOT_DIR}"
adb shell "chmod +x ${ROOT_DIR}/set_android_scaling_governor.sh"
adb shell "su root sh ${ROOT_DIR}/set_android_scaling_governor.sh performance"
adb shell "su root setprop persist.vendor.disable.thermal.control 1"

adb push "${TD}/../utils" "${ROOT_DIR}"
adb shell "chmod +x ${ROOT_DIR}/utils/run_tflite_benchmark.py"

BENCHMARK_BINARY_PATH="${ROOT_DIR}/android_aarch64_benchmark_model"

# Download benchmark tool if not set.
if [[ -z "${TFLITE_BENCHMARK_BINARY}" ]]; then
BENCHMARK_BINARY_URL="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model"
"${TD}/../../comparative_benchmark/scripts/adb_fetch_and_push.py" \
    --source_url="${BENCHMARK_BINARY_URL}" \
    --destination="${BENCHMARK_BINARY_PATH}" \
    --verbose
else
adb push "${TFLITE_BENCHMARK_BINARY}" "${BENCHMARK_BINARY_PATH}"
fi

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

# Thread-to-taskset config. Here we run 1 thread on the largest core, and 4 threads on the 4 largest (2 big, 2 medium) cores.
# Assumes the benchmarks run on Pixel-6-Pro.
THREAD_CONFIG="{1: '80', 4: 'F0'}"
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
