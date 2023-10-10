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
# OOBI_TOOL_DIR: path to save benchmark tools.
# OOBI_OUTPUT: path to output benchmark results, can also be specified the
#   third argument.
#
# Example usage:
# ./benchmark_tflite.sh <target-device> <tool-dir> <result-path>

set -xeuo pipefail

VENV_DIR="${OOBI_VENV_DIR:-tflite-benchmarks.venv}"
PYTHON="${PYTHON:-"$(which python3)"}"
TARGET_DEVICE="${1:-${OOBI_TARGET_DEVICE}}"
TOOL_DIR="${2:-${OOBI_TOOL_DIR}}"
OUTPUT_PATH="${3:-${OOBI_OUTPUT}}"

# Setup virtual environment.
TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

# Initialize results json.
OUTPUT_PATH="$(realpath ${OUTPUT_PATH})"
"${TD}/../../comparative_benchmark/scripts/create_results_json.sh" "${OUTPUT_PATH}"

declare -a args=(
  --tflite_benchmark_tool "${TOOL_DIR}/tflite_benchmark_model"
  --output "${OUTPUT_PATH}"
  --target_device "${TARGET_DEVICE}"
  --verbose
)

# Download TFLite benchmark tool depending on target.
if [[ "${TARGET_DEVICE}" =~ ^(pixel-4|pixel-6-pro|moto-edge-x30)$ ]]; then
  wget -O "${TOOL_DIR}/tflite_benchmark_model" https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex

  # Setup mobile device for benchmarking.
  adb push "${TD}/set_android_scaling_governor.sh" "/data/local/tmp"
  adb shell "chmod +x /data/local/tmp/set_android_scaling_governor.sh"
  adb shell "su root sh /data/local/tmp/set_android_scaling_governor.sh performance"

  BENCHMARK_SCRIPT="run_benchmarks_android.py"

  # Pixel 6 has a maximum of 8 cores.
  THREADS="1,4"
  TASKSETS="80,f0"

  args+=(
     --threads "${THREADS}"
     --tasksets "${TASKSETS}"
     --iterations 10
  )

  declare -a BENCHMARK_NAMES=(
    #"models/RESNET50_FP32_TF_.+_BATCH(1|8)/.+"
    "models/BERT_LARGE_FP32_TF_.+_BATCH(1|16|24|32)/.+"
    #"models/T5_LARGE_FP32_TF_.+_BATCH(1|16)/.+"
    #"models/EFFICIENTNETB7_FP32_TF_.+_BATCH1/.+"
  )
else
  wget -O "${TOOL_DIR}/tflite_benchmark_model" https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_plus_flex
  chmod +x "${TOOL_DIR}/tflite_benchmark_model"

  BENCHMARK_SCRIPT="run_benchmarks.py"

  # c2-standard-16 has 8 cores.
  THREADS="1,8"
  args+=(
     --threads "${THREADS}"
     --iterations 20
  )

  declare -a BENCHMARK_NAMES=(
    #"models/RESNET50_FP32_TF_.+_BATCH(1|64|128)/.+"
    "models/BERT_LARGE_FP32_TF_.+_BATCH(1|16|24|32)/.+"
    #"models/T5_LARGE_FP32_TF_.+_BATCH(1|16|24|32)/.+"
    #"models/EFFICIENTNETB7_FP32_TF_.+_BATCH(1|64|128)/.+"
  )
fi

for i in ${!BENCHMARK_NAMES[@]}; do
  args+=(
    --benchmark_name "${BENCHMARK_NAMES[$i]}"
  )
  "${TD}/${BENCHMARK_SCRIPT}" "${args[@]}"
done
