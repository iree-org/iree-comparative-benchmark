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
# OOBI_VENV_DIR: path to create Python virtualenv, default: ggml-benchmarks.venv
# OOBI_TARGET_DEVICE: target benchmark device, can also be specified the first
#   argument.
# OOBI_BUILD_DIR: path to the GGMl build directory.
# OOBI_OUTPUT: path to output benchmark results, can also be specified the
#   second argument.
#
# Example usage:
# ./benchmark_ggml.sh <target-device> <build-dir> <result-path>

set -xeuo pipefail

VENV_DIR="${OOBI_VENV_DIR:-ggml-benchmarks.venv}"
PYTHON="${PYTHON:-"$(which python3)"}"
TARGET_DEVICE_NAME="${1:-${OOBI_TARGET_DEVICE}}"
BUILD_DIR="${2:-${OOBI_BUILD_DIR}}"
OUTPUT_PATH="${3:-${OOBI_OUTPUT}}"

TD="$(cd $(dirname $0) && pwd)"

# Setup virtual environment.
VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

# Initialize results json.
OUTPUT_PATH="$(realpath ${OUTPUT_PATH})"
"${TD}/../../comparative_benchmark/scripts/create_results_json.sh" "${OUTPUT_PATH}"

pushd "${BUILD_DIR}"

PROMPT="Once upon a time"
BENCHMARK_BINARY="$(realpath bin/gpt-2)"
BENCHMARK_LIBRARY="$(realpath src/libggml.so)"
WARMUP_ITERAIONS=2
NUM_ITERATIONS=10

MODEL="$(realpath models/gpt-2-117M/ggml-model-f32.bin)"

declare -a BENCHMARK_NAMES=(
  "models/GPT2LMHEAD_FP32_GGML/inputs/INPUT_DATA_MODEL_DEFAULT"
  "models/GPT2LMHEAD_FP16_GGML/inputs/INPUT_DATA_MODEL_DEFAULT"
  "models/GPT2LMHEAD_INT4_GGML/inputs/INPUT_DATA_MODEL_DEFAULT"
)
declare -a MODELS=(
  ggml-model-f32.bin
  ggml-model-f16.bin
  ggml-model-q4_0.bin
)
declare -a DATA_TYPES=(
  "fp32"
  "fp16"
  "int4"
)

declare -a args=(
  --warmup_iterations "${WARMUP_ITERAIONS}"
  --iterations "${NUM_ITERATIONS}"
  --benchmark_binary "${BENCHMARK_BINARY}"
  --benchmark_library "${BENCHMARK_LIBRARY}"
  --prompt "${PROMPT}"
  --seed 0
  --output "${OUTPUT_PATH}"
  --target_device "${TARGET_DEVICE_NAME}"
  --verbose
)

if [[ "${TARGET_DEVICE_NAME}" =~ ^(pixel-4|pixel-6-pro|moto-edge-x30)$ ]]; then
  BENCHMARK_SCRIPT="run_benchmarks_android.py"
  # Pixel 6 has a maximum of 8 cores.
  THREADS="1,4,8"
  TASKSETS="80,f0,ff"

  args+=(
     --threads "${THREADS}"
     --tasksets "${TASKSETS}"
  )

  # Setup mobile device for benchmarking.
  adb push "${TD}/../../comparative_benchmark/scripts/set_android_scaling_governor.sh" "/data/local/tmp"
  adb shell "chmod +x /data/local/tmp/set_android_scaling_governor.sh"
  adb shell "su root sh /data/local/tmp/set_android_scaling_governor.sh performance"
else
  BENCHMARK_SCRIPT="run_benchmarks.py"
  # c2-standard-16 has 16 cores.
  THREADS="1,8,16"

  args+=(
     --threads "${THREADS}"
  )
fi

for i in ${!BENCHMARK_NAMES[@]}; do
  MODEL="$(realpath models/gpt-2-117M/${MODELS[$i]})"
  args+=(
    --benchmark_name "${BENCHMARK_NAMES[$i]}"
    --model "${MODEL}"
    --data_type "${DATA_TYPES[$i]}"
  )
  "${TD}/${BENCHMARK_SCRIPT}" "${args[@]}"
done

popd # BUILD_DIR
