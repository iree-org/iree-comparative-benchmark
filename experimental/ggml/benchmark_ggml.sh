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
# OOBI_OUTPUT: path to output benchmark results, can also be specified the
#   second argument.
# OOBI_SCRATCH_DIR: the directory to place temporary benchmarking artifacts.
#
# Example usage:
# ./benchmark_ggml.sh c2-standard-16 /tmp/results.json

set -xeuo pipefail

VENV_DIR="${OOBI_VENV_DIR:-ggml-benchmarks.venv}"
ROOT_DIR="${OOBI_SCRATCH_DIR:-/tmp}"
PYTHON="${PYTHON:-/usr/bin/python3}"
TARGET_DEVICE="${1:-${OOBI_TARGET_DEVICE}}"
OUTPUT_PATH="${2:-${OOBI_OUTPUT}}"

TD="$(cd $(dirname $0) && pwd)"

# Setup virtual environment.
VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

# Initialize results json.
OUTPUT_PATH="$(realpath ${OUTPUT_PATH})"
"${TD}/../../comparative_benchmark/scripts/create_results_json.sh" "${OUTPUT_PATH}"

pushd "${ROOT_DIR}"

# We clone a fork of ggml which includes additional benchmark logging.
git clone --branch benchmark https://github.com/mariecwhite/ggml.git
pushd ggml

# Build
mkdir build
pushd build
cmake ..
make -j8

# Generate FP32, FP16 and INT4 versions of GPT2 117M (Small).
GPT_VARIANT="117M"
../examples/gpt-2/download-model.sh "${GPT_VARIANT}"
# Generate FP32.
python ../examples/gpt-2/convert-ckpt-to-ggml.py models/gpt-2-${GPT_VARIANT}/ 0
# Generate FP16.
python ../examples/gpt-2/convert-ckpt-to-ggml.py models/gpt-2-${GPT_VARIANT}/ 1
# Generate INT4.
./bin/gpt-2-quantize models/gpt-2-${GPT_VARIANT}/ggml-model-f16.bin models/gpt-2-${GPT_VARIANT}/ggml-model-q4_0.bin 2

PROMPT="Once upon a time"
BENCHMARK_BINARY="$(realpath bin/gpt-2)"
WARMUP_ITERAIONS=2
NUM_ITERATIONS=10
declare -a NUM_THREADS=(1 8 16)

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

for i in ${!BENCHMARK_NAMES[@]}; do
  MODEL="$(realpath models/gpt-2-117M/${MODELS[$i]})"

  for threads in "${NUM_THREADS[@]}"; do
    "${TD}/benchmark.py" \
      --benchmark_name "${BENCHMARK_NAMES[$i]}" \
      --warmup_iterations "${WARMUP_ITERAIONS}" \
      --iterations "${NUM_ITERATIONS}" \
      --benchmark_binary "${BENCHMARK_BINARY}" \
      --model "${MODEL}" \
      --data_type "${DATA_TYPES[$i]}" \
      --prompt "${PROMPT}" \
      --seed 0 \
      --threads "${threads}" \
      --output "${OUTPUT_PATH}" \
      --target_device "${TARGET_DEVICE}" \
      --verbose
  done
done

popd # build
popd # ggml
popd # ROOT_DIR
