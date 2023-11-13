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
# IREE_COMPILE_PATH: the path to the `iree-compile` binary.
# OOBI_TARGET_DEVICE: target benchmark device, can also be specified the first
#   argument.
# OOBI_TEMP_DIR: directory to save intermediates.
# OOBI_VENV_DIR: name of the virtual environment.
#
# Example usage:
# ./compile_workloads.sh <target-device> <output-dir>

set -xeuo pipefail

PYTHON="${PYTHON:-/usr/bin/python3}"
VENV_DIR="${OOBI_VENV_DIR:-iree.venv}"
IREE_COMPILE_PATH="${IREE_COMPILE_PATH:-/tmp/iree-build/install/bin/iree-compile}"
TEMP_DIR="${OOBI_TEMP_DIR:-/tmp/openxla-benchmark}"
TARGET_DEVICE_NAME="${1:-${OOBI_TARGET_DEVICE}}"
OUTPUT_DIR="${2:-/tmp/compiled-artifacts/${TARGET_DEVICE_NAME}}"

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

mkdir -p "${OUTPUT_DIR}"

declare -a BENCHMARK_NAMES=(
  "models/BERT_BASE_FP32_TFLITE_I32_SEQLEN.+/.+"
  "models/BERT_BASE_FP16_TFLITE_I32_SEQLEN.+/.+"
  "models/BERT_BASE_DYN_QUANT_TFLITE_I32_SEQLEN.+/.+"
  "models/BERT_BASE_INT8_TFLITE_I32_SEQLEN.+/.+"
  "models/VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32/.+"
  "models/VIT_CLASSIFICATION_FP16_TFLITE_3X224X224XF32/.+"
  "models/VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32/.+"
  "models/VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8/.+"
  "models/BERT_BASE_FP32_JAX_I32_SEQLEN.+/.+"
  "models/BERT_BASE_FP16_JAX_I32_SEQLEN.+/.+"
  "models/BERT_BASE_BF16_JAX_I32_SEQLEN.+/.+"
  "models/T5_4CG_SMALL_FP32_JAX_1X128XI32_GEN.+/.+"
  "models/SD_PIPELINE_FP32_JAX_64XI32_BATCH.+/.+"
  "models/SD_PIPELINE_FP16_JAX_64XI32_BATCH.+/.+"
  "models/SD_PIPELINE_BF16_JAX_64XI32_BATCH.+/.+"
  "models/VIT_CLASSIFICATION_JAX_3X224X224XF32/.+"
)

for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
  "${TD}/compile_workloads.py" \
    --benchmark_name="${benchmark_name}" \
    --target_device="${TARGET_DEVICE_NAME}" \
    --output="${OUTPUT_DIR}" \
    --iree_compile_path="${IREE_COMPILE_PATH}" \
    --temp-dir="${TEMP_DIR}" \
    --verbose
done
