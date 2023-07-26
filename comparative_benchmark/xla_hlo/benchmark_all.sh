#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

VENV_DIR="${OOBI_VENV_DIR:-xla-hlo-benchmarks.venv}"
PYTHON="${PYTHON:-/usr/bin/python3}"
XLA_TOOLS_DIR="${OOBI_XLA_TOOLS_DIR:-xla-tools-dir}"
TARGET_DEVICE="${1:-"${OOBI_TARGET_DEVICE}"}"
OUTPUT_PATH="${2:-"${OOBI_OUTPUT}"}"

TD="$(cd $(dirname $0) && pwd)"

VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

declare -a GPU_BENCHMARK_NAMES=(
  "models/RESNET50_FP32_JAX_.+"
  "models/RESNET50_FP16_JAX_.+"
  "models/RESNET50_BF16_JAX_.+"
  "models/BERT_LARGE_FP32_JAX_.+"
  "models/BERT_LARGE_FP16_JAX_.+"
  "models/BERT_LARGE_BF16_JAX_.+"
  "models/T5_LARGE_FP32_JAX_.+"
  "models/T5_LARGE_FP16_JAX_.+"
  "models/T5_LARGE_BF16_JAX_.+"
  "models/T5_4CG_LARGE_FP32_JAX_.+"
  "models/RESNET50_FP32_TF_.+"
  "models/BERT_LARGE_FP32_TF_.+"
  "models/T5_LARGE_FP32_TF_.+"
)

declare -a CPU_BENCHMARK_NAMES=(
#  "models/RESNET50_FP32_JAX_.+_BATCH1/.+"
#  "models/RESNET50_FP32_JAX_.+_BATCH64/.+"
#  "models/RESNET50_FP32_JAX_.+_BATCH128/.+"
#  "models/BERT_LARGE_FP32_JAX_.+_BATCH1/.+"
#  "models/BERT_LARGE_FP32_JAX_.+_BATCH32/.+"
#  "models/BERT_LARGE_FP32_JAX_.+_BATCH64/.+"
#  "models/T5_LARGE_FP32_JAX_.+_BATCH1/.+"
#  "models/T5_LARGE_FP32_JAX_.+_BATCH16/.+"
#  "models/T5_LARGE_FP32_JAX_.+_BATCH32/.+"
#  "models/GPT2LMHEAD_FP32_JAX_.+_BATCH1/.+"
  "models/GPT2LMHEAD_FP32_JAX_.+_BATCH64/.+"
  "models/GPT2LMHEAD_FP32_JAX_.+_BATCH128/.+"
#  "models/T5_4CG_LARGE_FP32_JAX_.+_BATCH1/.+"
#  "models/T5_4CG_LARGE_FP32_JAX_.+_BATCH16/.+"
#  "models/T5_4CG_LARGE_FP32_JAX_.+_BATCH32/.+"
  "models/RESNET50_FP32_TF_.+_BATCH1/.+"
  "models/RESNET50_FP32_TF_.+_BATCH64/.+"
  "models/RESNET50_FP32_TF_.+_BATCH128/.+"
  "models/BERT_LARGE_FP32_TF_.+_BATCH1/.+"
  "models/BERT_LARGE_FP32_TF_.+_BATCH32/.+"
  "models/BERT_LARGE_FP32_TF_.+_BATCH64/.+"
  "models/T5_LARGE_FP32_TF_.+_BATCH1/.+"
  "models/T5_LARGE_FP32_TF_.+_BATCH16/.+"
  "models/T5_LARGE_FP32_TF_.+_BATCH32/.+"
  "models/EFFICIENTNETB7_FP32_TF_.+_BATCH1/.+"
  "models/EFFICIENTNETB7_FP32_TF_.+_BATCH64/.+"
  "models/EFFICIENTNETB7_FP32_TF_.+_BATCH128/.+"
)

if [ "${TARGET_DEVICE}" = "a2-highgpu-1g" ]; then
  BENCHMARK_NAMES=("${GPU_BENCHMARK_NAMES[@]}")
  HLO_TOOL="hlo_runner_main"
  ITERATIONS=50
elif [ "${TARGET_DEVICE}" = "c2-standard-16" ]; then
  # Since each iteration includes both compilation and inference, we keep the
  # total iterations small because of the amount of time it takes to do both.
  # Std deviation is <1ms.
  BENCHMARK_NAMES=("${CPU_BENCHMARK_NAMES[@]}")
  HLO_TOOL="run_hlo_module"
  ITERATIONS=11
elif [ "${TARGET_DEVICE}" = "c2-standard-60" ]; then
  # Since each iteration includes both compilation and inference, we keep the
  # total iterations small because of the amount of time it takes to do both.
  # Std deviation is <1ms.
  BENCHMARK_NAMES=("${CPU_BENCHMARK_NAMES[@]}")
  HLO_TOOL="run_hlo_module"
  ITERATIONS=11
else
  echo "Unsupported target device ${TARGET_DEVICE}."
  exit 1
fi

"${TD}/../scripts/create_results_json.sh" "${OUTPUT_PATH}"

for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
  "${TD}/run_benchmarks.py" \
    --benchmark_name="${benchmark_name}" \
    --target_device="${TARGET_DEVICE}" \
    --hlo-tool="${XLA_TOOLS_DIR}/${HLO_TOOL}" \
    --output="${OUTPUT_PATH}" \
    --iterations="${ITERATIONS}" \
    --verbose
done
