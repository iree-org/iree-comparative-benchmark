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
  "models/RESNET50_(FP32|FP16|BF16)_TF_.+"
  # FP16 disabled due to type errors: https://github.com/openxla/openxla-benchmark/issues/117
  "models/BERT_LARGE_(FP32|BF16)_JAX_.+"
  # Batch 512 disabled: https://github.com/openxla/openxla-benchmark/issues/125.
  "models/T5_LARGE_(FP32|FP16|BF16)_JAX_.+_BATCH(1|16|24|32|48|64)/.+"
  "models/T5_4CG_LARGE_FP32_JAX_.+"
  "models/GPT2LMHEAD_FP32_JAX_.+"
  "models/RESNET50_FP32_TF_.+"
  "models/BERT_LARGE_FP32_TF_.+"
  # Batch 512 disabled: https://github.com/openxla/openxla-benchmark/issues/125.
  "models/T5_LARGE_FP32_TF_.+_BATCH(1|16|24|32|48|64)/.+"
  "models/EFFICIENTNETB7_FP32_TF_.+"
)

declare -a CPU_BENCHMARK_NAMES=(
  "models/RESNET50_FP32_JAX_.+_BATCH(1|64|128)/.+"
  "models/BERT_LARGE_FP32_JAX_.+_BATCH(1|32|64)/.+"
  "models/T5_LARGE_FP32_JAX_.+_BATCH(1|16|32)/.+"
  "models/T5_4CG_LARGE_FP32_JAX_.+_BATCH(1|16|32)/.+"
  "models/GPT2LMHEAD_FP32_JAX_.+_BATCH(1|64|128)/.+"
  "models/RESNET50_FP32_TF_.+_BATCH(1|64|128)/.+"
  "models/BERT_LARGE_FP32_TF_.+_BATCH(1|32|64)/.+"
  "models/T5_LARGE_FP32_TF_.+_BATCH(1|16|32)/.+"
  "models/EFFICIENTNETB7_FP32_TF_.+_BATCH(1|64|128)/.+"
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
  ITERATIONS=5
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
    --compiler="xla" \
    --verbose
done

# Disable for now as it takes too long to run.
# If running on CPU, also benchmark XLA CPU-Next.
# Use a lower number of iterations since CPU-Next is slow.
# if [ "${TARGET_DEVICE}" = "c2-standard-16" ]; then
#   for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
#     "${TD}/run_benchmarks.py" \
#       --benchmark_name="${benchmark_name}" \
#       --target_device="${TARGET_DEVICE}" \
#       --hlo-tool="${XLA_TOOLS_DIR}/${HLO_TOOL}" \
#       --output="${OUTPUT_PATH}" \
#       --iterations=3 \
#       --compiler="xla_cpu_next" \
#       --verbose
#   done
# fi
