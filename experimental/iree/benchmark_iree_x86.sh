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
# OOBI_IREE_BENCHMARK_MODULE_PATH: the path to `iree-benchmark-module`.
# OOBI_IREE_RUN_MODULE_PATH: the path to `iree-run-module`.
# OOBI_IREE_COMPILED_ARTIFACTS_PATH: the path to the IREE vmfb files to benchmark.
#
# Example usage:
# ./benchmark_iree.sh <target-device> <path-to-compiled-artifacts> <results-path>

set -xeuo pipefail

VENV_DIR="${OOBI_VENV_DIR:-iree-benchmarks.venv}"
PYTHON="${PYTHON:-/usr/bin/python3}"
IREE_BENCHMARK_MODULE_PATH="${OOBI_IREE_BENCHMARK_MODULE_PATH:-/tmp/iree-build/install/bin/iree-benchmark-module}"
IREE_RUN_MODULE_PATH="${OOBI_IREE_RUN_MODULE_PATH:-/tmp/iree-build/install/bin/iree-run-module}"
TARGET_DEVICE="${1:-"${OOBI_TARGET_DEVICE}"}"
COMPILED_ARTIFACTS_PATH="${2:-"${OOBI_IREE_COMPILED_ARTIFACTS_PATH}"}"
OUTPUT_PATH="${3:-"${OOBI_OUTPUT}"}"

TD="$(cd $(dirname $0) && pwd)"

VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

"${TD}/../../comparative_benchmark/scripts/create_results_json.sh" "${OUTPUT_PATH}"

THREAD_CONFIG="{1: '0', 8: '0,1,2,3,4,5,6,7', 15: '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14', 30: '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29'}"

"${TD}/run_benchmarks.py" \
  --target_device="${TARGET_DEVICE}" \
  --output="${OUTPUT_PATH}" \
  --artifact_dir="${COMPILED_ARTIFACTS_PATH}" \
  --iree_run_module_path="${IREE_RUN_MODULE_PATH}" \
  --iree_benchmark_module_path="${IREE_BENCHMARK_MODULE_PATH}" \
  --thread_config="${THREAD_CONFIG}" \
  --verbose
