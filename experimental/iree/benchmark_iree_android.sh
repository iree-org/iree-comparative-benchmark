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
# OOBI_ANDROID_BENCHMARK_DIR: the on-device directory where benchmark artifacts are copied to.
#
# Example usage:
# ./benchmark_iree.sh <target-device> <path-to-compiled-artifacts> <results-path>

set -xeuo pipefail

VENV_DIR="${OOBI_VENV_DIR:-iree-benchmarks.venv}"
PYTHON="${PYTHON:-/usr/bin/python3}"
ROOT_DIR="${OOBI_ANDROID_BENCHMARK_DIR:-/data/local/tmp/oobi_benchmarks}"
IREE_BENCHMARK_MODULE_PATH="${OOBI_IREE_BENCHMARK_MODULE_PATH:-/tmp/iree-build-android/tools/iree-benchmark-module}"
IREE_RUN_MODULE_PATH="${OOBI_IREE_RUN_MODULE_PATH:-/tmp/iree-build-android/tools/iree-run-module}"
TARGET_DEVICE="${1:-"${OOBI_TARGET_DEVICE}"}"
COMPILED_ARTIFACTS_PATH="${2:-"${OOBI_IREE_COMPILED_ARTIFACTS_PATH}"}"
OUTPUT_PATH="${3:-"${OOBI_OUTPUT}"}"

TD="$(cd $(dirname $0) && pwd)"

# Setup benchmarking environment.
adb shell "rm -rf ${ROOT_DIR}"
adb shell "mkdir ${ROOT_DIR}"

adb push "${TD}/../../comparative_benchmark/scripts/set_android_scaling_governor.sh" "${ROOT_DIR}"
adb shell "chmod +x ${ROOT_DIR}/set_android_scaling_governor.sh"
adb shell "su root sh ${ROOT_DIR}/set_android_scaling_governor.sh performance"
#adb shell "su root sendhint -m DISPLAY_INACTIVE -e 0"
adb shell "su root setprop persist.vendor.disable.thermal.control 1"

adb push "${TD}/benchmark_lib.py" "${ROOT_DIR}"
adb shell "chmod +x ${ROOT_DIR}/benchmark_lib.py"

adb push "${IREE_RUN_MODULE_PATH}" "${ROOT_DIR}"
IREE_RUN_MODULE_PATH="${ROOT_DIR}/iree-run-module"

adb push "${IREE_BENCHMARK_MODULE_PATH}" "${ROOT_DIR}"
IREE_BENCHMARK_MODULE_PATH="${ROOT_DIR}/iree-benchmark-module"

VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

DEVICE_ARTIFACT_DIR="${ROOT_DIR}/artifacts"
adb shell mkdir "${DEVICE_ARTIFACT_DIR}"

if [[ "${COMPILED_ARTIFACTS_PATH}" = https* ]]; then
  archive_name=$(basename "${COMPILED_ARTIFACTS_PATH}")

  "${TD}/../../comparative_benchmark/scripts/adb_fetch_and_push.py" \
    --source_url="${COMPILED_ARTIFACTS_PATH}" \
    --destination="${ROOT_DIR}/${archive_name}" \
    --verbose

  adb shell "tar -xf ${ROOT_DIR}/${archive_name} --strip-components=1 -C ${DEVICE_ARTIFACT_DIR}"
else
  adb push "${COMPILED_ARTIFACTS_PATH}/." "${DEVICE_ARTIFACT_DIR}/"
fi

"${TD}/../../comparative_benchmark/scripts/create_results_json.sh" "${OUTPUT_PATH}"

# A num_threads to cpu_ids map. We use the biggest cores for each configuration.
THREAD_CONFIG="{1: '0', 4: '0,1,2,3'}"

"${TD}/run_benchmarks.py" \
  --target_device="${TARGET_DEVICE}" \
  --output="${OUTPUT_PATH}" \
  --artifact_dir="${DEVICE_ARTIFACT_DIR}" \
  --iree_run_module_path="${IREE_RUN_MODULE_PATH}" \
  --iree_benchmark_module_path="${IREE_BENCHMARK_MODULE_PATH}" \
  --thread_config="${THREAD_CONFIG}" \
  --verbose

# Cleanup.
adb shell rm -rf "${ROOT_DIR}"
