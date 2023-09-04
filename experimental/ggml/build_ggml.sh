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
# ANDROID_NDK: the path to the Android NDK if building for Android.
# OOBI_VENV_DIR: path to create Python virtualenv, default: ggml-build.venv
# OOBI_TARGET_DEVICE: target benchmark device, can also be specified the first
#   argument.
# OOBI_OUTPUT: path to output benchmark results, can also be specified the
#   second argument.
# OOBI_SCRATCH_DIR: the directory to place temporary benchmarking artifacts.
#
# Example usage:
# ./build_ggml.sh <target-device>> <build-dir>

set -xeuo pipefail

VENV_DIR="${OOBI_VENV_DIR:-ggml-build.venv}"
ROOT_DIR="${OOBI_SCRATCH_DIR:-/tmp}"
PYTHON="${PYTHON:-/usr/bin/python3}"
TARGET_DEVICE_NAME="${1:-${OOBI_TARGET_DEVICE}}"
BUILD_DIR="${2:-/tmp/ggml-build}"

TD="$(cd $(dirname $0) && pwd)"
BUILD_DIR="$(realpath ${BUILD_DIR})"

# Setup virtual environment.
VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

pushd "${ROOT_DIR}"

# We clone a fork of ggml which includes additional benchmark logging.
git clone --branch benchmark https://github.com/mariecwhite/ggml.git
pushd ggml

REPO_DIR="$(pwd)"

# Build gpt-2-quantize.
cmake -G Ninja -B local-build .
cmake --build local-build -t gpt-2-quantize

# Build gpt-2.
if [[ "${TARGET_DEVICE_NAME}" =~ ^(pixel-4|pixel-6-pro|moto-edge-x30)$ ]]; then
cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS=-march=armv8.4a+dotprod -B "${BUILD_DIR}" .
cmake --build "${BUILD_DIR}" -t gpt-2
else
cmake -G Ninja -B "${BUILD_DIR}" .
cmake --build "${BUILD_DIR}" -t gpt-2
fi

popd # ggml
popd # ROOT_DIR

# Generate FP32 and FP16 versions of GPT2 117M (Small).
pushd "${BUILD_DIR}"

GPT_VARIANT="117M"
${REPO_DIR}/examples/gpt-2/download-model.sh "${GPT_VARIANT}"
# Generate FP32.
python ${REPO_DIR}/examples/gpt-2/convert-ckpt-to-ggml.py models/gpt-2-${GPT_VARIANT}/ 0
# Generate FP16.
python ${REPO_DIR}/examples/gpt-2/convert-ckpt-to-ggml.py models/gpt-2-${GPT_VARIANT}/ 1
# Generate INT4. Keep this disabled until we want to use it.
${REPO_DIR}/local-build/bin/gpt-2-quantize models/gpt-2-${GPT_VARIANT}/ggml-model-f16.bin models/gpt-2-${GPT_VARIANT}/ggml-model-q4_0.bin 2

popd # BUILD_DIR
