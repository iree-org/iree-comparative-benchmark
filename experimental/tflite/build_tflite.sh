#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Builds the TFLite `benchmark_model` for x86 and Android.
#
# Environment variables:
# OOBI_TEMP_DIR: the path to store temporary files to.
# ANDROID_NDK: the path to the Android NDK.
# ANDROID_NDK_API_LEVEL: the version of Android NDK.
# ANDROID_SDK: the path to the Android SDK.
# ANDROID_SDK_API_LEVEL: the version of the Android SDK.
#
# Example usage:
# ./build_tflite.sh <x86-output-dir> <android-output-dir>

set -xeuo pipefail

ROOT_DIR="${OOBI_TEMP_DIR:-/tmp/tensorflow-src}"
X86_OUTPUT_DIR="${1:-/tmp/tflite-tools/x86}"
ANDROID_OUTPUT_DIR="${2:-/tmp/tflite-tools/android}"

mkdir -p "${X86_OUTPUT_DIR}"
mkdir -p "${ANDROID_OUTPUT_DIR}"

# Use absolute paths.
X86_OUTPUT_DIR=$(realpath ${X86_OUTPUT_DIR})
ANDROID_OUTPUT_DIR=$(realpath ${ANDROID_OUTPUT_DIR})

rm -rf "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}"
pushd "${ROOT_DIR}"

git clone https://github.com/tensorflow/tensorflow.git
pushd tensorflow

# Log the git version of Tensorflow repo.
git log --oneline --graph --max-count=1

bazel build -c opt --define xnnpack_use_latest_ops=true //tensorflow/lite/tools/benchmark:benchmark_model
cp "$(realpath bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model)" "${X86_OUTPUT_DIR}/"

# The compiler flags used here were retrieved by running ./configure in the root Tensorflow repo.
bazel build -c opt --config=android_arm64 \
    --define xnnpack_use_latest_ops=true \
    --action_env ANDROID_NDK_HOME="${ANDROID_NDK}" \
    --action_env ANDROID_NDK_VERSION="${ANDROID_NDK_API_LEVEL}" \
    --action_env ANDROID_NDK_API_LEVEL="26" \
    --action_env ANDROID_SDK_API_LEVEL="${ANDROID_SDK_API_LEVEL}" \
    --action_env ANDROID_SDK_HOME="${ANDROID_SDK}" \
    --action_env ANDROID_BUILD_TOOLS_VERSION="${ANDROID_BUILD_TOOLS_VERSION}" \
    --action_env CLANG_COMPILER_PATH=/usr/lib/llvm-14/bin/clang \
    --repo_env=CC=/usr/lib/llvm-14/bin/clang \
    --repo_env=BAZEL_COMPILER=/usr/lib/llvm-14/bin/clang \
    --copt=-Wno-gnu-offsetof-extensions \
    //tensorflow/lite/tools/benchmark:benchmark_model

cp "$(realpath bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model)" "${ANDROID_OUTPUT_DIR}/"

popd
popd
