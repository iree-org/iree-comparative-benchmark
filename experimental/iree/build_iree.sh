#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
# IREE_SOURCE_DIR: the directory to clone the IREE repo into.
# IREE_INSTALL_DIR: the directory to install IREE into.
# ANDROID_PLATFORM_VERSION: the version of the Android platform to use.
# ANDROID_NDK: the path to the Android NDK.
#
# Example usage:
# ./build_iree.sh <build-dir> <android-build-dir>

set -xeuo pipefail

ANDROID_PLATFORM_VERSION="${ANDROID_PLATFORM_VERSION:-34}"
IREE_SOURCE_DIR="${IREE_SOURCE_DIR:-/tmp/iree-src}"
IREE_INSTALL_DIR="${IREE_INSTALL_DIR:-install}"
BUILD_DIR="${1:-/tmp/iree-build}"
ANDROID_BUILD_DIR="${2:-/tmp/iree-build-android}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

rm -rf "${ANDROID_BUILD_DIR}"
mkdir -p "${ANDROID_BUILD_DIR}"

rm -rf "${IREE_SOURCE_DIR}"
mkdir -p "${IREE_SOURCE_DIR}"

pushd "${IREE_SOURCE_DIR}"
git clone https://github.com/openxla/iree.git
cd iree
git submodule update --init
popd

cmake -G Ninja -B "${BUILD_DIR}" -S "${IREE_SOURCE_DIR}/iree" \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_PYTHON_BINDINGS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DIREE_ENABLE_ASSERTIONS=OFF \
  -DIREE_BUILD_BINDINGS_TFLITE=OFF \
  -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF \
  -DIREE_ENABLE_LLD=ON \
  -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/${IREE_INSTALL_DIR}"
cmake --build "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target install

# We need the absolute path when being used as a parameter to `IREE_HOST_BIN_DIR`.
BUILD_DIR=$(realpath "${BUILD_DIR}")

cmake -GNinja -B "${ANDROID_BUILD_DIR}" -S "${IREE_SOURCE_DIR}/iree" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
  -DIREE_HOST_BIN_DIR="${BUILD_DIR}/${IREE_INSTALL_DIR}/bin" \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM="android-${ANDROID_PLATFORM_VERSION}" \
  -DIREE_BUILD_COMPILER=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_PYTHON_BINDINGS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DIREE_ENABLE_ASSERTIONS=OFF \
  -DIREE_BUILD_BINDINGS_TFLITE=OFF \
  -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF \
  -DIREE_ENABLE_LLD=ON
cmake --build "${ANDROID_BUILD_DIR}"
