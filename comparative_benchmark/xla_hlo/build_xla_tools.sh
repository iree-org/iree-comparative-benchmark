#!/bin/bash
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

BUILD_DIR="${XLA_BUILD_DIR:-build-xla}"
TOOLS_OUTPUT_DIR="${1:-"${XLA_TOOLS_OUTPUT_DIR}"}"
CUDA_VERSION="${2:-"${XLA_CUDA_VERSION}"}"

mkdir -p "${BUILD_DIR}"
pushd "${BUILD_DIR}"

git clone https://github.com/openxla/xla.git
cd xla
# Last passing commit.
#git checkout 5ec8b22

# Log the git version of XLA repo.
git log --oneline --graph --max-count=1

bazel build -c opt --config=cuda \
  --action_env TF_CUDA_COMPUTE_CAPABILITIES="8.0" \
  --action_env GCC_HOST_COMPILER_PATH="/usr/bin/x86_64-linux-gnu-gcc-11" \
  --action_env LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:" \
  --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-${CUDA_VERSION}" \
  --copt=-Wno-switch \
  xla/tools/multihost_hlo_runner:hlo_runner_main
RUN_HLO_RUNNER_MAIN_PATH="$(realpath bazel-bin/xla/tools/multihost_hlo_runner/hlo_runner_main)"

bazel build -c opt --copt=-Wno-switch \
  --action_env GCC_HOST_COMPILER_PATH="/usr/bin/x86_64-linux-gnu-gcc-11" \
  xla/tools:run_hlo_module
RUN_HLO_MODULE_PATH="$(realpath bazel-bin/xla/tools/run_hlo_module)"

popd

cp "${RUN_HLO_RUNNER_MAIN_PATH}" "${RUN_HLO_MODULE_PATH}" "${TOOLS_OUTPUT_DIR}/"
