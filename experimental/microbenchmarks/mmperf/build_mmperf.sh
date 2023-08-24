#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#   MKL_DIR=/opt/intel/mkl
#   BLIS_DIR=/opt/blis
#
# Example usage:
#   ./run_mmperf.sh <cpu|gpu> <build-dir>
#

set -xeuo pipefail

BACKEND=$1
BUILD_DIR=$2

# Clone mmperf.
git clone https://github.com/mmperf/mmperf.git
pushd mmperf
git submodule update --init --recursive --jobs 8 --depth 1

# Create virtual environment.
python3 -m venv mmperf.venv
source mmperf.venv/bin/activate
pip install -r requirements.txt
pip install -r ./external/llvm-project/mlir/python/requirements.txt

popd # mmperf.

# Build mmperf.
declare -a args=(
  -GNinja
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++
  -DCMAKE_C_COMPILER=/usr/bin/clang
  -DUSE_IREE=ON
)

if [[ ${BACKEND} == "cuda" ]]; then
  args+=(
    -DIREE_CUDA=ON
    -DUSE_CUBLAS=ON
  )
  cmake "${args[@]}" -B "${BUILD_DIR}" mmperf
elif [[ ${BACKEND} == "cpu" ]]; then
  args+=(
    -DMKL_DIR=/opt/intel/mkl
    -DBLIS_DIR=/opt/blis
    -DUSE_MKL=ON
    -DUSE_RUY=ON
    -DIREE_LLVMCPU=ON
    -DUSE_OPENBLAS=ON
    -DUSE_BLIS=ON
  )
  MKL_DIR=${MKL_DIR} BLIS_DIR=${BLIS_DIR} cmake "${args[@]}" -B "${BUILD_DIR}" mmperf
else
  echo "Error: Unsupported backend."
  exit 1
fi

cmake --build "${BUILD_DIR}"
