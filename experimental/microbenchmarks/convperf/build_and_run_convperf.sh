#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#   CC=clang
#   CXX=clang++
#
# Example usage:
#   ./build_convperf.sh <build-dir> <output-dir>

set -xeuo pipefail

BUILD_DIR=$1
OUTPUT_DIR=$2

git clone https://github.com/nod-ai/convperf.git
pushd convperf
git submodule update --init --recursive --jobs 8 --depth 1

# Create virtual environment.
python3 -m venv convperf.venv
source convperf.venv/bin/activate
pip install -r requirements.txt

popd # convperf.

# Build convperf.
cmake -GNinja \
  -DCMAKE_C_COMPILER="${CC:-clang}" \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -B "${BUILD_DIR}" convperf

cmake --build "${BUILD_DIR}"

# Run ConvPerf for several threading configurations.
# There is a non-deterministic bus in libxsmm that occurs when the number of threads > 1.
# We disable these threads for now.
# declare -a threads=( 1 2 4 8 16 )
declare -a threads=(1)

for i in "${threads[@]}"; do
  export NUM_THREADS=$i
  python3 "convperf/convperf.py" \
      --benchmark_tool="${BUILD_DIR}/tools/benchmark_conv" \
      --runners="iree,xsmm" \
      --benchmark_sizes="convperf/benchmark_sizes/resnet50.json" \
      --num_threads="$i"

  python "convperf/convperf.py" --visualize --runtimes_file="runtimes.json"
  mv runtimes.json "${OUTPUT_DIR}/resnet50_thread$i.json"
  mv convs.png "${OUTPUT_DIR}/resnet50_thread$i.png"
done
