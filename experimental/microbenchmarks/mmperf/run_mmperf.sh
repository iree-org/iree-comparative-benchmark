#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#  REPO_DIR=mmperf
#
# Example usage:
#   ./run_mmperf.sh <build-dir> <output-dir>

set -xeuo pipefail

BUILD_DIR=$1
OUTPUT_DIR=$2

git clone https://github.com/mmperf/mmperf.git
pushd mmperf
git submodule update --init --recursive --jobs 8 --depth 1
popd

python3 -m venv mmperf.venv
source mmperf.venv/bin/activate
pip install -r mmperf/requirements.txt
pip install -r mmperf/external/llvm-project/mlir/python/requirements.txt

python3 mmperf/mmperf.py ${BUILD_DIR}/matmul/ ${OUTPUT_DIR}
