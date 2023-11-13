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
# OOBI_VENV_DIR: name of the virtual environment.
#
# Example usage:
# ./generate_iree_vs_tflite_report.sh <iree-results> <tflite-results> <output-html>

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${OOBI_VENV_DIR:-comparisons.venv}"
PYTHON="${PYTHON:-/usr/bin/python3}"

IREE_RESULTS_PATH=$1
TFLITE_RESULTS_PATH=$2
OUTPUT_PATH=$3

VENV_DIR="${VENV_DIR}" PYTHON="${PYTHON}" source "${TD}/setup_venv.sh"

"${TD}/generate_iree_vs_tflite_report.py" \
    --iree_results_path "${IREE_RESULTS_PATH}" \
    --tflite_results_path "${TFLITE_RESULTS_PATH}" \
    --output_path "${OUTPUT_PATH}"
