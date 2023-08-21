#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs `generate_model_artifacts.py` on all registered PyTorch models and saves
# artifacts into the directory `/tmp/pt_models_<torch-mlir-version>_<timestamp>`.
#
# Once complete. please upload the output directory to
# `gs://iree-model-artifacts/pt`, preserving directory name.
#
# Usage:
#     bash generate_saved_models.sh <(optional) model name regex>
#
# Requires python-3.11 and above and python-venv.
#
# Environment variables:
#   VENV_DIR=pt-models.venv
#   PYTHON=/usr/bin/python3.11
#   WITH_CUDA=1
#   GCS_UPLOAD_DIR=gs://iree-model-artifacts/pytorch
#   AUTO_UPLOAD=1
#
# Positional arguments:
#   FILTER (Optional): Regex to match models, e.g., BERT_LARGE_FP32_.+

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR:-pt-models.venv}"
PYTHON="${PYTHON:-"$(which python)"}"
WITH_CUDA="${WITH_CUDA:-}"
AUTO_UPLOAD="${AUTO_UPLOAD:-0}"

FILTER="${1:-".*"}"

VENV_DIR=${VENV_DIR} PYTHON=${PYTHON} WITH_CUDA=${WITH_CUDA} "${TD}/setup_venv.sh"
source ${VENV_DIR}/bin/activate

# Generate unique output directory.
TORCH_MLIR_VERSION=$(pip show torch-mlir | grep Version | sed -e "s/^Version: \(.*\)$/\1/g")
DIR_NAME="pt_models_${TORCH_MLIR_VERSION}_$(date +'%s')"
OUTPUT_DIR="/tmp/${DIR_NAME}"
mkdir "${OUTPUT_DIR}"

pip list > "${OUTPUT_DIR}/models_version_info.txt"

declare -a args=(
  -o "${OUTPUT_DIR}"
  --filter="${FILTER}"
)

if (( AUTO_UPLOAD == 1 )); then
  args+=(
    --auto_upload
  )
fi

python "${TD}/generate_model_artifacts.py" "${args[@]}"

echo "Output directory: ${OUTPUT_DIR}"
