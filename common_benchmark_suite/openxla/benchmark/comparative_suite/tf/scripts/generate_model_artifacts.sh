#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs `generate_model_artifacts.py` on all registered JAX models and saves
# artifacts into the directory `/tmp/tf_models_<tf-version>_<timestamp>`.
#
# Once complete. please upload the output directory to
# `gs://iree-model-artifacts/tensorflow`, preserving directory name.
#
# Usage:
#     bash generate_saved_models.sh <(optional) model name regex>
#
# Requires python-3.10 and above and python-venv.
#
# Environment variables:
#   VENV_DIR=tf-models.venv
#   PYTHON=/usr/bin/python3.10
#   GCS_UPLOAD_DIR=gs://iree-model-artifacts/tensorflow
#   AUTO_UPLOAD=1
#
# Positional arguments:
#   FILTER (Optional): Regex to match models, e.g., BERT_LARGE_FP32_.+

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR:-tf-models.venv}"
PYTHON="${PYTHON:-"$(which python)"}"
AUTO_UPLOAD="${AUTO_UPLOAD:-0}"

FILTER="${1:-".*"}"

VENV_DIR=${VENV_DIR} PYTHON=${PYTHON} "${TD}/setup_venv.sh"
source ${VENV_DIR}/bin/activate

# Generate unique output directory.
TF_VERSION=$(pip show tf-nightly | grep Version | sed -e "s/^Version: \(.*\)$/\1/g")
DIR_NAME="tf_models_${TF_VERSION}_$(date +'%s')"
OUTPUT_DIR="/tmp/${DIR_NAME}"
mkdir ${OUTPUT_DIR}

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
