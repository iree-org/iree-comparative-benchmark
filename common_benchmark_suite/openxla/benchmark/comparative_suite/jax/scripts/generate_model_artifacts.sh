#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs `generate_model_artifacts.py` on all registered JAX models and saves
# artifacts into the directory
# `${OUTPUT_DIR}/jax_models_<jax-version>_<timestamp>`.
#
# Once complete. please upload the output directory to
# `gs://iree-model-artifacts/jax`, preserving directory name.
#
# Usage:
#     bash generate_saved_models.sh <(optional) model name regex>
#
# Requires python-3.10 and above and python-venv.
#
# Environment variables:
#   VENV_DIR=jax-models.venv
#   PYTHON=/usr/bin/python3.10
#   WITH_CUDA=1
#   GCS_UPLOAD_DIR=gs://iree-model-artifacts/jax
#   AUTO_UPLOAD=1
#
# Positional arguments:
#   FILTER (Optional): Regex to match models, e.g., BERT_LARGE_FP32_.+

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR:-jax-models.venv}"
PYTHON="${PYTHON:-"$(which python)"}"
WITH_CUDA="${WITH_CUDA:-}"
AUTO_UPLOAD="${AUTO_UPLOAD:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp}"
FILTER=( "$@" )

VENV_DIR=${VENV_DIR} PYTHON=${PYTHON} WITH_CUDA=${WITH_CUDA} "${TD}/setup_venv.sh"
source ${VENV_DIR}/bin/activate

VENV_DIR_PATH="$(realpath ${VENV_DIR})"
PYTHON_VERSION="$(python --version | sed -e "s/^Python \(.*\)\.\(.*\)\..*$/\1\.\2/g")"

# Generate unique output directory.
JAX_VERSION=$(pip show jax | grep Version | sed -e "s/^Version: \(.*\)$/\1/g")
DIR_NAME="jax_models_${JAX_VERSION}_$(date +'%s')"
VERSION_DIR="${OUTPUT_DIR}/${DIR_NAME}"
mkdir "${VERSION_DIR}"

pip list > "${VERSION_DIR}/models_version_info.txt"

declare -a args=(
  -o "${VERSION_DIR}"
  --iree_compile_path="$(which iree-compile)"
  --iree_ir_tool="$(which iree-ir-tool)"
)

if (( "${#FILTER[@]}" > 0 )); then
  args+=( --filter "${FILTER[@]}" )
fi

if (( AUTO_UPLOAD == 1 )); then
  args+=( --auto_upload )
fi

python "${TD}/generate_model_artifacts.py" "${args[@]}"

echo "Output directory: ${VERSION_DIR}"
