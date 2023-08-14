#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#   VENV_DIR=pt-models.venv
#   PYTHON=/usr/bin/python3.10
#   WITH_CUDA=1

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR:-pt-models.venv}"
WITH_CUDA="${WITH_CUDA:-}"
PYTHON="${PYTHON:-"$(which python)"}"

echo "Setting up venv dir: ${VENV_DIR}"

# Start with a fresh ${VENV_DIR} to ensure torch-mlir is updated.
rm -rf ${VENV_DIR}

${PYTHON} -m venv "${VENV_DIR}" || echo "Could not create venv."
source "${VENV_DIR}/bin/activate" || echo "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip || echo "Could not upgrade pip"

# Run through all model directories and install requirements.
PT_MODELS_DIR="$(dirname $(dirname $(dirname "${TD}")))/models/pt"
find "${PT_MODELS_DIR}" -type d | while read dir; do
  if [[ -f "${dir}/requirements.txt" ]]; then
    echo "Installing ${dir}/requirements.txt"
    python -m pip install --upgrade -r "${dir}/requirements.txt"
  fi
done

if [ -z "$WITH_CUDA" ]
then
  echo "Installing torch-mlir and dependencies without cuda support; set WITH_CUDA to enable cuda support."
  python -m pip install --pre torch-mlir torchvision -f https://llvm.github.io/torch-mlir/package-index/ -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
else
  echo "Installing torch-mlir and dependencies with cuda support"
  python -m pip install --pre torch-mlir torchvision -f https://llvm.github.io/torch-mlir/package-index/ --index-url https://download.pytorch.org/whl/nightly/cu118
fi

# Install accelerate after torch-mlir and torch installation to automatically pick a compatible version
python -m pip install accelerate

echo "Activate venv with:"
echo "  source $VENV_DIR/bin/activate"
