#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#   VENV_DIR=tf-models.venv
#   PYTHON=/usr/bin/python3.10

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR:-tf-models.venv}"
PYTHON="${PYTHON:-"$(which python)"}"

echo "Setting up venv dir: ${VENV_DIR}"

${PYTHON} -m venv "${VENV_DIR}" || echo "Could not create venv."
source "${VENV_DIR}/bin/activate" || echo "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip || echo "Could not upgrade pip"
python -m pip install iree-tools-tf -f https://openxla.github.io/iree/pip-release-links.html

# Run through all model directories and install requirements.
TF_MODELS_DIR="$(dirname $(dirname $(dirname "${TD}")))/models/tf"
find "${TF_MODELS_DIR}" -type d | while read dir; do
  if [[ -f "${dir}/requirements.txt" ]]; then
    echo "Installing ${dir}/requirements.txt"
    python -m pip install --upgrade -r "${dir}/requirements.txt"
  fi
done

echo "Activate venv with:"
echo "  source ${VENV_DIR}/bin/activate"
