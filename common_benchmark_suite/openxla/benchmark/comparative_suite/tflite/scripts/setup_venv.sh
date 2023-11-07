#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#   VENV_DIR=tflite-models.venv
#   PYTHON=/usr/bin/python3.11

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR:-tflite-models.venv}"
PYTHON="${PYTHON:-"$(which python)"}"

echo "Setting up venv dir: ${VENV_DIR}"

${PYTHON} -m venv "${VENV_DIR}" || echo "Could not create venv."
source "${VENV_DIR}/bin/activate" || echo "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip || echo "Could not upgrade pip"

# Get IREE tools for importing and binarizing.
python -m pip install \
  --find-links https://openxla.github.io/iree/pip-release-links.html \
  iree-compiler \
  iree-tools-tflite

python -m pip install pillow

echo "Activate venv with:"
echo "  source ${VENV_DIR}/bin/activate"
