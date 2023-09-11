#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#   VENV_DIR=jax-models.venv
#   PYTHON=/usr/bin/python3.10
#   WITH_CUDA=1

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR:-jax-models.venv}"
WITH_CUDA="${WITH_CUDA:-}"
PYTHON="${PYTHON:-"$(which python)"}"

echo "Setting up venv dir: ${VENV_DIR}"

${PYTHON} -m venv "${VENV_DIR}" || echo "Could not create venv."
source "${VENV_DIR}/bin/activate" || echo "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip || echo "Could not upgrade pip"

# Get iree-ir-tool for IR postprocessing.
python -m pip install \
  --find-links https://openxla.github.io/iree/pip-release-links.html \
  iree-compiler

if [ -z "$WITH_CUDA" ]; then
  echo "Installing jax and dependencies without cuda support; set WITH_CUDA to enable cuda support."
  python -m pip install --upgrade "jax[cpu]" "flax"
else
  echo "Installing jax and dependencies with cuda support"
  python -m pip install --upgrade "jax[cuda11_local]" "flax" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

# Run through all model directories and install requirements.
JAX_MODELS_DIR="$(dirname $(dirname $(dirname "${TD}")))/models/jax"
find "${JAX_MODELS_DIR}" -type d | while read dir; do
  if [[ -f "${dir}/requirements.txt" ]]; then
    echo "Installing ${dir}/requirements.txt"
    python -m pip install --upgrade -r "${dir}/requirements.txt"
  fi
done

echo "Activate venv with:"
echo "  source ${VENV_DIR}/bin/activate"
