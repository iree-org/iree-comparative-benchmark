#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#   VENV_DIR=jax-benchmarks.venv
#   PYTHON=/usr/bin/python3.10
#   WITH_CUDA=1

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${VENV_DIR:-jax-benchmarks.venv}"
PYTHON="${PYTHON:-"$(which python3)"}"
WITH_CUDA="${WITH_CUDA:-}"
echo "Setting up venv dir: ${VENV_DIR}"

"${PYTHON}" -m venv "${VENV_DIR}" || echo "Could not create venv."
source "${VENV_DIR}/bin/activate" || echo "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip || echo "Could not upgrade pip"

if [ -z "$WITH_CUDA" ]; then
  echo "Installing jax and dependencies without cuda support; set WITH_CUDA to enable cuda support."
  python -m pip install --upgrade "jax[cpu]" "flax"
else
  echo "Installing jax and dependencies with cuda support"
  python -m pip install --upgrade "jax[cuda11_local]" "flax" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

python -m pip install --upgrade -r "${TD}/requirements.txt"

echo "Activate venv with:"
echo "  source ${VENV_DIR}/bin/activate"
