#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#   VENV_DIR=tf-benchmarks.venv
#   PYTHON=/usr/bin/python3.10

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR=${VENV_DIR:-pt-benchmarks.venv}
PYTHON="${PYTHON:-"$(which python3)"}"

"${PYTHON}" -m venv "${VENV_DIR}" || echo "Could not create venv."
source "${VENV_DIR}/bin/activate" || echo "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip || die "Could not upgrade pip"
python -m pip install -r "${TD}/requirements.txt"

echo "Activate venv with:"
echo "  source ${VENV_DIR}/bin/activate"
