#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Create a results JSON file with the information of execution environment.

set -euo pipefail

OUTPUT_PATH="${1}"

# Create json file and populate with global information.
PACKAGE_VERSIONS="$(python -m pip list --format json)"
TIMESTAMP="$(date +'%s')"
cat <<EOF > "${OUTPUT_PATH}"
{
  "trigger": {
    "timestamp": "${TIMESTAMP}"
  },
  "execution_environment": {
    "python_environment": $PACKAGE_VERSIONS
  },
  "benchmarks": []
}
EOF
