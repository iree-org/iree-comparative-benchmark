#!/bin/bash

# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs Markdownlint on Markdown (.md) files
# https://github.com/igorshubovych/markdownlint-cli

set -euo pipefail

declare -a included_files_patterns=(
  # All .md files.
  "./**/*.md"
)

declare -a excluded_files_patterns=(
  "**/third_party/**"
  "**/node_modules/**"
)

# ${excluded_files_patterns} is expanded into
# "--ignore pattern1 --ignore pattern2 ...", since markdownlint doesn't accept
# "--ignore pattern1 pattern2 ...".
markdownlint \
    "${included_files_patterns[*]}" \
    --config ./.markdownlint.yml \
    ${excluded_files_patterns[*]/#/--ignore }
