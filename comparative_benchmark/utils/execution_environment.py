# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities to collect the information of execution environment."""

import subprocess
import re


def get_python_environment_info():
  """ Returns a dictionary of package versions in the python virtual
  environment.
  """
  output = subprocess.check_output(["pip", "list"]).decode("utf-8")
  # The first few lines are the table headers so we remove that.
  output = output[output.rindex("---\n") + 4:]
  output = output.split("\n")
  package_dict = {}
  for item in output:
    split = re.split(r"\s+", item)
    if len(split) == 2:
      package_dict[split[0]] = split[1]
  return package_dict
