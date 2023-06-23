#!/usr/bin/env python3

# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
from typing import Any, Sequence


def run_command(
    command: Sequence[Any],
    dry_run: bool = False,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
    **run_kwargs,
) -> subprocess.CompletedProcess:
  """Thin wrapper around subprocess.run"""
  print(f"Running: `{' '.join(str(arg) for arg in command)}`")
  if dry_run:
    # Dummy CompletedProess with successful returncode.
    return subprocess.CompletedProcess(command, returncode=0)

  completed_process = subprocess.run(
      command,
      text=text,
      check=check,
      capture_output=capture_output,
      **run_kwargs,
  )
  return completed_process


def check_gcloud_auth(dry_run: bool = False):
  # Ensure the user has the correct authorization if they try to push to GCR.
  try:
    run_command(["which", "gcloud"])
  except subprocess.CalledProcessError as error:
    raise RuntimeError(
        "gcloud not found. See https://cloud.google.com/sdk/install for "
        "installation.") from error
  run_command(["gcloud", "auth", "configure-docker"], dry_run)
