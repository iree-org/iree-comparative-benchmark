#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import requests
import tarfile


def download_file(source_url: str,
                  save_path: pathlib.Path,
                  unpack: bool = True):
  """Downloads `source_url` to `saved_path`.

  NEVER use this function to download from untrusted sources, it doesn't unpack
  the file safely.

  Args:
    source_url: URL to download.
    save_path: Path to save the file.
    unpack: Unarchive the .tgz if set. `x/y.tgz` will be unarchived to `x/y`.
  """

  save_path.parent.mkdir(parents=True, exist_ok=True)
  with requests.get(source_url, stream=True) as response:
    with save_path.open("wb") as f:
      for chunk in response.iter_content(chunk_size=65536):
        f.write(chunk)

  if not unpack:
    return

  if save_path.suffix == ".tgz":
    with tarfile.open(save_path) as tar_file:
      # If the tgz is at `x/y.tgz`, unpack at `x/y`.
      tar_file.extractall(save_path.with_suffix(""))
