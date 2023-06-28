#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import concurrent.futures
import numpy as np
import pathlib
import requests
import tarfile
from typing import List, Sequence, Tuple


def download_file(source_url: str,
                  save_path: pathlib.Path,
                  unpack: bool = True,
                  verbose: bool = False):
  """Downloads `source_url` to `saved_path`.

  NEVER use this function to download from untrusted sources, it doesn't unpack
  the file safely.

  Args:
    source_url: URL to download.
    save_path: Path to save the file.
    unpack: Unarchive the .tgz if set. `x/y.tgz` will be unarchived to `x/y`.
    verbose: Show downloading message.
  """

  save_path.parent.mkdir(parents=True, exist_ok=True)

  if verbose:
    print(f"Downloading '{source_url}' to '{save_path}'.")

  # requests doesn't clearly state that its session is thread-safe. In order to
  # download in parallel, don't use session here.
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


def download_files(urls_to_paths: List[Tuple[str, pathlib.Path]],
                   max_workers: int = 8,
                   verbose: bool = False):
  """Fetch a list of URLs in parallel."""

  with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
    futures = []
    for source_url, save_path in urls_to_paths:
      futures.append(
          executor.submit(download_file,
                          source_url=source_url,
                          save_path=save_path,
                          verbose=verbose))
    concurrent.futures.wait(futures)


def compare_tensors(outputs: Sequence[np.ndarray],
                    expects: Sequence[np.ndarray],
                    absolute_tolerance: float = 0,
                    relative_tolerance: float = 0) -> List[Tuple[bool, float]]:
  """Compares numpy tensors and returns a list of (is_equal, max_diff).
  
  See numpy.allclose for the meaning of absolute_tolerance and
  relative_tolerance.
  """

  verdicts = []
  for output, expect in zip(outputs, expects):
    is_equal = np.allclose(output,
                           expect,
                           rtol=relative_tolerance,
                           atol=absolute_tolerance)
    max_diff = np.max(np.abs(expect - output))
    verdicts.append((is_equal, max_diff))

  return verdicts
