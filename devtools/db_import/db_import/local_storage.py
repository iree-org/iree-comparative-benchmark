## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""This API mimics the Google Cloud Storage Python APIs.
   Instead of accessing files in GCP, it's backed by the local
   filesystem. Only reading and listing files is supported.
"""

import os
import pathlib


class Blob:

  def __init__(self, file: pathlib.Path, bucket):
    self.file: pathlib.Path = file
    self.bucket = bucket

  @property
  def name(self):
    return os.path.relpath(self.file, self.bucket.path)

  def open(self):
    return open(self.file)


class Bucket:

  def __init__(self, directory: pathlib.Path, name: str):
    self.name: str = name
    self.path: pathlib.Path = directory / name

  def list_blobs(self):
    for root, dirs, files in os.walk(self.path):
      for file in files:
        yield Blob(pathlib.Path(root) / file, self)

  def blob(self, filepath: str):
    full_path = self.path / filepath
    return Blob(full_path, self)


class Client:

  def __init__(self, dir: pathlib.Path):
    self._dir: pathlib.Path = dir

  def get_bucket(self, name: str):
    return Bucket(self._dir, name)
