## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os


class MockBlob:

  def __init__(self, file : str, bucket):
    self.file = file
    self.bucket = bucket

  @property
  def name(self):
    return os.path.relpath(self.file, self.bucket.path)

  def open(self):
    return open(self.file)


class MockBucket:
  def __init__(self, dir : str, name : str):
    self.name : str = name
    self.path : str = os.path.join(dir, name)

  def list_blobs(self):
    for root, dirs, files in os.walk(self.path):
      for file in files:
        yield MockBlob(os.path.join(root, file), self)

  def blob(self, filepath : str):
    full_path = os.path.join(self.path, filepath)
    return MockBlob(full_path, self)


class MockClient:
  def __init__(self, dir : str):
    self._dir : str = dir

  def bucket(self, name : str):
    return MockBucket(self._dir, name)

  def get_bucket(self, name : str):
    return MockBucket(self._dir, name)
