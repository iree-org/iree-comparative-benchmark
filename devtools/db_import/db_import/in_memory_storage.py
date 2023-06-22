## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
from typing import Dict, Iterable, Optional


class Blob:
  """This class mocks a Blob from the google-cloud-storage package.
     So far it only supports reading files. The contents is provided as a string.
  """

  def __init__(self, name: str, bucket: "Bucket", contents: str):
    self.name = name
    self.bucket = bucket
    self.contents = contents

  def open(self):
    return io.StringIO(self.contents)


class Bucket:
  """This class mocks a Bucket from the google-cloud-storage package.
     Blobs can be registered with `register_blob` and will then appear as present in the bucket.
  """

  def __init__(self):
    self.contents: dict[str, str] = {}

  def blob(self, name: str) -> Blob:
    return Blob(name, self, self.contents[name])

  def register_blob(self, name: str, contents: str):
    self.contents[name] = contents
    return self.blob(name)

  def list_blobs(self, prefix: Optional[str] = None) -> Iterable[Blob]:
    for name, content in self.contents.items():
      if prefix and not name.startswith(prefix):
        continue
      yield Blob(name, self, content)


class Client:
  """This class mocks a Client from the google-cloud-storage package.
     Buckets can be registered with `register_bucket` and will then appear as available.
  """

  def __init__(self):
    self._buckets: Dict[str, Bucket] = {}

  def register_bucket(self, name: str) -> Bucket:
    return self._buckets.setdefault(name, Bucket())

  def get_bucket(self, name: str) -> Bucket:
    return self._buckets[name]
