## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import tempfile
import unittest

from db_import import local_storage
from db_import import storage


class TestLocalStorage(unittest.TestCase):

  def run(self, result=None):
    with tempfile.TemporaryDirectory() as directory_path_str:
      self.directory_path = pathlib.Path(directory_path_str)
      super(TestLocalStorage, self).run(result)

  def test_read_file(self):
    client = local_storage.Client(pathlib.Path(self.directory_path))
    self.assertTrue(isinstance(client, storage.Client))

    bucket_name = "bucket"
    (self.directory_path / bucket_name).mkdir()
    bucket = client.get_bucket(bucket_name)
    self.assertTrue(isinstance(bucket, storage.Bucket))

    content = "blibablub"
    filename = "filename.txt"
    (self.directory_path / bucket_name / filename).write_text(content)
    blob = bucket.blob(filename)

    self.assertTrue(isinstance(blob, storage.Blob))
    with blob.open() as fd:
      self.assertEqual(content, fd.read())

  def test_list_blobs(self):
    client = local_storage.Client(pathlib.Path(self.directory_path))

    bucket_name = "bucket"
    (self.directory_path / bucket_name).mkdir()
    bucket = client.get_bucket(bucket_name)

    content = "blibablub"
    filename1 = "filename1.txt"
    filename2 = "filename2.txt"
    (self.directory_path / bucket_name / filename1).write_text(content)
    (self.directory_path / bucket_name / filename2).write_text(content)

    self.assertEqual(set(blob.name for blob in bucket.list_blobs()),
                     set([filename1, filename2]))
