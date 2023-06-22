#!/usr/bin/env python
## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import in_memory_storage
import storage


class TestInMemoryStorage(unittest.TestCase):

  def test_read_file(self):
    client = in_memory_storage.Client()
    self.assertTrue(isinstance(client, storage.Client))

    bucket_name = "bucket"
    client.register_bucket(bucket_name)
    bucket = client.get_bucket(bucket_name)
    self.assertTrue(isinstance(bucket, storage.Bucket))

    content = "blibablub"
    filename = "filename.txt"
    bucket.register_blob(filename, content)
    blob = bucket.blob(filename)

    self.assertTrue(isinstance(blob, storage.Blob))
    with blob.open() as fd:
      self.assertEqual(content, fd.read())

  def test_list_blobs(self):
    client = in_memory_storage.Client()

    bucket_name = "bucket"
    client.register_bucket(bucket_name)
    bucket = client.get_bucket(bucket_name)

    content = "blibablub"
    filename1 = "filename1.txt"
    filename2 = "filename2.txt"
    bucket.register_blob(filename1, content)
    bucket.register_blob(filename2, content)

    self.assertEqual(set(blob.name for blob in bucket.list_blobs()),
                     set([filename1, filename2]))


if __name__ == "__main__":
  unittest.main()
