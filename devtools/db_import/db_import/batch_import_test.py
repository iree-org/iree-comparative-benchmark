## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import io
import unittest

from db_import.batch_import import import_entire_bucket
from db_import import in_memory_storage
from db_import import in_memory_database


class TestImportEntireBucket(unittest.TestCase):

  def test_empty_bucket(self):
    rule = {"filepath_regex": "\.json$", "result": "{}"}
    config = {
        "bucket_name": "random_bucket_name",
        "rules": [rule],
        "cloud_function_name": "random_cloud_function_name",
        "table_name": "random_table_name",
    }

    db_client = in_memory_database.Client()
    table = db_client.register_table("random_table_name")

    storage_client = in_memory_storage.Client()
    storage_client.register_bucket("random_bucket_name")

    import_entire_bucket(db_client,
                         storage_client,
                         config,
                         snippets={},
                         check_for_presence=False)
    self.assertEqual(table.rows, [])

  def test_no_match(self):
    rule = {"filepath_regex": "^unknown\.json$", "result": "{}"}
    config = {
        "bucket_name": "random_bucket_name",
        "rules": [rule],
        "cloud_function_name": "random_cloud_function_name",
        "table_name": "random_table_name",
    }

    db_client = in_memory_database.Client()
    table = db_client.register_table("random_table_name")

    storage_client = in_memory_storage.Client()
    storage_client.register_bucket("random_bucket_name").register_blob(
        "hello.json", "")

    import_entire_bucket(db_client,
                         storage_client,
                         config,
                         snippets={},
                         check_for_presence=False)
    self.assertEqual(table.rows, [])

  def test_single_file(self):
    rule = {"filepath_regex": "^hello\.json$", "result": "{}"}
    config = {
        "bucket_name": "random_bucket_name",
        "rules": [rule],
        "cloud_function_name": "random_cloud_function_name",
        "table_name": "random_table_name",
    }

    db_client = in_memory_database.Client()
    table = db_client.register_table("random_table_name")

    storage_client = in_memory_storage.Client()
    storage_client.register_bucket("random_bucket_name").register_blob(
        "hello.json", "")

    with contextlib.redirect_stdout(io.StringIO()):
      import_entire_bucket(db_client,
                           storage_client,
                           config,
                           snippets={},
                           check_for_presence=False)

    self.assertEqual(table.rows, [{}])

  def test_multiple_files(self):
    rule = {"filepath_regex": "^hello.\.json$", "result": "{}"}
    config = {
        "bucket_name": "random_bucket_name",
        "rules": [rule],
        "cloud_function_name": "random_cloud_function_name",
        "table_name": "random_table_name",
    }

    db_client = in_memory_database.Client()
    table = db_client.register_table("random_table_name")

    storage_client = in_memory_storage.Client()
    bucket = storage_client.register_bucket("random_bucket_name")
    bucket.register_blob("hello1.json", "")
    bucket.register_blob("hello2.json", "")
    bucket.register_blob("hello3.json", "")

    with contextlib.redirect_stdout(io.StringIO()):
      import_entire_bucket(db_client,
                           storage_client,
                           config,
                           snippets={},
                           check_for_presence=False)

    self.assertEqual(table.rows, [{}, {}, {}])
