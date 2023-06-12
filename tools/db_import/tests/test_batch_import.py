#!/usr/bin/env python
## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest, io
from db_import.batch_import import import_entire_bucket
import contextlib
from typing import Iterable


class Blob:

  def __init__(self, name, bucket, contents):
    self.name = name
    self.bucket = bucket
    self.contents = contents

  def open(self):
    return io.StringIO(self.contents)


class Bucket:

  def __init__(self):
    self.contents: dict[str, str] = {}

  def blob(self, name):
    return Blob(name, self, self.contents[name])

  def list_blobs(self, prefix: str = None) -> Iterable[Blob]:
    return (self.blob(name)
            for name in self.contents
            if not prefix or name.startswith(prefix))

  def registerBlob(self, name: str, contents: str):
    self.contents[name] = contents
    return self.blob(name)


class StorageClient:

  def __init__(self):
    self.buckets: dict[str, Bucket] = {}

  def get_bucket(self, name):
    return self.buckets[name]

  def registerBucket(self, name) -> Bucket:
    self.buckets[name] = Bucket()
    return self.get_bucket(name)


class Table:

  def __init__(self):
    self.rows: list = []


class DbClient:

  def __init__(self):
    self.tables: dict[str, Table] = {}

  def insert_rows(self, table: Table, rows: list):
    table.rows += rows

  def get_table(self, table_name) -> Table:
    return self.tables[table_name]

  def register_table(self, table_name):
    self.tables[table_name] = Table()
    return self.get_table(table_name)


EMPTY_CONFIG = {"bucket_name": "", "cloud_function_name": ""}
BASIC_SNIPPETS = {
    "getFilepathCapture":
        "function(name) std.parseJson(std.extVar('filepath_captures'))[name]",
    "loadJson":
        "function(path) std.parseJson(std.native('readFile')(path))",
    "loadCsv":
        "function(path) std.native('parseCsv')(std.native('readFile')(path))",
}


class TestImportEntireBucket(unittest.TestCase):

  def test_empty_bucket(self):
    rule = {"filepath_regex": "\.json$", "result": "{}"}
    config = {
        "bucket_name": "random_bucket_name",
        "rules": [rule],
        "cloud_function_name": "random_cloud_function_name",
        "table_name": "random_table_name",
    }

    db_client = DbClient()
    table = db_client.register_table("random_table_name")

    storage_client = StorageClient()
    storage_client.registerBucket("random_bucket_name")

    import_entire_bucket(db_client, storage_client, config, BASIC_SNIPPETS,
                         False, False)
    self.assertEqual(table.rows, [])

  def test_no_match(self):
    rule = {"filepath_regex": "^unknown\.json$", "result": "{}"}
    config = {
        "bucket_name": "random_bucket_name",
        "rules": [rule],
        "cloud_function_name": "random_cloud_function_name",
        "table_name": "random_table_name",
    }

    db_client = DbClient()
    table = db_client.register_table("random_table_name")

    storage_client = StorageClient()
    storage_client.registerBucket("random_bucket_name").registerBlob(
        "hello.json", "")

    import_entire_bucket(db_client, storage_client, config, BASIC_SNIPPETS,
                         False, False)
    self.assertEqual(table.rows, [])

  def test_single_file(self):
    rule = {"filepath_regex": "^hello\.json$", "result": "{}"}
    config = {
        "bucket_name": "random_bucket_name",
        "rules": [rule],
        "cloud_function_name": "random_cloud_function_name",
        "table_name": "random_table_name",
    }

    db_client = DbClient()
    table = db_client.register_table("random_table_name")

    storage_client = StorageClient()
    storage_client.registerBucket("random_bucket_name").registerBlob(
        "hello.json", "")

    with contextlib.redirect_stdout(io.StringIO()):
      import_entire_bucket(db_client, storage_client, config, BASIC_SNIPPETS,
                           False, False)

    self.assertEqual(table.rows, [{}])

  def test_multiple_files(self):
    rule = {"filepath_regex": "^hello.\.json$", "result": "{}"}
    config = {
        "bucket_name": "random_bucket_name",
        "rules": [rule],
        "cloud_function_name": "random_cloud_function_name",
        "table_name": "random_table_name",
    }

    db_client = DbClient()
    table = db_client.register_table("random_table_name")

    storage_client = StorageClient()
    bucket = storage_client.registerBucket("random_bucket_name")
    bucket.registerBlob("hello1.json", "")
    bucket.registerBlob("hello2.json", "")
    bucket.registerBlob("hello3.json", "")

    with contextlib.redirect_stdout(io.StringIO()):
      import_entire_bucket(db_client, storage_client, config, BASIC_SNIPPETS,
                           False, False)

    self.assertEqual(table.rows, [{}, {}, {}])


if __name__ == "__main__":
  unittest.main()
