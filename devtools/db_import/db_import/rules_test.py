#!/usr/bin/env python
## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import pathlib
import tempfile
import unittest

from db_import.rules import apply_rule_to_file, BenchmarkRunAlreadyPresentError
from db_import.in_memory_storage import Bucket

EMPTY_CONFIG = {
    "bucket_name": "random_bucket_name",
    "cloud_function_name": "random_cloud_function_name"
}


class TestApplyRuleToFile(unittest.TestCase):

  def test_no_match(self):
    empty_file = Bucket().register_blob("hello.json", "")

    rule = {"filepath_regex": "unknown\.json", "result": "{}"}

    result = apply_rule_to_file(rule, empty_file, EMPTY_CONFIG, dict(), None,
                                None)
    self.assertEqual(result, False)

  def test_matches_filepath(self):
    empty_file = Bucket().register_blob("hello.json", "")

    rule = {"filepath_regex": "hello\.json", "result": "{}"}

    result = apply_rule_to_file(rule, empty_file, EMPTY_CONFIG, dict(), None,
                                None)
    self.assertEqual(result, dict())

  def test_already_present(self):
    empty_file = Bucket().register_blob("hello.json", "")

    rule = {"filepath_regex": "hello\.json", "result": "{}"}

    # This is usually making a DB query to check whether this data has already been imported.
    def presence_check(rule, parameters):
      return True

    with self.assertRaises(BenchmarkRunAlreadyPresentError):
      apply_rule_to_file(rule, empty_file, EMPTY_CONFIG, dict(), presence_check,
                         None)

  def test_filepath_capture(self):
    empty_file = Bucket().register_blob("hello.json", "")

    rule = {
        "filepath_regex": "(?P<capture>hello)\.json",
        "result": """std.parseJson(std.extVar("filepath_captures"))""",
    }

    result = apply_rule_to_file(rule, empty_file, EMPTY_CONFIG, dict(), None,
                                None)
    self.assertEqual(result, {"capture": "hello"})

  def test_import_snippet(self):
    empty_file = Bucket().register_blob("hello.json", "")

    rule = {
        "filepath_regex": "hello\.json",
        "result": """local func = import "func"; func("hello")""",
    }

    result = apply_rule_to_file(rule, empty_file, EMPTY_CONFIG,
                                {"func": "function(value) 'world'"}, None, None)
    self.assertEqual(result, "world")

  def test_load_json(self):
    contents = {"hello": 42, "world": 43}
    file_with_contents = Bucket().register_blob("hello.json",
                                                json.dumps(contents))

    rule = {
        "filepath_regex": "(?P<filepath>.*\.json)",
        "result": "std.parseJson(std.native('readFile')('hello.json'))",
    }

    result = apply_rule_to_file(rule, file_with_contents, EMPTY_CONFIG, dict(),
                                None, None)
    self.assertEqual(result, contents)

  def test_load_csv(self):
    csv_contents = """"col1", "col2"
"hello", "world"
"""
    file_with_contents = Bucket().register_blob("hello.json", csv_contents)

    rule = {
        "filepath_regex":
            "(?P<filepath>.*\.json)",
        "result":
            "std.native('parseCsv')(std.native('readFile')('hello.json'))",
    }

    result = apply_rule_to_file(rule, file_with_contents, EMPTY_CONFIG, dict(),
                                None, None)
    self.assertEqual(result, [{"col1": "hello", "col2": "world"}])

  def test_dump_files(self):
    bucket = Bucket()
    file1_with_content = bucket.register_blob("hello.json",
                                              "hello.json content")
    file2_with_content = bucket.register_blob("world.json",
                                              "world.json content")

    rule = {
        "filepath_regex":
            "hello\.json",
        "result":
            "std.native('readFile')('hello.json') + std.native('readFile')('world.json')",
    }

    with tempfile.TemporaryDirectory() as dir:
      temp_dir = pathlib.Path(dir)
      apply_rule_to_file(rule, file1_with_content, EMPTY_CONFIG, dict(), None,
                         temp_dir)

      path1 = temp_dir / EMPTY_CONFIG["bucket_name"] / "hello.json"
      self.assertTrue(path1.is_file())
      self.assertEqual(path1.read_text(), file1_with_content.contents)

      path2 = temp_dir / EMPTY_CONFIG["bucket_name"] / "world.json"
      self.assertTrue(path2.is_file())
      self.assertEqual(path2.read_text(), file2_with_content.contents)

  def test_timestamp_to_iso8601(self):
    empty_file = Bucket().register_blob("hello.json", "")

    rule = {
        "filepath_regex":
            "hello\.json",
        "result":
            """
            {
              epoch: std.native('timestampToIso8601')(0),
              random_time: std.native('timestampToIso8601')(1686644993)
            }
            """,
    }

    result = apply_rule_to_file(rule, empty_file, EMPTY_CONFIG, dict(), None,
                                None)
    self.assertEqual(
        result,
        {
            "epoch": "1970-01-01T00:00:00+00:00",
            "random_time": "2023-06-13T08:29:53+00:00",
        },
    )

  def test_parse_numbers(self):
    empty_file = Bucket().register_blob("hello.json", "")

    rule = {
        "filepath_regex":
            "hello\.json",
        "result":
            """
            std.assertEqual({
              int1: std.native('parseNumber')('0'),
              int2: std.native('parseNumber')('-2'),
              int3: std.native('parseNumber')('42'),
              float1: std.native('parseNumber')('0.1'),
              float2: std.native('parseNumber')('-2.1'),
              float3: std.native('parseNumber')('400.0'),
            },
            {
              int1: 0,
              int2: -2,
              int3: 42,
              float1: 0.1,
              float2: -2.1,
              float3: 400.0
            })
            """,
    }

    apply_rule_to_file(rule, empty_file, EMPTY_CONFIG, dict(), None, None)

  def test_try_reading_files_until_success(self):
    bucket = Bucket()
    file1_with_content = bucket.register_blob("hello.json",
                                              "hello.json content")
    bucket.register_blob("world.json", "world.json content")

    rule = {
        "filepath_regex":
            "hello\.json",
        "result":
            """
              std.native('tryReadingFilesUntilSuccess')([
                'does_not_exist.json',
                'does_not_exist_either.csv',
                'hello.json',
                'world.json' // This one will NOT be read because the previous one succeeds
              ])""",
    }

    result = apply_rule_to_file(rule, file1_with_content, EMPTY_CONFIG, dict(),
                                None, None)
    self.assertEqual(result, file1_with_content.contents)
