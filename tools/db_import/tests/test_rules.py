#!/usr/bin/env python
## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest, io, json
from db_import.rules import apply_rule_to_file


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

  def registerBlob(self, name, contents):
    self.contents[name] = contents
    return self.blob(name)


EMPTY_CONFIG = {"bucket_name": "", "cloud_function_name": ""}
BASIC_SNIPPETS = {
    "getFilepathCapture":
        "function(name) std.parseJson(std.extVar('filepath_captures'))[name]",
    "loadJson":
        "function(path) std.parseJson(std.native('readFile')(path))",
    "loadCsv":
        "function(path) std.native('parseCsv')(std.native('readFile')(path))",
}


class TestApplyRuleToFile(unittest.TestCase):

  def test_no_match(self):
    empty_file = Blob("hello.json", Bucket(), "")

    rule = {"filepath_regex": "^unknown\.json$", "result": "{}"}

    result = apply_rule_to_file(rule, empty_file, EMPTY_CONFIG, dict(), None,
                                None)
    self.assertEqual(result, False)

  def test_matches_filepath(self):
    # file = Blob("hello.json", bucket, '{ "hello" : "world" }')
    empty_file = Blob("hello.json", Bucket(), "")

    rule = {"filepath_regex": "^hello\.json$", "result": "{}"}

    result = apply_rule_to_file(rule, empty_file, EMPTY_CONFIG, dict(), None,
                                None)
    self.assertEqual(result, dict())

  def test_filepath_capture(self):
    # file = Blob("hello.json", bucket, '{ "hello" : "world" }')
    empty_file = Blob("hello.json", Bucket(), "")

    rule = {
        "filepath_regex":
            "^(?P<capture>hello)\.json$",
        "result":
            """local getFilepathCapture = import "getFilepathCapture";
                         { capture: getFilepathCapture("capture")}""",
    }

    result = apply_rule_to_file(rule, empty_file, EMPTY_CONFIG, BASIC_SNIPPETS,
                                None, None)
    self.assertEqual(result, {"capture": "hello"})

  def test_read_json(self):
    # file = Blob("hello.json", bucket, '{ "hello" : "world" }')
    contents = {"hello": 42, "world": 43}
    bucket = Bucket()
    file_with_contents = bucket.registerBlob("hello.json", json.dumps(contents))

    rule = {
        "filepath_regex":
            "^(?P<filepath>.*\.json)$",
        "result":
            """
            local loadJson = import "loadJson";
            local getFilepathCapture = import "getFilepathCapture";
            loadJson(getFilepathCapture('filepath'))
        """,
    }

    result = apply_rule_to_file(rule, file_with_contents, EMPTY_CONFIG,
                                BASIC_SNIPPETS, None, None)
    self.assertEqual(result, contents)

  def test_read_csv(self):
    # file = Blob("hello.json", bucket, '{ "hello" : "world" }')
    csv_contents = """"col1", "col2"
"hello", "world"
"""
    bucket = Bucket()
    file_with_contents = bucket.registerBlob("hello.json", csv_contents)

    rule = {
        "filepath_regex":
            "^(?P<filepath>.*\.json)$",
        "result":
            """
            local loadCsv = import "loadCsv";
            local getFilepathCapture = import "getFilepathCapture";
            loadCsv(getFilepathCapture('filepath'))
        """,
    }

    result = apply_rule_to_file(rule, file_with_contents, EMPTY_CONFIG,
                                BASIC_SNIPPETS, None, None)
    self.assertEqual(result, [{"col1": "hello", "col2": "world"}])


if __name__ == "__main__":
  unittest.main()
