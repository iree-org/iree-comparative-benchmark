## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import pathlib
import tempfile
import unittest

from db_import import config


class TestConfig(unittest.TestCase):

  def run(self, result=None):
    with tempfile.TemporaryDirectory() as directory_path_str:
      self.directory_path = pathlib.Path(directory_path_str)
      super(TestConfig, self).run(result)

  def test_load_empty_config(self):
    fd = io.StringIO("---\n")
    self.assertEqual(config.load_config(fd), None)

  def test_load_basic_config(self):
    fd = io.StringIO("---\nkey: value\n")
    self.assertEqual(config.load_config(fd), {"key": "value"})

  def test_load_config_with_relative_path_embedding(self):
    with tempfile.TemporaryDirectory() as directory_path_str:
      directory_path = pathlib.Path(directory_path_str)
      (directory_path / "file.txt").write_text("42")

      # The embedding feature only works when the YAML is loaded from a file, so no StringIO here:
      config_file_path = directory_path / "config.yml"
      config_file_path.write_text("---\nkey: !embed file.txt\n")

      with open(config_file_path) as fd:
        self.assertEqual(config.load_config(fd), {"key": "42"})

  def test_load_config_with_absolute_path_embedding(self):
    with tempfile.TemporaryDirectory() as directory_path_str:
      directory_path = pathlib.Path(directory_path_str)
      (directory_path / "file.txt").write_text("42")

      # The embedding feature only works when the YAML is loaded from a file, so no StringIO here:
      config_file_path = directory_path / "config.yml"
      config_file_path.write_text(
          f"---\nkey: !embed {directory_path / 'file.txt'}\n")

      with open(config_file_path) as fd:
        self.assertEqual(config.load_config(fd), {"key": "42"})

  def test_load_config_with_forbidden_path(self):
    with tempfile.TemporaryDirectory() as directory_path_str:
      directory_path = pathlib.Path(directory_path_str)
      (directory_path / "file.txt").write_text("42")

      # The embedding feature only works when the YAML is loaded from a file, so no StringIO here:
      config_file_path = directory_path / "config.yml"
      config_file_path.write_text(f"---\nkey: !embed /etc/passwd\n")

      with open(config_file_path) as fd:
        with self.assertRaises(Exception):
          config.load_config(fd)
