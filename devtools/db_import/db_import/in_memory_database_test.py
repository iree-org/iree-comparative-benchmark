#!/usr/bin/env python
## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import db
import in_memory_database


class TestInMemoryDatabase(unittest.TestCase):

  def test_insert_rows(self):
    client = in_memory_database.Client()
    self.assertTrue(isinstance(client, db.Client))

    table = client.get_table('dataset_id.table_name')
    self.assertTrue(isinstance(table, db.Table))

    self.assertEqual(table.table_id, 'table_name')
    self.assertEqual(table.dataset_id, 'dataset_id')
    self.assertEqual([], table.get_rows())

    rows = [{"col1": "hello", "col2": "world"}]
    client.insert_rows(table, rows)
    self.assertEqual(rows, table.get_rows())

  def test_query(self):
    client = in_memory_database.Client()
    with self.assertRaises(RuntimeError):
      # Querying is not supported
      client.query('SELECT * FROM table')


if __name__ == "__main__":
  unittest.main()
