## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import unittest

from google.cloud import bigquery

from db_import import db
from db_import import bigquery_emulator


@unittest.skipUnless(bigquery_emulator.is_bigquery_emulator_available(
), "The BigQuery emulator is not available on this machine. Check out https://github.com/goccy/bigquery-emulator for more details."
                    )
class TestDbQueries(unittest.TestCase):

  def setUp(self):
    self.project_name: str = "test"
    self.dataset_name: str = "dataset"

    with contextlib.ExitStack() as stack:
      self.client = stack.enter_context(
          bigquery_emulator.emulate_bigquery(self.project_name,
                                             self.dataset_name))
      self.addCleanup(stack.pop_all().close)

  def test_query_returns_non_empty_result(self):
    table_name = "table"
    table = self.client.create_table(
        bigquery.Table(f"{self.project_name}.{self.dataset_name}.{table_name}",
                       [bigquery.SchemaField("benchmark_id", "STRING")]))

    sql = f"SELECT benchmark_id FROM {self.dataset_name}.{table_name} LIMIT 1"
    sql_with_param = f"SELECT benchmark_id FROM {self.dataset_name}.{table_name} WHERE benchmark_id = @benchmark_id LIMIT 1"

    self.assertFalse(
        db.query_returns_non_empty_result(self.client, sql, parameters={}))
    self.assertFalse(
        db.query_returns_non_empty_result(
            self.client,
            sql_with_param,
            parameters={"benchmark_id": "hello_world"}))

    self.client.insert_rows(table, [{"benchmark_id": "hello_world"}])
    self.assertTrue(
        db.query_returns_non_empty_result(self.client, sql, parameters={}))
    self.assertTrue(
        db.query_returns_non_empty_result(
            self.client,
            sql_with_param,
            parameters={"benchmark_id": "hello_world"}))
    self.assertFalse(
        db.query_returns_non_empty_result(
            self.client,
            sql_with_param,
            parameters={"benchmark_id": "hello_other_world"}))
