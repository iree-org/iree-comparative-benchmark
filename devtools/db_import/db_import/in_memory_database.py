## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This API mimics the Google BigQuery API and implements the protocols defined in `db_import.db`.
# It only supports adding rows with the `insert_rows` stream commands.
# The data will be stored in memory and querying is not supported.

from typing import Dict, Sequence, Optional, Union
from google.cloud import bigquery


class Table:

  def __init__(self, name: str):
    self.name: str = name
    self.rows = []

  @property
  def table_id(self):
    return self.name.split('.')[1]

  @property
  def dataset_id(self):
    return self.name.split('.')[0]

  def get_rows(self) -> Sequence:
    return self.rows


class QueryJob:

  def result(self):
    return []

  def __iter__(self):
    return iter([])


class Client:

  def __init__(self):
    self.tables: Dict[str, Table] = {}

  def get_table(self, name: str) -> Table:
    if name not in self.tables:
      self.tables[name] = Table(name)
    return self.tables[name]

  def insert_rows(self, table: Union[str, Table], rows: Sequence):
    table_obj = table if isinstance(table, Table) else self.get_table(table)
    table_obj.rows.extend(rows)

  def query(self,
            sql: str,
            job_config: Optional[bigquery.QueryJobConfig] = None) -> QueryJob:
    raise RuntimeError("Client.query is not implemented")
