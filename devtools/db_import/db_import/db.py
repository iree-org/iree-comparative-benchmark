## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines protocols for a subset of types from `google.cloud.bigquery`.
   Check out in_memory_database for an alternative implementation of these protocols.
"""

from typing import Any, Protocol, Mapping, Optional, Iterable, Union, runtime_checkable
from google.cloud import bigquery


@runtime_checkable
class Table(Protocol):

  @property
  def dataset_id(self) -> str:
    ...

  @property
  def table_id(self) -> str:
    ...


@runtime_checkable
class QueryJob(Protocol):

  def result(self):
    ...

  def __iter__(self):
    ...


@runtime_checkable
class Client(Protocol):

  def query(self,
            query: str,
            job_config: Optional[bigquery.QueryJobConfig] = None) -> QueryJob:
    ...

  def insert_rows(self, table: Union[str, Table], rows: Iterable[Mapping[str,
                                                                         Any]]):
    ...

  def get_table(self, table: Union[str, Table]) -> Table:
    ...


def query_returns_non_empty_result(
    client: Client,
    sql: str,
    parameters: dict[str, str],
) -> bool:
  job_config = bigquery.QueryJobConfig(query_parameters=[
      bigquery.ScalarQueryParameter(key, "STRING", value)
      for key, value in parameters.items()
  ])

  rows = list(
      client.query(
          f"SELECT COUNT(*) as count FROM ({sql})",
          job_config=job_config,
      ).result())
  assert len(rows) == 1
  return rows[0].get("count") > 0


def delete_all_preexisting_data(client: Client, config: dict[str, Any]):
  table = client.get_table(config["table_name"])

  job_config = bigquery.QueryJobConfig(query_parameters=[
      bigquery.ScalarQueryParameter("bucket_name", "STRING",
                                    config["bucket_name"])
  ])

  sql = config["sql_delete"].format(table=table.table_id,
                                    dataset=table.dataset_id)
  client.query(
      sql,
      job_config=job_config,
  ).result()
