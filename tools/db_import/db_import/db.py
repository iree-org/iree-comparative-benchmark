## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
from google.cloud import bigquery


def is_benchmark_run_already_present_in_db(
    client: bigquery.Client,
    table: bigquery.TableReference | bigquery.Table,
    sql: str,
    parameters: dict[str, str],
) -> bool:
  job_config = bigquery.QueryJobConfig(query_parameters=[
      bigquery.ScalarQueryParameter(key, "STRING", value)
      for key, value in parameters.items()
  ])

  rows = list(
      client.query(
          sql.format(table=table.table_id, dataset=table.dataset_id),
          job_config=job_config,
      ))
  return len(rows) == 1


def import_result_set(client: bigquery.Client, table: bigquery.Table,
                      rows: list[dict]):
  if len(rows) > 0:
    client.insert_rows(table, rows)


def is_there_already_some_data_in_table(client: bigquery.Client,
                                        config) -> bool:
  parameters = {"bucket_name": config["bucket_name"]}
  table = client.get_table(config["table_name"])

  return is_benchmark_run_already_present_in_db(client, table,
                                                config["sql_data_present"],
                                                parameters)


def delete_all_preexisting_data(client: bigquery.Client, config: dict[str,
                                                                      Any]):
  parameters = {"bucket_name": config["bucket_name"]}
  table = client.get_table(config["table_name"])

  job_config = bigquery.QueryJobConfig(query_parameters=[
      bigquery.ScalarQueryParameter(key, "STRING", value)
      for key, value in parameters.items()
  ])

  sql = config["sql_delete"].format(table=table.table_id,
                                    dataset=table.dataset_id)
  client.query(
      sql,
      job_config=job_config,
  ).result()
