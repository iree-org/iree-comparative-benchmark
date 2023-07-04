## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains the cloud function entry point

import functions_framework
import os
import pathlib
from typing import Dict
import yaml

from google.cloud import bigquery, storage
from cloudevents.http import event

from db_import import db
from db_import import process

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

with open(SCRIPT_DIR / "config.yml") as fd:
  config_file = yaml.safe_load(fd)

config = config_file["cloud_functions"][os.environ["config_name"]]

db_client = bigquery.Client()
table = db_client.get_table(config["table_name"])

storage_client = storage.Client()
bucket = storage_client.bucket(config["bucket_name"])


def presence_check(rule: Dict, parameters: Dict) -> bool:
  return db.query_returns_non_empty_result(
      db_client, rule["sql_condition"].format(dataset=table.dataset_id,
                                              table=table.table_id), parameters)


@functions_framework.cloud_event
def entry_point(event: event.CloudEvent):
  assert event["type"] == "google.cloud.storage.object.v1.finalized"
  assert event.data["name"]

  file = bucket.get_blob(event.data["name"])
  if not file:
    raise RuntimeError(
        f"File {event.data['name']} does not exist in bucket {config['bucket_name']}."
    )

  rows = process.process_single_file(config["rules"], file, config,
                                     config_file["snippets"], presence_check)

  if len(rows) > 0:
    db_client.insert_rows(table, rows)
