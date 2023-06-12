## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains the cloud function entry point

import os, yaml
import functions_framework
from google.cloud import bigquery, storage
from cloudevents.http.event import CloudEvent
from db_import.process import process_single_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "config.yml")) as fd:
  config_file = yaml.safe_load(fd)

config = config_file['cloud_functions'][os.environ['config_name']]

db_client = bigquery.Client()
table = db_client.get_table(config["table_name"])

storage_client = storage.Client()
bucket = storage_client.bucket(config["bucket_name"])


@functions_framework.cloud_event
def entry_point(event: CloudEvent):
  assert event["type"] == "google.cloud.storage.object.v1.finalized"
  assert event.data["name"]

  file = bucket.get_blob(event.data["name"])
  if not file:
    raise RuntimeError("File {} does not exist in bucket {}.".format(
        event.data["name"], config["bucket_name"]))
  process_single_file(
      db_client,
      table,
      config["rules"],
      file,
      config,
      config_file['snippets'],
      check_for_presence=True,
  )
