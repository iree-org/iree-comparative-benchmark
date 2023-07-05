## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import sys
import pathlib

from google.cloud import bigquery
from google.cloud import storage
from typing import Optional, Any, Dict

from db_import import process
from db_import import rules
from db_import import db


def configure_parser(parser: argparse.ArgumentParser):
  parser.set_defaults(command_handler=_batch_import)
  parser.add_argument("config_name")
  parser.add_argument(
      "-s",
      "--check",
      help=
      "Checks for each record if it already exists in the DB before importing (Warning: Slow!)",
      action="store_true",
  )


def _batch_import(config_file, args: argparse.Namespace):
  try:
    config = config_file["pipelines"][args.config_name]
  except KeyError:
    sys.exit(f"No configuration with the name {args.config_name} found.")

  db_client = bigquery.Client()
  storage_client = storage.Client()

  import_entire_bucket(
      db_client,
      storage_client,
      config,
      config_file.get("snippets", {}),
      check_for_presence=args.check,
  )


# This function tries to import all files from a GCS bucket.
# It lists all the files from the given bucket and behaves as
# if each file was triggered seperately. The order is not enforced.
def import_entire_bucket(
    db_client: db.Client,
    storage_client: storage.Client,
    config: Dict[str, Any],
    snippets: Dict[str, str],
    check_for_presence: bool,
    prefix_filter: Optional[str] = None,
    dump_files_to: Optional[pathlib.Path] = None,
):
  bucket = storage_client.get_bucket(config["bucket_name"])
  table = db_client.get_table(config["table_name"])

  def presence_check(rule: Dict, parameters: Dict) -> bool:
    return db.query_returns_non_empty_result(
        db_client, rule["sql_condition"].format(table=table.table_id,
                                                dataset=table.dataset_id),
        parameters)

  def process_single_file_with_status_output(file):
    try:
      print(f"Processing {file.name}... ", end="")

      rows = process.process_single_file(
          config["rules"],
          file,
          config,
          snippets,
          presence_check if check_for_presence else None,
          dump_files_to,
      )
      if len(rows) > 0:
        db_client.insert_rows(table, rows)
      print("Done.")
    except process.NoRuleAppliesError:
      # This happens so often, that we don't wanna clutter the output, so
      # we rather do a carriage return and re-use the same line.
      print("No rule applies.\r" + " " * 250, end="\r")
    except rules.BenchmarkRunAlreadyPresentError:
      print("Data already present.")
    except Exception as e:
      print("Failure:")
      print(e)

  for file in bucket.list_blobs(
      prefix=prefix_filter if prefix_filter else None):
    process_single_file_with_status_output(file)
