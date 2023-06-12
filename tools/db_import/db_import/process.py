## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os, sys, argparse
from typing import Optional, Any
from google.cloud import bigquery, storage
from db_import.mock_storage import MockClient
from db_import.mock_database import MockClient as MockDbClient
from db_import.rules import apply_rule_to_file
from db_import.db import is_benchmark_run_already_present_in_db

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def configure_parser(parser: argparse.ArgumentParser):
  parser.add_argument(
      "-c",
      "--config",
      help="Use given config",
      required=True,
  )
  parser.add_argument(
      "-f",
      "--no-check",
      help="Do not check if a clashing record already exists in the DB",
      action="store_true",
  )
  parser.add_argument(
      "-d",
      "--dry-run",
      help="Do not import into the DB. Print the rows instead",
      action="store_true",
  )
  parser.add_argument(
      "-s",
      "--source",
      help=
      "Read files from given directory (<dir>/<bucket_name>/<filepath>) instead of from the bucket",
  )
  parser.add_argument(
      "-t",
      "--trigger",
      help="The trigger filepath",
      required=True,
  )


def process(config_file, args: argparse.Namespace):
  try:
    config = config_file["cloud_functions"][args.config]
  except KeyError:
    sys.exit(f"No configuration with the name {args.config_name} found.")

  if args.source:
    storage_client = MockClient(os.path.join(SCRIPT_DIR, "test_data"))
  else:
    storage_client = storage.Client()

  bucket = storage_client.bucket(config["bucket_name"])
  file = bucket.blob(args.trigger)

  if args.dry_run:
    db_client = MockDbClient()
  else:
    db_client = bigquery.Client()

  db_table = db_client.get_table(config["table_name"])

  result = process_single_file(
      db_client,
      db_table,
      config["rules"],
      file,
      config,
      config_file["snippets"],
      False,
  )

  print("Done." if result else "No rule applies.")


class NoRuleAppliesError(Exception):
  pass


def process_single_file(
    client: bigquery.Client,
    table: bigquery.Table,
    rules: list[dict],
    file: storage.Blob,
    config: dict,
    snippets: dict[str, str],
    check_for_presence: bool,
    dry_run: bool = False,
    dump_files_to: Optional[str] = None,
) -> list[Any]:

  def presence_check(rule, parameters):
    return is_benchmark_run_already_present_in_db(client, table,
                                                  rule["sql_condition"],
                                                  parameters)

  for rule in rules:
    rows = apply_rule_to_file(
        rule,
        file,
        config,
        snippets,
        presence_check if check_for_presence else None,
        dump_files_to,
    )
    if rows != False:
      return rows if type(rows) is list else [rows]

  raise NoRuleAppliesError()
