## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys, argparse
from google.cloud import bigquery, storage
from typing import Optional, Any
from db_import.process import process_single_file, NoRuleAppliesError
from db_import.db import import_result_set
from db_import.rules import BenchmarkRunAlreadyPresentError


def configure_parser(parser: argparse.ArgumentParser):
  parser.add_argument("config_name")
  parser.add_argument(
      "-s",
      "--check",
      help="Checks if each record already exists in the DB",
      action="store_true",
  )


def batch_import(config_file, args: argparse.Namespace):
  try:
    config = config_file["cloud_functions"][args.config_name]
  except KeyError:
    sys.exit(f"No configuration with the name {args.config_name} found.")

  db_client = bigquery.Client()
  storage_client = storage.Client()

  import_entire_bucket(
      db_client,
      storage_client,
      config,
      config_file.get("snippets", {}),
      args.check,
      False,
  )


# This function tries to import all files from a GCS bucket.
# It lists all the files from the given bucket and behaves as
# if each file was triggered seperately. The order is not enforced.
def import_entire_bucket(
    db_client: bigquery.Client,
    storage_client: storage.Client,
    config: dict[str, Any],
    snippets: dict[str, str],
    check_for_presence: bool,
    dry_run: bool,
    prefix_filter: Optional[str] = None,
    dump_files_to: Optional[str] = None,
):
  bucket = storage_client.get_bucket(config["bucket_name"])
  table = db_client.get_table(config["table_name"])

  def process_single_file_with_status_output(file):
    try:
      print(f"Processing {file.name}... ", end="")
      rows = process_single_file(
          db_client,
          table,
          config["rules"],
          file,
          config,
          snippets,
          check_for_presence,
          dry_run,
          dump_files_to,
      )
      import_result_set(db_client, table, rows)
      print("Done.")
    except NoRuleAppliesError:
      # This happens so often, so that we don't wanna clutter the output
      print("No rule applies.\r" + " " * 250, end="\r")
    except BenchmarkRunAlreadyPresentError:
      print("Data already present.")
    except Exception:
      print("Failure.")

  for file in bucket.list_blobs(
      prefix=prefix_filter if prefix_filter else None):
    process_single_file_with_status_output(file)
