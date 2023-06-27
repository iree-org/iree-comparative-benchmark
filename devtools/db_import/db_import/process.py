## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import os
import sys

from google.cloud import bigquery, storage as CloudStorage
from typing import Optional, Any, Callable, Dict

from local_storage import Client as LocalStorageClient
from in_memory_database import Client as InMemoryDbClient
from rules import apply_rule_to_file
import storage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def configure_parser(parser: argparse.ArgumentParser):
  parser.set_defaults(command_handler=_process)
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


def _process(config_file, args: argparse.Namespace):
  try:
    config = config_file["cloud_functions"][args.config]
  except KeyError:
    sys.exit(f"No configuration with the name {args.config_name} found.")

  if args.source:
    storage_client = LocalStorageClient(args.source)
  else:
    storage_client = CloudStorage.Client()

  bucket = storage_client.get_bucket(config["bucket_name"])
  file = bucket.blob(args.trigger)

  if args.dry_run:
    db_client = InMemoryDbClient()
  else:
    db_client = bigquery.Client()

  db_table = db_client.get_table(config["table_name"])

  result = process_single_file(db_client, db_table, config["rules"], file,
                               config, config_file["snippets"])

  if args.dry_run:
    print(json.dumps(result))

  print("Done." if result else "No rule applies.", file=sys.stderr)


class NoRuleAppliesError(Exception):
  pass


def process_single_file(
    rules: list[dict],
    file: storage.Blob,
    config: dict,
    snippets: dict[str, str],
    presence_check: Optional[Callable[[Dict, Dict], bool]] = None,
    dump_files_to: Optional[str] = None,
) -> list[Any]:

  for rule in rules:
    rows = apply_rule_to_file(
        rule,
        file,
        config,
        snippets,
        presence_check,
        dump_files_to,
    )
    if rows != False:
      return rows if type(rows) is list else [rows]

  raise NoRuleAppliesError()
