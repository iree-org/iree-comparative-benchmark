## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os, sys, argparse
from google.cloud import storage, bigquery

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from db_import.batch_import import import_entire_bucket


def configure_parser(parser: argparse.ArgumentParser):
  parser.add_argument("config_name")
  parser.add_argument(
      "-c",
      "--config",
      help="Read config from the given file",
      default=os.path.join(SCRIPT_DIR, "config.yml"),
  )
  parser.add_argument(
      "-d",
      "--destination",
      help="Directory path where to store the files",
      default=os.path.join(SCRIPT_DIR, "test_data"),
  )
  parser.add_argument(
      "-p",
      "--prefix",
      help="Filter triggering files in bucket by this regular expression",
  )


def download(config_file, args: argparse.Namespace):
  try:
    config = config_file["cloud_functions"][args.config_name]
  except KeyError:
    sys.exit(f"No configuration with the name {args.config_name} found.")

  storage_client = storage.Client()
  db_client = bigquery.Client()

  import_entire_bucket(db_client,
                       storage_client,
                       config,
                       config_file.get("snippets", {}),
                       check_for_presence=False,
                       dry_run=True,
                       prefix_filter=args.prefix if args.prefix else None,
                       dump_files_to=args.destination)
