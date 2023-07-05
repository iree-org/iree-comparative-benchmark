## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import pathlib
import sys

from google.cloud import storage

from db_import import in_memory_database
from db_import import batch_import


def configure_parser(parser: argparse.ArgumentParser):
  parser.set_defaults(command_handler=_download)
  parser.add_argument("config_name")
  parser.add_argument(
      "-d",
      "--destination",
      help="Directory path where to store the files (default=./testdata)",
      type=pathlib.Path,
      default=pathlib.Path('.').resolve() / "testdata",
  )
  parser.add_argument(
      "-p",
      "--prefix",
      help="Filter triggering files in bucket by this regular expression",
  )


def _download(config_file, args: argparse.Namespace):
  try:
    config = config_file["pipelines"][args.config_name]
  except KeyError:
    sys.exit(f"No configuration with the name {args.config_name} found.")

  storage_client = storage.Client()
  db_client = in_memory_database.Client()

  # We are simulating a batch import here, by streaming the results into
  # the in memory database implementation. In addition we use the
  # `dump_files_to` which writes each downloaded GCS blob also into a file
  # on the filesystem. This way we end up with exactly the files we need
  # to repeat an import locally.
  batch_import.import_entire_bucket(
      db_client,
      storage_client,
      config,
      config_file.get("snippets", {}),
      check_for_presence=False,
      prefix_filter=args.prefix if args.prefix else None,
      dump_files_to=args.destination)
