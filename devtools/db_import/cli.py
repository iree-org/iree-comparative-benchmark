#!/usr/bin/env python
## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import pathlib
import yaml

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

from db_import import download
from db_import import batch_import
from db_import import process

parser = argparse.ArgumentParser(prog="cli",
                                 description="Manages the cloud functions")
parser.add_argument(
    "-F",
    "--config-file",
    help=
    "Read configuration from the given file. The default is <SCRIPT_DIR>/config.yml",
    type=pathlib.Path,
    default=SCRIPT_DIR / "config.yml",
)
subparsers = parser.add_subparsers(required=True)

download_parser = subparsers.add_parser(
    "download", help="Download files from bucket for local usage")
download.configure_parser(download_parser)

batch_import_parser = subparsers.add_parser(
    "batch_import", help="Batch import an entire bucket")
batch_import.configure_parser(batch_import_parser)

process_parser = subparsers.add_parser(
    "process", help="Process a single file from a bucket")
process.configure_parser(process_parser)

args = parser.parse_args()

with open(args.config_file) as fd:
  config_file = yaml.safe_load(fd)

args.command_handler(config_file, args)
