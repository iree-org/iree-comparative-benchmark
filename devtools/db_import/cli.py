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

from db_import.batch_import import configure_parser as configure_batch_import_parser
from db_import.process import configure_parser as configure_process_parser

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

batch_import_parser = subparsers.add_parser(
    "batch_import", help="Batch import an entire bucket")
configure_batch_import_parser(batch_import_parser)

process_parser = subparsers.add_parser(
    "process", help="Process a single file from a bucket")
configure_process_parser(process_parser)

args = parser.parse_args()

with open(args.config_file) as fd:
  config_file = yaml.safe_load(fd)

args.command_handler(config_file, args)
