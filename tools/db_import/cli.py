#!/usr/bin/env python
## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os, yaml, sys, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(prog="cli",
                                 description="Manages the cloud functions")
parser.add_argument(
    "-F",
    "--config-file",
    help=
    "Read configuration from the given file. The default is <SCRIPT_DIR>/config.yml",
    default=os.path.join(SCRIPT_DIR, "config.yml"),
)
subparsers = parser.add_subparsers(help='Choose a subcommand',
                                   required=True,
                                   dest='command')

from db_import.deploy import configure_parser as configure_deploy_parser, deploy

deploy_parser = subparsers.add_parser('deploy',
                                      help='Deploy one or more cloud functions')
configure_deploy_parser(deploy_parser)

from db_import.download import configure_parser as configure_download_parser, download

download_parser = subparsers.add_parser(
    'download', help='Download files from bucket for local usage')
configure_download_parser(download_parser)

from db_import.batch_import import configure_parser as configure_batch_import_parser, batch_import

batch_import_parser = subparsers.add_parser(
    'batch_import', help='Batch import an entire bucket')
configure_batch_import_parser(batch_import_parser)

from db_import.process import configure_parser as configure_process_parser, process

process_parser = subparsers.add_parser(
    'process', help='Process a single file from a bucket')
configure_process_parser(process_parser)

args = parser.parse_args()

try:
  with open(args.config) as fd:
    config_file = yaml.safe_load(fd)
except:
  sys.exit("Failed to open config file with path: {}".format(args.config))

if args.command == 'deploy':
  deploy(config_file, args)
elif args.command == 'download':
  download(config_file, args)
elif args.command == 'batch_import':
  batch_import(config_file, args)
elif args.command == 'process':
  process(config_file, args)
