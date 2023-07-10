## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import pathlib
import yaml

from typing import TextIO

PIPELINES_KEY: str = "pipelines"


def _embed(loader: yaml.Loader, node):
  """This tag handler allows loading the contents of file as string into a node.

    The initial node's content is interpreted as a file path.
    Relative file path are expected to be relative to the YAML file's location.
    Absolute paths are also allowed but must refer to a location below the YAML
    file's parent directory.

    Example:
    ```yaml
    ---
    property: !embed filename.txt
    """
  yaml_filepath = pathlib.Path(loader.name).resolve()
  allowed_directory = yaml_filepath.parent
  filepath = (allowed_directory /
              pathlib.Path(loader.construct_scalar(node))).resolve()

  # We use os.path here to check whether filepath is below allowed_directory,
  # because the equivalent function in pathlib is not available before Python 3.9.
  assert os.path.commonpath([allowed_directory]) == os.path.commonpath(
      [allowed_directory, filepath])
  assert filepath.is_file()
  return filepath.read_text()


yaml.SafeLoader.add_constructor("!embed", _embed)


def load_config(file: TextIO):
  return yaml.safe_load(file)


def configure_parser(parser: argparse.ArgumentParser):
  subcommands = parser.add_subparsers(required=True)
  dump_command = subcommands.add_parser(
      "dump",
      help=
      "Dump the entire config with all YAML references and embeddings expanded."
  )
  dump_command.set_defaults(command_handler=_dump)

  list_pipelines_command = subcommands.add_parser(
      "list_pipelines", help="Get a machine readable list of all pipelines")
  list_pipelines_command.set_defaults(command_handler=_list_pipelines)


class _Dumper(yaml.SafeDumper):

  def ignore_aliases(self, data):
    return True


def _use_block_mode_for_multiline_strings(dumper: _Dumper, data):
  if '\n' in data:
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
  return dumper.represent_scalar('tag:yaml.org,2002:str', data)


_Dumper.add_representer(str, _use_block_mode_for_multiline_strings)


def _dump(config_file, args: argparse.Namespace):
  print(yaml.dump(config_file, Dumper=_Dumper))


def _list_pipelines(config_file, args: argparse.Namespace):
  for name in config_file['pipelines'].keys():
    print(name)
