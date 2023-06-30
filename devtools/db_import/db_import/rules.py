## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import csv
import datetime
import io
import json
import pathlib
import re
import rjsonnet

from google.cloud import storage
from typing import Any, Callable, Dict, Optional

from db_import import utils


class BenchmarkRunAlreadyPresentError(Exception):
  pass


def apply_rule_to_file(
    rule: Dict,
    file: storage.Blob,
    config: Dict,
    snippets: Dict[str, str],
    presence_check: Optional[Callable[[Any, Any], bool]],
    dump_files_to: Optional[pathlib.Path],
):
  """Applies the given `rule` to `file`.

  - Returns False if `rule[filepath_regex]` does not match the filepath of `file`.
  - If it matches, it evaluates the JSONNet expression `rule[result]` and returns its result (parsed as JSON).
  """
  regex = re.compile(rule["filepath_regex"])
  match = regex.match(file.name)

  if not match:
    return False

  parameters = match.groupdict() | {
      "bucket_name": config["bucket_name"],
      "cloud_function_name": config["cloud_function_name"],
  }

  if presence_check and presence_check(rule, parameters):
    raise BenchmarkRunAlreadyPresentError()

  def import_callback(dir, rel):
    return rel, snippets[rel]

  def timestamp_to_iso8601(timestamp):
    return datetime.datetime.fromtimestamp(int(timestamp),
                                           datetime.timezone.utc).isoformat()

  def parse_csv(contents):
    dialect = csv.Sniffer().sniff(contents)
    csvfile = io.StringIO(contents)
    reader = csv.DictReader(csvfile, dialect=dialect)
    return list(reader)

  def parse_number(str):
    return float(str)

  def read_file(filepath):
    blob = file.bucket.blob(filepath)
    with blob.open() as fd:
      contents = fd.read()

    if dump_files_to:
      full_filepath = dump_files_to / config["bucket_name"] / filepath
      full_filepath.parent.mkdir(parents=True, exist_ok=True)
      full_filepath.write_text(contents)

    return contents

  def try_reading_files_until_success(filepaths):
    return utils.first_no_except(read_file, filepaths)

  native_callbacks = {
      "timestampToIso8601": (("timestamp",), timestamp_to_iso8601),
      "parseCsv": (("contents",), parse_csv),
      "parseNumber": (("number",), parse_number),
      "readFile": (("filepath",), read_file),
      "tryReadingFilesUntilSuccess": (
          ("filepaths",),
          try_reading_files_until_success,
      ),
  }

  additional_config_params = {
      "config.bucket_name": config["bucket_name"],
      "config.cloud_function_name": config["cloud_function_name"],
  }
  additional_config_params["filepath_captures"] = json.dumps(match.groupdict())

  return json.loads(
      rjsonnet.evaluate_snippet(
          file.name,
          rule["result"],
          ext_vars=additional_config_params,
          import_callback=import_callback,
          native_callbacks=native_callbacks,
      ))
