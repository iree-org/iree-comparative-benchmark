## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import csv, datetime, io, json, re, os
from typing import Any, Optional, Callable
import rjsonnet
from google.cloud import storage
from db_import.utils import first_no_except


class BenchmarkRunAlreadyPresentError(Exception):
  pass


# Applies the given `rule`` to `file`.
# - Returns False if `rule[filepath_regex]` does not match the filepath of `file`.
# - If it matches, it evaluates the JSONNet expression `rule[result]` and returns its result (parsed as JSON).
def apply_rule_to_file(
    rule: dict,
    file: storage.Blob,
    config: dict,
    snippets: dict[str, str],
    presence_check: Optional[Callable[[Any, Any], bool]],
    dump_files_to: Optional[str],
):
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
    try:
      return int(str)
    except ValueError:
      return float(str)

  def read_file(filepath):
    blob = file.bucket.blob(filepath)
    with blob.open() as fd:
      contents = fd.read()

    if dump_files_to:
      dump_filepath = os.path.join(dump_files_to, config["bucket_name"],
                                   filepath)
      os.makedirs(os.path.dirname(dump_filepath), exist_ok=True)
      with open(dump_filepath, "w") as fd:
        fd.write(contents)

    return contents

  def try_reading_files_until_success(filepaths):
    return first_no_except(read_file, filepaths)

  native_callbacks = {
      "timestampToIso8601": (("timestamp",), timestamp_to_iso8601),
      "parseCsv": (("contents",), parse_csv),
      "parseNumber": (("parseNumber",), parse_number),
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

  return json.loads(
      rjsonnet.evaluate_snippet(
          file.name,
          rule["result"],
          ext_vars=additional_config_params |
          {"filepath_captures": json.dumps(match.groupdict())},
          import_callback=import_callback,
          native_callbacks=native_callbacks,
      ))
