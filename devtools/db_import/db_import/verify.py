## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import re
import sys

from google.cloud import storage
from typing import Dict

from db_import import db
from db_import import process
from db_import import bigquery_emulator


def configure_parser(parser: argparse.ArgumentParser):
  parser.set_defaults(command_handler=_verify)
  parser.add_argument("config_name")
  parser.add_argument("--benchmark_id_re")
  parser.add_argument("--overwrite_triggers", nargs='*')


def _verify(config_file: Dict, args: argparse.Namespace):
  """The verify subcommand allows testing of a pipeline without actually
     deploying a cloud function or re-uploading a file to the GCS bucket.

     There are 2 main uses case:
     1. Test a change to the pipeline configuration
     2. Test a change to the input file format

     Each pipeline config allows the definition of a series of tests
     (under the 'tests' node in YAML). Each test

     - must have an `id`,
     - can have a more descriptive `name`,
     - can have a series of setup SQL statements under `setup`,
     - must have one or multiple SQL-based asserts under `checks`.

     In addition the pipeline needs to define how the destination
     database table should be created under `sql_create_table`.

     The verify command will then simulate the operation of the cloud function
     and will verify the integrity of the imported data by making SQL queries
     to BigQuery (using the bigquery-emulator).

     Example:

     ```yaml
     pipelines:
       example_pipeline:
         bucket_name: my_bucket
         cloud_function_name: my_cloud_function
         table_name: my_dataset.my_table
         rules:
         - # Missing in this example
         tests:
         - id: my_test
           name: My Test
           setup:
           # `setup` can contain arbitrary SQL commands. You can use that to create arbitrary views or user defined functions, for example.
           - CREATE VIEW my_view AS (SELECT * FROM `{dataset}.{table}`)
           triggers:
           # Note that we will process all triggers in the given order before we evaluate the `checks` below.
           # Hence the `checks` will always "see" the data from all triggers.
           - path/to/file/in/bucket.json
           - path/to/other/file/in/bucket.json
           checks:
           # The checks are arbitrary SQL commands, but it's good practice to make them non-mutating (SELECT queries only!)
           # Checks are executed in given order and the actual result is NOT processed in any way. An empty result is not an error.
           # You can use the `ERROR` SQL expression to make your query fail in all the cases you wanted - as demonstrated below:
           - SELECT CASE WHEN COUNT(*) = 42 THEN true ELSE ERROR(FORMAT('Expected 42 rows in view, but found %t'), COUNT(*))) END FROM {dataset}.my_view
           - SELECT ERROR('Field `my_field` was NULL, but not allowed') FROM {dataset}.my_view WHERE my_field IS NULL
     ```

     The `triggers` can be overridden with the `--overwrite_triggers` command line option.
     This is useful for testing whether a certain new JSON file succeeds in all the tests.

     The `--benchmark_id_re` command line option also allows to filter by benchmark ID and only
     execute a subset of tests. This is particular useful in conjunction with `--overwrite_triggers`
     in a CI setup where we want to check the imported data for certain invariants (things like fields
     must not be NULL, etc.), but not check for an exact number of rows or even specific field values.
     """
  try:
    config = config_file["pipelines"][args.config_name]
  except KeyError:
    sys.exit(f"No configuration with the name {args.config_name} found.")

  if "tests" not in config:
    raise ValueError(f"No tests configured for {args.config_name}")

  storage_client = storage.Client()
  bucket = storage_client.get_bucket(config["bucket_name"])

  for test in config["tests"]:
    if args.benchmark_id_re and not re.search(args.benchmark_id_re, test['id']):
      continue

    print(f"Preparing test '{test.get('name', test['id'])}'...")
    dataset_name, table_name = config["table_name"].split(".")
    with bigquery_emulator.emulate_bigquery("project",
                                            dataset_name) as db_client:
      db_client.query(config["sql_create_table"].format(
          dataset=dataset_name, table=table_name)).result()
      db_table = db_client.get_table(config["table_name"])

      sql_data_present = config["sql_data_present"].format(
          dataset=db_table.dataset_id, table=db_table.table_id)
      sql_data_present_params = {"bucket_name": config["bucket_name"]}

      print("  Checking if `sql_data_present` returns an empty result...")
      try:
        if db.query_returns_non_empty_result(db_client, sql_data_present,
                                             sql_data_present_params):
          sys.exit(
              f"The `sql_data_present` SQL query '{config['sql_data_present']}' returned a non-empty result even though the destination table was just created and should not contain any data."
          )
      except Exception as e:
        sys.exit(
            f"The SQL query '{sql_data_present}' failed with the following error:\n{e}\n"
        )

      for setup in test.get("setup", []):
        db_client.query(
            setup.format(dataset=db_table.dataset_id,
                         table=db_table.table_id)).result()

      triggers = args.overwrite_triggers if args.overwrite_triggers else test[
          "triggers"]
      for trigger_path in triggers:
        print(f"  Importing '{trigger_path}'...")
        file = bucket.blob(trigger_path)
        rows = process.process_single_file(
            config["rules"],
            file,
            config,
            config_file.get("snippets", {}),
        )
        if len(rows) > 0:
          db_client.insert_rows(db_table, rows)

      try:
        if not test.get('expect_no_import', False):
          print(
              "  Checking if `sql_data_present` returns a non-empty result...")
          if not db.query_returns_non_empty_result(db_client, sql_data_present,
                                                   sql_data_present_params):
            sys.exit(
                f"The `sql_data_present` SQL query '{sql_data_present}' returned an empty result even though we just imported some data."
            )
        else:
          print(
              "  Checking if `sql_data_present` still returns an empty result after attempted import..."
          )
          if db.query_returns_non_empty_result(db_client, sql_data_present,
                                               sql_data_present_params):
            sys.exit(
                f"The `sql_data_present` SQL query '{sql_data_present}' returned a non-empty result even though we expected no import to happen."
            )
      except Exception as e:
        sys.exit(
            f"The SQL query '{sql_data_present}' failed with the following error:\n{e}\n"
        )

      print(f"  Running checks...")
      for check in test["checks"]:
        sql = check.format(dataset=db_table.dataset_id, table=db_table.table_id)
        try:
          db_client.query(sql).result()
        except Exception as e:
          sys.exit(
              f"The SQL query '{sql}' failed with the following error:\n{e}\n")

      print("  Checking whether `sql_delete` query works...")
      db.delete_all_preexisting_data(db_client, config)
      print("  Checking if `sql_data_present` now returns an empty result...")
      if db.query_returns_non_empty_result(db_client, sql_data_present,
                                           sql_data_present_params):
        sys.exit(
            f"The `sql_data_present` SQL query '{sql_data_present}' returned a non-empty result even though we just deleted all the data. This is either an issue with the `sql_data_present` query or with the `sql_delete` query."
        )

    print("Done.\n\n")
