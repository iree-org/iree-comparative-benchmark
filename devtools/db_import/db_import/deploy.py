## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import __main__
import argparse
import pathlib
import plumbum
import sys

from google.cloud import storage, bigquery

from db_import import batch_import
from db_import import db

MAIN_MODULE_DIR = pathlib.Path(__main__.__file__).resolve().parent


def configure_parser(parser: argparse.ArgumentParser):
  parser.set_defaults(command_handler=_deploy)
  parser.add_argument("config_names", nargs="+")
  parser.add_argument(
      "-r",
      "--region",
      help="The region where the cloud function(s) should be deployed to.",
      default="us-central1",
  )
  parser.add_argument(
      "--force-data-import",
      help=
      "Always do an initial data import even if the destination table already exists. By default data is only imported if there is no data yet from the same bucket in the table.",
      action="store_true",
      default=False,
  )
  parser.add_argument(
      "--force-data-deletion",
      help="Force deletion of all pre-existing data from the destination table",
      action="store_true",
      default=False,
  )


def _deploy(config_file, args: argparse.Namespace):
  gcloud = plumbum.local["gcloud"]
  bq = plumbum.local["bq"]

  project_id = gcloud("config", "get-value", "project").strip()

  # This is a special SA that all GCS events originate from.
  gcs_service_account = gcloud("storage", "service-agent").strip()

  # This grants the cloud storage service account the permission
  # to send pubsub events
  gcloud("projects", "add-iam-policy-binding", project_id, "--member",
         f"serviceAccount:{gcs_service_account}", "--role",
         "roles/pubsub.publisher")

  # This is the SA we run the cloud function as.
  my_service_account = config_file["service_account"]
  if "@" not in my_service_account:
    my_service_account += f"@{project_id}.iam.gserviceaccount.com"

  for config_name in args.config_names:
    print(f"Processing config {config_name}...")
    config = config_file["pipelines"][config_name]
    bucket_name = config["bucket_name"]
    cloud_function_name = config["cloud_function_name"]
    table_name = config["table_name"]
    table_exists = bool(bq["show", table_name] & plumbum.TF)
    db_client = bigquery.Client()

    if not table_exists:
      print(f"Table {table_name} does not exist. Creating it...")
      try:
        schema = config_file["table_schemas"][table_name]
      except KeyError:
        sys.exit(
            f"Table schema for table {table_name} not found in config file.")

      bq("mk", "-t", table_name, schema)
      bq(
          "add-iam-policy-binding",
          f"--member=serviceAccount:{my_service_account}",
          "--role=roles/bigquery.dataEditor",
          f"{project_id}:{table_name}",
      )
    else:
      if args.force_data_deletion:
        print(
            f"Deleting pre-existing data from destination table {table_name} as requested."
        )
        db.delete_all_preexisting_data(db_client, config)
      else:
        print(
            f"The destination table {table_name} already exists. Checking if data is already present."
        )

    table = db_client.get_table(table_name)
    data_exists = db.query_returns_non_empty_result(
        db_client, config["sql_data_present"].format(table=table.table_id,
                                                     dataset=table.dataset_id),
        {"bucket_name": bucket_name})

    if not data_exists or args.force_data_import:
      if args.force_data_import and data_exists:
        print(
            "There is already some data in the table, but the batch data import was forced."
        )
      print("Importing existing data now...")
      storage_client = storage.Client()

      batch_import.import_entire_bucket(db_client,
                                        storage_client,
                                        config,
                                        config_file.get("snippets", {}),
                                        check_for_presence=False)

    # We need the bucket's location because the cloud function event trigger needs to be registered in the same location
    trigger_location = gcloud(
        "storage",
        "buckets",
        "describe",
        f"gs://{bucket_name}",
        "--format",
        "value(location)",
    ).strip()

    print(
        f"Deploying cloud function {cloud_function_name} through gcloud now...")
    gcloud(
        "functions",
        "deploy",
        cloud_function_name,
        "--gen2",
        "--runtime=python311",
        f"--service-account={my_service_account}",
        f"--source={MAIN_MODULE_DIR}",
        "--entry-point=entry_point",
        f"--region={args.region.lower()}",
        "--trigger-event-filters=type=google.cloud.storage.object.v1.finalized",
        f"--trigger-event-filters=bucket={bucket_name}",
        f"--trigger-location={trigger_location.lower()}",
        f"--set-env-vars=config_name={config_name}",
    )

    # Note that the previous command automatically creates the event trigger but does not take care of
    # authorizing it to call the cloud function. We have to do that manually here.
    print(
        f"Authorizing the Event Arc trigger to call the cloud function {cloud_function_name}..."
    )
    gcloud(
        "functions",
        "add-invoker-policy-binding",
        cloud_function_name,
        f"--region={args.region.lower()}",
        f"--member=serviceAccount:{my_service_account}",
    )
