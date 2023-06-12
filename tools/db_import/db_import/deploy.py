## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os, sys, argparse
from plumbum import local, TF
from google.cloud import storage, bigquery

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from db_import.batch_import import import_entire_bucket
from db_import.db import is_there_already_some_data_in_table, delete_all_preexisting_data


def configure_parser(parser: argparse.ArgumentParser):
  parser.add_argument("config_names", nargs="+")
  parser.add_argument(
      "-c",
      "--config",
      help="Read config from the given file",
      default=os.path.join(SCRIPT_DIR, "config.yml"),
  )
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


def deploy(config_file, args: argparse.Namespace):
  gcloud = local["gcloud"]
  gsutil = local["gsutil"]
  bq = local["bq"]

  project_id = gcloud("config", "get-value", "project").strip()

  # This is a special SA that all GCS events originate from.
  gcs_service_account = gsutil("kms", "serviceaccount", "-p",
                               project_id).strip()

  # This is the SA we run the cloud function as.
  my_service_account = config_file['service_account']
  if "@" not in my_service_account:
    my_service_account += "@{}.iam.gserviceaccount.com".format(project_id)

  for config_name in args.config_names:
    print("Processing config {}...".format(config_name))
    config = config_file["cloud_functions"][config_name]
    bucket_name = config["bucket_name"]
    cloud_function_name = config["cloud_function_name"]
    table_name = config["table_name"]
    table_exists = bool(bq["show", table_name] & TF)
    db_client = bigquery.Client()

    if not table_exists:
      print("Table {} does not exist. Creating it...".format(table_name))
      try:
        schema = config_file["table_schemas"][table_name]
      except KeyError:
        sys.exit("Table schema for table {} not found in config file.".format(
            table_name))

      bq("mk", "-t", table_name, schema)
      bq(
          "add-iam-policy-binding",
          "--member=serviceAccount:{}".format(my_service_account),
          "--role=roles/bigquery.dataEditor",
          "{}:{}".format(project_id, table_name),
      )
    else:
      if args.force_data_deletion:
        print("Deleting pre-existing data from destination table {}".format(
            table_name))
        delete_all_preexisting_data(db_client, config)
      else:
        print(
            "The destination table {} already exists. Checking if data is already present."
            .format(table_name))

    data_exists = is_there_already_some_data_in_table(db_client, config)

    if not data_exists or args.force_data_import:
      if args.force_data_import and data_exists:
        print(
            "There is already some data in the table, but the data import was forced."
        )
      print("Importing existing data now...")
      storage_client = storage.Client()

      import_entire_bucket(
          db_client,
          storage_client,
          config,
          config_file.get("snippets", {}),
          False,
      )

    # We need the bucket's location because the cloud function event trigger needs to be registered in the same location
    trigger_location = gcloud(
        "storage",
        "buckets",
        "describe",
        "gs://{}".format(bucket_name),
        "--format",
        "value(location)",
    ).strip()

    print("Deploying cloud function {} through gcloud now...".format(
        cloud_function_name))
    gcloud(
        "functions",
        "deploy",
        cloud_function_name,
        "--gen2",
        "--runtime=python311",
        "--service-account={}".format(my_service_account),
        "--source={}".format(SCRIPT_DIR),
        "--entry-point=entry_point",
        "--region={}".format(args.region.lower()),
        "--trigger-event-filters=type=google.cloud.storage.object.v1.finalized",
        "--trigger-event-filters=bucket={}".format(bucket_name),
        "--trigger-location={}".format(trigger_location.lower()),
        "--set-env-vars=config_name={}".format(config_name),
    )

    # Note that the previous command automatically creates the event trigger but does not take care of
    # authorizing it to call the cloud function. We have to do that manually here.
    print("Authorizing the Event Arc trigger to call the cloud function {}...".
          format(cloud_function_name))
    gcloud(
        "functions",
        "add-invoker-policy-binding",
        cloud_function_name,
        "--region={}".format(args.region.lower()),
        "--member=serviceAccount:{}".format(my_service_account),
    )
