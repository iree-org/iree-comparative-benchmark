# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import pathlib
import re
import requests
import shutil
import subprocess
import sys
from typing import Any, List, Optional

# Add openxla dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parents[5]))
from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.tflite import model_definitions
from openxla.benchmark.models import utils

GCS_UPLOAD_DIR = os.getenv("GCS_UPLOAD_DIR", "gs://iree-model-artifacts/tflite")


def _download_file(url: str, save_path: pathlib.Path):
  print(f"Downloading {url} to {save_path}")
  with requests.get(url, stream=True) as response:
    if not response.ok:
      raise ValueError(f"Failed to download '{url}'."
                       f" Error: '{response.status_code} - {response.text}'")

    with save_path.open("wb") as f:
      for chunk in response.iter_content(chunk_size=65536):
        f.write(chunk)


def _generate_mlir(tflite_file: pathlib.Path, iree_import_tool: pathlib.Path,
                   model_dir: pathlib.Path,
                   iree_ir_tool: Optional[pathlib.Path]):
  mlir_path = model_dir / "tosa.mlir"
  command = [iree_import_tool, tflite_file, "-o", mlir_path]
  subprocess.run(command, check=True)

  if iree_ir_tool:
    binary_mlir_path = model_dir / "tosa.mlirbc"
    subprocess.run(
        [
            iree_ir_tool, "cp", "--emit-bytecode", mlir_path, "-o",
            binary_mlir_path
        ],
        check=True,
    )
    mlir_path.unlink()


def _generate_artifacts(model: def_types.Model, save_dir: pathlib.Path,
                        iree_import_tool: pathlib.Path,
                        iree_ir_tool: Optional[pathlib.Path],
                        auto_upload: bool):
  print(f"Generating artifacts for {model.name}")
  model_dir = save_dir / model.name
  model_dir.mkdir(exist_ok=True)

  try:
    model_obj = utils.create_model_obj(model)

    # Download artifacts.
    tflite_file = model_dir / model_obj.model_filename
    _download_file(model_obj.model_uri, tflite_file)
    _download_file(model_obj.input_data_uri, model_dir / "inputs_npy.tgz")
    _download_file(model_obj.output_data_uri, model_dir / "outputs_npy.tgz")

    # Generate mlir.
    _generate_mlir(tflite_file, iree_import_tool, model_dir, iree_ir_tool)
    print(f"Completed generating artifacts {model.name}\n")

    if auto_upload:
      utils.gcs_upload(str(model_dir),
                       f"{GCS_UPLOAD_DIR}/{save_dir.name}/{model_dir.name}")
      shutil.rmtree(model_dir)

  except Exception as e:
    print(f"Failed to import model {model.name}. Exception: {e}")
    # Remove all generated files.
    shutil.rmtree(model_dir)


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generates TFLite model artifacts for benchmarking.")
  parser.add_argument("-o",
                      "--output_dir",
                      type=pathlib.Path,
                      required=True,
                      help="Directory to save model artifacts.")
  parser.add_argument("-f",
                      "--filter",
                      dest="filters",
                      nargs="+",
                      default=[".*"],
                      help="The regex patterns to filter model names.")
  parser.add_argument(
      "--iree-import-tool",
      "--iree_import_tool",
      type=pathlib.Path,
      default=None,
      help=
      "Path to `iree-import-tflite`. Used to import the TFLite flatbuffer to mlir."
  )
  parser.add_argument("--iree-ir-tool",
                      "--iree_ir_tool",
                      type=pathlib.Path,
                      default=None,
                      help="Path to `iree-ir-tool`. Used to binarize mlir.")
  parser.add_argument(
      "--auto-upload",
      "--auto_upload",
      action="store_true",
      help=
      f"If set, uploads artifacts automatically to {GCS_UPLOAD_DIR} and removes them locally once uploaded."
  )
  return parser.parse_args()


def main(output_dir: pathlib.Path, filters: List[str],
         iree_import_tool: pathlib.Path, iree_ir_tool: pathlib.Path,
         auto_upload: bool):
  combined_filters = "|".join(f"({name_filter})" for name_filter in filters)
  name_pattern = re.compile(f"^{combined_filters}$")
  models = [
      model for model in model_definitions.ALL_MODELS
      if name_pattern.match(model.name)
  ]

  if not models:
    all_models_list = "\n".join(
        model.name for model in model_definitions.ALL_MODELS)
    raise ValueError(f'No model matches "{filters}".'
                     f' Available models:\n{all_models_list}')

  output_dir.mkdir(parents=True, exist_ok=True)

  for model in models:
    _generate_artifacts(model, output_dir, iree_import_tool, iree_ir_tool,
                        auto_upload)

  if auto_upload:
    utils.gcs_upload(f"{output_dir}/**", f"{GCS_UPLOAD_DIR}/{output_dir.name}/")


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
