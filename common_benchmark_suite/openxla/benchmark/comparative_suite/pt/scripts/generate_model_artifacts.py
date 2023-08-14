# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import pathlib
import re
import multiprocessing
import shutil
import sys
import torch
import torch_mlir

from import_utils import import_torch_module_with_fx, import_torch_module

# Add openxla dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parents[5]))
from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.pt import model_definitions
from openxla.benchmark.models import utils

GCS_UPLOAD_DIR = os.getenv("GCS_UPLOAD_DIR",
                           "gs://iree-model-artifacts/pytorch")


def _generate_artifacts(model: def_types.Model, save_dir: pathlib.Path,
                        auto_upload: bool):
  # Remove all gradient info from models and tensors since these models are
  # inference only.
  with torch.no_grad():
    import_on_gpu = model.model_parameters.get("import_on_gpu", False)
    import_with_fx = model.model_parameters.get("import_with_fx", True)
    if import_on_gpu and not torch.cuda.is_available():
      raise RuntimeError("Model can only be exported on CUDA.")

    model_obj = utils.create_model_obj(model)

    model_dir = save_dir / model.name
    model_dir.mkdir(exist_ok=True)
    print(f"Created {model_dir}")
    try:
      inputs = utils.generate_and_save_inputs(model_obj, model_dir)
      if import_on_gpu:
        model_obj.cuda()
        inputs = tuple(input.cuda() for input in inputs)

      output_obj = model_obj.forward(*inputs)

      outputs = utils.canonicalize_to_tuple(output_obj)
      outputs = tuple(output.cpu() for output in outputs)
      utils.save_outputs(outputs, model_dir)
    except Exception as e:
      # Remove all generated files.
      shutil.rmtree(model_dir)
      raise

    # Try to export the mlir with torch mlir. Skip if failed.
    try:
      if import_with_fx:
        mlir_data = import_torch_module_with_fx(
            model_obj, inputs, torch_mlir.OutputType.LINALG_ON_TENSORS)
      else:
        graph = torch.jit.trace(model_obj, inputs)
        mlir_data = import_torch_module(graph, inputs,
                                        torch_mlir.OutputType.LINALG_ON_TENSORS)

      mlir_path = model_dir / "linalg.mlirbc"
      print(f"Saving mlir to {mlir_path}")
      mlir_path.write_bytes(mlir_data)

    except Exception as e:
      print(f"WARNING: Failed to import model {model.name} into MLIR."
            f" Exception: {e}")

    if auto_upload:
      utils.gcs_upload(str(model_dir),
                       f"{GCS_UPLOAD_DIR}/{save_dir.name}/{model_dir.name}")
      shutil.rmtree(model_dir)


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generates PyTorch model artifacts for benchmarking.")
  parser.add_argument("-o",
                      "--output_dir",
                      type=pathlib.Path,
                      required=True,
                      help="Directory to save model artifacts.")
  parser.add_argument("-f",
                      "--filter",
                      type=str,
                      default=".*",
                      help="The regex pattern to filter model names.")
  parser.add_argument(
      "--auto-upload",
      "--auto_upload",
      action="store_true",
      help=
      f"If set, uploads artifacts automatically to {GCS_UPLOAD_DIR} and removes them locally once uploaded."
  )
  return parser.parse_args()


def main(output_dir: pathlib.Path, filter: str, auto_upload: bool):
  name_pattern = re.compile(f"^{filter}$")
  models = [
      model for model in model_definitions.ALL_MODELS
      if name_pattern.match(model.name)
  ]

  if not models:
    all_models_list = "\n".join(
        model.name for model in model_definitions.ALL_MODELS)
    raise ValueError(f'No model matches "{filter}".'
                     f' Available models:\n{all_models_list}')

  output_dir.mkdir(parents=True, exist_ok=True)
  for model in models:
    p = multiprocessing.Process(target=_generate_artifacts,
                                args=(model, output_dir, auto_upload))
    p.start()
    p.join()

  if auto_upload:
    utils.gcs_upload(f"{output_dir}/**", f"{GCS_UPLOAD_DIR}/{output_dir.name}/")


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
