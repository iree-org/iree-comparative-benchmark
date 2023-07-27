# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
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


def _generate_artifacts(model: def_types.Model, save_dir: pathlib.Path):
  model_dir = save_dir / model.name
  model_dir.mkdir(exist_ok=True)
  print(f"Created {model_dir}")

  try:
    # Remove all gradient info from models and tensors since these models are inference only.
    with torch.no_grad():
      import_on_gpu = model.model_parameters.get("import_on_gpu", False)
      import_with_fx = model.model_parameters.get("import_with_fx", True)
      model_obj = utils.create_model_obj(model)

      if import_on_gpu and not torch.cuda.is_available():
        raise RuntimeError("Model can only be exported on CUDA.")

      inputs = utils.generate_and_save_inputs(model_obj, model_dir)
      if import_on_gpu:
        model_obj.cuda()
        inputs = [input.cuda() for input in inputs]

      if import_with_fx:
        mlir_data = import_torch_module_with_fx(
            model_obj, inputs, torch_mlir.OutputType.LINALG_ON_TENSORS)
      else:
        graph = torch.jit.trace(model_obj, inputs)
        mlir_data = import_torch_module(graph, inputs,
                                        torch_mlir.OutputType.LINALG_ON_TENSORS)

      # Save mlir.
      mlir_path = model_dir / "linalg.mlirbc"
      print(f"Saving mlir to {mlir_path}")
      mlir_path.write_bytes(mlir_data)

      output = model_obj.forward(*inputs)
      output = output.cpu()
      outputs = (output,)
      utils.save_outputs(outputs, model_dir)

  except Exception as e:
    print(f"Failed to import model {model.name}. Exception: {e}")
    # Remove all generated files.
    shutil.rmtree(model_dir)
    raise


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generates JAX model artifacts for benchmarking.")
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
  return parser.parse_args()


def main(output_dir: pathlib.Path, filter: str):
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
                                args=(model, output_dir))
    p.start()
    p.join()


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
