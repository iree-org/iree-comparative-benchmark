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

# Add openxla dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parents[5]))
from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.pt import model_definitions
from openxla.benchmark.models import model_interfaces, utils


def _import_torch_mlir(model_obj: model_interfaces.InferenceModel,
                       inputs: tuple, import_with_fx: bool) -> bytes:
  # Import torch_mlir here to make it optional if mlir generation is not
  # required.
  import torch_mlir
  from import_utils import import_torch_module_with_fx, import_torch_module

  if import_with_fx:
    return import_torch_module_with_fx(model_obj, inputs,
                                       torch_mlir.OutputType.LINALG_ON_TENSORS)

  graph = torch.jit.trace(model_obj, inputs)
  return import_torch_module(graph, inputs,
                             torch_mlir.OutputType.LINALG_ON_TENSORS)


def _generate_artifacts(model: def_types.Model, save_dir: pathlib.Path,
                        skip_torch_mlir_gen: bool):
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

      model_obj = utils.create_model_obj(model)
      inputs = utils.generate_and_save_inputs(model_obj, model_dir)
      if import_on_gpu:
        model_obj.cuda()
        inputs = tuple(input.cuda() for input in inputs)

      output_obj = model_obj.forward(*inputs)

      outputs = utils.canonicalize_to_tuple(output_obj)
      outputs = tuple(output.cpu() for output in outputs)
      utils.save_outputs(outputs, model_dir)

      if skip_torch_mlir_gen:
        return

      # Generate and save mlir.
      mlir_data = _import_torch_mlir(model_obj=model_obj,
                                     inputs=inputs,
                                     import_with_fx=import_with_fx)
      mlir_path = model_dir / "linalg.mlirbc"
      print(f"Saving mlir to {mlir_path}")
      mlir_path.write_bytes(mlir_data)

  except Exception as e:
    print(f"Failed to import model {model.name}. Exception: {e}")
    # Remove all generated files.
    shutil.rmtree(model_dir)
    raise


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
  parser.add_argument("--skip-torch-mlir-gen",
                      "--skip_torch_mlir_gen",
                      action="store_true",
                      help="Don't generate torch mlir. Use cautiously when"
                      " torch-mlir fails to import some models.")
  return parser.parse_args()


def main(output_dir: pathlib.Path, filter: str, skip_torch_mlir_gen: bool):
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
                                args=(model, output_dir, skip_torch_mlir_gen))
    p.start()
    p.join()


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
