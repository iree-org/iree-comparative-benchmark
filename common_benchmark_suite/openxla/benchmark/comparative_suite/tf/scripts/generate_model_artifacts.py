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
import subprocess
import sys
import tarfile
import tensorflow as tf

from typing import Any, Optional, Tuple

# Add openxla dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parents[5]))
from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.tf import model_definitions
from openxla.benchmark.models import model_interfaces, utils

HLO_FILENAME_REGEX = r".*inference_forward.*before_optimizations.txt"


def _generate_saved_model(inputs: Tuple[Any, ...],
                          model_obj: model_interfaces.InferenceModel,
                          model_dir: pathlib.Path) -> pathlib.Path:
  tensor_specs = []
  for input in inputs:
    tensor_specs.append(tf.TensorSpec.from_tensor(input))
  call_signature = model_obj.forward.get_concrete_function(*tensor_specs)

  saved_model_dir = model_dir.joinpath("saved_model")
  saved_model_dir.mkdir(exist_ok=True)
  print(f"Saving {saved_model_dir} with call signature: {call_signature}")
  tf.saved_model.save(model_obj,
                      saved_model_dir,
                      signatures={"serving_default": call_signature})
  return saved_model_dir


def _generate_mlir(model_dir: pathlib.Path, saved_model_dir: pathlib.Path,
                   iree_import_tf_path: pathlib.Path,
                   iree_opt_path: Optional[pathlib.Path]):
  mlir_path = model_dir.joinpath("stablehlo.mlir")
  subprocess.run(
      f"{iree_import_tf_path} --output-format=mlir-bytecode --tf-import-type=savedmodel_v2 --tf-savedmodel-exported-names=forward {saved_model_dir} -o {mlir_path}",
      shell=True,
      check=True)

  if iree_opt_path:
    binary_mlir_path = model_dir.joinpath("stablehlo.mlirbc")
    subprocess.run(
        f"{iree_opt_path} --emit-bytecode {mlir_path} -o {binary_mlir_path}",
        shell=True,
        check=True)
    mlir_path.unlink()


def _generate_artifacts(model: def_types.Model, save_dir: pathlib.Path,
                        iree_import_tf_path: pathlib.Path,
                        iree_opt_path: pathlib.Path):
  model_dir = save_dir.joinpath(model.name)
  model_dir.mkdir(exist_ok=True)
  print(f"Created {model_dir}")

  try:
    # Configure to dump hlo.
    hlo_dir = model_dir.joinpath("hlo")
    hlo_dir.mkdir(exist_ok=True)
    # Only dump hlo for the inference function `jit_model_jitted`.
    os.environ[
        "XLA_FLAGS"] = f"--xla_dump_to={hlo_dir} --xla_dump_hlo_module_re=.*inference_forward.*"

    model_obj = utils.create_model_obj(model)

    inputs = utils.generate_and_save_inputs(model_obj, model_dir)
    output_obj = model_obj.forward(*inputs)
    outputs = utils.canonicalize_to_tuple(output_obj)
    utils.save_outputs(outputs, model_dir)

    utils.cleanup_hlo(hlo_dir, model_dir, HLO_FILENAME_REGEX)
    os.unsetenv("XLA_FLAGS")

    saved_model_dir = _generate_saved_model(inputs, model_obj, model_dir)
    _generate_mlir(model_dir,
                   saved_model_dir,
                   iree_import_tf_path=iree_import_tf_path,
                   iree_opt_path=iree_opt_path)

    with tarfile.open(model_dir.joinpath("tf-model.tgz"), "w:gz") as tar:
      tar.add(f"{saved_model_dir}/", arcname="")
    shutil.rmtree(saved_model_dir)

    print(f"Completed generating artifacts {model.name}\n")

  except Exception as e:
    print(f"Failed to import model {model.name}. Exception: {e}")
    # Remove all generated files.
    shutil.rmtree(model_dir)
    raise


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generates TF model artifacts for benchmarking.")
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
  parser.add_argument("--iree_import_tf_path",
                      type=pathlib.Path,
                      default="iree-import-tf",
                      help="Path to `iree-import-tf`. Used to binarize mlir.")
  parser.add_argument("--iree_opt_path",
                      type=pathlib.Path,
                      default=None,
                      help="Path to `iree-opt`. Used to binarize mlir.")
  return parser.parse_args()


def main(output_dir: pathlib.Path, filter: str,
         iree_import_tf_path: pathlib.Path, iree_opt_path: pathlib.Path):
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
    # We need to generate artifacts in a separate proces each time in order for
    # XLA to update the HLO dump directory.
    p = multiprocessing.Process(target=_generate_artifacts,
                                args=(model, output_dir, iree_import_tf_path,
                                      iree_opt_path))
    p.start()
    p.join()


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
