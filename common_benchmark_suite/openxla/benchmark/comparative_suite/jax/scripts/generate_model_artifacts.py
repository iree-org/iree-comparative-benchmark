# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import importlib
import jax
import jax.numpy as jnp
import os
import pathlib
import re
import multiprocessing
import shutil
import subprocess
import sys
import tarfile

from typing import Any, Optional, Tuple, Union

# Add openxla dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parents[5]))
from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.jax import model_definitions
from openxla.benchmark.models.jax import model_interfaces

HLO_FILENAME_REGEX = r".*jit_forward.before_optimizations.txt"
HLO_STATIC_FILENAME = "xla_hlo_before_optimizations.txt"


def _generate_and_save_inputs(model_obj: Any,
                              model_dir: pathlib.Path) -> Tuple[Any, ...]:
  # TODO(#44): Support multiple raw inputs.
  raw_inputs = model_obj.generate_default_inputs()
  inputs = model_obj.preprocess(raw_inputs)

  # Save inputs.
  inputs_dir = model_dir.joinpath("inputs")
  inputs_dir.mkdir(exist_ok=True)
  for idx, input in enumerate(inputs):
    input_path = inputs_dir.joinpath(f"input_{idx}.npy")
    print(f"Saving input {jnp.shape(input)} to {input_path}")
    jnp.save(input_path, input)

  with tarfile.open(model_dir.joinpath("inputs_npy.tgz"), "w:gz") as tar:
    tar.add(f"{inputs_dir}/", arcname="")
  shutil.rmtree(inputs_dir)

  return inputs


def _save_outputs(outputs: Tuple[Any, ...], model_dir: pathlib.Path):
  outputs_dir = model_dir.joinpath("outputs")
  outputs_dir.mkdir(exist_ok=True)

  for idx, output in enumerate(outputs):
    output_path = outputs_dir.joinpath(f"output_{idx}.npy")
    print(f"Saving output {jnp.shape(outputs)} to {output_path}")
    jnp.save(output_path, output)

  with tarfile.open(model_dir.joinpath("outputs_npy.tgz"), "w:gz") as tar:
    tar.add(f"{outputs_dir}/", arcname="")
  shutil.rmtree(outputs_dir)


def _cleanup_hlo(hlo_dir: pathlib.Path, model_dir: pathlib.Path):
  # The filename of the input HLO varies for each model so we rename it to a
  # a known name.
  hlo_files = [
      f for f in os.listdir(hlo_dir) if re.search(HLO_FILENAME_REGEX, f)
  ]
  if len(hlo_files) != 1:
    raise RuntimeError("Could not find HLO file")

  shutil.move(str(hlo_dir.joinpath(hlo_files[0])),
              str(model_dir.joinpath(HLO_STATIC_FILENAME)))
  shutil.rmtree(hlo_dir)


def _generate_mlir(jit_function: Any, jit_inputs: Any, model_dir: pathlib.Path,
                   iree_opt_path: Optional[pathlib.Path]):
  mlir = jit_function.lower(jit_inputs).compiler_ir(dialect="stablehlo")
  mlir_path = model_dir.joinpath("stablehlo.mlir")
  print(f"Saving mlir to {mlir_path}")
  with open(mlir_path, "w") as f:
    f.write(str(mlir))

  if iree_opt_path:
    binary_mlir_path = model_dir.joinpath("stablehlo.mlirbc")
    subprocess.run(
        f"{iree_opt_path} --emit-bytecode {mlir_path} -o {binary_mlir_path}",
        shell=True,
        check=True)
    mlir_path.unlink()


def _generate_artifacts(model: def_types.Model, save_dir: pathlib.Path,
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
        "XLA_FLAGS"] = f"--xla_dump_to={hlo_dir} --xla_dump_hlo_module_re=.*jit_forward.*"

    model_module = importlib.import_module(model.model_impl.module_path)
    model_obj: model_interfaces.InferenceModel = model_module.create_model(
        **model.model_parameters)

    inputs = _generate_and_save_inputs(model_obj, model_dir)
    jit_inputs = jax.device_put(inputs)
    jit_function = jax.jit(model_obj.forward)
    outputs = jit_function(jit_inputs)
    _save_outputs(outputs, model_dir)

    _cleanup_hlo(hlo_dir, model_dir)
    os.unsetenv("XLA_FLAGS")

    _generate_mlir(jit_function,
                   jit_inputs,
                   model_dir,
                   iree_opt_path=iree_opt_path)

    print(f"Completed generating artifacts {model.name}\n")

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
  parser.add_argument("--iree_opt_path",
                      type=pathlib.Path,
                      default=None,
                      help="Path to `iree-opt`. Used to binarize mlir.")
  return parser.parse_args()


def main(output_dir: pathlib.Path, filter: str, iree_opt_path: pathlib.Path):
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
                                args=(model, output_dir, iree_opt_path))
    p.start()
    p.join()


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
