# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import concurrent.futures
import jax
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any, List, Optional

# Add openxla dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parents[5]))
from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.jax import model_definitions
from openxla.benchmark.models import utils

HLO_FILENAME_REGEX = r".*jit_forward.before_optimizations.txt"
GCS_UPLOAD_DIR = os.getenv("GCS_UPLOAD_DIR", "gs://iree-model-artifacts/jax")


def _generate_mlir(jit_function: Any, jit_inputs: Any, model_dir: pathlib.Path,
                   iree_ir_tool: Optional[pathlib.Path]):
  mlir = jit_function.lower(*jit_inputs).compiler_ir(dialect="stablehlo")
  mlir_path = model_dir / "stablehlo.mlir"
  print(f"Saving mlir to {mlir_path}")
  with open(mlir_path, "w") as f:
    f.write(str(mlir))

  if iree_ir_tool:
    binary_mlir_path = model_dir / "stablehlo.mlirbc"
    subprocess.run(
        [
            iree_ir_tool, "cp", "--emit-bytecode", mlir_path, "-o",
            binary_mlir_path
        ],
        check=True,
    )
    mlir_path.unlink()


def _generate_artifacts(model: def_types.Model, save_dir: pathlib.Path,
                        iree_ir_tool: Optional[pathlib.Path],
                        auto_upload: bool):
  model_dir = save_dir / model.name
  model_dir.mkdir(exist_ok=True)
  print(f"Created {model_dir}")

  try:
    # Configure to dump hlo.
    hlo_dir = model_dir / "hlo"
    hlo_dir.mkdir(exist_ok=True)
    # Only dump hlo for the inference function `jit_model_jitted`.
    os.environ[
        "XLA_FLAGS"] = f"--xla_dump_to={hlo_dir} --xla_dump_hlo_module_re=.*jit_forward.*"

    model_obj = utils.create_model_obj(model)

    inputs = utils.generate_and_save_inputs(model_obj, model_dir)

    jit_inputs = jax.device_put(inputs)
    jit_function = jax.jit(model_obj.forward)
    jit_output_obj = jit_function(*jit_inputs)
    jax.block_until_ready(jit_output_obj)
    output_obj = jax.device_get(jit_output_obj)

    outputs = utils.canonicalize_to_tuple(output_obj)
    utils.save_outputs(outputs, model_dir)

    utils.cleanup_hlo(hlo_dir, model_dir, HLO_FILENAME_REGEX)
    os.unsetenv("XLA_FLAGS")

    _generate_mlir(jit_function=jit_function,
                   jit_inputs=jit_inputs,
                   model_dir=model_dir,
                   iree_ir_tool=iree_ir_tool)

    print(f"Completed generating artifacts {model.name}\n")

    if auto_upload:
      utils.gcs_upload(str(model_dir),
                       f"{GCS_UPLOAD_DIR}/{save_dir.name}/{model_dir.name}")
      shutil.rmtree(model_dir)

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
                      dest="filters",
                      nargs="+",
                      default=[".*"],
                      help="The regex patterns to filter model names.")
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
  parser.add_argument(
      "-j",
      "--jobs",
      type=int,
      default=1,
      help="Max number of concurrent jobs to generate artifacts. Be cautious"
      " when generating with GPU.")
  return parser.parse_args()


def main(output_dir: pathlib.Path, filters: List[str],
         iree_ir_tool: pathlib.Path, auto_upload: bool, jobs: int):
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

  with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
    for model in models:
      # We need to generate artifacts in a separate proces each time in order for
      # XLA to update the HLO dump directory.
      executor.submit(_generate_artifacts,
                      model=model,
                      save_dir=output_dir,
                      iree_ir_tool=iree_ir_tool,
                      auto_upload=auto_upload)

  if auto_upload:
    utils.gcs_upload(f"{output_dir}/**", f"{GCS_UPLOAD_DIR}/{output_dir.name}/")


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
