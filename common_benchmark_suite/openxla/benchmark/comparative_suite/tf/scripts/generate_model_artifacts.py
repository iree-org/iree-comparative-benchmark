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
import tarfile
import tensorflow as tf

from tensorflow.mlir.experimental import convert_saved_model, run_pass_pipeline, write_bytecode
from typing import Any, Optional, Tuple

# Add openxla dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parents[5]))
from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.tf import model_definitions
from openxla.benchmark.models import model_interfaces, utils

HLO_FILENAME_REGEX = r".*inference_forward.*before_optimizations.txt"
GCS_UPLOAD_DIR = os.getenv("GCS_UPLOAD_DIR",
                           "gs://iree-model-artifacts/tensorflow")


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


def _generate_mlir(model_dir: pathlib.Path, saved_model_dir: pathlib.Path):
  result = convert_saved_model(saved_model_dir,
                               exported_names="forward",
                               show_debug_info=False)

  # The import to MLIR produces public functions like __inference__{name}_2222
  # but the conversion pipeline requires a single public @main function.
  # Not sure how this was allowed to happen, but regex to the rescue.
  # This is fine and normal, and totally to be expected. :(
  result = re.sub(r"func @__inference_(.+)_[0-9]+\(", r"func @\1(", result)
  pipeline = ["tf-lower-to-mlprogram-and-hlo"]
  result = run_pass_pipeline(result, ",".join(pipeline), show_debug_info=False)
  mlir_path = model_dir.joinpath("stablehlo.mlirbc")
  write_bytecode(str(mlir_path), result)


def _generate_artifacts(model: def_types.Model, save_dir: pathlib.Path,
                        auto_upload: bool):
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
    _generate_mlir(model_dir, saved_model_dir)

    with tarfile.open(model_dir.joinpath("tf-model.tgz"), "w:gz") as tar:
      tar.add(f"{saved_model_dir}/", arcname="")
    shutil.rmtree(saved_model_dir)

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
    # We need to generate artifacts in a separate proces each time in order for
    # XLA to update the HLO dump directory.
    p = multiprocessing.Process(target=_generate_artifacts,
                                args=(model, output_dir, auto_upload))
    p.start()
    p.join()

  if auto_upload:
    utils.gcs_upload(f"{output_dir}/**", f"{GCS_UPLOAD_DIR}/{output_dir.name}/")


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
