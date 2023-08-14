# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from PIL import Image
import importlib
import io
import numpy as np
import os
import pathlib
import re
import requests
import shutil
import subprocess
import tarfile
from typing import Any, Tuple, Union

from openxla.benchmark import def_types
from openxla.benchmark.models import model_interfaces


def download_and_read_img(url: str) -> Image.Image:
  """Downloads an image and reads it into memory."""
  data = requests.get(url).content
  img = Image.open(io.BytesIO(data))
  return img


def create_model_obj(model: def_types.Model) -> model_interfaces.InferenceModel:
  """Create model object with the bound parameters."""
  model_module = importlib.import_module(model.model_impl.module_path)
  return model_module.create_model(**model.model_parameters)


def canonicalize_to_tuple(return_value: Union[tuple, Any]) -> tuple:
  """Canonicalize the return value of a model interface method, which may return
  either a single value or a tuple for multiple values, to tuple.
  """
  if isinstance(return_value, tuple):
    return return_value
  return (return_value,)


def generate_and_save_inputs(model_obj: model_interfaces.InferenceModel,
                             model_dir: pathlib.Path,
                             archive: bool = True) -> Tuple[Any, ...]:
  """Generates and preprocesses inputs, then saves it into `model_dir/input_npy.tz`."""
  # TODO(#44): Support multiple raw inputs.
  raw_input_obj = model_obj.generate_default_inputs()
  input_obj = model_obj.preprocess(raw_input_obj)
  inputs = canonicalize_to_tuple(input_obj)

  # Save inputs.
  inputs_dir = model_dir / "inputs_npy"
  inputs_dir.mkdir(exist_ok=True)
  for idx, input in enumerate(inputs):
    input_path = inputs_dir / f"input_{idx}.npy"
    np.save(input_path, input)

  if archive:
    with tarfile.open(model_dir.joinpath("inputs_npy.tgz"), "w:gz") as tar:
      tar.add(f"{inputs_dir}/", arcname="")
    shutil.rmtree(inputs_dir)

  return inputs


def save_outputs(outputs: Tuple[Any, ...], model_dir: pathlib.Path) -> None:
  """Saves `outputs` into `model_dir/output_npy.tgz`."""
  outputs_dir = model_dir.joinpath("outputs")
  outputs_dir.mkdir(exist_ok=True)

  for idx, output in enumerate(outputs):
    output_path = outputs_dir.joinpath(f"output_{idx}.npy")
    np.save(output_path, output)

  with tarfile.open(model_dir.joinpath("outputs_npy.tgz"), "w:gz") as tar:
    tar.add(f"{outputs_dir}/", arcname="")
  shutil.rmtree(outputs_dir)


def cleanup_hlo(hlo_dir: pathlib.Path, model_dir: pathlib.Path,
                hlo_filename_regex: str):
  """Takes a HLO dump of files, and renamese and removes relevant files."""
  HLO_STATIC_FILENAME = "xla_hlo_before_optimizations.txt"

  # The filename of the input HLO varies for each model so we rename it to a
  # a known name.
  hlo_files = [
      f for f in os.listdir(hlo_dir) if re.search(hlo_filename_regex, f)
  ]
  if len(hlo_files) != 1:
    raise RuntimeError("Could not find HLO file")

  shutil.move(str(hlo_dir.joinpath(hlo_files[0])),
              str(model_dir.joinpath(HLO_STATIC_FILENAME)))
  shutil.rmtree(hlo_dir)


def gcs_upload(local_path: str, gcs_path: str):
  subprocess.run(["gcloud", "storage", "cp", "-r", local_path, gcs_path],
                 check=True)
  print(f"Uploaded {local_path} to {gcs_path}")
