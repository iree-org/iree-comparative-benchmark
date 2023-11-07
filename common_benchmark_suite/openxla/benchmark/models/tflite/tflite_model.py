# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import urllib


class TFLiteModel:
  """A wrapper around a TFLite flatbuffer and its accuracy data."""

  model_uri: str
  model_filename: str
  input_data_uri: str
  output_data_uri: str

  def __init__(self, model_uri: str):
    self.model_uri = model_uri
    parsed_url = urllib.parse.urlparse(model_uri)
    self.model_filename = os.path.basename(parsed_url.path)

    parent_url = model_uri.replace(f"/{self.model_filename}", "")
    self.input_data_uri = f"{parent_url}/inputs_npy.tgz"
    self.output_data_uri = f"{parent_url}/outputs_npy.tgz"


def create_model(model_uri: str):
  return TFLiteModel(model_uri)
