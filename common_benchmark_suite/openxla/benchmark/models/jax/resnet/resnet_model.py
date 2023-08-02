# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from PIL import Image
import jax.numpy as jnp
from transformers import AutoImageProcessor, FlaxResNetModel
from typing import Any, Tuple

from openxla.benchmark.models import model_interfaces, utils

DEFAULT_IMAGE_URL = "https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG"


class ResNet(model_interfaces.InferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/resnet for more information."""

  def __init__(
      self,
      model_name: str,
      batch_size: int,
      dtype: Any,
  ):
    model: FlaxResNetModel = FlaxResNetModel.from_pretrained(
        model_name,
        dtype=dtype,
    )
    if dtype == jnp.float32:
      # The original model is fp32.
      pass
    elif dtype == jnp.float16:
      model.params = model.to_fp16(model.params)
    elif dtype == jnp.bfloat16:
      model.params = model.to_bf16(model.params)
    else:
      raise ValueError(f"Unsupported data type '{dtype}'.")

    self.model = model
    self.model_name = model_name
    self.batch_size = batch_size

  def generate_default_inputs(self) -> Image.Image:
    # TODO(#44): This should go away once we support different raw inputs.
    return utils.download_and_read_img(DEFAULT_IMAGE_URL)

  def preprocess(self, input_image: Image.Image) -> Any:
    resized_image = input_image.resize((224, 224))
    image_processor = AutoImageProcessor.from_pretrained(self.model_name)
    inputs = image_processor(images=resized_image, return_tensors="jax")
    tensor = inputs["pixel_values"]
    tensor = jnp.asarray(jnp.tile(tensor, [self.batch_size, 1, 1, 1]),
                         dtype=self.model.dtype)
    return tensor

  def forward(self, input_tensor: Any) -> Any:
    return self.model(input_tensor).last_hidden_state


DTYPE_MAP = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}


def create_model(model_name: str = "microsoft/resnet-50",
                 batch_size: int = 1,
                 data_type: str = "fp32",
                 **_unused_params) -> ResNet:
  """Configure and create a JAX ResNet model instance.
  
  Args:
    batch_size: input batch size.
    data_type: model data type. Supported options include: fp32, fp16, bf16.
  Returns:
    A JAX ResNet model.
  """
  return ResNet(model_name=model_name,
                batch_size=batch_size,
                dtype=DTYPE_MAP[data_type])
