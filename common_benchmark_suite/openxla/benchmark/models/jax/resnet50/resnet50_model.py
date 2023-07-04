# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from PIL import Image
import io
import jax.numpy as jnp
import requests
from transformers import AutoImageProcessor, FlaxResNetModel
from typing import Any, Tuple

from openxla.benchmark.models.jax import model_interfaces

MODEL_NAME = "microsoft/resnet-50"


def _get_image_input(width=224, height=224):
  """Returns a sample image in the Imagenet2012 Validation Dataset.
  Input size 224x224x3 is used, as stated in the MLPerf Inference Rules:
  https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#41-benchmarks
  """
  # We use an image of 5 applies since this is an easy example.
  img_path = "https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG"
  data = requests.get(img_path).content
  img = Image.open(io.BytesIO(data))
  img = img.resize((width, height))
  return img


class ResNet50(model_interfaces.InferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/resnet for more information."""

  def __init__(
      self,
      batch_size: int,
      dtype: Any,
  ):
    self.model = FlaxResNetModel.from_pretrained(MODEL_NAME, dtype=dtype)
    if dtype == jnp.float32:
      # The original model is fp32.
      pass
    elif dtype == jnp.float16:
      self.model.params = self.model.to_fp16(self.model.params)
    elif dtype == jnp.bfloat16:
      self.model.params = self.model.to_bf16(self.model.params)
    else:
      raise ValueError(f"Unsupported data type '{dtype}'.")

    self.batch_size = batch_size

  def generate_default_inputs(self) -> Tuple[Any, ...]:
    # TODO(#44): This should go away once we support different raw inputs.
    image = _get_image_input()
    return (image,)

  def preprocess(self, raw_inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    image, = raw_inputs
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    inputs = image_processor(images=image, return_tensors="jax")
    tensor = inputs["pixel_values"]
    tensor = jnp.asarray(jnp.tile(tensor, [self.batch_size, 1, 1, 1]),
                         dtype=self.model.dtype)
    return (tensor,)

  def forward(self, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    tensor, = inputs
    output = self.model(tensor)[0]
    return (output,)

  def postprocess(self, outputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    # No-op.
    return outputs


DTYPE_MAP = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}


def create_model(batch_size: int = 1,
                 data_type: str = "fp32",
                 **_unused_params) -> ResNet50:
  """Configure and create a JAX ResNet50 model instance.
  
  Args:
    batch_size: input batch size.
    data_type: model data type. Supported options include: fp32, fp16, bf16.
  Returns:
    A JAX ResNet50 model.
  """
  return ResNet50(batch_size=batch_size, dtype=DTYPE_MAP[data_type])
