# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import flax
from PIL import Image
import jax.numpy as jnp
from transformers import AutoImageProcessor, FlaxViTForImageClassification
from typing import Any, List, Tuple

from openxla.benchmark.models import utils
from openxla.benchmark.models.jax import jax_model_interface

DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


class VitForClassification(jax_model_interface.JaxInferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/vit#transformers.FlaxViTForImageClassification for more information."""

  batch_size: int
  model: FlaxViTForImageClassification
  model_name: str

  def __init__(self, batch_size: int, model_name: str):
    self.batch_size = batch_size
    self.model_name = model_name
    self.model = FlaxViTForImageClassification.from_pretrained(
        model_name, hidden_act="relu")
    self.image_processor = AutoImageProcessor.from_pretrained(model_name)

  def generate_default_inputs(self) -> Image.Image:
    return utils.download_and_read_img(DEFAULT_IMAGE_URL)

  def preprocess(self, input_image: Image.Image) -> Tuple[Any, Any]:
    resized_image = input_image.resize((224, 224))
    inputs = self.image_processor(images=resized_image, return_tensors="jax")
    tensor = inputs["pixel_values"]
    tensor = jnp.asarray(jnp.tile(tensor, [self.batch_size, 1, 1, 1]),
                         dtype=self.model.dtype)
    return tensor

  def apply(self, input_tensor: Any) -> Any:
    input_tensor = jnp.transpose(input_tensor, (0, 2, 3, 1))
    outputs = self.model.module.apply(
        {'params': flax.core.freeze(self.model.params)}, input_tensor)
    return outputs.logits

  def forward(self, input_tensor: Any) -> Any:
    return self.model(input_tensor).logits


def create_model(batch_size: int = 1,
                 model_name: str = "google/vit-base-patch16-224",
                 **_unused_params) -> VitForClassification:
  """Configure and create a JAX Vision Transformer.

  Args:
    batch_size: input batch size.
    model_name: The name of the ViT variant to use. Supported variants include:
      google/vit-base-patch16-224.
  Returns:
    A JAX VitForClassification model.
  """
  return VitForClassification(batch_size=batch_size, model_name=model_name)
