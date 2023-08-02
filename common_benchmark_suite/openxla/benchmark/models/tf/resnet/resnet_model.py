# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from PIL import Image
import tensorflow as tf
from transformers import AutoImageProcessor, TFResNetModel
from typing import Any

from openxla.benchmark.models import model_interfaces, utils

DEFAULT_IMAGE_URL = "https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG"


class ResNet(tf.Module, model_interfaces.InferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/resnet for more information."""

  batch_size: int
  model: TFResNetModel
  model_name: str

  def __init__(self, batch_size: int, model_name: str):
    self.model = TFResNetModel.from_pretrained(model_name)
    self.batch_size = batch_size
    self.model_name = model_name

  def generate_default_inputs(self) -> Image.Image:
    # TODO(#44): This should go away once we support different raw inputs.
    return utils.download_and_read_img(DEFAULT_IMAGE_URL)

  def preprocess(self, input_image: Image.Image) -> Any:
    resized_image = input_image.resize((224, 224))
    image_processor = AutoImageProcessor.from_pretrained(self.model_name)
    inputs = image_processor(images=resized_image, return_tensors="tf")
    tensor = inputs["pixel_values"]
    tensor = tf.tile(tensor, [self.batch_size, 1, 1, 1])
    return tensor

  @tf.function(jit_compile=True)
  def forward(self, input_tensor: Any) -> Any:
    return self.model(input_tensor, training=False).last_hidden_state


def create_model(batch_size: int = 1,
                 model_name: str = "microsoft/resnet-50",
                 **_unused_params) -> ResNet:
  """Configure and create a TF ResNet model instance.

  Args:
    batch_size: input batch size.
    model_name: The name of the ResNet variant to use.
      Supported variants include:
  Returns:
    A TF ResNet model.
  """
  return ResNet(batch_size=batch_size, model_name=model_name)
