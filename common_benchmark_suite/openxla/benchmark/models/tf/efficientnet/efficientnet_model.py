# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from PIL import Image
import tensorflow as tf
from typing import Any

from openxla.benchmark.models import model_interfaces, utils

DEFAULT_IMAGE_URL = "https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG"

# EfficientNet uses different image input sizes depending on the variant.
# The larger the model, the larger the image size.
MODEL_NAME_TO_INPUT_SIZE = {
    "efficientnetb0": 224,
    "efficientnetb1": 240,
    "efficientnetb2": 260,
    "efficientnetb3": 300,
    "efficientnetb4": 380,
    "efficientnetb5": 456,
    "efficientnetb6": 528,
    "efficientnetb7": 600,
}

MODEL_NAME_TO_CLASS = {
    "efficientnetb0": tf.keras.applications.efficientnet.EfficientNetB0,
    "efficientnetb1": tf.keras.applications.efficientnet.EfficientNetB1,
    "efficientnetb2": tf.keras.applications.efficientnet.EfficientNetB2,
    "efficientnetb3": tf.keras.applications.efficientnet.EfficientNetB3,
    "efficientnetb4": tf.keras.applications.efficientnet.EfficientNetB4,
    "efficientnetb5": tf.keras.applications.efficientnet.EfficientNetB5,
    "efficientnetb6": tf.keras.applications.efficientnet.EfficientNetB6,
    "efficientnetb7": tf.keras.applications.efficientnet.EfficientNetB7,
}


class EfficientNet(tf.Module, model_interfaces.InferenceModel):
  """See https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet for more information."""

  batch_size: int
  model: Any
  model_name: str
  input_size: int

  def __init__(self, batch_size: int, model_name: str):
    self.batch_size = batch_size
    self.model_name = model_name
    self.input_size = MODEL_NAME_TO_INPUT_SIZE[model_name]
    self.model = MODEL_NAME_TO_CLASS[model_name]()

  def generate_default_inputs(self) -> Image.Image:
    # TODO(#44): This should go away once we support different raw inputs.
    return utils.download_and_read_img(DEFAULT_IMAGE_URL)

  def preprocess(self, input_image: Image.Image) -> Any:
    image_size = MODEL_NAME_TO_INPUT_SIZE[self.model_name]
    resized_image = input_image.resize((image_size, image_size))
    tensor = tf.convert_to_tensor(resized_image)
    tensor = tf.image.convert_image_dtype(tensor, dtype=tf.float32)
    tensor = tf.keras.applications.efficientnet.preprocess_input(tensor)
    tensor = tf.expand_dims(tensor, 0)
    tensor = tf.tile(tensor, [self.batch_size, 1, 1, 1])
    return tensor

  @tf.function(jit_compile=True)
  def forward(self, input_tensor: Any) -> Any:
    return self.model(input_tensor, training=False)


def create_model(batch_size: int = 1,
                 model_name: str = "efficientb7",
                 **_unused_params) -> EfficientNet:
  """Configure and create a TF EfficientNet model instance.
  Args:
    batch_size: input batch size.
    model_name: The name of the EfficientNet variant to use.
      Supported variants include: efficientnetb[0, 1, 2, 3, 4, 6, 7].
  Returns:
    A TF EfficientNet model.
  """
  return EfficientNet(batch_size=batch_size, model_name=model_name)
