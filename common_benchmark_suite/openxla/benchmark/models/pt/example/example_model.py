# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torchvision.models as models
import PIL.Image

from openxla.benchmark.models import model_interfaces, utils

DEFAULT_IMAGE_URL = "https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG"


class ExampleModel(model_interfaces.InferenceModel, torch.nn.Module):

  def __init__(self, batch_size: int, dtype: torch.dtype):
    super().__init__()

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    self.model = models.mobilenet_v3_small(weights=weights).to(dtype)
    self.input_transfomer = weights.transforms()
    self.batch_size = batch_size
    self.dtype = dtype

  def generate_default_inputs(self) -> PIL.Image.Image:
    """Provides the default raw input."""
    return utils.download_and_read_img(DEFAULT_IMAGE_URL, width=224, height=224)

  def preprocess(self, image: PIL.Image.Image) -> torch.Tensor:
    """Preprocess the raw input into tensor."""
    tensor = self.input_transfomer(image).to(dtype=self.dtype).unsqueeze(0)
    dim_scalars = [1 for _ in range(tensor.dim())]
    dim_scalars[0] = self.batch_size
    return tensor.repeat(dim_scalars)

  def forward(self, batch_tensor: torch.Tensor) -> torch.Tensor:
    """Run the model."""
    return self.model(batch_tensor)

  def postprocess(self, batch_output: torch.Tensor) -> torch.Tensor:
    """Posprocess the tensor output."""
    # No-op.
    return batch_output


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


def create_model(batch_size: int = 1,
                 data_type: str = "fp32",
                 **_unused_params) -> ExampleModel:
  """Configure and create a PyTorch example model instance.

  Args:
    batch_size: input batch size
    data_type: model data type. Available options: `fp32`, `fp16`
  Returns:
    A PyTorch example model.
  """
  dtype = DTYPE_MAP.get(data_type)
  if dtype is None:
    raise ValueError(f"Unsupported data type: '{data_type}'.")

  return ExampleModel(batch_size=batch_size, dtype=dtype)
