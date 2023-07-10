# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torchvision.models
from typing import Any, Callable, Tuple

from openxla.benchmark.models import model_interfaces, utils

DEFAULT_IMAGE_URL = "https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG"


class ResNet(model_interfaces.InferenceModel, torch.nn.Module):
  """We use the ResNet variant listed in MLPerf here:
  https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection
  Input size 3x224x224 is used, as stated in the MLPerf Inference Rules:
  https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#41-benchmarks
  """

  model: torchvision.models.ResNet
  preprocessor: Callable[[Any], torch.Tensor]
  batch_size: int
  dtype: torch.dtype
  import_on_gpu: bool
  import_with_fx: bool

  def __init__(self, batch_size: int, dtype: torch.dtype, model_name: str,
               import_on_gpu: bool, import_with_fx: bool):
    super().__init__()

    if model_name == "torchvision/resnet50":
      weights = torchvision.models.ResNet50_Weights.DEFAULT
      model = torchvision.models.resnet50(weights=weights).to(dtype)
      preprocessor = weights.transforms()
    else:
      raise ValueError(f"Unsupported model: '{model_name}'")

    self.model = model
    self.preprocessor = preprocessor
    self.batch_size = batch_size
    self.dtype = dtype
    self.import_on_gpu = import_on_gpu
    self.import_with_fx = import_with_fx
    self.train(False)

  def generate_default_inputs(self) -> Tuple[Any, ...]:
    # TODO(#44): This should go away once we support different raw inputs.
    image = utils.download_and_read_img(DEFAULT_IMAGE_URL,
                                        width=224,
                                        height=224)
    return (image,)

  def preprocess(self, raw_inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    image, = raw_inputs
    tensor = self.preprocessor(image).to(dtype=self.dtype).unsqueeze(0)
    tensor = tensor.repeat(self.batch_size, 1, 1, 1)
    return (tensor,)

  def forward(self, input):
    return self.model(input)

  def postprocess(self, outputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    # No-op.
    return outputs


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


def create_model(batch_size: int = 1,
                 data_type: str = "fp32",
                 model_name: str = "torchvision/resnet50",
                 import_on_gpu: bool = False,
                 import_with_fx: bool = True,
                 **_unused_params) -> ResNet:
  """Configure and create a PyTorch ResNet model instance.

  Args:
    batch_size: input batch size
    data_type: model data type. Available options: `fp32`, `fp16`
    model_name: The name of the ResNet variant to use. Supported variants
      include: `torchvision/resnet50`
    import_on_gpu: Whether to generate model artifacts on a GPU.
    import_with_fx: Whether to lower to mlir using fx.
  Returns:
    A PyTorch ResNet model.
  """
  dtype = DTYPE_MAP.get(data_type)
  if dtype is None:
    raise ValueError(f"Unsupported data type: '{data_type}'.")

  return ResNet(batch_size=batch_size,
                dtype=dtype,
                model_name=model_name,
                import_on_gpu=import_on_gpu,
                import_with_fx=import_with_fx)
