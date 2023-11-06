# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import abstractmethod
from openxla.benchmark.models import model_interfaces, utils
from typing import Any


class JaxInferenceModel(model_interfaces.InferenceModel):

  def apply(self, *preprocessed_inputs: Any) -> model_interfaces.U:
    """Evaluates the JAX module given a set of parameters (which is never
    stored with the model) and input. This is used to generate a Tensorflow
    function.

    Returns:
      A output object.
    """
    ...
