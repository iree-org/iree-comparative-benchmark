# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Interfaces to interact with models."""

from abc import abstractmethod
import typing
from typing import Any, Protocol, Union


@typing.runtime_checkable
class InferenceModel(Protocol):
  """Interface to interact with a inference model."""

  @abstractmethod
  def generate_default_inputs(self) -> Union[tuple, Any]:
    """Returns default inputs in its raw form.

    Returns:
      A single raw input or a tuple for multi-value raw input.
    """
    ...

  @abstractmethod
  def preprocess(self, *raw_inputs: Any) -> Union[tuple, Any]:
    """Converts raw inputs into a form that is understandable by the model.

    It can have multiple parameters for multi-value raw input, e.g., encoder and
    decoder texts.

    Returns:
      A single preprocessed input or a tuple for multi-value preprocessed input.
    """
    ...

  @abstractmethod
  def forward(self, *preprocessed_inputs: Any) -> Union[tuple, Any]:
    """Model inference function.

    It can have multiple parameters for multi-value preprocessed input, e.g.,
    encoder and decoder input tensors.

    Returns:
      A single output or a tuple for multi-value output.
    """
    ...

  def postprocess(self, *outputs: Any) -> Union[tuple, Any]:
    """Converts raw outputs to a more human-understandable form.

    It can have multiple parameters for multi-value output.

    The default implementation is no-op.

    Returns:
      A single postprocessed output or a tuple for multi-value postprocessed
      output.
    """
    # Returns an object instead of tuple if there is only a single argument.
    if len(outputs) == 1:
      return outputs[0]
    return outputs
