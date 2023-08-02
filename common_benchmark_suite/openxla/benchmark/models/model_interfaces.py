# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Interfaces to interact with models."""

from abc import abstractmethod
import typing
from typing import Any, Protocol, Union

T = typing.TypeVar("T")
U = typing.TypeVar("U")


@typing.runtime_checkable
class InferenceModel(Protocol[T, U]):
  """Interface to interact with a inference model."""

  @abstractmethod
  def generate_default_inputs(self) -> T:
    """Returns default inputs in its raw form.

    Returns:
      A raw input object, can be a tuple for multi-value raw data.
    """
    ...

  @abstractmethod
  def preprocess(self, raw_input_obj: T) -> Union[tuple, Any]:
    """Converts the raw input object into a form that is understandable by the
    model.

    Due to the compatibility of many ML frameworks, when returning a tuple, it
    will be treated as an argument list for the forward method.

    Returns:
      A single preprocessed input or a tuple for multi-value preprocessed input.
    """
    ...

  @abstractmethod
  def forward(self, *preprocessed_inputs: Any) -> U:
    """Model inference function.

    It can have multiple parameters for multi-value preprocessed input, e.g.,
    encoder and decoder input tensors.

    Returns:
      A output object.
    """
    ...

  def postprocess(self, output: U) -> Any:
    """Converts the output object to a more human-understandable form.

    The default implementation is no-op.

    Returns:
      A postprocessed output.
    """
    return output
