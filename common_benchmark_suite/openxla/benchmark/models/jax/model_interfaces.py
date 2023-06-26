# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Interfaces to interact with JAX models."""

import abc
import typing
from typing import Any

T = typing.TypeVar("T")
U = typing.TypeVar("U")


class InferenceModel(abc.ABC, typing.Generic[T, U]):
  """Interface to interact with a JAX inference model."""

  @abc.abstractmethod
  def generate_default_inputs(self) -> Any:
    """Returns default inputs in its raw form."""
    pass

  @abc.abstractmethod
  def preprocess(self, raw_input: Any) -> T:
    """Converts raw inputs into a form that is understandable by the model."""
    pass

  @abc.abstractmethod
  def forward(self, inputs: T) -> U:
    """Model inference function."""
    pass

  @abc.abstractmethod
  def postprocess(self, outputs: U) -> Any:
    """Converts raw outputs to a more human-understandable form."""
    pass
