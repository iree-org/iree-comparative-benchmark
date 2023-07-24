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
V = typing.TypeVar("V")


class InferenceModel(abc.ABC, typing.Generic[T, U, V]):
  """Interface to interact with a JAX inference model."""

  @abc.abstractmethod
  def generate_default_inputs(self) -> T:
    """Returns default inputs in its raw form."""
    pass

  @abc.abstractmethod
  def preprocess(self, raw_input: T) -> U:
    """Converts raw inputs into a form that is understandable by the model."""
    pass

  @abc.abstractmethod
  def forward(self, inputs: U) -> V:
    """Model inference function."""
    # TODO(#60): Refactor to not use a tuple as inputs.
    pass

  @abc.abstractmethod
  def postprocess(self, outputs: V) -> Any:
    """Converts raw outputs to a more human-understandable form."""
    pass
