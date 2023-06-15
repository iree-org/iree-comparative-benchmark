# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Interfaces to interact with JAX models."""

import abc
import typing

T = typing.TypeVar("T")
U = typing.TypeVar("U")


class InferenceModel(abc.ABC, typing.Generic[T, U]):
  """Interface to interact with a JAX inference model."""

  @abc.abstractmethod
  def generate_inputs(self) -> T:
    """Transforms raw input data into model inputs."""
    pass

  @abc.abstractmethod
  def forward(self, inputs: T) -> U:
    """Model inference function."""
    pass
