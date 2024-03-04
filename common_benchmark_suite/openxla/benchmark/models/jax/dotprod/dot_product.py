# Copyright 2024 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import flax, jax
from flax import linen as nn
from PIL import Image
import jax.numpy as jnp
import jax.random as random
from transformers import AutoImageProcessor, FlaxViTForImageClassification
from typing import Any, List, Tuple

from openxla.benchmark.models import utils
from openxla.benchmark.models.jax import jax_model_interface


class DotProductModule(nn.Module):
  """A Flax module that includes a single dot product.

  The RHS is fixed and can be thought of as weights.
  The LHS values are dynamic and can be thought of as activations.
  """

  static_rhs: jnp.ndarray

  def __call__(self, lhs):
    # If operands are integer, use a return type of int32. Otherwise, use the
    # same floating point type.
    return_type = jnp.int32
    if jnp.issubdtype(self.static_rhs.dtype, jnp.floating):
      assert (self.static_rhs.dtype == lhs.dtype)
      return_type = self.static_rhs.dtype

    if self.static_rhs.dtype == jnp.int4:
      # jnp.dot does not take int4 arrays as input. We cast it to int8 and IREE
      # compiler will optimize the constant to be int4 type.
      self.static_rhs = self.static_rhs.astype(jnp.int8)

    return jnp.dot(lhs, self.static_rhs, preferred_element_type=return_type)


class DotProduct(jax_model_interface.JaxInferenceModel):
  """A Flax module that includes a single dot product."""

  model: DotProductModule
  model_name: str

  def __init__(self, model_name: str, lhs_shape: Tuple[int, int],
               lhs_type: jnp.dtype, rhs_shape: Tuple[int,
                                                     int], rhs_type: jnp.dtype):
    self.model_name = model_name
    self.lhs_shape = lhs_shape
    self.lhs_type = lhs_type

    self.key = random.PRNGKey(0)
    rhs = self._generate_random(self.key, rhs_shape, rhs_type)
    self.model = DotProductModule(static_rhs=rhs)

  def _generate_random(self, key: Any, shape: Tuple[int, int],
                       type: jnp.dtype) -> jnp.ndarray:

    def get_min_max(dtype: jnp.dtype) -> Tuple[Any, Any]:
      if dtype == jnp.int8:
        return (-127, 127)
      elif dtype == jnp.int4:
        return (-7, 7)
      elif dtype == jnp.int32:
        # Use arbitrarily big numbers.
        return (-65536, 65536)
      else:  # floating point.
        return (-1.0, 1.0)

    min, max = get_min_max(type)

    if jnp.issubdtype(type, jnp.floating):
      return random.uniform(self.key,
                            shape=shape,
                            minval=min,
                            maxval=max,
                            dtype=type)

    if type == jnp.int4:
      # `randint` does not accept int4 as a dtype so we generate int8 values
      # and later cast it back to int4.
      randint_dtype = jnp.int8
    else:
      randint_dtype = type

    random_array = random.randint(self.key,
                                  shape=shape,
                                  minval=min,
                                  maxval=max,
                                  dtype=randint_dtype)
    return random_array.astype(type)

  def generate_default_inputs(self) -> Any:
    return self._generate_random(self.key, self.lhs_shape, self.lhs_type)

  def preprocess(self, inputs) -> Any:
    return inputs

  def apply(self, lhs: Any) -> Any:
    params = self.model.init(self.key, lhs)
    output = self.model.apply(params, lhs)
    return output

  def forward(self, lhs: Any) -> Any:
    return self.model(lhs)


DTYPE_MAP = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
    "int32": jnp.int32,
    "int8": jnp.int8,
    "int4": jnp.int4,
}


def create_model(model_name: str = "dotprod",
                 lhs_shape: Tuple[int, int] = (1, 256),
                 lhs_type: str = "int8",
                 rhs_shape: Tuple[int, int] = (256, 2048),
                 rhs_type: str = "int8",
                 **_unused_params) -> DotProduct:
  """Configure and create a JAX dot product.

  Args:
    model_name: The name of the model.
    lhs_shape: The shape of the left hand side of the dot product.
    lhs_type: The type of the left hand side of the dot product.
    rhs_shape: The shape of the right hand side of the dot product.
    rhs_type: The type of the right hand side of the dot product.
  Returns:
    A JAX DotProduct model.
  """
  return DotProduct(model_name=model_name,
                    lhs_shape=lhs_shape,
                    lhs_type=DTYPE_MAP[lhs_type],
                    rhs_shape=rhs_shape,
                    rhs_type=DTYPE_MAP[rhs_type])
