# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax.numpy as jnp
from transformers import BertTokenizer, FlaxBertModel
from typing import Any, Tuple

from openxla.benchmark.models.jax import model_interfaces


class Bert(model_interfaces.InferenceModel):

  def __init__(
      self,
      batch_size: int,
      seq_len: int,
      dtype: Any,
      model_name: str,
  ):
    self.model = FlaxBertModel.from_pretrained(model_name, dtype=dtype)
    if dtype == jnp.float16:
      self.model.params = self.model.to_fp16(self.model.params)
    elif dtype == jnp.bfloat16:
      self.model.params = self.model.to_bf16(self.model.params)
    else:
      raise ValueError(f"Unsupported data type '{dtype}'.")

    self.model_name = model_name
    self.batch_size = batch_size
    self.seq_len = seq_len

  def generate_default_inputs(self) -> Tuple[Any, ...]:
    input_text = ["a photo of a cat"] * self.batch_size
    return (input_text,)

  def preprocess(self, raw_input: Tuple[Any, ...]) -> Tuple[Any, ...]:
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    input_text, = raw_input
    inputs = tokenizer(text=input_text,
                       padding="max_length",
                       max_length=self.seq_len,
                       return_tensors="jax")
    return (inputs["input_ids"], inputs["attention_mask"])

  def forward(self, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    input_ids, attention_mask = inputs
    output = self.model(input_ids, attention_mask)[0]
    return (output,)

  def postprocess(self, outputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    # No-op.
    return outputs


DTYPE_MAP = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}


def create_model(batch_size: int = 1,
                 seq_len: int = 384,
                 data_type: str = "fp32",
                 model_name: str = "bert-large-uncased",
                 **_unused_params) -> Bert:
  """Configure and create a JAX Bert model instance.
  
  Args:
    batch_size: input batch size.
    seq_len: input sequence length.
    data_type: model data type. Supported options include: fp32, fp16, bf16.
    model_name: The name of the Bert variant to use. Supported variants include:
      bert-large-uncased.
  Returns:
    A JAX Bert model.
  """
  return Bert(batch_size=batch_size,
              seq_len=seq_len,
              dtype=DTYPE_MAP[data_type],
              model_name=model_name)
