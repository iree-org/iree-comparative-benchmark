# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax.numpy as jnp
from transformers import AutoTokenizer, BertTokenizer, FlaxBertModel
from typing import Any, Tuple

from openxla.benchmark.models import model_interfaces


class Bert(model_interfaces.InferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/bert for more information."""

  batch_size: int
  seq_len: int
  model: FlaxBertModel
  model_name: str
  tokenizer: BertTokenizer
  tokenization_kwargs: dict[str, Any]

  def __init__(
      self,
      batch_size: int,
      seq_len: int,
      dtype: Any,
      model_name: str,
  ):
    model: FlaxBertModel = FlaxBertModel.from_pretrained(
        model_name,
        dtype=dtype,
    )
    if dtype == jnp.float32:
      # The original model is fp32.
      pass
    elif dtype == jnp.float16:
      model.params = model.to_fp16(self.model.params)
    elif dtype == jnp.bfloat16:
      model.params = model.to_bf16(self.model.params)
    else:
      raise ValueError(f"Unsupported data type '{dtype}'.")

    self.model = model
    self.model_name = model_name
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=self.seq_len)
    self.tokenization_kwargs = {
        "pad_to_multiple_of": self.seq_len,
        "padding": True,
        "return_tensors": "jax",
    }

  def generate_default_inputs(self) -> str:
    return "a photo of a cat"

  def preprocess(self, input_text: str) -> Tuple[Any, Any]:
    batch_input_text = [input_text] * self.batch_size
    inputs = self.tokenizer(text=batch_input_text, **self.tokenization_kwargs)
    return (inputs["input_ids"], inputs["attention_mask"])

  def forward(self, input_ids: Any, attention_mask: Any) -> Any:
    return self.model(input_ids, attention_mask).last_hidden_state


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
