# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax.numpy as jnp
from transformers import AutoTokenizer, GPT2Tokenizer, FlaxGPT2LMHeadModel
from typing import Any, Tuple

from openxla.benchmark.models import model_interfaces


class GPT2LMHead(model_interfaces.InferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/gpt2 for more information."""

  batch_size: int
  seq_len: int
  model: FlaxGPT2LMHeadModel
  model_name: str
  tokenizer: GPT2Tokenizer
  tokenization_kwargs: dict[str, Any]

  def __init__(
      self,
      batch_size: int,
      seq_len: int,
      dtype: Any,
      model_name: str,
  ):
    if dtype == jnp.int8:
      self.model = FlaxGPT2LMHeadModel.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
    else:
      self.model = FlaxGPT2LMHeadModel.from_pretrained(model_name, dtype=dtype)

    if dtype == jnp.float32 or dtype == jnp.int8:
      # The model is already in the requested dtype.
      pass
    elif dtype == jnp.float16:
      self.model.params = self.model.to_fp16(self.model.params)
    elif dtype == jnp.bfloat16:
      self.model.params = self.model.to_bf16(self.model.params)
    else:
      raise ValueError(f"Unsupported data type '{dtype}'.")

    self.model_name = model_name
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=self.seq_len)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenization_kwargs = {
        "pad_to_multiple_of": self.seq_len,
        "padding": True,
        "return_tensors": "jax",
    }

  def generate_default_inputs(self) -> Tuple[Any, ...]:
    input_text = ["a photo of a cat"] * self.batch_size
    return (input_text,)

  def preprocess(self, raw_input: Tuple[Any, ...]) -> Tuple[Any, ...]:
    input_text, = raw_input
    inputs = self.tokenizer(text=input_text, **self.tokenization_kwargs)
    return (inputs["input_ids"], inputs["attention_mask"])

  def forward(self, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    input_ids, attention_mask = inputs
    output = self.model(input_ids, attention_mask).logits
    return (output,)

  def postprocess(self, outputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    # No-op.
    return outputs


DTYPE_MAP = {
    "int8": jnp.int8,
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}


def create_model(batch_size: int = 1,
                 seq_len: int = 512,
                 data_type: str = "fp32",
                 model_name: str = "gpt2",
                 **_unused_params) -> GPT2LMHead:
  """Configure and create a JAX GPT2LMHead model instance.
  
  Args:
    batch_size: input batch size.
    seq_len: input sequence length.
    data_type: model data type. Supported options include: fp32, fp16, bf16.
    model_name: The name of the GPT2 variant to use. Supported variants include:
      gpt2.
  Returns:
    A JAX GPT2LMHead model.
  """
  return GPT2LMHead(batch_size=batch_size,
                    seq_len=seq_len,
                    dtype=DTYPE_MAP[data_type],
                    model_name=model_name)
