# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import flax

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration, GenerationConfig, T5Tokenizer
from typing import Any, List, Tuple

from openxla.benchmark.models.jax import jax_model_interface


class T5ForConditionalGeneration(jax_model_interface.JaxInferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/t5 for more information."""

  batch_size: int
  seq_len: int
  model: FlaxT5ForConditionalGeneration
  model_name: str
  tokenizer: T5Tokenizer
  tokenization_kwargs: dict[str, Any]
  generation_config: GenerationConfig

  def __init__(self, batch_size: int, seq_len: int, model_name: str,
               max_new_tokens: int):
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.model_name = model_name
    self.model = FlaxT5ForConditionalGeneration.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=self.seq_len)
    self.tokenization_kwargs = {
        "pad_to_multiple_of": self.seq_len,
        "padding": True,
        "return_tensors": "jax",
    }
    self.generation_config = GenerationConfig.from_pretrained(
        model_name, max_new_tokens=max_new_tokens)

  def generate_default_inputs(self) -> str:
    return "summarize: My friends are cool but they eat too many carbs."

  def preprocess(self, input_text: str) -> Any:
    batch_input = [input_text] * self.batch_size
    inputs = self.tokenizer(batch_input, **self.tokenization_kwargs)
    return inputs["input_ids"]

  def forward(self, input_ids: Any) -> Any:
    return self.model.generate(
        input_ids, generation_config=self.generation_config).sequences

  def postprocess(self, output: Any) -> List[str]:
    return self.tokenizer.batch_decode(output, skip_special_tokens=True)

  def apply(self, input_ids: Any) -> Any:
    outputs = self.model.module.apply(
        {'params': flax.core.freeze(self.model.params)},
        input_ids,
        method=self.model.generate)
    return outputs.sequences


def create_model(batch_size: int = 1,
                 seq_len: int = 512,
                 model_name: str = "t5-large",
                 max_new_tokens: int = 128,
                 **_unused_params) -> T5ForConditionalGeneration:
  """Configure and create a JAX T5 model instance with a language modeling head
  on top.
  
  Args:
    batch_size: input batch size.
    seq_len: input sequence length. Default to 512, which is the default in the
      T5 config.
    model_name: The name of the T5 variant to use. Supported variants include:
      t5-small, t5-base, t5-large, t5-3b and t5-11b.
  Returns:
    A JAX T5ForConditionalGeneration model.
  """
  return T5ForConditionalGeneration(batch_size=batch_size,
                                    seq_len=seq_len,
                                    model_name=model_name,
                                    max_new_tokens=max_new_tokens)
