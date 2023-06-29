# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration, T5Tokenizer
from typing import Any, Tuple

from openxla.benchmark.models.jax import model_interfaces


class T5ForConditionalGeneration(model_interfaces.InferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/t5 for more information."""

  batch_size: int
  seq_len: int
  model: FlaxT5ForConditionalGeneration
  model_name: str
  tokenizer: T5Tokenizer
  tokenization_kwargs: dict[str, Any]

  def __init__(self, batch_size: int, seq_len: int, model_name: str):
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

  def generate_default_inputs(self) -> Tuple[Any, ...]:
    text = "summarize: My friends are cool but they eat too many carbs."
    return ([text] * self.batch_size,)

  def preprocess(self, raw_input: Tuple[Any, ...]) -> Tuple[Any, ...]:
    inputs = self.tokenizer(*raw_input, **self.tokenization_kwargs)
    return (inputs["input_ids"], inputs["attention_mask"])

  def forward(
      self,
      inputs: Tuple[Any, ...],
      max_new_tokens: int = 50,
  ) -> Tuple[Any, ...]:
    input_ids, attention_mask = inputs
    # Calls `generate()` which takes care of running the encoder and decoder
    # auto-regressively.
    output = self.model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 max_new_tokens=max_new_tokens)
    return (output,)

  def postprocess(self, outputs: Any) -> Any:
    return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def create_model(batch_size: int = 1,
                 seq_len: int = 512,
                 model_name: str = "t5-large",
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
                                    model_name=model_name)
