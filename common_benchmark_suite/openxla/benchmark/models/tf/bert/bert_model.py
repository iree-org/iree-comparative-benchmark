# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import tensorflow as tf

from transformers import AutoTokenizer, BertTokenizer, TFBertModel
from typing import Any, Tuple

from openxla.benchmark.models import model_interfaces


class Bert(model_interfaces.InferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/bert for more information."""

  batch_size: int
  seq_len: int
  model: TFBertModel
  model_name: str
  tokenizer: BertTokenizer
  tokenization_kwargs: dict[str, Any]

  def __init__(self, batch_size: int, seq_len: int, model_name: str):
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.model_name = model_name
    self.model = TFBertModel.from_pretrained(model_name)
    self.tokenizer = BertTokenizer.from_pretrained(
        model_name, model_max_length=self.seq_len)
    self.tokenization_kwargs = {
        "pad_to_multiple_of": self.seq_len,
        "padding": True,
        "return_tensors": "tf",
    }

  def generate_default_inputs(self) -> Tuple[Any, ...]:
    input_text = ["a photo of a cat"] * self.batch_size
    return (input_text,)

  def preprocess(self, raw_input: Tuple[Any, ...]) -> Tuple[Any, ...]:
    input_text, = raw_input
    inputs = self.tokenizer(text=input_text, **self.tokenization_kwargs)
    return (inputs["input_ids"], inputs["attention_mask"])

  @tf.function(jit_compile=True)
  def forward(self, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    input_ids, attention_mask = inputs
    output = self.model(input_ids, attention_mask,
                        training=False).last_hidden_state
    return (output,)

  def postprocess(self, outputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    # No-op.
    return outputs


def create_model(batch_size: int = 1,
                 seq_len: int = 384,
                 model_name: str = "bert-large-uncased",
                 **_unused_params) -> Bert:
  """Configure and create a TF Bert model instance.

  Args:
    batch_size: input batch size.
    seq_len: input sequence length. Default to 384.
    model_name: The name of the T5 variant to use. Supported variants include:
      bert-base-[un]cased, bert-large-[un]cased, bert-base-chinese, etc.
  Returns:
    A TF Bert model.
  """
  return Bert(batch_size=batch_size, seq_len=seq_len, model_name=model_name)
