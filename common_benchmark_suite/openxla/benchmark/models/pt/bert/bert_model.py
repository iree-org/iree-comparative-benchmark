# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from transformers import BertTokenizer, BertModel
from typing import Any, Dict, Tuple

from openxla.benchmark.models import model_interfaces


class Bert(model_interfaces.InferenceModel, torch.nn.Module):
  """See https://huggingface.co/docs/transformers/model_doc/bert for more
  information.
  """

  batch_size: int
  seq_len: int
  dtype: torch.dtype
  model: BertModel
  model_name: str
  tokenizer: BertTokenizer
  tokenization_kwargs: Dict[str, Any]

  def __init__(
      self,
      batch_size: int,
      seq_len: int,
      dtype: torch.dtype,
      model_name: str,
  ):
    super().__init__()

    self.batch_size = batch_size
    self.seq_len = seq_len
    self.dtype = dtype
    self.model_name = model_name
    self.model = BertModel.from_pretrained(model_name)
    self.tokenizer = BertTokenizer.from_pretrained(
        model_name, model_max_length=self.seq_len, dtype=dtype)
    self.tokenization_kwargs = {
        "pad_to_multiple_of": self.seq_len,
        "padding": True,
        "return_tensors": "pt",
    }

  def generate_default_inputs(self) -> Tuple[Any, ...]:
    input_text = ["a photo of a cat"] * self.batch_size
    return (input_text,)

  def preprocess(self, raw_input: Tuple[Any, ...]) -> Tuple[Any, ...]:
    input_text, = raw_input
    inputs = self.tokenizer(text=input_text, **self.tokenization_kwargs)
    return (inputs["input_ids"], inputs["attention_mask"])

  def forward(self, input_ids, attention_mask):
    return self.model(input_ids, attention_mask)[0]

  def postprocess(self, outputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    # No-op.
    return outputs


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


def create_model(batch_size: int = 1,
                 seq_len: int = 384,
                 data_type: str = "fp32",
                 model_name: str = "bert-large-uncased",
                 **_unused_params) -> Bert:
  """Configure and create a PyTorch Bert model instance.

  Args:
    batch_size: input batch size.
    seq_len: input sequence length. Default to 384.
    data_type: model data type.
    model_name: The name of the T5 variant to use. Supported variants include:
      bert-base-[un]cased, bert-large-[un]cased, bert-base-chinese, etc.
  Returns:
    A PyTorch Bert model.
  """
  dtype = DTYPE_MAP.get(data_type)
  if dtype is None:
    raise ValueError(f"Unsupported data type: '{data_type}'.")

  model = Bert(batch_size=batch_size,
               seq_len=seq_len,
               dtype=dtype,
               model_name=model_name)

  return model.to(dtype=dtype)
