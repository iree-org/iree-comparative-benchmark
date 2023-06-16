# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from transformers import AutoTokenizer, FlaxT5Model
from typing import Any, Tuple

from openxla.benchmark.models.jax_models import model_interfaces


class T5Large(model_interfaces.InferenceModel):
  """See https://huggingface.co/t5-large for more information."""

  batch_size: int
  seq_len: int
  model: FlaxT5Model

  def __init__(self, batch_size: int, seq_len: int):
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.model = FlaxT5Model.from_pretrained("t5-large", return_dict=True)

  def generate_inputs(self) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    tokenization_kwargs = {
        "pad_to_multiple_of": self.seq_len,
        "padding": True,
        "return_tensors": "jax",
    }

    text = "Studies have been shown that owning a dog is good for you"
    batched_text = [text] * self.batch_size
    encoded_input_ids = tokenizer(batched_text, **tokenization_kwargs).input_ids

    text = "Studies show that"
    batched_text = [text] * self.batch_size
    decoder_input_ids = tokenizer(batched_text, **tokenization_kwargs).input_ids
    # The HuggingFace documentation reports that _shift_right() exists for
    # `FlaxT5Model` but we get an attribute does not exist error. Disabling for
    # now.
    # decoder_input_ids = self.model._shift_right(decoder_input_ids)

    return (encoded_input_ids, decoder_input_ids)

  def forward(self, inputs: Tuple[Any, Any]) -> Any:
    encoder_input_ids, decoder_input_ids = inputs
    return self.model(
        input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
    )[0]


def create_model(batch_size: int = 1,
                 seq_len: int = 512,
                 **_unused_params) -> T5Large:
  """Configure and create a JAX T5 large model instance.
  
  Args:
    batch_size: input batch size.
    seq_len: input sequence length. Default to 512, which is the default in the
      T5 config.
  Returns:
    A JAX T5 large model.
  """
  return T5Large(batch_size=batch_size, seq_len=seq_len)
