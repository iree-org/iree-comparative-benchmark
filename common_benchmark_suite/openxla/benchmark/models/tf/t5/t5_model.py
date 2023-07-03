# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import tensorflow as tf

from transformers import AutoTokenizer, TFT5Model, T5Tokenizer
from typing import Any, Tuple

from openxla.benchmark.models import model_interfaces


class T5(model_interfaces.InferenceModel, tf.Module):
  """See https://huggingface.co/docs/transformers/model_doc/t5 for more information."""

  batch_size: int
  seq_len: int
  model: TFT5Model
  model_name: str
  tokenizer: T5Tokenizer
  tokenization_kwargs: dict[str, Any]

  def __init__(self, batch_size: int, seq_len: int, model_name: str):
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.model_name = model_name
    self.model = TFT5Model.from_pretrained(model_name, return_dict=True)
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=self.seq_len)
    self.tokenization_kwargs = {
        "pad_to_multiple_of": self.seq_len,
        "padding": True,
        "return_tensors": "tf",
    }

  def generate_default_inputs(self) -> Tuple[Any, ...]:
    encoder_text = "Studies have been shown that owning a dog is good for you"
    decoder_text = "Studies show that"
    return ([encoder_text] * self.batch_size, [decoder_text] * self.batch_size)

  def preprocess(self, raw_input: Tuple[Any, ...]) -> Tuple[Any, ...]:
    encoder_text, decoder_text = raw_input
    encoder_input_ids = self.tokenizer(encoder_text,
                                       **self.tokenization_kwargs).input_ids

    decoder_input_ids = self.tokenizer(decoder_text,
                                       **self.tokenization_kwargs).input_ids
    decoder_input_ids = self.model._shift_right(decoder_input_ids)

    return (encoder_input_ids, decoder_input_ids)

  @tf.function(jit_compile=True)
  def forward(self, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    encoder_input_ids, decoder_input_ids = inputs
    output = self.model(
        input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
    ).last_hidden_state
    return (output,)

  @tf.function(jit_compile=True)
  def forward_sm(self, input_ids, decoder_input_ids):
    """ Provides an inference interface amenable to generating a TF SavedModel
    and lowering to MLIR.
    """
    return self.model(input_ids,
                      decoder_input_ids=decoder_input_ids).last_hidden_state

  def postprocess(self, outputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    # No-op.
    return outputs


def create_model(batch_size: int = 1,
                 seq_len: int = 512,
                 model_name: str = "t5-large",
                 **_unused_params) -> T5:
  """Configure and create a TF T5 model instance.

  Args:
    batch_size: input batch size.
    seq_len: input sequence length. Default to 512, which is the default in the
      T5 config.
    model_name: The name of the T5 variant to use. Supported variants include:
      t5-small, t5-base, t5-large, t5-3b and t5-11b.
  Returns:
    A TF T5 model.
  """
  return T5(batch_size=batch_size, seq_len=seq_len, model_name=model_name)
