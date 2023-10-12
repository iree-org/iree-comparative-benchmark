# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from openxla.benchmark.models import model_interfaces, utils
from transformers import AutoTokenizer, GPT2Tokenizer, FlaxGPT2LMHeadModel, GenerationConfig
from typing import Any, List, Tuple


class GPT2Pipeline(model_interfaces.InferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/gpt2 for more information.
  This version of GPT2 is configured to match the GGML implementation in https://github.com/ggerganov/ggml/blob/7b5fcf5f2a676e6d7c018c6a15dcb6338d9b2a38/examples/gpt-2/main.cpp.
  """

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
      model_name: str,
  ):
    self.model = FlaxGPT2LMHeadModel.from_pretrained(model_name)
    self.model_name = model_name
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=self.seq_len,
        padding_side="left",
    )
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenization_kwargs = {
        "return_tensors": "jax",
    }

    self.generation_config = GenerationConfig.from_pretrained(
        model_name,
        max_new_tokens=200,
        do_sample=True,
        use_cache=True,
        temperature=0.9,
        top_k=40,
        top_p=0.9)

  def generate_default_inputs(self) -> str:
    return "Once upon a time"

  def preprocess(self, input_text: str) -> Tuple[Any,]:
    batch_input_text = [input_text] * self.batch_size
    inputs = self.tokenizer(text=batch_input_text, **self.tokenization_kwargs)
    return (inputs["input_ids"],)

  def forward(self, input_text: Any) -> Any:
    return self.model.generate(input_text,
                               generation_config=self.generation_config)

  def postprocess(self, output: Any) -> List[str]:
    return self.tokenizer.batch_decode(output, skip_special_tokens=True)


def create_model(batch_size: int = 1,
                 seq_len: int = 1024,
                 model_name: str = "gpt2",
                 **_unused_params) -> GPT2Pipeline:
  """Configure and create a JAX GPT2LMHead pipeline.
  Args:
    batch_size: input batch size.
    seq_len: input sequence length.
    model_name: The name of the GPT2 variant to use. Supported variants include:
      gpt2 (117M params), gpt2-medium (345M params), gpt2-large (774M params),
      gpt2-xl (1558M params).
  Returns:
    A JAX GPT2Pipeline.
  """
  return GPT2Pipeline(batch_size=batch_size,
                      seq_len=seq_len,
                      model_name=model_name)
