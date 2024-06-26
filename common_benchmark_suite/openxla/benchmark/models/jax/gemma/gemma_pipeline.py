# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax.numpy as jnp

from transformers import AutoTokenizer, GemmaTokenizer, FlaxPreTrainedModel, FlaxGemmaForCausalLM, GenerationConfig
from typing import Any, List, Tuple

from openxla.benchmark.models.jax import jax_model_interface


class GemmaPipeline(jax_model_interface.JaxInferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/gemma for more information."""

  batch_size: int
  seq_len: int
  model: FlaxGemmaForCausalLM
  params: dict[str, Any]
  model_name: str
  tokenizer: GemmaTokenizer
  tokenization_kwargs: dict[str, Any]

  def __init__(
      self,
      batch_size: int,
      dtype: Any,
      seq_len: int,
      max_new_tokens: int,
      model_name: str,
  ):
    self.model, self.params = FlaxGemmaForCausalLM.from_pretrained(
        model_name, revision="flax", _do_init=False)

    if dtype == jnp.float32:
      self.params = self.model.to_fp32(self.params)
    elif dtype == jnp.float16:
      self.params = self.model.to_fp16(self.params)
    elif dtype == jnp.bfloat16:
      self.params = self.model.to_bf16(self.params)
    else:
      raise ValueError(f"Unsupported data type '{dtype}'.")

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
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True)

  def generate_default_inputs(self) -> str:
    return "Once upon a time"

  def preprocess(self, input_text: str) -> Tuple[Any,]:
    batch_input_text = [input_text] * self.batch_size
    inputs = self.tokenizer(text=batch_input_text, **self.tokenization_kwargs)
    return (inputs["input_ids"],)

  def forward(self, input_text: Any) -> Any:
    output = self.model.generate(input_text,
                                 params=self.params,
                                 generation_config=self.generation_config)
    print(f"output: {output}")

  def postprocess(self, output: Any) -> List[str]:
    return self.tokenizer.batch_decode(output, skip_special_tokens=True)

  def apply(self, input_text: Any) -> Any:
    raise Exception("Not implemented.")


DTYPE_MAP = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}


def create_model(batch_size: int = 1,
                 data_type: str = "fp32",
                 seq_len: int = 1024,
                 max_new_tokens: int = 256,
                 model_name: str = "google/gemma-2b-it",
                 **_unused_params) -> GemmaPipeline:
  """Configure and create a JAX Gemma pipeline.
  Args:
    batch_size: input batch size.
    seq_len: input sequence length.
    max_new_tokens: the maximum number of new tokens to generate.
    model_name: The name of the Gemma variant to use. Supported variants include:
      google/gemma-2b-it, google/gemma-7b-it.
  Returns:
    A JAX GemmaPipeline.
  """
  return GemmaPipeline(batch_size=batch_size,
                       dtype=DTYPE_MAP[data_type],
                       seq_len=seq_len,
                       max_new_tokens=max_new_tokens,
                       model_name=model_name)
