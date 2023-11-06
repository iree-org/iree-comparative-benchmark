# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax
import jax.numpy as jnp
import numpy as np

from typing import Any, List, Optional, Tuple
from diffusers import FlaxStableDiffusionPipeline

from openxla.benchmark.models.jax import jax_model_interface


class StableDiffusionPipeline(jax_model_interface.JaxInferenceModel):
  """See https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.FlaxStableDiffusionPipeline for more information."""

  batch_size: int
  seq_len: int
  pipeline: FlaxStableDiffusionPipeline
  params: Any
  model_name: str
  num_inference_steps: int

  def __init__(self, batch_size: int, seq_len: int, dtype: Any, model_name: str,
               num_inference_steps: int):
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.model_name = model_name
    self.num_inference_steps = num_inference_steps
    self.guidance_scale = 7.5
    self.prng_seed = jax.random.PRNGKey(0)

    # We disable safety checker because this adds steps at the end of the pipeline that will increase latency and not be reflective of runtime.
    self.pipeline, self.params = FlaxStableDiffusionPipeline.from_pretrained(
        model_name, revision="flax", safety_checker=None)

    self.height = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
    self.width = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
    self.pipeline.tokenizer.model_max_length = seq_len

    if dtype == jnp.float32:
      # The original model is fp32.
      pass
    elif dtype == jnp.float16:
      self.params["unet"] = self.pipeline.unet.to_fp16(self.params["unet"])
      self.params["text_encoder"] = self.pipeline.text_encoder.to_fp16(
          self.params["text_encoder"])
      self.params["vae"] = self.pipeline.vae.to_fp16(self.params["vae"])
    elif dtype == jnp.bfloat16:
      self.params["unet"] = self.pipeline.unet.to_bf16(self.params["unet"])
      self.params["text_encoder"] = self.pipeline.text_encoder.to_bf16(
          self.params["text_encoder"])
      self.params["vae"] = self.pipeline.vae.to_bf16(self.params["vae"])
    else:
      raise ValueError(f"Unsupported data type '{dtype}'.")

  def generate_default_inputs(self) -> str:
    return "a photo of an astronaut riding a horse on mars"

  def preprocess(self, input_prompt: str) -> Any:
    batch_input = [input_prompt] * self.batch_size
    prompt_ids = self.pipeline.prepare_inputs(batch_input)
    return prompt_ids

  def forward(self, prompt_ids: Any) -> Any:
    return self.pipeline._generate(prompt_ids, self.params, self.prng_seed,
                                   self.num_inference_steps, self.height,
                                   self.width, self.guidance_scale)

  def postprocess(self, output: Any) -> Any:
    output = np.asarray(output)
    return self.pipeline.numpy_to_pil(
        np.asarray(output.reshape((self.batch_size,) + output.shape[-3:])))


DTYPE_MAP = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}


def create_model(batch_size: int = 1,
                 seq_len: int = 64,
                 model_name: str = "runwayml/stable-diffusion-v1-5",
                 data_type: str = "fp32",
                 num_inference_steps: int = 5,
                 **_unused_params) -> StableDiffusionPipeline:
  """Configure and create a JAX StableDiffusionPipeline instance.

  Args:
    batch_size: input batch size.
    seq_len: input sequence length.
    model_name: The name of the StableDiffusionPipeline variant to use.
    data_type: The data type of the models in the pipeline.
  Returns:
    A JAX StableDiffusionPipeline model.
  """
  return StableDiffusionPipeline(batch_size=batch_size,
                                 seq_len=seq_len,
                                 dtype=DTYPE_MAP[data_type],
                                 model_name=model_name,
                                 num_inference_steps=num_inference_steps)
