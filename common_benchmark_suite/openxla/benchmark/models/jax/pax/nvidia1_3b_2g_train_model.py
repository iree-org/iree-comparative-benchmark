# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from etils import epath
import tempfile
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from paxml import partitioning
from paxml import programs
from paxml import trainer_lib
from paxml.tasks.lm.params import nvidia
from praxis import base_layer
from praxis import py_utils

from openxla.benchmark.models import model_interfaces

instantiate = base_layer.instantiate
NestedMap = py_utils.NestedMap


class NVIDIA1_3B2g(nvidia.NVIDIA1_3B):
  ICI_MESH_SHAPE = [2, 1, 1]


class NVIDIA1_3B2gTrain(model_interfaces.InferenceModel):
  batch_size: int

  def __init__(self, batch_size: int):
    self.batch_size = batch_size

    self.experiment_config = NVIDIA1_3B2g()
    self.task = instantiate(self.experiment_config.task())
    self.partitioner = partitioning.create_partitioner(self.task)
    prng_key = jax.random.PRNGKey(123)

    train_input_p = self.experiment_config.datasets()[0]
    train_input_p = self.partitioner.preprocess_input_config(train_input_p)
    self.train_input = instantiate(train_input_p)

    with tempfile.TemporaryDirectory() as d:
      job_log_dir = epath.Path(d)
      prng_key, setup_key = jax.random.split(prng_key)
      self.partitioner.setup(
          self.task,
          setup_key,
          train_inputs_shape_dtype=None,
          train_input_pipeline=self.train_input,
          job_log_dir=job_log_dir,
      )

      # Initialize the partitioned train state.
      prng_key, state_key = jax.random.split(prng_key)
      _, self.train_state, _ = self.partitioner.initialize_prng_key_and_train_state(
          state_key,
          train_state=None,
          checkpoint_type=None,
      )

      prng_key, train_prng_seed, eval_prng_seed = jax.random.split(prng_key, 3)
      self.train_program = programs.SingleTaskTrainProgram()
      self.train_program.setup(
          self.task,
          self.train_input,
          self.partitioner,
          job_log_dir,
          train_prng_seed,
          eval_prng_seed,
          init_step=0,
      )
      self.partitioned_prng_key = self.partitioner.preprocess_prng_key(prng_key)

  def generate_default_inputs(self) -> NestedMap:
    train_input_p = self.experiment_config.datasets()[0]
    train_input_p = self.partitioner.preprocess_input_config(train_input_p)
    train_input_p.input.batch_size = self.batch_size
    train_input = instantiate(train_input_p)
    train_batch = train_input.get_next()
    train_batch = self.partitioner.preprocess_inputs(
        train_input,
        train_batch,
        self.train_program.train_input_partition_spec(train_batch)
    )
    return train_batch

  def preprocess(self, raw_input: Any) -> Any:
    return raw_input

  def forward(self, inputs: NestedMap) -> Tuple[NestedMap]:
    step, train_state, step_fn_output = self.train_program.train_step(
        step=0,
        state=self.train_state,
        prng_key=self.partitioned_prng_key,
        inputs=inputs,
        static_args=trainer_lib.BaseStepFnStaticArgs(
            unpadded_global_batch_size=self.batch_size)
    )
    return (step_fn_output,)

  def postprocess(self, outputs: Any) -> Any:
    return outputs


def create_model(batch_size: int = 1,
                 **_unused_params) -> NVIDIA1_3B2gTrain:
  """Configure and create a NVIDIA1_3B model instance.

  Args:
    batch_size: input batch size.
  Returns:
    A NVIDIA1_3B model.
  """
  return NVIDIA1_3B2gTrain(batch_size=batch_size)
