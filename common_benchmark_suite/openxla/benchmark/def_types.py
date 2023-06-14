# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Types of the benchmark definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class ModelFrameworkType(Enum):
  """Type of framework a model is implemented in."""
  TF_V1 = "tensorflow_v1"
  TF_V2 = "tensorflow_v2"
  PYTORCH = "pytorch"
  JAX = "jax"


class ModelDataType(Enum):
  """Model data type used in the model."""
  FP32 = "fp32"
  FP16 = "fp16"
  BF16 = "bf16"
  INT8 = "int8"
  UINT8 = "uint8"


@dataclass(frozen=True)
class ModelImplementation:
  """Model implementation with configurable parameters."""
  id: str
  # Friendly unique name.
  name: str
  # Tags that describe the model characteristics.
  tags: List[str]
  # Framework the model is implemented in.
  framework_type: ModelFrameworkType
  # Data type used in the model.
  data_type: ModelDataType
  # Source of the model.
  source_info: str

  def __str__(self):
    return self.name


class ModelArtifactType(Enum):
  """Type of derived model artifact."""
  TF_SAVEDMODEL_V1 = "tf_savedmodel_v1"
  TF_SAVEDMODEL_V2 = "tf_savedmodel_v2"
  XLA_HLO_DUMP = "xla_hlo_dump"
  STABLEHLO = "stablehlo"


@dataclass(frozen=True)
class Model:
  """A model with concrete parameters to initialize."""
  id: str
  # Friendly unique name.
  name: str
  # Tags that describe the characteristics additional to model_impl.tags.
  tags: List[str]
  # Source model implementation.
  model_impl: ModelImplementation
  # Parameters to initialize the model, e.g., input batch size, sequence length.
  model_parameters: Dict[str, Any]
  # URLs to download the derived models of the initialized model.
  artifact_sources: Dict[ModelArtifactType, str]

  def __str__(self):
    return self.name


class ModelTestDataFormat(Enum):
  """Model input or output data format."""
  # Pack of npy tensor files.
  NUMPY_TENSORS = "npy_tensors"


@dataclass(frozen=True)
class ModelTestData:
  """Model input or expected output data."""
  id: str
  # Friendly name.
  name: str
  # Tags that describe the data characteristics.
  tags: List[str]
  # Information on where the data was originally sourced.
  source_info: str
  # URLs to download the data in multiple formats.
  data_sources: Dict[ModelTestDataFormat, str]
  # Parameters for output verifiers if applicable.
  output_verify_params: Dict[ModelTestDataFormat, Dict[str, Any]]

  def __str__(self):
    return self.name


@dataclass(frozen=True)
class DeviceSpec:
  """Device specification to run benchmarks."""
  id: str
  # Friendly unique name.
  name: str
  # Describes the host that runs the runtime and talks to the accelerator.
  # E.g., GCP
  host_type: str
  # E.g., c2-standard-60
  host_model: str
  # E.g., linux-x86_64
  host_environment: str

  # Describes the target accelerator (can be the same as the host for CPU
  # benchmarks).
  # E.g., cpu, gpu, my-custom-accelerator
  accelerator_type: str
  # E.g., nvidia-a100-40g
  accelerator_model: str
  # E.g., intel-cascadelake, nvidia-ampere
  accelerator_architecture: str
  # E.g., "num_of_gpus": 4, "cpu_mask": "0-3"
  accelerator_attributes: Dict[str, Any]

  def __str__(self):
    return self.name


@dataclass(frozen=True)
class BenchmarkCase:
  """A benchmark case."""
  model: Model
  input_data: ModelTestData
  expected_output: ModelTestData
  target_device: DeviceSpec
