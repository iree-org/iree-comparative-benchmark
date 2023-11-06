# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Types of the benchmark definitions."""

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelFrameworkType(Enum):
  """Type of framework a model is implemented in."""
  TF_V1 = "tensorflow_v1"
  TF_V2 = "tensorflow_v2"
  TFLITE = "tflite"
  PYTORCH = "pytorch"
  JAX = "jax"
  GGML = "ggml"


@dataclass(frozen=True)
class ModelImplementation:
  """Model implementation with configurable parameters."""
  # Friendly unique name.
  name: str
  # Tags that describe the model characteristics.
  tags: List[str]
  # Framework the model is implemented in.
  framework_type: ModelFrameworkType
  # Source of the model.
  source_info: str
  # Model module path
  module_path: str

  def __str__(self):
    return self.name


class ModelArtifactType(Enum):
  """Type of derived model artifact."""
  TF_SAVEDMODEL_V1 = "tf_savedmodel_v1"
  TF_SAVEDMODEL_V2 = "tf_savedmodel_v2"
  TFLITE_FP32 = "tflite_fp32"
  TFLITE_FP32_STABLEHLO = "tflite_fp32_stablehlo"
  TFLITE_FP16 = "tflite_fp16"
  TFLITE_DYNAMIC_RANGE_QUANT = "tflite_dynamic_range_quant"
  TFLITE_INT8 = "tflite_int8"
  XLA_HLO_DUMP = "xla_hlo_dump"
  STABLEHLO_MLIR = "stablehlo_mlir"
  LINALG_MLIR = "linalg_mlir"


@dataclass(frozen=True)
class Model:
  """A model with concrete parameters to initialize."""
  # Friendly unique name.
  name: str
  # Tags that describe the characteristics additional to `model_impl.tags`.
  tags: List[str]
  # Source model implementation.
  model_impl: ModelImplementation
  # Parameters to initialize the model, e.g., input batch size, sequence length.
  model_parameters: Dict[str, Any]
  # URLs to download exported models and generated test data.
  artifacts_dir_url: Optional[str] = None
  # Types of exported models.
  exported_model_types: List[ModelArtifactType] = dataclasses.field(
      default_factory=list)

  def __str__(self):
    return self.name


@dataclass(frozen=True)
class ModelTestData:
  """Raw input data to test models."""
  # Friendly name.
  name: str
  # URL to download the test data.
  source_url: str
  # Source information.
  source_info: str = ""
  # Tags that describe characteristics additional to `data_source.tags`.
  tags: List[str] = dataclasses.field(default_factory=list)

  def __str__(self):
    return self.name


@dataclass(frozen=True)
class DeviceSpec:
  """Device specification to run benchmarks."""
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
  # Unique name.
  name: str
  model: Model
  input_data: ModelTestData
  # Parameters for output verifiers if applicable.
  verify_parameters: Dict[str, Any] = dataclasses.field(default_factory=dict)

  @classmethod
  def build(cls,
            model: Model,
            input_data: ModelTestData,
            verify_parameters: Optional[Dict[str, Any]] = None):
    name = "/".join([
        "models",
        model.name,
        "inputs",
        input_data.name,
    ])
    verify_parameters = {} if verify_parameters is None else verify_parameters
    return cls(
        name=name,
        model=model,
        input_data=input_data,
        verify_parameters=verify_parameters,
    )
