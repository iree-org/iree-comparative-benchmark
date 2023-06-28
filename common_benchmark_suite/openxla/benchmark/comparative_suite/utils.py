# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Common utilities to define comparative benchmarks."""

import string
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union, Sequence

from openxla.benchmark import def_types


@dataclass(frozen=True)
class TemplateFunc:
  """Function to return an arbitrary object with substitutions."""
  func: Callable[..., Any]


# Module path to find all model implementations.
MODELS_MODULE_PATH = "openxla.benchmark.models"

# Constants and functions help build batch templates.
BATCH_ID = lambda model_id: string.Template(model_id + "-batch${batch_size}")
BATCH_NAME = lambda name: string.Template(name + "_BATCH${batch_size}")
BATCH_TAG = string.Template("batch-${batch_size}")
BATCH_SIZE_PARAM = TemplateFunc(func=lambda batch_size: batch_size)
BATCH_TENSOR_DIMS = lambda dims: string.Template("${batch_size}x" + dims)


@dataclass(frozen=True)
class ModelArtifactTemplate:
  """Template of def_types.ModelArtifact."""
  artifact_type: def_types.ModelArtifactType
  source_url: string.Template


@dataclass(frozen=True)
class ModelTemplate:
  """Template of def_types.Model."""
  id: string.Template
  name: string.Template
  tags: List[Union[str, string.Template]]
  model_impl: def_types.ModelImplementation
  model_parameters: Dict[str, Any]
  artifacts: Dict[def_types.ModelArtifactType, ModelArtifactTemplate]


def _substitute_template(obj: Any, **substitutions) -> Any:
  """Recursively substitute `string.Template` in an object.

  Supports traversing in list and dictionary.
  """

  if isinstance(obj, TemplateFunc):
    return obj.func(**substitutions)

  if obj is None or any(
      isinstance(obj, primitive_type)
      for primitive_type in [int, float, str, bool]):
    return obj
  if isinstance(obj, string.Template):
    return obj.substitute(**substitutions)
  if isinstance(obj, list):
    return [_substitute_template(value, **substitutions) for value in obj]
  if isinstance(obj, dict):
    return dict((key, _substitute_template(value, **substitutions))
                for key, value in obj.items())

  raise ValueError(f"Unsupported object type: {type(obj)} of {obj}.")


def build_batch_models(
    template: ModelTemplate,
    batch_sizes: Sequence[int]) -> Dict[int, def_types.Model]:
  """Build model with batch sizes by replacing `${batch_size}` in the template.

  Args:
    template: model template with "${batch_size}" to replace.
    batch_sizes: list of batch sizes to generate.

  Returns:
    Map of batch size to model.
  """

  batch_models = {}
  for batch_size in batch_sizes:
    substitute = lambda obj: _substitute_template(obj=obj,
                                                  batch_size=batch_size)
    artifacts = {}
    for artifact_type, artifact_template in template.artifacts.items():
      artifact = def_types.ModelArtifact(
          artifact_type=artifact_template.artifact_type,
          source_url=substitute(artifact_template.source_url))
      artifacts[artifact_type] = artifact

    model = def_types.Model(id=substitute(template.id),
                            name=substitute(template.name),
                            tags=substitute(template.tags),
                            model_impl=template.model_impl,
                            model_parameters=substitute(
                                template.model_parameters),
                            artifacts=artifacts)
    batch_models[batch_size] = model

  return batch_models


@dataclass(frozen=True)
class ModelTestDataArtifactTemplate:
  """Template of def_types.ModelTestDataArtifact."""
  data_format: def_types.ModelTestDataFormat
  data_parameters: Dict[str, Any]
  verify_parameters: Dict[str, Any]
  source_url: string.Template


@dataclass(frozen=True)
class ModelTestDataTemplate:
  """Template of def_types.ModelTestData."""
  id: string.Template
  name: string.Template
  tags: List[Union[str, string.Template]]
  source_info: str
  artifacts: Dict[def_types.ModelTestDataFormat, ModelTestDataArtifactTemplate]


def build_batch_model_test_data(
    template: ModelTestDataTemplate,
    batch_sizes: Sequence[int]) -> Dict[int, def_types.ModelTestData]:
  """Build model test data with batch sizes by replacing `${batch_size}` in the
  template.

  Args:
    template: model test data template with "${batch_size}" to replace.
    batch_sizes: list of batch sizes to generate.

  Returns:
    Map of batch size to model test data.
  """

  batch_test_data = {}
  for batch_size in batch_sizes:
    substitute = lambda obj: _substitute_template(obj=obj,
                                                  batch_size=batch_size)
    artifacts = {}
    for artifact_type, artifact_template in template.artifacts.items():
      artifact = def_types.ModelTestDataArtifact(
          data_format=artifact_template.data_format,
          data_parameters=substitute(artifact_template.data_parameters),
          verify_parameters=substitute(artifact_template.verify_parameters),
          source_url=substitute(artifact_template.source_url))
      artifacts[artifact_type] = artifact

    test_data = def_types.ModelTestData(id=substitute(template.id),
                                        name=substitute(template.name),
                                        tags=substitute(template.tags),
                                        source_info=template.source_info,
                                        artifacts=artifacts)
    batch_test_data[batch_size] = test_data

  return batch_test_data


def build_batch_benchmark_cases(
    batch_models: Dict[int, def_types.Model],
    batch_inputs: Dict[int, def_types.ModelTestData],
    batch_expected_outputs: Dict[int, def_types.ModelTestData],
    target_device: def_types.DeviceSpec,
    batch_sizes: Sequence[int]) -> Dict[int, def_types.BenchmarkCase]:
  """Build benchmark cases for multiple batch sizes."""
  benchmark_cases = {}
  for batch_size in batch_sizes:
    benchmark_case = def_types.BenchmarkCase.build(
        model=batch_models[batch_size],
        input_data=batch_inputs[batch_size],
        expected_output=batch_expected_outputs[batch_size],
        target_device=target_device,
    )
    benchmark_cases[batch_size] = benchmark_case

  return benchmark_cases
