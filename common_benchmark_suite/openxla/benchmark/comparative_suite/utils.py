# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Common utilities to define comparative benchmarks."""

import string
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from openxla.benchmark import def_types, testdata


@dataclass(frozen=True)
class TemplateFunc:
  """Function to return an arbitrary object with substitutions."""
  func: Callable[..., Any]


# Module path to find all model implementations.
MODELS_MODULE_PATH = "openxla.benchmark.models"

# Constants and functions help build batch templates.
BATCH_NAME = lambda name: string.Template(name + "_BATCH${batch_size}")
BATCH_TAG = string.Template("batch-${batch_size}")
BATCH_SIZE_PARAM = TemplateFunc(func=lambda batch_size, **_unused: batch_size)
BATCH_TENSOR_DIMS = lambda dims: string.Template("${batch_size}x" + dims)

GEN_NAME = lambda name: string.Template(name + "_GEN${gen_size}")
GEN_TAG = string.Template("gen-${gen_size}")
GEN_SIZE_PARAM = TemplateFunc(func=lambda gen_size, **_unused: gen_size)


@dataclass(frozen=True)
class ModelTemplate:
  """Template of def_types.Model."""
  name: string.Template
  tags: List[Union[str, string.Template]]
  model_impl: def_types.ModelImplementation
  model_parameters: Dict[str, Any]
  artifacts_dir_url: Optional[string.Template] = None
  exported_model_types: List[def_types.ModelArtifactType] = dataclasses.field(
      default_factory=list)


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
  """Build model with batch sizes by replacing `${batch_size}`, `${name}` in the
  template.

  The `${name}` will be replaced by model name. Note that the model name
  template can't contain `${name}`.

  Args:
    template: model template to replace.
    batch_sizes: list of batch sizes to generate.

  Returns:
    Map of batch size to model.
  """

  batch_models = {}
  for batch_size in batch_sizes:
    name = _substitute_template(obj=template.name, batch_size=batch_size)
    substitute = lambda obj: _substitute_template(
        obj=obj, batch_size=batch_size, name=name)
    model = def_types.Model(
        name=substitute(template.name),
        tags=substitute(template.tags),
        model_impl=template.model_impl,
        model_parameters=substitute(template.model_parameters),
        exported_model_types=template.exported_model_types,
        artifacts_dir_url=substitute(template.artifacts_dir_url),
    )
    batch_models[batch_size] = model

  return batch_models


def build_gen_models(template: ModelTemplate,
                     gen_sizes: Sequence[int]) -> Dict[int, def_types.Model]:
  """Build model with gen sizes by replacing `${gen_size}`, `${name}` in the
  template.

  The `${name}` will be replaced by model name. Note that the model name
  template can't contain `${name}`.

  Args:
    template: model template to replace.
    gen_sizes: list of gen sizes to generate.

  Returns:
    Map of gen size to model.
  """

  gen_models = {}
  for gen_size in gen_sizes:
    name = _substitute_template(obj=template.name, gen_size=gen_size)
    substitute = lambda obj: _substitute_template(
        obj=obj, gen_size=gen_size, name=name)
    model = def_types.Model(
        name=substitute(template.name),
        tags=substitute(template.tags),
        model_impl=template.model_impl,
        model_parameters=substitute(template.model_parameters),
        exported_model_types=template.exported_model_types,
        artifacts_dir_url=substitute(template.artifacts_dir_url),
    )
    gen_models[gen_size] = model

  return gen_models


def build_batch_benchmark_cases(
    batch_models: Dict[int, def_types.Model],
    batch_sizes: Sequence[int],
    input_data: def_types.ModelTestData = testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters: Optional[Dict[str, Any]] = None,
) -> Dict[int, def_types.BenchmarkCase]:
  """Build benchmark cases for multiple batch sizes."""
  benchmark_cases: Dict[int, def_types.BenchmarkCase] = {}
  for batch_size in batch_sizes:
    benchmark_case = def_types.BenchmarkCase.build(
        model=batch_models[batch_size],
        input_data=input_data,
        verify_parameters=verify_parameters)
    benchmark_cases[batch_size] = benchmark_case

  return benchmark_cases
