#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import importlib
import numpy as np
import pathlib
import statistics
import sys
import time
import torch
from typing import Any, Dict, Sequence, Tuple

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))

from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.pt import benchmark_definitions
from openxla.benchmark.models import model_interfaces
import benchmark_lib


def _run_framework_benchmark(
    model: def_types.Model,
    input_npys: Sequence[pathlib.Path],
    warmup_iterations: int,
    benchmark_iterations: int,
    backend: str,
    verbose: bool,
) -> Tuple[Dict[str, Any], Any]:
  try:

    data_type = model.model_parameters["data_type"]

    if backend == "gpu":
      if data_type == "fp16":
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
      elif data_type == "fp32":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
      else:
        raise ValueError(f"Datatype {data_type} not supported.")
    elif backend == "cpu":
      if data_type != "fp32":
        raise ValueError(f"Datatype other than FP32 is not supported on CPU.")
      torch.set_default_tensor_type(torch.FloatTensor)
    else:
      raise ValueError(f"Backend {backend} not supported.")

    model_module = importlib.import_module(model.model_impl.module_path)
    model_obj: model_interfaces.InferenceModel = model_module.create_model(
        **model.model_parameters)

    inputs = [np.load(path) for path in input_npys]
    pt_inputs = [torch.from_numpy(input_data) for input_data in inputs]

    if backend == "gpu":
      model_obj.cuda()
      pt_inputs = [input_data.cuda() for input_data in pt_inputs]

    if data_type == "fp16":
      # Autotuning not supported with FP16 datatypes.
      compiled_model = torch.compile(model_obj, backend="inductor")
      autotuning_enabled = False
    else:
      compiled_model = torch.compile(model_obj,
                                     mode="max-autotune",
                                     backend="inductor")
      autotuning_enabled = True

    def run_one_iter():
      start = time.perf_counter()
      output = compiled_model.forward(*pt_inputs)
      if backend == "gpu":
        torch.cuda.synchronize()
        output = output.cpu()
      end = time.perf_counter()
      latency = 1000 * (end - start)
      return (output, latency)

    # Run warmup.
    warmup_latencies = []
    # Run at least one warmup iteration to compile the model.
    warmup_iterations = min(1, warmup_iterations)
    compile_time_ms = None
    for i in range(warmup_iterations):
      _, latency = run_one_iter()
      if i == 0:
        compile_time_ms = latency
      warmup_latencies.append(latency)

    # Run benchmark.
    latencies = []
    output = None
    for i in range(benchmark_iterations):
      output, latency = run_one_iter()
      latencies.append(latency)
      output = output.detach().numpy()

    if output is None:
      raise ValueError("No benchmark runs.")

    last_outputs = (output,)

  except Exception as e:
    print(f"Failed to benchmark model {model.name}.")
    raise

  metrics = {
      "min_warmup_latency_ms":
          min(warmup_latencies, default=None),
      "max_warmup_latency_ms":
          max(warmup_latencies, default=None),
      "mean_warmup_latency_ms":
          None if not warmup_latencies else statistics.mean(warmup_latencies),
      "median_warmup_latency_ms":
          None if not warmup_latencies else statistics.median(warmup_latencies),
      "stddev_warmup_latency_ms":
          None
          if len(warmup_latencies) < 2 else statistics.stdev(warmup_latencies),
      "warmup_iterations":
          warmup_iterations,
      "min_latency_ms":
          min(latencies, default=None),
      "max_latency_ms":
          max(latencies, default=None),
      "mean_latency_ms":
          None if not latencies else statistics.mean(latencies),
      "median_latency_ms":
          None if not latencies else statistics.median(latencies),
      "stddev_latency_ms":
          None if len(latencies) < 2 else statistics.stdev(latencies),
      "benchmark_iterations":
          benchmark_iterations,
      "compile_time_ms":
          compile_time_ms,
      "autotuning_enabled":
          autotuning_enabled,
  }
  return (metrics, last_outputs)


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run PyTorch benchmarks with Inductor backend.")
  benchmark_lib.configure_parser(parser)
  return parser.parse_args()


def main(**kwargs):
  benchmark_lib.benchmark(benchmark_function=_run_framework_benchmark,
                          benchmark_cases=benchmark_definitions.ALL_BENCHMARKS,
                          compiler="inductor",
                          **kwargs)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
