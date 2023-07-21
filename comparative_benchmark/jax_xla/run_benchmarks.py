#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import importlib
import jax
import numpy as np
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, Sequence, Tuple

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))

from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.jax import benchmark_definitions
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

  model_module = importlib.import_module(model.model_impl.module_path)
  model_obj: model_interfaces.InferenceModel = model_module.create_model(
      **model.model_parameters)

  inputs = [np.load(path) for path in input_npys]

  try:
    with jax.default_device(jax.devices(backend)[0]):

      # Create jits.
      start = time.perf_counter()
      jit_inputs = jax.device_put(inputs)
      end = time.perf_counter()
      input_data_transfer_ms = 1000 * (end - start)

      jit_function = jax.jit(model_obj.forward)

      # Run warmup.
      warmup_latencies = []
      compile_time_ms = -1
      for i in range(warmup_iterations):
        start = time.perf_counter()
        jax.block_until_ready(jit_function(jit_inputs))
        end = time.perf_counter()
        latency = 1000 * (end - start)
        if i == 0:
          compile_time_ms = latency
        warmup_latencies.append(latency)

      # Run benchmark.
      latencies = []
      last_outputs = None
      for i in range(benchmark_iterations):
        start = time.perf_counter()
        last_outputs = jit_function(jit_inputs)
        jax.block_until_ready(last_outputs)
        end = time.perf_counter()
        latencies.append(1000 * (end - start))

      if last_outputs is None:
        raise ValueError("No benchmark runs.")

  except Exception as e:
    print(f"Failed to benchmark model {model.name}. Exception: {e}")
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
          None if not warmup_latencies else statistics.stdev(warmup_latencies),
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
          None if not latencies else statistics.stdev(latencies),
      "benchmark_iterations":
          benchmark_iterations,
      "compile_time_ms":
          compile_time_ms,
      "input_data_transfer_ms":
          input_data_transfer_ms,
  }
  return (metrics, last_outputs)


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run JAX benchmarks with XLA backend.")
  benchmark_lib.configure_parser(parser)
  return parser.parse_args()


def main(**kwargs):
  benchmark_lib.benchmark(benchmark_function=_run_framework_benchmark,
                          benchmark_cases=benchmark_definitions.ALL_BENCHMARKS,
                          compiler="xla",
                          **kwargs)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
