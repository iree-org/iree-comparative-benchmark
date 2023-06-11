#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import dataclasses
from dataclasses import dataclass
import importlib
import jax
import json
import multiprocessing
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, List

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))

from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.jax import benchmark_definitions
from openxla.benchmark.models.jax import model_interfaces
from utils import execution_environment


def run_framework_benchmark(model: def_types.Model, warmup_iterations: int,
                            benchmark_iterations: int, backend: str,
                            shared_dict) -> None:

  model_module = importlib.import_module(model.model_impl.module_path)
  model_obj: model_interfaces.InferenceModel = model_module.create_model(
      **model.model_parameters)

  try:
    with jax.default_device(jax.devices(backend)[0]):

      # TODO(#14): Benchmark should load the dumped model input instead of
      # generating it.
      inputs = model_obj.generate_inputs()

      # Create jits.
      start = time.perf_counter()
      jit_inputs = jax.device_put(inputs)
      end = time.perf_counter()
      input_data_transfer_ms = 1000 * (end - start)

      jit_function = jax.jit(model_obj.forward)

      # Run warmup.
      warmup_latencies = []
      compile_time_s = -1
      for i in range(warmup_iterations):
        start = time.perf_counter()
        jax.block_until_ready(jit_function(jit_inputs))
        end = time.perf_counter()
        latency = 1000 * (end - start)
        if i == 0:
          compile_time_s = latency / 1000
        warmup_latencies.append(latency)

      # Run benchmark.
      latencies = []
      for i in range(benchmark_iterations):
        start = time.perf_counter()
        jax.block_until_ready(jit_function(jit_inputs))
        end = time.perf_counter()
        latencies.append(1000 * (end - start))

      # TODO(#11): Verify the model output.

  except Exception as e:
    print(f"Failed to benchmark model {model.name}. Exception: {e}")
    raise RuntimeError(e)

    # Save results.
  result_dict = {
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
      "compile_time_s":
          compile_time_s,
      "input_data_transfer_ms":
          input_data_transfer_ms,
  }
  shared_dict.update(result_dict)


@dataclass
class BenchmarkResult:
  execution_environment: Dict[str, Any]
  benchmarks: List[Dict[str, Any]]


def _append_result(result_path: pathlib.Path, result: BenchmarkResult) -> None:
  merged_result = result

  if result_path.exists():
    prev_result = BenchmarkResult(**json.loads(result_path.read_text()))
    if prev_result.execution_environment != result.execution_environment:
      raise ValueError(
          "Can't merge benchmark results from different environments.")

    merged_result = dataclasses.replace(
        merged_result,
        benchmarks=prev_result.benchmarks + result.benchmarks,
    )

  result_path.write_text(json.dumps(dataclasses.asdict(merged_result)))


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run JAX benchmarks with XLA backend.")
  parser.add_argument(
      "-o",
      "--output_path",
      type=pathlib.Path,
      required=True,
      help=
      "Path to results json file. Expects this file to have been pre-populated."
  )
  parser.add_argument("-name",
                      "--benchmark_name",
                      type=str,
                      required=True,
                      help="The unique name that defines a benchmark.")
  parser.add_argument("-w",
                      "--warmup_iterations",
                      type=int,
                      default=5,
                      help="The number of warmup steps.")
  parser.add_argument("-iter",
                      "--iterations",
                      type=int,
                      default=100,
                      help="The number of iterations to benchmark.")
  parser.add_argument(
      "--run_in_process",
      action="store_true",
      help=("Whether to run the benchmark under the same process. Set this to"
            " true when profiling a single workload."))
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")

  return parser.parse_args()


def main(
    benchmark_name: str,
    run_in_process: bool,
    warmup_iterations: int,
    iterations: int,
    output_path: pathlib.Path,
    verbose: bool,
):
  benchmark = benchmark_definitions.BENCHMARK_NAME_MAP.get(benchmark_name)
  if benchmark is None:
    raise ValueError(f'Benchmark "{benchmark_name}" not found.')

  model = benchmark.model
  input_data = benchmark.input_data.artifacts[
      def_types.ModelTestDataFormat.NUMPY_TENSORS]
  expected_output = benchmark.expected_output.artifacts[
      def_types.ModelTestDataFormat.NUMPY_TENSORS]

  data_type = model.model_parameters["data_type"]
  batch_size = model.model_parameters["batch_size"]
  input_dims = input_data.data_parameters["tensor_dimensions"]
  output_dims = expected_output.data_parameters["tensor_dimensions"]
  benchmark_definition = {
      "benchmark_id": benchmark.id,
      "benchmark_name": benchmark.name,
      "framework": str(model.model_impl.framework_type),
      "data_type": data_type,
      "batch_size": batch_size,
      "inputs": input_dims,
      "outputs": output_dims,
      "compiler": "xla",
      "device": benchmark.target_device.name,
      "tags": model.model_impl.tags + model.tags,
  }

  print(f"\n\n--- {benchmark_name} ---")

  # Retrieve framework-level benchmarks.
  with multiprocessing.Manager() as manager:
    shared_dict = manager.dict()

    backend = benchmark.target_device.accelerator_type
    args: Dict[str, Any] = dict(
        model=model,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=iterations,
        backend=backend,
        shared_dict=shared_dict,
    )
    if run_in_process:
      run_framework_benchmark(**args)
    else:
      p = multiprocessing.Process(target=run_framework_benchmark, args=args)
      p.start()
      p.join()

    framework_metrics = dict(shared_dict)

  result = BenchmarkResult(
      execution_environment={
          "python_environment":
              execution_environment.get_python_environment_info()
      },
      benchmarks=[{
          "definition": benchmark_definition,
          "metrics": {
              "framework_level": framework_metrics,
          }
      }],
  )
  _append_result(output_path, result)

  if verbose:
    print(json.dumps(dataclasses.asdict(result), indent=2))


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
