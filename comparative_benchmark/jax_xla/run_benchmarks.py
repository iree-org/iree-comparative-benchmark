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
import numpy as np
import pathlib
import re
import statistics
import sys
import time
from typing import Any, Dict, Sequence

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))

from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.jax import benchmark_definitions
from openxla.benchmark.models.jax import model_interfaces
import utils


def _run_framework_benchmark(
    model: def_types.Model,
    input_npys: Sequence[pathlib.Path],
    warmup_iterations: int,
    benchmark_iterations: int,
    backend: str,
    shared_dict,
) -> None:

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
  definition: Dict[str, Any]
  metrics: Dict[str, Any]


def _append_result(result_path: pathlib.Path, result: BenchmarkResult) -> None:
  result_obj = {}
  if result_path.exists():
    result_obj = json.loads(result_path.read_text())

  benchmarks = result_obj.get("benchmarks", [])
  result_obj["benchmarks"] = benchmarks + [dataclasses.asdict(result)]

  result_path.write_text(json.dumps(result_obj))


def _run(benchmark: def_types.BenchmarkCase, run_in_process: bool,
         warmup_iterations: int, iterations: int,
         input_npys: Sequence[pathlib.Path]) -> BenchmarkResult:
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

  print(f"\n\n--- {benchmark.name} ---")

  # Retrieve framework-level benchmarks.
  with multiprocessing.Manager() as manager:
    shared_dict = manager.dict()

    backend = benchmark.target_device.accelerator_type
    kwargs: Dict[str, Any] = dict(
        model=model,
        input_npys=list(input_npys),
        warmup_iterations=warmup_iterations,
        benchmark_iterations=iterations,
        backend=backend,
        shared_dict=shared_dict,
    )
    if run_in_process:
      _run_framework_benchmark(**kwargs)
    else:
      p = multiprocessing.Process(target=_run_framework_benchmark,
                                  kwargs=kwargs)
      p.start()
      p.join()

    framework_metrics = dict(shared_dict)

  return BenchmarkResult(
      definition=benchmark_definition,
      metrics={
          "framework_level": framework_metrics,
      },
  )


def _download_artifacts(benchmarks: Sequence[def_types.BenchmarkCase],
                        root_dir: pathlib.Path,
                        verbose: bool = False):
  """Download benchmark artifacts."""

  download_list = []
  for benchmark in benchmarks:
    artifact = benchmark.input_data.artifacts[
        def_types.ModelTestDataFormat.NUMPY_TENSORS]
    input_path = root_dir / benchmark.model.name / "input_npy.tgz"
    download_list.append((artifact.source_url, input_path))

  utils.download_files(download_list, verbose=verbose)


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run JAX benchmarks with XLA backend.")
  parser.add_argument("-o",
                      "--output",
                      type=pathlib.Path,
                      required=True,
                      help="JSON file path to merge the results.")
  parser.add_argument("-name",
                      "--benchmark_name",
                      type=str,
                      required=True,
                      help="The regex pattern to match benchmark names.")
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
  parser.add_argument("--root-dir",
                      "--root_dir",
                      type=pathlib.Path,
                      default=pathlib.Path("/tmp/openxla-benchmark/jax_xla"),
                      help="Root directory stores benchmark artifacts.")
  parser.add_argument("--no-download",
                      "--no_download",
                      action="store_true",
                      help="Don't automatically download benchmark artifacts.")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")

  return parser.parse_args()


def main(
    benchmark_name: str,
    run_in_process: bool,
    warmup_iterations: int,
    iterations: int,
    output: pathlib.Path,
    root_dir: pathlib.Path,
    no_download: bool,
    verbose: bool,
):
  name_pattern = re.compile(f"^{benchmark_name}$")
  benchmarks = [
      benchmark for benchmark in benchmark_definitions.ALL_BENCHMARKS
      if name_pattern.match(benchmark.name)
  ]

  if not benchmarks:
    all_benchmark_list = "\n".join(
        benchmark.name for benchmark in benchmark_definitions.ALL_BENCHMARKS)
    raise ValueError(f'No benchmark matches "{benchmark_name}".'
                     f' Available benchmarks:\n{all_benchmark_list}')

  if not no_download:
    _download_artifacts(benchmarks=benchmarks,
                        root_dir=root_dir,
                        verbose=verbose)

  benchmarks_to_inputs = {}
  for benchmark in benchmarks:
    artifact = benchmark.input_data.artifacts[
        def_types.ModelTestDataFormat.NUMPY_TENSORS]
    model_dir = root_dir / benchmark.model.name

    num_of_tensors = len(artifact.data_parameters["tensor_dimensions"])
    input_npys = []

    # Check and gather input npy paths.
    for idx in range(num_of_tensors):
      tensor_path = model_dir / "input_npy" / f"input_{idx}.npy"
      if not tensor_path.exists():
        raise ValueError(f"Missing input data '{tensor_path}'.")

      input_npys.append(tensor_path)

    benchmarks_to_inputs[benchmark.id] = input_npys

  for benchmark in benchmarks:
    result = _run(benchmark,
                  run_in_process=run_in_process,
                  warmup_iterations=warmup_iterations,
                  iterations=iterations,
                  input_npys=benchmarks_to_inputs[benchmark.id])
    if verbose:
      print(json.dumps(dataclasses.asdict(result), indent=2))

    _append_result(output, result)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
