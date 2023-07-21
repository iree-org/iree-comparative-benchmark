# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import dataclasses
import json
import multiprocessing
import pathlib
import re
import sys
from typing import Any, Callable, Dict, Sequence

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))

from openxla.benchmark import def_types, devices
import utils

ALL_DEVICE_NAMES = [device.name for device in devices.ALL_DEVICES]


def _run(
    benchmark: def_types.BenchmarkCase,
    target_device: def_types.DeviceSpec,
    run_in_process: bool,
    warmup_iterations: int,
    iterations: int,
    input_npys: Sequence[pathlib.Path],
    expect_npys: Sequence[pathlib.Path],
    benchmark_function: Callable,
    compiler: str,
    verbose: bool,
) -> utils.BenchmarkResult:
  model = benchmark.model
  input_data = benchmark.input_data.artifacts[
      def_types.ModelTestDataFormat.NUMPY_TENSORS]

  data_type = model.model_parameters["data_type"]
  batch_size = model.model_parameters["batch_size"]
  input_dims = input_data.data_parameters["tensor_dimensions"]
  benchmark_definition = {
      "benchmark_name": benchmark.name,
      "framework": str(model.model_impl.framework_type),
      "data_type": data_type,
      "batch_size": batch_size,
      "inputs": input_dims,
      "compiler": compiler,
      "device": target_device.name,
      "tags": model.model_impl.tags + model.tags,
  }

  print(f"\n\n--- {benchmark.name} ---")

  # Retrieve framework-level benchmarks.
  with multiprocessing.Manager() as manager:
    backend = target_device.accelerator_type
    kwargs: Dict[str, Any] = dict(
        model=model,
        input_npys=list(input_npys),
        expect_npys=list(expect_npys),
        verify_params=benchmark.verify_parameters,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=iterations,
        backend=backend,
        verbose=verbose,
    )
    if run_in_process:
      framework_metrics = benchmark_function(**kwargs)
    else:
      shared_dict = manager.dict()

      def wrapped_benchmark_function() -> None:
        shared_dict.update(benchmark_function(**kwargs))

      p = multiprocessing.Process(target=wrapped_benchmark_function)
      p.start()
      p.join()

      framework_metrics = dict(shared_dict)

  return utils.BenchmarkResult(
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
    model = benchmark.model

    input_artifact = benchmark.input_data.artifacts[
        def_types.ModelTestDataFormat.NUMPY_TENSORS]
    input_path = root_dir / model.name / "inputs_npy.tgz"
    download_list.append((input_artifact.source_url, input_path))

    expect_path = root_dir / model.name / "outputs_npy.tgz"
    expect_url = model.artifacts_url + "/outputs_npy.tgz"
    download_list.append((expect_url, expect_path))

  utils.download_files(download_list, verbose=verbose)


def configure_parser(parser: argparse.ArgumentParser):
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
  parser.add_argument("-device",
                      "--target_device",
                      dest="target_device_name",
                      type=str,
                      required=True,
                      choices=ALL_DEVICE_NAMES,
                      help="The target device to benchmark.")
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
                      default=pathlib.Path("/tmp/openxla-benchmark"),
                      help="Root directory stores benchmark artifacts.")
  parser.add_argument("--no-download",
                      "--no_download",
                      action="store_true",
                      help="Don't automatically download benchmark artifacts.")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")


def benchmark(
    benchmark_name: str,
    target_device_name: str,
    run_in_process: bool,
    warmup_iterations: int,
    iterations: int,
    output: pathlib.Path,
    root_dir: pathlib.Path,
    no_download: bool,
    verbose: bool,
    benchmark_function: Callable,
    benchmark_cases: Sequence[def_types.BenchmarkCase],
    compiler: str,
):
  name_pattern = re.compile(f"^{benchmark_name}$")
  benchmarks = [
      benchmark for benchmark in benchmark_cases
      if name_pattern.match(benchmark.name)
  ]

  if not benchmarks:
    all_benchmark_list = "\n".join(
        benchmark.name for benchmark in benchmark_cases)
    raise ValueError(f'No benchmark matches "{benchmark_name}".'
                     f' Available benchmarks:\n{all_benchmark_list}')

  try:
    target_device = next(device for device in devices.ALL_DEVICES
                         if device.name == target_device_name)
  except StopIteration:
    raise ValueError(f'Target device "{target_device_name}" is not defined.'
                     f' Available device options:\n{ALL_DEVICE_NAMES}')

  if not no_download:
    _download_artifacts(benchmarks=benchmarks,
                        root_dir=root_dir,
                        verbose=verbose)

  benchmarks_to_inputs = {}
  benchmarks_to_expects = {}
  for benchmark in benchmarks:
    model_dir = root_dir / benchmark.model.name

    input_artifact = benchmark.input_data.artifacts[
        def_types.ModelTestDataFormat.NUMPY_TENSORS]
    num_of_inputs = len(input_artifact.data_parameters["tensor_dimensions"])
    input_npys = []
    # Check and gather input npy paths.
    for idx in range(num_of_inputs):
      path = model_dir / "inputs_npy" / f"input_{idx}.npy"
      if not path.exists():
        raise ValueError(f"Missing input data '{path}'.")
      input_npys.append(path)

    outputs_npy_dir = model_dir / "outputs_npy"
    expect_npys = [outputs_npy_dir.glob("output_*.npy")]

    benchmarks_to_inputs[benchmark.name] = input_npys
    benchmarks_to_expects[benchmark.name] = expect_npys

  for benchmark in benchmarks:
    result = _run(benchmark=benchmark,
                  target_device=target_device,
                  run_in_process=run_in_process,
                  warmup_iterations=warmup_iterations,
                  iterations=iterations,
                  input_npys=benchmarks_to_inputs[benchmark.name],
                  expect_npys=benchmarks_to_expects[benchmark.name],
                  benchmark_function=benchmark_function,
                  compiler=compiler,
                  verbose=verbose)
    if verbose:
      print(json.dumps(dataclasses.asdict(result), indent=2))

    utils.append_benchmark_result(output, result)
