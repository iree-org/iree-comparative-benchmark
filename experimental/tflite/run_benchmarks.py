#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import dataclasses
import json
import pathlib
import re
import subprocess
import sys
from typing import Sequence

import benchmark_lib

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
from openxla.benchmark import def_types, devices
from openxla.benchmark.comparative_suite.tf import benchmark_definitions as tf_benchmark_definitions

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))
import utils

ALL_DEVICE_NAMES = [device.name for device in devices.ALL_DEVICES]
TFLITE_FP32_FILENAME = "model_fp32.tflite"

LATENCY_REGEXP = re.compile(
    "INFO: count=\d+ first=\d+ curr=\d+ min=(.*) max=(.*) avg=(.*) std=(.*)")
PEAK_MEMORY_REGEXP = re.compile(
    "INFO: Overall peak memory footprint \(MB\) via periodic monitoring: (.*)")


def _run(
    benchmark: def_types.BenchmarkCase,
    target_device: def_types.DeviceSpec,
    iterations: int,
    num_threads: str,
    tflite_benchmark_tool: pathlib.Path,
    tflite_model_path: pathlib.Path,
    verbose: bool,
) -> utils.BenchmarkResult:
  model = benchmark.model
  data_type = model.model_parameters["data_type"]
  batch_size = model.model_parameters["batch_size"]
  benchmark_definition = {
      "benchmark_name": benchmark.name,
      "framework": str(model.model_impl.framework_type),
      "data_type": data_type,
      "batch_size": batch_size,
      "compiler": "TFLite",
      "device": target_device.name,
      "num_threads": num_threads,
      "num_iterations": iterations,
      "tags": model.model_impl.tags + model.tags,
  }
  cmd = [
      tflite_benchmark_tool,
      f"--graph={tflite_model_path}",
      f"--num_runs={iterations}",
      f"--num_threads={num_threads}",
      f"--report_peak_memory_footprint=true",
  ]

  return benchmark_lib.benchmark(cmd, benchmark_definition, iterations, verbose)


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run TFLite benchmarks.")
  benchmark_lib.configure_parser(parser)
  return parser.parse_args()


def main(
    benchmark_name: str,
    target_device_name: str,
    output: pathlib.Path,
    root_dir: pathlib.Path,
    threads: str,
    tflite_benchmark_tool: pathlib.Path,
    iterations: int,
    no_download: bool,
    verbose: bool,
):
  name_pattern = re.compile(f"^{benchmark_name}$")
  all_benchmarks = tf_benchmark_definitions.ALL_BENCHMARKS
  benchmarks = [
      benchmark for benchmark in all_benchmarks
      if name_pattern.match(benchmark.name)
  ]

  if not benchmarks:
    all_benchmark_names = "\n".join(
        benchmark.name for benchmark in all_benchmarks)
    raise ValueError(f'No benchmark matches "{benchmark_name}".'
                     f' Available benchmarks:\n{all_benchmark_names}')

  try:
    target_device = next(device for device in devices.ALL_DEVICES
                         if device.name == target_device_name)
  except StopIteration:
    raise ValueError(f'Target device "{target_device_name}" is not defined.'
                     f' Available device options:\n{ALL_DEVICE_NAMES}')

  if not no_download:
    benchmark_lib.download_artifacts(benchmarks=benchmarks,
                                     root_dir=root_dir,
                                     verbose=verbose)

  threads = threads.split(",")
  for benchmark in benchmarks:
    tflite_model_path = root_dir / benchmark.model.name / TFLITE_FP32_FILENAME
    if not tflite_model_path.exists():
      raise ValueError(f"TFLite model not found: '{tflite_model_path}'.")

    for num_threads in threads:
      result = _run(benchmark=benchmark,
                    target_device=target_device,
                    iterations=iterations,
                    num_threads=num_threads,
                    tflite_benchmark_tool=tflite_benchmark_tool,
                    tflite_model_path=tflite_model_path,
                    verbose=verbose)
      if verbose:
        print(json.dumps(dataclasses.asdict(result), indent=2))

      utils.append_benchmark_result(output, result)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
