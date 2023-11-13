#!/usr/bin/env python3
#
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys

import benchmark_lib

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))

from openxla.benchmark import def_types, devices
import openxla.benchmark.comparative_suite.jax.benchmark_definitions as jax_benchmark_definitions
import openxla.benchmark.comparative_suite.tflite.benchmark_definitions as tflite_benchmark_definitions
import utils

ALL_DEVICE_NAMES = [device.name for device in devices.ALL_DEVICES]


def check_accuracy(artifact_dir: pathlib.Path,
                   iree_run_module_path: pathlib.Path,
                   atol: float,
                   num_threads: str,
                   verbose: bool = False) -> bool:
  module_path = artifact_dir / f"module.vmfb"
  output_npy = artifact_dir / "outputs_npy" / "output_0.npy"
  command = [
      str(iree_run_module_path),
      f"--module={module_path}",
      f"--task_topology_group_count={num_threads}",
      "--device=local-task",
      "--function=main",
      f"--expected_output=@{output_npy}",
      f"--expected_f32_threshold={atol}",
      f"--expected_f16_threshold={atol}",
      f"--expected_f64_threshold={atol}",
  ]

  inputs_dir = artifact_dir / "inputs_npy"
  num_inputs = len(list(inputs_dir.glob("*.npy")))
  for i in range(num_inputs):
    command.append(f"--input=@{inputs_dir}/input_{i}.npy")

  command_str = " ".join(command)
  print(f"Running command: {command_str}")

  try:
    output = subprocess.run(command, check=True, capture_output=True)
    if verbose:
      print(output.stdout.decode())
    return True
  except subprocess.CalledProcessError as e:
    print(f"Error running command: {e}")

  return False


def benchmark_one(benchmark: def_types.BenchmarkCase,
                  target_device: def_types.DeviceSpec,
                  artifact_dir: pathlib.Path,
                  iree_benchmark_module_path: pathlib.Path, num_threads: str,
                  verbose: bool) -> utils.BenchmarkResult:
  model = benchmark.model
  benchmark_definition = {
      "benchmark_name": benchmark.name,
      "model_name": model.name,
      "framework": str(model.model_impl.framework_type),
      "device": target_device.name,
      "num_threads": num_threads,
      "tags": model.model_impl.tags + model.tags,
  }

  inputs_dir = artifact_dir / "inputs_npy"
  num_inputs = len(list(inputs_dir.glob("*.npy")))

  module_path = artifact_dir / "module.vmfb"
  command = [
      str(iree_benchmark_module_path),
      f"--module={module_path}",
      f"--task_topology_group_count={num_threads}",
      "--device=local-task",
      "--function=main",
      "--print_statistics",
  ]

  for i in range(num_inputs):
    command.append(f"--input=@{inputs_dir}/input_{i}.npy")

  metrics = benchmark_lib.run_benchmark_command(" ".join(command), verbose)
  return utils.BenchmarkResult(
      definition=benchmark_definition,
      metrics={
          "compiler_level": metrics,
      },
  )


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Runs IREE benchmarks.")
  parser.add_argument("-o",
                      "--output",
                      type=pathlib.Path,
                      required=True,
                      help="JSON file path to merge the results.")
  parser.add_argument(
      "--artifact_dir",
      type=pathlib.Path,
      required=True,
      help="The directory containing all required benchmark artifacts.")
  parser.add_argument("-device",
                      "--target_device",
                      dest="target_device_name",
                      type=str,
                      required=True,
                      choices=ALL_DEVICE_NAMES,
                      help="The target device to benchmark.")
  parser.add_argument("--iree_run_module_path",
                      type=pathlib.Path,
                      required=True,
                      help="Path to the iree-run-module binary.")
  parser.add_argument("--iree_benchmark_module_path",
                      type=pathlib.Path,
                      required=True,
                      help="Path to the iree-benchmark-module binary.")
  parser.add_argument("--threads",
                      type=str,
                      help="A comma-separated list of threads.")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")
  return parser.parse_args()


def main(output: pathlib.Path, artifact_dir: pathlib.Path,
         target_device_name: str, iree_run_module_path: pathlib.Path,
         iree_benchmark_module_path: pathlib.Path, threads: str, verbose: bool):

  try:
    target_device = next(device for device in devices.ALL_DEVICES
                         if device.name == target_device_name)
  except StopIteration:
    raise ValueError(f'Target device "{target_device_name}" is not defined.'
                     f' Available device options:\n{ALL_DEVICE_NAMES}')

  # Walk the artifacts dir to get benchmark names.
  all_benchmarks = jax_benchmark_definitions.ALL_BENCHMARKS + tflite_benchmark_definitions.ALL_BENCHMARKS

  benchmarks = {}
  contents = os.listdir(artifact_dir)
  for item in contents:
    if os.path.isdir(artifact_dir / item):
      name_pattern = re.compile(f".*{item}.*")
      for benchmark in all_benchmarks:
        if name_pattern.match(benchmark.name):
          benchmarks[item] = benchmark

  threads = threads.split(",")
  for directory, benchmark in benchmarks.items():
    benchmark_artifacts = artifact_dir / directory

    for num_thread in threads:
      atol = benchmark.verify_parameters["absolute_tolerance"]
      is_accurate = check_accuracy(benchmark_artifacts, iree_run_module_path,
                                   atol, num_thread, verbose)

      result = benchmark_one(benchmark, target_device, benchmark_artifacts,
                             iree_benchmark_module_path, num_thread, verbose)
      result.metrics["compiler_level"]["accuracy"] = is_accurate

      if verbose:
        print(json.dumps(dataclasses.asdict(result), indent=2))

      utils.append_benchmark_result(output, result)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
