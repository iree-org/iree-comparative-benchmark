#!/usr/bin/env python3
#
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import ast
import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys

from typing import Any, Dict, List

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))

from openxla.benchmark import def_types, devices
import openxla.benchmark.comparative_suite.jax.benchmark_definitions as jax_benchmark_definitions
import openxla.benchmark.comparative_suite.tflite.benchmark_definitions as tflite_benchmark_definitions

# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))
import utils

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "utils"))
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from experimental.utils import run_iree_benchmark

ALL_DEVICE_NAMES = [device.name for device in devices.ALL_DEVICES]


def get_directory_names(target_device: def_types.DeviceSpec,
                        directory: pathlib.Path):
  if target_device in devices.mobile_devices.ALL_DEVICES:
    output = subprocess.run(
        ["adb", "shell", "ls", str(directory)], check=True, capture_output=True)
    output = output.stdout.decode()
    contents = output.split("\n")
    # Remove empty elements.
    return [item for item in contents if item]
  else:
    return os.listdir(directory)


def get_common_command_parameters(target_device: def_types.DeviceSpec,
                                  artifacts_dir: pathlib.Path,
                                  task_topology_cpu_ids: str) -> List[str]:
  module_path = artifacts_dir / "module.vmfb"
  parameters = [
      f"--module={module_path}",
      f"--task_topology_cpu_ids={task_topology_cpu_ids}",
      "--device=local-task",
      "--function=main",
  ]

  inputs_dir = artifacts_dir / "inputs_npy"
  inputs = get_directory_names(target_device, inputs_dir)
  for input in inputs:
    parameters.append(f"--input=@{inputs_dir}/{input}")

  return parameters


def generate_accuracy_check_command(target_device: def_types.DeviceSpec,
                                    artifacts_dir: pathlib.Path,
                                    iree_run_module_path: pathlib.Path,
                                    atol: float,
                                    task_topology_cpu_ids: str) -> str:
  output_npy = artifacts_dir / "outputs_npy" / "output_0.npy"
  command = [str(iree_run_module_path)] + get_common_command_parameters(
      target_device, artifacts_dir, task_topology_cpu_ids) + [
          f"--expected_output=@{output_npy}",
          f"--expected_f32_threshold={atol}",
          f"--expected_f16_threshold={atol}",
          f"--expected_f64_threshold={atol}",
      ]
  return " ".join(command)


def benchmark_on_x86(target_device: def_types.DeviceSpec,
                     benchmark: def_types.BenchmarkCase,
                     artifacts_dir: pathlib.Path,
                     iree_run_module_path: pathlib.Path,
                     iree_benchmark_module_path: pathlib.Path,
                     task_topology_cpu_ids: str,
                     verbose: bool) -> Dict[str, Any]:
  # Check accuracy.
  atol = benchmark.verify_parameters["absolute_tolerance"]
  command = generate_accuracy_check_command(target_device, artifacts_dir,
                                            iree_run_module_path, atol,
                                            task_topology_cpu_ids)

  try:
    output = subprocess.run(command,
                            shell=True,
                            check=True,
                            capture_output=True)
    if verbose:
      print(output.stdout.decode())
    is_accurate = True
  except subprocess.CalledProcessError as e:
    print(f"Error running command: {e}")
    is_accurate = False

  # Run benchmark.
  command = [str(iree_benchmark_module_path)] + get_common_command_parameters(
      target_device, artifacts_dir,
      task_topology_cpu_ids) + ["--print_statistics"]
  metrics = run_iree_benchmark.run_benchmark_command(command, verbose)
  metrics["accuracy"] = is_accurate
  metrics["command"] = " ".join(command)
  return metrics


def benchmark_on_android(target_device: def_types.DeviceSpec,
                         benchmark: def_types.BenchmarkCase,
                         artifacts_dir: pathlib.Path,
                         iree_run_module_device_path: pathlib.Path,
                         iree_benchmark_module_device_path: pathlib.Path,
                         task_topology_cpu_ids: str,
                         verbose: bool) -> Dict[str, Any]:
  # Check accuracy.
  atol = benchmark.verify_parameters["absolute_tolerance"]
  command = generate_accuracy_check_command(target_device, artifacts_dir,
                                            iree_run_module_device_path, atol,
                                            task_topology_cpu_ids)
  command = f"adb shell su root {command}"

  try:
    output = subprocess.run(command,
                            shell=True,
                            check=True,
                            capture_output=True)
    if verbose:
      print(output.stdout.decode())
    is_accurate = True
  except subprocess.CalledProcessError as e:
    print(f"Error running command: {e}")
    is_accurate = False

  # Run benchmark.
  root_dir = iree_benchmark_module_device_path.parent
  benchmark_command = [str(iree_benchmark_module_device_path)
                      ] + get_common_command_parameters(
                          target_device, artifacts_dir,
                          task_topology_cpu_ids) + ["--print_statistics"]
  benchmark_command = " ".join(benchmark_command)

  command_path = root_dir / "command.txt"
  subprocess.run(f"adb shell \"echo '{benchmark_command}' > {command_path}\"",
                 shell=True,
                 check=True,
                 capture_output=True)

  benchmark_binary_path = root_dir / "utils" / "run_iree_benchmark.py"
  command = f"adb shell su root /data/data/com.termux/files/usr/bin/python {benchmark_binary_path} --command_path=\"{command_path}\""
  if verbose:
    command += " --verbose"

  output = subprocess.run(command, shell=True, check=True, capture_output=True)
  output = output.stdout.decode()
  if verbose:
    print(output)

  match = re.search(r"results_dict: (\{.*\})", output)
  if match:
    dictionary_string = match.group(1)
    metrics = ast.literal_eval(dictionary_string)
  else:
    metrics = {"error": f"Could not parse results"}

  metrics["accuracy"] = is_accurate
  metrics["command"] = benchmark_command
  return metrics


def benchmark_one(benchmark: def_types.BenchmarkCase,
                  target_device: def_types.DeviceSpec,
                  artifacts_dir: pathlib.Path,
                  iree_run_module_path: pathlib.Path,
                  iree_benchmark_module_path: pathlib.Path, num_threads: str,
                  task_topology_cpu_ids: str,
                  verbose: bool) -> utils.BenchmarkResult:
  model = benchmark.model
  benchmark_definition = {
      "benchmark_name": benchmark.name,
      "model_name": model.name,
      "framework": str(model.model_impl.framework_type),
      "device": target_device.name,
      "num_threads": num_threads,
      "task_topology_cpu_ids": task_topology_cpu_ids,
      "compiler": "iree",
      "tags": model.model_impl.tags + model.tags,
  }

  if target_device in devices.mobile_devices.ALL_DEVICES:
    metrics = benchmark_on_android(target_device, benchmark, artifacts_dir,
                                   iree_run_module_path,
                                   iree_benchmark_module_path,
                                   task_topology_cpu_ids, verbose)
  else:
    metrics = benchmark_on_x86(target_device, benchmark, artifacts_dir,
                               iree_run_module_path, iree_benchmark_module_path,
                               task_topology_cpu_ids, verbose)

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
      help=
      "The directory containing all required benchmark artifacts on the host.")
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
  parser.add_argument(
      "--thread_config",
      type=str,
      help=
      "A string dictionary of num_threads to cpu_ids. If cpu_ids is empty, does not pin threads to a specific CPU. Example: {1: '0', 4: '1,2,3,4', 5: '0,1,2,3,4'}"
  )
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")
  return parser.parse_args()


def main(output: pathlib.Path, artifact_dir: pathlib.Path,
         target_device_name: str, iree_run_module_path: pathlib.Path,
         iree_benchmark_module_path: pathlib.Path, thread_config: str,
         verbose: bool):

  try:
    target_device = next(device for device in devices.ALL_DEVICES
                         if device.name == target_device_name)
  except StopIteration:
    raise ValueError(f'Target device "{target_device_name}" is not defined.'
                     f' Available device options:\n{ALL_DEVICE_NAMES}')

  # Walk the artifacts dir to get benchmark names.
  all_benchmarks = jax_benchmark_definitions.ALL_BENCHMARKS + tflite_benchmark_definitions.ALL_BENCHMARKS

  benchmarks = {}
  contents = get_directory_names(target_device, artifact_dir)
  for item in contents:
    name_pattern = re.compile(f".*{item}.*")
    for benchmark in all_benchmarks:
      if name_pattern.match(benchmark.name):
        benchmarks[item] = benchmark

  thread_config = ast.literal_eval(thread_config.replace("'", '"'))
  for directory, benchmark in benchmarks.items():
    model_artifact_dir = artifact_dir / directory
    for num_thread, cpu_ids in thread_config.items():
      result = benchmark_one(benchmark, target_device, model_artifact_dir,
                             iree_run_module_path, iree_benchmark_module_path,
                             num_thread, cpu_ids, verbose)
      if verbose:
        print(json.dumps(dataclasses.asdict(result), indent=2))
      utils.append_benchmark_result(output, result)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
