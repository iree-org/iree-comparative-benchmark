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
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List, Sequence, Tuple

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))

from openxla.benchmark import def_types, devices
from openxla.benchmark.comparative_suite.tflite import benchmark_definitions
import openxla.benchmark.models.utils as model_utils

# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))
from scripts import adb_fetch_and_push
import utils

# Add command libs.
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "utils"))
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from experimental.utils import run_tflite_benchmark

ALL_DEVICE_NAMES = [device.name for device in devices.ALL_DEVICES]


def _download_artifacts_x86(benchmarks: Sequence[def_types.BenchmarkCase],
                            root_dir: pathlib.Path,
                            verbose: bool = False):
  """Download benchmark artifacts."""
  download_list = []
  for benchmark in benchmarks:
    model = benchmark.model
    tflite_model = model_utils.create_model_obj(benchmark.model)

    tflite_model_path = root_dir / model.name / tflite_model.model_filename
    download_list.append((tflite_model.model_uri, tflite_model_path))

  utils.download_files(download_list, verbose=verbose)


def _download_artifacts_android(benchmarks: Sequence[def_types.BenchmarkCase],
                                root_dir: pathlib.Path,
                                verbose: bool = False):
  for benchmark in benchmarks:
    model = benchmark.model
    tflite_model = model_utils.create_model_obj(benchmark.model)

    tflite_model_path = root_dir / model.name / tflite_model.model_filename
    adb_fetch_and_push.adb_download_and_push_file(tflite_model.model_uri,
                                                  tflite_model_path, verbose)


def _run_benchmark_command_x86(command: List[str], verbose: bool):
  metrics = run_tflite_benchmark.run_benchmark_command(command, verbose)
  metrics["command"] = " ".join(command)
  return metrics


def _run_benchmark_command_android(command: List[str], root_dir: pathlib.Path,
                                   verbose: bool):
  command_path = root_dir / "command.txt"
  command_str = " ".join(command)
  subprocess.run(f"adb shell \"echo '{command_str}' > {command_path}\"",
                 shell=True,
                 check=True,
                 capture_output=True)

  benchmark_binary_path = root_dir / "utils" / "run_tflite_benchmark.py"
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
  metrics["command"] = command_str

  return metrics


def _download_artifacts(target_device: def_types.DeviceSpec,
                        benchmarks: Sequence[def_types.BenchmarkCase],
                        root_dir: pathlib.Path,
                        verbose: bool = False):
  if target_device in devices.mobile_devices.ALL_DEVICES:
    _download_artifacts_android(benchmarks, root_dir, verbose)
  else:
    _download_artifacts_x86(benchmarks, root_dir, verbose)


def _benchmark(benchmark: def_types.BenchmarkCase,
               target_device: def_types.DeviceSpec,
               tflite_benchmark_binary: pathlib.Path, root_dir: pathlib.Path,
               num_threads: int, taskset: str, num_iterations: int,
               verbose: bool) -> utils.BenchmarkResult:
  model = benchmark.model
  tflite_model = model_utils.create_model_obj(benchmark.model)
  tflite_model_path = root_dir / model.name / tflite_model.model_filename

  benchmark_definition = {
      "benchmark_name": benchmark.name,
      "model_name": model.name,
      "model_source": tflite_model.model_uri,
      "framework": str(model.model_impl.framework_type),
      "device": target_device.name,
      "num_threads": num_threads,
      "taskset": taskset,
      "compiler": "tflite",
      "tags": model.model_impl.tags + model.tags,
  }

  command = []
  # Check if taskset is empty.
  if not bool(re.match(r'^$|^\\s*$', taskset)):
    taskset = taskset.split(" ")
    command += ["taskset"] + taskset
  command += [
      str(tflite_benchmark_binary),
      f"--graph={tflite_model_path}",
      f"--num_threads={num_threads}",
      f"--num_runs={num_iterations}",
  ]

  if target_device in devices.mobile_devices.ALL_DEVICES:
    metrics = _run_benchmark_command_android(command, root_dir, verbose)
  else:
    metrics = _run_benchmark_command_x86(command, verbose)

  return utils.BenchmarkResult(
      definition=benchmark_definition,
      metrics={
          "compiler_level": metrics,
      },
  )


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Runs TFLite benchmarks.")
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
  parser.add_argument("--tflite_benchmark_binary",
                      type=pathlib.Path,
                      required=True,
                      help="The path to the TFLite benchmark binary.")
  parser.add_argument("-device",
                      "--target_device",
                      dest="target_device_name",
                      type=str,
                      required=True,
                      choices=ALL_DEVICE_NAMES,
                      help="The target device to benchmark.")
  parser.add_argument(
      "--thread_config",
      type=str,
      help=
      "A string dictionary of num_threads to tasksets. If tasksets are blank, tasksets are not used. Example: {1: '100', 4: 'F0', 5: '1F0'}."
  )
  parser.add_argument("-iter",
                      "--iterations",
                      type=int,
                      default=50,
                      help="The number of iterations to benchmark.")
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
  return parser.parse_args()


def main(output: pathlib.Path, benchmark_name: str,
         tflite_benchmark_binary: pathlib.Path, target_device_name: str,
         thread_config: str, iterations: int, root_dir: pathlib.Path,
         no_download: bool, verbose: bool):

  try:
    target_device = next(device for device in devices.ALL_DEVICES
                         if device.name == target_device_name)
  except StopIteration:
    raise ValueError(f'Target device "{target_device_name}" is not defined.'
                     f' Available device options:\n{ALL_DEVICE_NAMES}')

  all_benchmarks = benchmark_definitions.ALL_BENCHMARKS
  name_pattern = re.compile(f"^{benchmark_name}$")
  benchmarks = [
      benchmark for benchmark in all_benchmarks
      if name_pattern.match(benchmark.name)
  ]

  if not benchmarks:
    all_benchmark_list = "\n".join(
        benchmark.name for benchmark in all_benchmarks)
    raise ValueError(f'No benchmark matches "{benchmark_name}".'
                     f' Available benchmarks:\n{all_benchmark_list}')

  if not no_download:
    if verbose:
      print("Downloading artifacts...")
    _download_artifacts(target_device=target_device,
                        benchmarks=benchmarks,
                        root_dir=root_dir,
                        verbose=verbose)

  thread_config = ast.literal_eval(thread_config.replace("'", '"'))
  for benchmark in benchmarks:
    for num_threads, tasksets in thread_config.items():
      result = _benchmark(benchmark, target_device, tflite_benchmark_binary,
                          root_dir, num_threads, tasksets, iterations, verbose)
      if verbose:
        print(json.dumps(dataclasses.asdict(result), indent=2))
      utils.append_benchmark_result(output, result)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
