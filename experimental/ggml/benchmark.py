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
import statistics
import subprocess
import sys

# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))
import utils

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
from openxla.benchmark import def_types, devices

ALL_DEVICE_NAMES = [device.name for device in devices.ALL_DEVICES]


def _parse_output(output_text):
  # Example output.
  # main:      mem per token =  2011380 bytes
  # main:          load time =   120.92 ms
  # main:        sample time =    73.86 ms
  # main: first predict time =    14.71 ms
  # main:  loop predict time =  2261.72 ms / 11.20 ms per token
  # main:       predict time =  2276.43 ms / 11.21 ms per token
  # main:         total time =  2494.66 ms

  LOAD_TIME_REGEXP = re.compile(f"main:          load time =   (.+) ms")
  match = LOAD_TIME_REGEXP.search(output_text)
  if not match:
    "Unable to parse first prediction time"
    return
  load_time_ms = float(match.group(1))

  SAMPLE_TIME_REGEXP = re.compile(f"main:        sample time =    (.+) ms")
  match = SAMPLE_TIME_REGEXP.search(output_text)
  if not match:
    "Unable to parse first prediction time"
    return
  sample_time_ms = float(match.group(1))

  FIRST_PREDICTION_TIME_REGEXP = re.compile(
      f"main: first predict time = (.+) ms")
  match = FIRST_PREDICTION_TIME_REGEXP.search(output_text)
  if not match:
    "Unable to parse first prediction time"
    return
  first_prediction_ms = float(match.group(1))

  LOOP_PREDICTION_TIME_REGEXP = re.compile(
      f"main:  loop predict time =  .+ ms / (.+) ms per token")
  match = LOOP_PREDICTION_TIME_REGEXP.search(output_text)
  if not match:
    "Unable to parse loop prediction time"
    return
  loop_prediction_ms = float(match.group(1))

  TOTAL_PREDICTION_TIME_REGEXP = re.compile(
      f"main:       predict time =  (.+) ms / .+ ms per token")
  match = TOTAL_PREDICTION_TIME_REGEXP.search(output_text)
  if not match:
    "Unable to parse total prediction time"
    return
  total_prediction_ms = float(match.group(1))

  E2E_TIME_REGEXP = re.compile(f"main:         total time =  (.+) ms")
  match = E2E_TIME_REGEXP.search(output_text)
  if not match:
    "Unable to parse total prediction time"
    return
  e2e_prediction_ms = float(match.group(1))

  return {
      "load_time_ms": load_time_ms,
      "first_prediction_ms": first_prediction_ms,
      "loop_prediction_ms": loop_prediction_ms,
      "total_prediction_ms": total_prediction_ms,
      "sample_time_ms": sample_time_ms,
      "e2e_prediction_ms": e2e_prediction_ms,
  }


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run GGML benchmarks.")
  parser.add_argument("-name",
                      "--benchmark_name",
                      type=str,
                      required=True,
                      help="The regex pattern to match benchmark names.")
  parser.add_argument(
      "-b",
      "--benchmark_binary",
      type=pathlib.Path,
      required=True,
      help="Path to benchmark binary e.g. /tmp/ggml/build/bin/gpt2")
  parser.add_argument(
      "-m",
      "--model",
      type=str,
      required=True,
      help=
      "The GGML model to benchmark e.g. /tmp/ggml/build/models/gpt-2-117M/ggml-model.bin"
  )
  parser.add_argument("--data_type", type=str, help="The model data type.")
  parser.add_argument("-p",
                      "--prompt",
                      type=str,
                      default="Once upon a time",
                      help="The input prompt to the model.")
  parser.add_argument("-s",
                      "--seed",
                      type=int,
                      default=0,
                      help="The seed to use for the RNG.")
  parser.add_argument("-t",
                      "--threads",
                      type=int,
                      default=8,
                      help="The number of threads to use.")
  parser.add_argument("-o",
                      "--output",
                      type=pathlib.Path,
                      required=True,
                      help="JSON file path to merge the results.")
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
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")
  return parser.parse_args()


def main(benchmark_name: str, benchmark_binary: pathlib.Path,
         warmup_iterations: int, iterations: int, model: str, data_type: str,
         prompt: str, seed: int, threads: int, output: pathlib.Path,
         target_device_name: str, verbose: bool):

  try:
    target_device = next(device for device in devices.ALL_DEVICES
                         if device.name == target_device_name)
  except StopIteration:
    raise ValueError(f'Target device "{target_device_name}" is not defined.'
                     f' Available device options:\n{ALL_DEVICE_NAMES}')

  benchmark_definition = {
      "benchmark_name": benchmark_name,
      "framework": str(def_types.ModelFrameworkType.GGML),
      "data_type": data_type,
      "batch_size": 1,
      "compiler": str(def_types.ModelFrameworkType.GGML),
      "device": target_device.name,
      "num_threads": threads,
      "warmup_iterations": warmup_iterations,
      "num_iterations": iterations,
      "tags": ["gpt2", "ggml"],
  }

  cmd = [
      benchmark_binary,
      "--model",
      f"{model}",
      "--prompt",
      f"{prompt}",
      "--seed",
      f"{seed}",
      "--threads",
      f"{threads}",
  ]

  # Run warmup iterations.
  for i in range(warmup_iterations):
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  load_times = []
  first_prediction_times = []
  loop_prediction_times = []
  total_prediction_times = []
  sample_times = []
  e2e_prediction_times = []

  # Run iterations.
  for i in range(iterations):
    raw_result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    raw_result = raw_result.stdout.decode("utf-8")
    metrics = _parse_output(raw_result)

    load_times.append(metrics["load_time_ms"])
    first_prediction_times.append(metrics["first_prediction_ms"])
    loop_prediction_times.append(metrics["loop_prediction_ms"])
    total_prediction_times.append(metrics["total_prediction_ms"])
    sample_times.append(metrics["sample_time_ms"])
    e2e_prediction_times.append(metrics["e2e_prediction_ms"])

  benchmark_metrics = {
      "median_load_time_ms":
          statistics.median(load_times) if load_times else None,
      "median_first_prediction_ms":
          statistics.median(first_prediction_times)
          if first_prediction_times else None,
      "median_loop_prediction_ms":
          statistics.median(loop_prediction_times)
          if loop_prediction_times else None,
      "median_total_prediction_ms":
          statistics.median(total_prediction_times)
          if total_prediction_times else None,
      "median_sample_time_ms":
          statistics.median(sample_times) if sample_times else None,
      "median_e2e_prediction_times":
          statistics.median(e2e_prediction_times)
          if e2e_prediction_times else None,
  }

  benchmark_result = utils.BenchmarkResult(
      definition=benchmark_definition,
      metrics={
          "compiler_level": benchmark_metrics,
      },
  )

  if verbose:
    print(json.dumps(dataclasses.asdict(benchmark_result), indent=2))
  utils.append_benchmark_result(output, benchmark_result)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
