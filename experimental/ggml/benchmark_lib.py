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
from openxla.benchmark import devices

# Regular expressions to parse GGML benchmark output.
# Example output:
# main:      mem per token =  2011380 bytes
# main:          load time =   120.92 ms
# main:        sample time =    73.86 ms
# main: first predict time =    14.71 ms
# main:  loop predict time =  2261.72 ms / 11.20 ms per token
# main:       predict time =  2276.43 ms / 11.21 ms per token
# main:         total time =  2494.66 ms
LOAD_TIME_REGEXP = re.compile(f".+ load time = (.+) ms")
SAMPLE_TIME_REGEXP = re.compile(f".+ sample time = (.+) ms")
FIRST_PREDICTION_TIME_REGEXP = re.compile(f".+ first predict time = (.+) ms")
LOOP_PREDICTION_TIME_REGEXP = re.compile(
    f".+ loop predict time = .+ ms / (.+) ms per token")
TOTAL_PREDICTION_TIME_REGEXP = re.compile(
    f".+ predict time = (.+) ms / .+ ms per token")
E2E_TIME_REGEXP = re.compile(f".+ total time = (.+) ms")


def _parse_output(output_text):
  match = LOAD_TIME_REGEXP.search(output_text)
  load_time_ms = float(match.group(1)) if match else print(
      "Unable to parse first prediction time")

  match = SAMPLE_TIME_REGEXP.search(output_text)
  sample_time_ms = float(match.group(1)) if match else print(
      "Unable to parse first prediction time")

  match = FIRST_PREDICTION_TIME_REGEXP.search(output_text)
  first_prediction_ms = float(match.group(1)) if match else print(
      "Unable to parse first prediction time")

  match = LOOP_PREDICTION_TIME_REGEXP.search(output_text)
  loop_prediction_ms = float(match.group(1)) if match else print(
      "Unable to parse loop prediction time")

  match = TOTAL_PREDICTION_TIME_REGEXP.search(output_text)
  total_prediction_ms = float(match.group(1)) if match else print(
      "Unable to parse total prediction time")

  match = E2E_TIME_REGEXP.search(output_text)
  e2e_prediction_ms = float(match.group(1)) if match else print(
      "Unable to parse total prediction time")

  return {
      "load_time_ms": load_time_ms,
      "first_prediction_ms": first_prediction_ms,
      "loop_prediction_ms": loop_prediction_ms,
      "total_prediction_ms": total_prediction_ms,
      "sample_time_ms": sample_time_ms,
      "e2e_prediction_ms": e2e_prediction_ms,
  }


def configure_parser(parser: argparse.ArgumentParser):
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
      "--benchmark_library",
      type=pathlib.Path,
      required=True,
      help="Path to benchmark library e.g. /tmp/ggml/build/src/libggml.so")
  parser.add_argument(
      "-m",
      "--model",
      type=pathlib.Path,
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
                      type=str,
                      default="1,4",
                      help="A comma-delimited list of threads.")
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
                      choices=devices.ALL_DEVICE_NAMES,
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


def benchmark(benchmark_command: str, benchmark_definition: dict,
              warmup_iterations: int, iterations: int, output: pathlib.Path,
              verbose: bool):

  # Run warmup iterations.
  for i in range(warmup_iterations):
    subprocess.run(benchmark_command,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT)

  load_times = []
  first_prediction_times = []
  loop_prediction_times = []
  total_prediction_times = []
  sample_times = []
  e2e_prediction_times = []

  # Run iterations.
  for i in range(iterations):
    raw_result = subprocess.run(benchmark_command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    raw_result = raw_result.stdout.decode("utf-8")
    if verbose:
      print(raw_result)

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
