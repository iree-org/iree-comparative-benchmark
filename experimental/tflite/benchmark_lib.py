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
from typing import Sequence, List

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

MIN_MAX_LATENCY_REGEXP = re.compile(
    "INFO: count=\d+ first=\d+ curr=\d+ min=(.*) max=(.*) avg=(.*) std=(.*)")
AVG_LATENCY_REGEXP = re.compile(
    "INFO: Inference timings in us: .* Inference \(avg\): (.*)")
PEAK_MEMORY_REGEXP = re.compile(
    "INFO: Overall peak memory footprint \(MB\) via periodic monitoring: (.*)")


def download_artifacts(benchmarks: Sequence[def_types.BenchmarkCase],
                       root_dir: pathlib.Path,
                       verbose: bool = False):
  """Download benchmark artifacts."""
  download_list = []
  for benchmark in benchmarks:
    model = benchmark.model
    if (model.artifacts_dir_url is None or
        def_types.ModelArtifactType.TFLITE_FP32
        not in model.exported_model_types):
      raise ValueError(f"XLA HLO dump isn't provided by '{model.name}'.")
    model_url = model.artifacts_dir_url + "/" + TFLITE_FP32_FILENAME
    model_path = root_dir / model.name / TFLITE_FP32_FILENAME
    download_list.append((model_url, model_path))

  utils.download_files(download_list, verbose=verbose)


def configure_parser(parser: argparse.ArgumentParser):
  parser.add_argument("-o",
                      "--output",
                      type=pathlib.Path,
                      required=True,
                      help="JSON file path to merge the results.")
  parser.add_argument("-name",
                      "--benchmark_name",
                      required=True,
                      help="The unique id that defines a benchmark.")
  parser.add_argument("-device",
                      "--target_device",
                      dest="target_device_name",
                      type=str,
                      required=True,
                      choices=ALL_DEVICE_NAMES,
                      help="The target device to benchmark.")
  parser.add_argument("--tflite-benchmark-tool",
                      "--tflite_benchmark_tool",
                      type=pathlib.Path,
                      required=True,
                      help="The path to the TFLite `benchmrk_model` tool.")
  parser.add_argument("-t",
                      "--threads",
                      type=str,
                      default="1,4",
                      help="A comma-delimited list of threads.")
  parser.add_argument("-iter",
                      "--iterations",
                      type=int,
                      default=10,
                      help="The number of iterations to benchmark.")
  parser.add_argument("--root-dir",
                      "--root_dir",
                      type=pathlib.Path,
                      default=pathlib.Path("/tmp/openxla-benchmark/tflite"),
                      help="Root directory stores benchmark artifacts.")
  parser.add_argument("--no-download",
                      "--no_download",
                      action="store_true",
                      help="Don't automatically download benchmark artifacts.")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")


def benchmark(benchmark_command: List[str], benchmark_definition: dict,
              iterations: int, verbose: bool) -> utils.BenchmarkResult:
  if verbose:
    print(f"Run command: {benchmark_command}")

  result = subprocess.run(benchmark_command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
  result_text = result.stdout.decode("utf-8")

  if verbose:
    print(result_text)

  min_latency_ms = None
  max_latency_ms = None
  mean_latency_ms = None
  stddev_latency_ms = None
  device_memory_peak_mb = None

  match = AVG_LATENCY_REGEXP.search(result_text)
  if match:
    mean_latency_ms = float(match.group(1)) * 1e-3

  match = MIN_MAX_LATENCY_REGEXP.search(result_text)
  if match:
    min_latency_ms = float(match.group(1)) * 1e-3
    max_latency_ms = float(match.group(2)) * 1e-3
    stddev_latency_ms = float(match.group(4)) * 1e-3

  match = PEAK_MEMORY_REGEXP.search(result_text)
  if match:
    device_memory_peak_mb = float(match.group(1))

  metrics = {
      "min_latency_ms": min_latency_ms,
      "max_latency_ms": max_latency_ms,
      "mean_latency_ms": mean_latency_ms,
      "stddev_latency_ms": stddev_latency_ms,
      "benchmark_iterations": iterations,
      "device_memory_peak_mb": device_memory_peak_mb,
  }

  return utils.BenchmarkResult(
      definition=benchmark_definition,
      metrics={
          "compiler_level": metrics,
      },
  )
