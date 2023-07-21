#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import importlib
import numpy as np
import pathlib
import statistics
import sys
import tensorflow as tf
import time

from typing import Any, Dict, Optional, Sequence, Tuple

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))

from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.tf import benchmark_definitions
from openxla.benchmark.models import model_interfaces
import benchmark_lib

_HLO_DUMP_DIR = "/tmp/hlo_dump"
_TF_CPU_DEVICE = "/CPU:0"
_TF_GPU_DEVICE = "/GPU:0"


def bytes_to_mb(bytes: Optional[int]) -> Optional[float]:
  return None if bytes is None else bytes / 1e6


def _run_framework_benchmark(
    model: def_types.Model,
    input_npys: Sequence[pathlib.Path],
    warmup_iterations: int,
    benchmark_iterations: int,
    backend: str,
    verbose: bool,
) -> Tuple[Dict[str, Any], Any]:
  tf_device = _TF_GPU_DEVICE if backend == "gpu" else _TF_CPU_DEVICE

  try:
    with tf.device(tf_device):
      if tf_device == _TF_GPU_DEVICE:
        tf.config.experimental.reset_memory_stats(tf_device)

      model_module = importlib.import_module(model.model_impl.module_path)
      model_obj: model_interfaces.InferenceModel = model_module.create_model(
          **model.model_parameters)

      inputs = [np.load(path) for path in input_npys]

      # Run warmup.
      warmup_latencies = []
      for i in range(warmup_iterations):
        start = time.perf_counter()
        outputs = model_obj.forward(inputs)
        tf.test.experimental.sync_devices()
        end = time.perf_counter()
        warmup_latencies.append(1000 * (end - start))

      # Run benchmark.
      latencies = []
      last_outputs = None
      for i in range(benchmark_iterations):
        start = time.perf_counter()
        last_outputs = model_obj.forward(inputs)
        tf.test.experimental.sync_devices()
        end = time.perf_counter()
        latencies.append(1000 * (end - start))

      if last_outputs is None:
        raise ValueError("No benchmark runs.")

      # Retrieve memory stats.
      if tf_device == _TF_GPU_DEVICE:
        memory_info = tf.config.experimental.get_memory_info(tf_device)
        device_peak_b = memory_info["peak"]
      else:
        # tf.config.experimental does not currently support measuring CPU memory usage.
        device_peak_b = None
      device_peak_mb = bytes_to_mb(device_peak_b)

      # Roughly calculate compile time.
      compile_time_ms = None if not warmup_latencies else (
          max(warmup_latencies) - statistics.median(latencies))

  except Exception as e:
    print(f"Failed to benchmark model {model.name}. Exception: {e}")
    raise

  metrics = {
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
      "compile_time_ms":
          compile_time_ms,
      "device_memory_peak_mb":
          device_peak_mb,
  }
  return (metrics, last_outputs)


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run TF benchmarks with XLA backend.")
  benchmark_lib.configure_parser(parser)
  return parser.parse_args()


def main(**kwargs):
  benchmark_lib.benchmark(benchmark_function=_run_framework_benchmark,
                          benchmark_cases=benchmark_definitions.ALL_BENCHMARKS,
                          compiler="xla",
                          **kwargs)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
