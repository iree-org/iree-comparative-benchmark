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
from typing import Any, Dict, List, Sequence

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))

from openxla.benchmark import def_types, devices
import openxla.benchmark.comparative_suite.jax.benchmark_definitions as jax_benchmark_definitions
import openxla.benchmark.comparative_suite.tf.benchmark_definitions as tf_benchmark_definitions
import utils

ALL_DEVICE_NAMES = [device.name for device in devices.ALL_DEVICES]

TIME_UNITS = {"us": 1e-3, "ms": 1, "s": 1e3, "min": 60 * 1e3, "h": 3600 * 1e3}
TIME_REGEXP = re.compile(r"time: (\d+\.?\d*) (%s)" % "|".join(TIME_UNITS))
SIZE_REGEXP = re.compile(r" (\d+) bytes")
LOG_TIME_REGEXP = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2}\.\d+):")

GPU_COMPILE_TIME_REGEXP = re.compile(
    r"NVPTXCompiler::CompileTargetBinary - CompileToPtx.*")
GPU_PEAK_MEMORY_REGEXP = re.compile(
    r"New Peak memory usage of \d+ bytes for GPU")
GPU_LATENCY_START_REGEXP = re.compile(r".+HloRunner: ExecuteOnDevices started")
GPU_LATENCY_STOP_REGEXP = re.compile(r".+HloRunner: ExecuteOnDevices succeeded")

CPU_COMPILE_TIME_REGEXP = re.compile(r"... compiled and ran in (.*)s.")
CPU_LATENCY_REGEXP = re.compile(r"execution time for runner [A-Za-z]*: (.*)s.")

HLO_FILENAME = "xla_hlo_before_optimizations.txt"


def _parse_log_time(line: str) -> float:
  """Parses timestamp from the standard log."""
  match = LOG_TIME_REGEXP.search(line)
  assert match, "Unable to parse log time: %s" % line
  _, h, m, s = match.groups()
  return 1000 * (int(h) * 3600 + int(m) * 60 + float(s))


def _parse_log_elapsed_time(line1: str, line2: str) -> float:
  """Calculates elapsed time between two log lines."""
  start, end = _parse_log_time(line1), _parse_log_time(line2)
  end += 86400 if end < start else 0  # next day correction
  return end - start


def _parse_gpu_latencies(raw_output: str,
                         expected_iterations: int) -> List[float]:
  """Returns a list of latencies in milliseconds parsed from XLA logs."""
  start_matches = GPU_LATENCY_START_REGEXP.findall(raw_output)
  stop_matches = GPU_LATENCY_STOP_REGEXP.findall(raw_output)

  if len(start_matches) != len(stop_matches):
    print(
        f"Error: Unequal number of start and stop logs. {len(start_matches)} start logs != {len(stop_matches)} stop logs."
    )
    return []

  if len(start_matches) != expected_iterations:
    print(
        f"Error: Number of iterations not equal to the number of expected iteration. Expected {expected_iterations}. Found {len(start_matches)}."
    )
    return []

  latencies = [
      _parse_log_elapsed_time(t1, t2)
      for t1, t2 in zip(start_matches, stop_matches)
  ]
  return latencies


def _parse_log_duration(time_str: str) -> float:
  """Returns the time in milliseconds parsed from XLA logs."""
  match = TIME_REGEXP.search(time_str)
  assert match, "Unable to parse the time on log line"
  exp = TIME_UNITS[match.group(2)]
  return float(match.group(1)) * exp


def _parse_log_size(size_str: str) -> float:
  """Returns the size in bytes parsed from XLA logs."""
  match = SIZE_REGEXP.search(size_str)
  assert match, "Unable to parse the size on log line"
  return float(match.group(1)) * 1e-6


def _parse_gpu_compile_time_ms(raw_output: str) -> float:
  matches = GPU_COMPILE_TIME_REGEXP.findall(raw_output)
  total_compile_time_ms = sum([_parse_log_duration(t1) for t1 in matches])
  return total_compile_time_ms


def _parse_gpu_peak_memory(raw_output: str) -> float:
  matches = GPU_PEAK_MEMORY_REGEXP.findall(raw_output)
  assert matches, "Unable to find peak memory"
  return _parse_log_size(matches[-1])


def _run_compiler_benchmark_gpu(
    hlo_benchmark_tool_path: pathlib.Path,
    hlo_input_path: pathlib.Path,
    benchmark_iterations: int,
    verbose: bool,
) -> Dict[str, Any]:
  cmd = [
      hlo_benchmark_tool_path,
      f"--hlo_file={hlo_input_path}",
      f"--device_type=gpu",
      f"--num_repeats={benchmark_iterations}",
      "--input_format=text",
      "--num_replicas=1",
      "--num_partitions=1",
      "--logtostderr",
  ]
  if verbose:
    print(f"Run command: {cmd}")
  result = subprocess.run(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      # Timings are logged under VLOG so we need to enable this for the modules
      # we are interested in.
      env={
          "TF_CPP_MIN_LOG_LEVEL":
              "0",
          "TF_CPP_VMODULE":
              "nvptx_compiler=1,gpu_compiler=1,parse_flags_from_env=1,bfc_allocator=2,functional_hlo_runner=1",
      })
  result_text = result.stdout.decode("utf-8")

  latencies = _parse_gpu_latencies(result_text, benchmark_iterations)
  compile_time_ms = _parse_gpu_compile_time_ms(result_text)
  peak_memory_usage = _parse_gpu_peak_memory(result_text)

  results_dict = {
      "compile_time_ms": compile_time_ms,
      "min_latency_ms": min(latencies, default=None),
      "max_latency_ms": max(latencies, default=None),
      "mean_latency_ms": statistics.mean(latencies) if latencies else None,
      "median_latency_ms": statistics.median(latencies) if latencies else None,
      "stddev_latency_ms": statistics.stdev(latencies) if latencies else None,
      "benchmark_iterations": benchmark_iterations,
      "device_memory_peak_mb": peak_memory_usage,
  }
  return results_dict


def _run_compiler_benchmark_cpu(
    hlo_benchmark_tool_path: pathlib.Path,
    hlo_input_path: pathlib.Path,
    benchmark_iterations: int,
    verbose: bool,
) -> Dict[str, Any]:
  cmd = [
      hlo_benchmark_tool_path,
      "--input_format=hlo",
      f"--platform=cpu",
      "--reference_platform=",
      "--logtostderr",
      f"--input_module={hlo_input_path}",
      f"--iterations={benchmark_iterations}",
  ]
  if verbose:
    print(f'Run command: {" ".join(cmd)}')
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  result_text = result.stdout.decode("utf-8")

  matches = CPU_COMPILE_TIME_REGEXP.findall(result_text)
  # Take the first iteration compile-time latency. Profiles show that this is
  # where tuning and other initialization occurs. Subsequent calls to compile
  # in the same process will reuse these results.
  compile_time_latency = float(matches[0]) if matches else None

  matches = CPU_LATENCY_REGEXP.findall(result_text)
  assert len(matches) == benchmark_iterations, (
      f"Expected to find {benchmark_iterations} latencies but found "
      f"{len(matches)} instead:\n{result_text}")
  latencies = [float(match) * 1000 for match in matches]

  results_dict = {
      "compile_time_s": compile_time_latency,
      "min_latency_ms": min(latencies, default=None),
      "max_latency_ms": max(latencies, default=None),
      "mean_latency_ms": statistics.mean(latencies) if latencies else None,
      "median_latency_ms": statistics.median(latencies) if latencies else None,
      "stddev_latency_ms": statistics.stdev(latencies) if latencies else None,
      "benchmark_iterations": benchmark_iterations,
  }
  return results_dict


def _run(
    benchmark: def_types.BenchmarkCase,
    target_device: def_types.DeviceSpec,
    iterations: int,
    hlo_tool: pathlib.Path,
    hlo_dump: pathlib.Path,
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
      "compiler": "xla",
      "device": target_device.name,
      "tags": model.model_impl.tags + model.tags,
  }

  # We use different binaries for benchmarking gpu and cpu.
  accelerator = target_device.accelerator_type
  if accelerator == "gpu":
    metrics = _run_compiler_benchmark_gpu(hlo_benchmark_tool_path=hlo_tool,
                                          hlo_input_path=hlo_dump,
                                          benchmark_iterations=iterations,
                                          verbose=verbose)
  elif accelerator == "cpu":
    metrics = _run_compiler_benchmark_cpu(hlo_benchmark_tool_path=hlo_tool,
                                          hlo_input_path=hlo_dump,
                                          benchmark_iterations=iterations,
                                          verbose=verbose)
  else:
    raise ValueError(f"Unsupported accelerator: '{accelerator}'.")

  return utils.BenchmarkResult(
      definition=benchmark_definition,
      metrics={
          "compiler_level": metrics,
      },
  )


def _download_artifacts(benchmarks: Sequence[def_types.BenchmarkCase],
                        root_dir: pathlib.Path,
                        verbose: bool = False):
  """Download benchmark artifacts."""

  download_list = []
  for benchmark in benchmarks:
    model = benchmark.model
    if (model.artifacts_dir_url is None or
        def_types.ModelArtifactType.XLA_HLO_DUMP
        not in model.exported_model_types):
      raise ValueError(f"XLA HLO dump isn't provided by '{model.name}'.")
    model_url = model.artifacts_dir_url + "/" + HLO_FILENAME
    model_path = root_dir / model.name / HLO_FILENAME
    download_list.append((model_url, model_path))

  utils.download_files(download_list, verbose=verbose)


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run XLA-HLO benchmarks.")
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
  parser.add_argument("-iter",
                      "--iterations",
                      type=int,
                      default=10,
                      help="The number of iterations to benchmark.")
  parser.add_argument("--hlo-tool",
                      "--hlo_tool",
                      required=True,
                      help="The path to `run_hlo_module` for CPU benchmarking"
                      " and `hlo_runner_main` for GPU benchmarking.")
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
    target_device_name: str,
    output: pathlib.Path,
    root_dir: pathlib.Path,
    hlo_tool: pathlib.Path,
    iterations: int,
    no_download: bool,
    verbose: bool,
):
  name_pattern = re.compile(f"^{benchmark_name}$")
  all_benchmarks = jax_benchmark_definitions.ALL_BENCHMARKS + tf_benchmark_definitions.ALL_BENCHMARKS
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
    _download_artifacts(benchmarks=benchmarks,
                        root_dir=root_dir,
                        verbose=verbose)

  for benchmark in benchmarks:
    hlo_dump = root_dir / benchmark.model.name / HLO_FILENAME
    if not hlo_dump.exists():
      raise ValueError(f"HLO dump not found: '{hlo_dump}'.")

    result = _run(benchmark=benchmark,
                  target_device=target_device,
                  iterations=iterations,
                  hlo_tool=hlo_tool,
                  hlo_dump=hlo_dump,
                  verbose=verbose)
    if verbose:
      print(json.dumps(dataclasses.asdict(result), indent=2))

    utils.append_benchmark_result(output, result)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
