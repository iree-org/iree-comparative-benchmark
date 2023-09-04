#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import pathlib
import subprocess
import sys

import benchmark_lib

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
from openxla.benchmark import def_types, devices


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run GGML benchmarks.")
  parser.add_argument(
      "--tasksets",
      type=str,
      default="f0",
      help=
      "A comma-separated list of tasksets to run under each thread configuration."
  )
  benchmark_lib.configure_parser(parser)
  return parser.parse_args()


def main(benchmark_name: str, benchmark_binary: pathlib.Path,
         warmup_iterations: int, iterations: int, model: pathlib.Path,
         data_type: str, prompt: str, seed: int, threads: str, tasksets: str,
         output: pathlib.Path, target_device_name: str, verbose: bool):
  try:
    target_device = next(device for device in devices.ALL_DEVICES
                         if device.name == target_device_name)
  except StopIteration:
    raise ValueError(f'Target device "{target_device_name}" is not defined.'
                     f' Available device options:\n{devices.ALL_DEVICE_NAMES}')

  threads = threads.split(",")
  tasksets = tasksets.split(",")
  if len(threads) != len(tasksets):
    raise ValueError(
        "The number of tasksets specified must be equal to the number of threads."
    )

  # Push artifacts to the Android device.
  subprocess.run(["adb", "push", benchmark_binary, "/data/local/tmp"])
  subprocess.run([
      "adb", "shell", "chmod", "+x", f"/data/local/tmp/{benchmark_binary.name}"
  ])
  subprocess.run(["adb", "push", model, "/data/local/tmp"])

  for taskset, thread in zip(tasksets, threads):
    benchmark_definition = {
        "benchmark_name": benchmark_name,
        "framework": str(def_types.ModelFrameworkType.GGML),
        "data_type": data_type,
        "batch_size": 1,
        "compiler": str(def_types.ModelFrameworkType.GGML),
        "device": target_device.name,
        "taskset": taskset,
        "num_threads": thread,
        "warmup_iterations": warmup_iterations,
        "num_iterations": iterations,
        "tags": ["gpt2", "ggml"],
    }

    cmd = [
        "adb", "shell", "taskset", taskset,
        f"/data/local/tmp/{benchmark_binary.name}", "--model",
        f"/data/local/tmp/{model.name}", "--prompt", f"\"{prompt}\"", "--seed",
        str(seed), "--threads",
        str(thread)
    ]

    benchmark_lib.benchmark(cmd, benchmark_definition, warmup_iterations,
                            iterations, output, verbose)

  # Cleanup.
  subprocess.run(["adb", "rm", f"/data/local/tmp/{benchmark_binary.name}"])
  subprocess.run(["adb", "rm", f"/data/local/tmp/{model.name}"])


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
