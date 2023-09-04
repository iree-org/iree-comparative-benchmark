#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import pathlib
import sys

import benchmark_lib

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
from openxla.benchmark import def_types, devices


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run GGML benchmarks.")
  benchmark_lib.configure_parser(parser)
  return parser.parse_args()


def main(benchmark_name: str, benchmark_binary: pathlib.Path,
         warmup_iterations: int, iterations: int, model: pathlib.Path,
         data_type: str, prompt: str, seed: int, threads: str,
         output: pathlib.Path, target_device_name: str, verbose: bool):

  try:
    target_device = next(device for device in devices.ALL_DEVICES
                         if device.name == target_device_name)
  except StopIteration:
    raise ValueError(f'Target device "{target_device_name}" is not defined.'
                     f' Available device options:\n{devices.ALL_DEVICE_NAMES}')

  threads = threads.split(",")
  for thread in threads:
    benchmark_definition = {
        "benchmark_name": benchmark_name,
        "framework": str(def_types.ModelFrameworkType.GGML),
        "data_type": data_type,
        "batch_size": 1,
        "compiler": str(def_types.ModelFrameworkType.GGML),
        "device": target_device.name,
        "num_threads": thread,
        "warmup_iterations": warmup_iterations,
        "num_iterations": iterations,
        "tags": ["gpt2", "ggml"],
    }

    cmd = [
        benchmark_binary, "--model", model, "--prompt", f"\"{prompt}\"",
        "--seed",
        str(seed), "--threads",
        str(thread)
    ]

    benchmark_lib.benchmark(cmd, benchmark_definition, warmup_iterations,
                            iterations, output, verbose)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
