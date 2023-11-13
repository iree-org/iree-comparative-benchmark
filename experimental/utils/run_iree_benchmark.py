#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import pathlib
import re

from typing import Any, Dict

from common import command_lib

_LATENCY_REGEX = re.compile(r".*?BM_main/process_time/real_time\s+(.*?) ms.*")
_IREE_MEM_PEAK_REGEX = re.compile(r".*?DEVICE_LOCAL: (.*?)B peak .*")


def run_benchmark_command(benchmark_command: str,
                          verbose: bool = False) -> Dict[str, Any]:
  """Runs `benchmark_command` and polls for memory consumption statistics.
  Args:
    benchmark_command: A bash command string that runs the benchmark.
  Returns:
    An dictionary containing latency and memory usage metrics.
  """
  try:
    output, mem_stats = command_lib.run_command_and_monitor_memory_usage(
        benchmark_command, verbose)

    match = _LATENCY_REGEX.search(output)
    if not match:
      return {"error": f"Could not parse latency: {output}"}
    latency_ms = float(match.group(1))

    match = _IREE_MEM_PEAK_REGEX.search(output)
    if not match:
      return {"error": f"Could not parse IREE peak memory: {output}"}
    # Convert bytes to MB.
    iree_mem_peak = float(match.group(1).strip()) * 1e-6

    return {
        "median_latency_ms": latency_ms,
        "device_memory_peak_mb": iree_mem_peak,
        "system_memory_vmhwm_mb": mem_stats[0],
        "system_memory_vmrss_mb": mem_stats[1],
        "system_memory_rssfile_mb": mem_stats[2],
    }
  except Exception as e:
    return {"error": f"Failed to run command: {e}."}


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Runs benchmarks.")
  # We need to store the command in a text file because argparse is unable to
  # ignore quoted string with dashes in it, instead interpreting them as
  # arguments.
  parser.add_argument("--command_path",
                      type=pathlib.Path,
                      required=True,
                      help="The command to run stored in a text file.")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")
  return parser.parse_args()


def main(command_path: pathlib.Path, verbose: bool = False):
  results = run_benchmark_command(command_path.read_text(), verbose)
  print(f"results_dict: {results}")


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
