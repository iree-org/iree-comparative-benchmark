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
import subprocess
import time

from typing import Any, Dict

# Regexes for retrieving memory information.
_VMHWM_REGEX = re.compile(r".*?VmHWM:.*?(\d+) kB.*")
_VMRSS_REGEX = re.compile(r".*?VmRSS:.*?(\d+) kB.*")
_RSSFILE_REGEX = re.compile(r".*?RssFile:.*?(\d+) kB.*")


def run_benchmark_command(benchmark_command: str,
                          verbose: bool = False) -> Dict[str, Any]:
  """Runs `benchmark_command` and polls for memory consumption statistics.
  Args:
    benchmark_command: A bash command string that runs the benchmark.
  Returns:
    An array containing values for [`latency`, `vmhwm`, `vmrss`, `rssfile`]
  """
  print(f"Running command: {benchmark_command}")
  benchmark_process = subprocess.Popen(benchmark_command,
                                       shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)

  # Keep a record of the highest VmHWM corresponding VmRSS and RssFile values.
  vmhwm = 0
  vmrss = 0
  rssfile = 0
  while benchmark_process.poll() is None:
    pid_status = subprocess.run(
        ["cat", "/proc/" + str(benchmark_process.pid) + "/status"],
        capture_output=True,
    )
    output = pid_status.stdout.decode()
    vmhwm_matches = _VMHWM_REGEX.search(output)
    vmrss_matches = _VMRSS_REGEX.search(output)
    rssfile_matches = _RSSFILE_REGEX.search(output)

    if vmhwm_matches and vmrss_matches and rssfile_matches:
      curr_vmhwm = float(vmhwm_matches.group(1))
      if curr_vmhwm > vmhwm:
        vmhwm = curr_vmhwm
        vmrss = float(vmrss_matches.group(1))
        rssfile = float(rssfile_matches.group(1))

    time.sleep(0.5)

  stdout_data, _ = benchmark_process.communicate()

  if benchmark_process.returncode != 0:
    print(f"Warning! Benchmark command failed with return code:"
          f" {benchmark_process.returncode}")
    return {"error": f"Return code: {benchmark_process.returncode}."}

  output = stdout_data.decode()
  if verbose:
    print(output)

  LATENCY_REGEX = re.compile(r".*?BM_main/process_time/real_time\s+(.*?) ms.*")
  match = LATENCY_REGEX.search(output)
  if not match:
    return {"error": f"Could not parse latency: {output}"}
  latency_ms = float(match.group(1))

  IREE_MEM_PEAK_REGEX = re.compile(r".*?DEVICE_LOCAL: (.*?)B peak .*")
  match = IREE_MEM_PEAK_REGEX.search(output)
  if not match:
    return {"error": f"Could not parse IREE peak memory: {output}"}
  # Convert bytes to MB.
  iree_mem_peak = float(match.group(1).strip()) * 1e-6

  results_dict = {
      "median_latency_ms": latency_ms,
      "device_memory_peak_mb": iree_mem_peak,
      "system_memory_vmhwm_mb": vmhwm * 1e-3,
      "system_memory_vmrss_mb": vmrss * 1e-3,
      "system_memory_rssfile_mb": rssfile * 1e-3,
  }
  return results_dict


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
