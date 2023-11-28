#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import subprocess
import time
from typing import Any, Tuple

# Regexes for retrieving memory information.
_VMHWM_REGEX = re.compile(r".*?VmHWM:.*?(\d+) kB.*")
_VMRSS_REGEX = re.compile(r".*?VmRSS:.*?(\d+) kB.*")
_RSSFILE_REGEX = re.compile(r".*?RssFile:.*?(\d+) kB.*")


def run_command_and_monitor_memory_usage(command: str,
                                         verbose: bool = False
                                        ) -> Tuple[str, Any]:
  """Runs `command` and polls for memory consumption statistics.
  Args:
    command: A bash command string that runs the benchmark.
  Returns:
    A tuple containing output and memory usage in the form of [`vmhwm_mb`, `vmrss_mb`, `rssfile_mb`]
  """
  print(f"Running command: {command}")
  benchmark_process = subprocess.Popen(command,
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
    raise RuntimeError(f"Command returned error {benchmark_process.returncode}")

  output = stdout_data.decode()
  if verbose:
    print(output)
  return (output, [vmhwm * 1e-3, vmrss * 1e-3, rssfile * 1e-3])
