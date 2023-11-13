#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# A simple hard-coded script that generates tables comparing IREE vs TFLite.

import argparse
import json
import pandas as pd
import pathlib
import sys

from datetime import date
from typing import Any

from common import html_utils

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))

from openxla.benchmark.comparative_suite.tflite import benchmark_definitions

_TABLES = [
    "BERT_BASE_FP32_TFLITE_I32",
    "BERT_BASE_FP16_TFLITE_I32",
    "BERT_BASE_DYN_QUANT_TFLITE",
    "VIT_CLASSIFICATION",
]

_MODEL = "model"
_DEVICE = "device"
_IREE_LATENCY = "IREE latency (ms)"
_TFLITE_LATENCY = "TFLite latency (ms)"
_IREE_VS_TFLITE_LATENCY = "IREE vs TFLite Latency"
_IREE_MEMORY = "IREE Peak Memory Usage (MB)"
_TFLITE_MEMORY = "TFLite Peak Memory Usage (MB)"
_IREE_VS_TFLITE_MEMORY = "IREE vs TFLite Memory"

_COMPARISON_COLUMNS = [_IREE_VS_TFLITE_LATENCY, _IREE_VS_TFLITE_MEMORY]
_LATENCY_COLUMNS = [_IREE_LATENCY, _TFLITE_LATENCY]
_MEMORY_COLUMNS = [_IREE_MEMORY, _TFLITE_MEMORY]


def read_json_file(path):
  with open(path, "r") as json_file:
    data = json.load(json_file)
  return data


def _get_best_result(benchmark_name: str, json_data: Any) -> Any:
  benchmarks = []
  for benchmark in json_data:
    definition = benchmark["definition"]
    if benchmark_name in definition["benchmark_name"]:
      benchmarks.append(benchmark)

  best_results_index = -1
  lowest_latency = -1

  for i, benchmark in enumerate(benchmarks):
    metrics = benchmark["metrics"]["compiler_level"]
    if "median_latency_ms" in metrics:
      latency = metrics["median_latency_ms"]
    elif "mean_latency_ms" in metrics:
      latency = metrics["mean_latency_ms"]
    else:
      # No latency found, keep going.
      continue

    if lowest_latency < 0 or latency < lowest_latency:
      lowest_latency = latency
      best_results_index = i

  if best_results_index < 0:
    return None
  else:
    return benchmarks[best_results_index]


def format_table(table: pd.DataFrame) -> pd.DataFrame:
  table = table.set_properties(
      subset=[_MODEL],
      **{
          "width": "500px",
          "text-align": "left",
      },
  )
  table = table.set_properties(
      subset=[_DEVICE],
      **{
          "width": "150",
          "text-align": "center",
      },
  )
  table = table.set_properties(
      subset=_LATENCY_COLUMNS,
      **{
          "width": "100",
          "text-align": "right",
      },
  )
  table = table.set_properties(
      subset=_COMPARISON_COLUMNS,
      **{
          "width": "150px",
          "text-align": "right",
          "color": "#ffffff"
      },
  )
  table = table.set_properties(
      subset=_MEMORY_COLUMNS,
      **{
          "width": "100",
          "text-align": "right",
      },
  )
  table = table.applymap(html_utils.style_latency,
                         subset=[_IREE_VS_TFLITE_LATENCY])
  table = table.applymap(html_utils.style_memory,
                         subset=[_IREE_VS_TFLITE_MEMORY])
  return table


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generates a report comparing IREE with TFLite.")
  parser.add_argument("--iree_results_path",
                      type=pathlib.Path,
                      required=True,
                      help="The path to the IREE json results file.")
  parser.add_argument("--tflite_results_path",
                      type=pathlib.Path,
                      required=True,
                      help="The path to the TFLite json results file.")
  parser.add_argument(
      "--output_path",
      type=pathlib.Path,
      default="/tmp/summary.html",
      help="The path to the output html file that summarizes results.",
  )
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")
  return parser.parse_args()


def main(iree_results_path: pathlib.Path,
         tflite_results_path: pathlib.Path,
         output_path: pathlib.Path,
         verbose: bool = False):
  iree_results = read_json_file(iree_results_path)
  tflite_results = read_json_file(tflite_results_path)

  # Create dataframe per table.
  tables = {}
  for table in _TABLES:
    tables[table] = pd.DataFrame(columns=[
        _MODEL,
        _DEVICE,
        _IREE_LATENCY,
        _TFLITE_LATENCY,
        _IREE_VS_TFLITE_LATENCY,
        _IREE_MEMORY,
        _TFLITE_MEMORY,
        _IREE_VS_TFLITE_MEMORY,
    ])

  for benchmark in benchmark_definitions.ALL_BENCHMARKS:
    iree_benchmark = _get_best_result(benchmark.name,
                                      iree_results["benchmarks"])
    tflite_benchmark = _get_best_result(benchmark.name,
                                        tflite_results["benchmarks"])

    if iree_benchmark and tflite_benchmark:
      # Add result to table group.
      for table_name, table in tables.items():
        if table_name in benchmark.name:
          definition = iree_benchmark["definition"]

          iree_latency = iree_benchmark["metrics"]["compiler_level"][
              "median_latency_ms"]
          tflite_latency = tflite_benchmark["metrics"]["compiler_level"][
              "mean_latency_ms"]
          iree_vs_tflite_latency = html_utils.format_latency_comparison(
              iree_latency, tflite_latency)

          iree_memory = iree_benchmark["metrics"]["compiler_level"][
              "system_memory_vmhwm_mb"]
          tflite_memory = tflite_benchmark["metrics"]["compiler_level"][
              "system_memory_vmhwm_mb"]
          iree_vs_tflite_memory = html_utils.format_memory_comparison(
              iree_memory, tflite_memory)

          table.loc[len(table)] = [
              definition["model_name"],
              definition["device"],
              f"{iree_latency:.1f}",
              f"{tflite_latency:.1f}",
              iree_vs_tflite_latency,
              f"{iree_memory:.3f}",
              f"{tflite_memory:.3f}",
              iree_vs_tflite_memory,
          ]
          break

  version_html = f"<i>last updated: {date.today().isoformat()}</i><br/>"
  version_html += f"<i>raw results: <a href='iree.json'>IREE</a>, <a href='tflite.json'>TFLITE</a></i><br/><br/>"
  html = html_utils.generate_header_and_legend(version_html)

  for table_name, table in tables.items():
    if verbose:
      print(f"{table_name}: {table}")

    table = table.round(2)
    table = table.style.set_table_styles(html_utils.get_table_css())
    table = table.hide(axis="index")
    table = table.set_caption(table_name)
    table = format_table(table)
    html += table.to_html() + "<br/>"

  output_path.write_text(html)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
