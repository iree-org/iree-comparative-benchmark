#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import concurrent.futures
import pathlib
import re
import subprocess
import sys

from typing import Sequence

sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
from openxla.benchmark import def_types, devices
import openxla.benchmark.comparative_suite.jax.benchmark_definitions as jax_benchmark_definitions
import openxla.benchmark.comparative_suite.tflite.benchmark_definitions as tflite_benchmark_definitions

# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))
import utils

ALL_DEVICE_NAMES = [device.name for device in devices.ALL_DEVICES]

IREE_COMPILE_COMMON_FLAGS = [
    "--iree-hal-target-backends=llvm-cpu",
    "--iree-llvmcpu-link-embedded=true",
    "--iree-input-demote-f64-to-f32=false",
    "--iree-input-demote-i64-to-i32=false",
    "--iree-llvmcpu-debug-symbols=false",
    "--iree-vm-emit-polyglot-zip=false",
    "--iree-opt-data-tiling",
    "--iree-llvmcpu-enable-ukernels=all",
    "--iree-llvmcpu-use-decompose-softmax-fuse=false",
]

FRAMEWORK_TO_DIALECT = {
    def_types.ModelFrameworkType.JAX: "stablehlo",
    def_types.ModelFrameworkType.TF_V1: "stablehlo",
    def_types.ModelFrameworkType.TF_V2: "stablehlo",
    def_types.ModelFrameworkType.TFLITE: "tosa",
    def_types.ModelFrameworkType.PYTORCH: "linalg",
}

COMPILE_FLAGS_PER_TARGET = {
    "host-cpu": [
        "--iree-llvmcpu-target-cpu-features=host",
        "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    ],
    "c2-standard-16": [
        "--iree-llvmcpu-target-cpu=cascadelake",
        "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    ],
    "c2-standard-60": [
        "--iree-llvmcpu-target-cpu=cascadelake",
        "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    ],
    "pixel-6-pro": [
        "--iree-llvmcpu-target-triple=aarch64-none-linux-android31",
        "--iree-llvmcpu-target-cpu-features=+fp-armv8,+neon,+crc,+lse,+rdm,+v8.2a,+aes,+sha2,+fullfp16,+dotprod,+rcpc,+ssbs"
    ],
    "pixel-8-pro": [
        "--iree-llvmcpu-target-triple=aarch64-none-linux-android34",
        "--iree-llvmcpu-target-cpu-features=+v9a,+fullfp16,fp-armv8,+neon,+aes,+sha2,+crc,+lse,+rdm,+complxnum,+rcpc,+sha3,+sm4,+dotprod,+fp16fml,+dit,+flagm,+ssbs,+sb,+sve2-aes,+sve2-bitperm,+sve2-sha3,+sve2-sm4,+altnzcv,+fptoint,+bf16,+i8mm,+bti,+mte,+pauth,+perfmon,+predres,+spe,+ras"
    ],
}


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Compile MLIR workloads with IREE.")
  parser.add_argument("-o",
                      "--output",
                      dest="output_dir",
                      type=pathlib.Path,
                      required=True,
                      help="Directory where output artifacts are saved.")
  parser.add_argument("-name",
                      "--benchmark_name",
                      type=str,
                      required=True,
                      help="The regex pattern to match benchmark names.")
  parser.add_argument("-device",
                      "--target_device",
                      dest="target_device_name",
                      type=str,
                      required=True,
                      choices=ALL_DEVICE_NAMES,
                      help="The target device to benchmark.")
  parser.add_argument("--iree-compile-path",
                      "--iree_compile_path",
                      type=pathlib.Path,
                      required=True,
                      help="Path to the IREE Compiler tool.")
  parser.add_argument("--temp-dir",
                      "--temp_dir",
                      type=pathlib.Path,
                      default=pathlib.Path("/tmp/openxla-benchmark"),
                      help="Temporary directory where intermediates are saved.")
  parser.add_argument("--no-download",
                      "--no_download",
                      action="store_true",
                      help="Don't automatically download benchmark artifacts.")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")
  return parser.parse_args()


def _download_artifacts(benchmarks: Sequence[def_types.BenchmarkCase],
                        temp_dir: pathlib.Path,
                        verbose: bool = False):
  """Download benchmark artifacts."""

  download_list = []
  for benchmark in benchmarks:
    model = benchmark.model
    artifacts_dir_url = model.artifacts_dir_url
    if artifacts_dir_url is None:
      raise ValueError(f"Artifacts dir URL is not found in '{model.name}'.")

    if def_types.ModelArtifactType.STABLEHLO_MLIR in model.exported_model_types:
      mlir_path = temp_dir / model.name / "stablehlo.mlirbc"
      mlir_url = artifacts_dir_url + "/stablehlo.mlirbc"
      download_list.append((mlir_url, mlir_path))

    if def_types.ModelArtifactType.TOSA_MLIR in model.exported_model_types:
      mlir_path = temp_dir / model.name / "tosa.mlirbc"
      mlir_url = artifacts_dir_url + "/tosa.mlirbc"
      download_list.append((mlir_url, mlir_path))

    input_path = temp_dir / model.name / "inputs_npy.tgz"
    input_url = artifacts_dir_url + "/inputs_npy.tgz"
    download_list.append((input_url, input_path))

    expect_path = temp_dir / model.name / "outputs_npy.tgz"
    expect_url = artifacts_dir_url + "/outputs_npy.tgz"
    download_list.append((expect_url, expect_path))

  utils.download_files(download_list, verbose=verbose)


def _compile_model(benchmark: def_types.BenchmarkCase,
                   iree_compile_path: pathlib.Path, temp_dir: pathlib.Path,
                   output_dir: pathlib.Path, target_device_name: str,
                   verbose: bool):
  try:
    model = benchmark.model
    model_dir = output_dir / model.name
    model_dir.mkdir(exist_ok=True)
    artifacts_dir = temp_dir / model.name

    # Copy input and output data.
    subprocess.run(
        ["cp", "-r",
         str(artifacts_dir / "inputs_npy"),
         str(model_dir)],
        check=True)
    subprocess.run(
        ["cp", "-r",
         str(artifacts_dir / "outputs_npy"),
         str(model_dir)],
        check=True)

    dialect = FRAMEWORK_TO_DIALECT[model.model_impl.framework_type]
    target_specific_flags = COMPILE_FLAGS_PER_TARGET[target_device_name]

    # Compile.
    compile_command = [
        str(iree_compile_path),
        str(artifacts_dir / f"{dialect}.mlirbc"),
        f"--iree-input-type={dialect}",
    ] + IREE_COMPILE_COMMON_FLAGS + target_specific_flags + [
        "-o",
        str(model_dir / "module.vmfb"),
    ]
    if verbose:
      print(" ".join(compile_command))

    output = subprocess.run(compile_command, check=True, capture_output=True)
    if verbose:
      print(output.stdout.decode())
  except Exception as e:
    print(f"Failed to compile model {benchmark.model.name}. Exception: {e}")


def _compile_models(benchmarks: Sequence[def_types.BenchmarkCase],
                    iree_compile_path: pathlib.Path,
                    temp_dir: pathlib.Path,
                    output_dir: pathlib.Path,
                    target_device_name: str,
                    verbose: bool = False):
  with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    for benchmark in benchmarks:
      executor.submit(_compile_model, benchmark, iree_compile_path, temp_dir,
                      output_dir, target_device_name, verbose)


def main(
    benchmark_name: str,
    target_device_name: str,
    iree_compile_path: pathlib.Path,
    output_dir: pathlib.Path,
    temp_dir: pathlib.Path,
    no_download: bool,
    verbose: bool,
):
  name_pattern = re.compile(f"^{benchmark_name}$")
  all_benchmarks = jax_benchmark_definitions.ALL_BENCHMARKS + tflite_benchmark_definitions.ALL_BENCHMARKS
  benchmarks = [
      benchmark for benchmark in all_benchmarks
      if name_pattern.match(benchmark.name)
  ]

  if not benchmarks:
    all_benchmark_names = "\n".join(
        benchmark.name for benchmark in all_benchmarks)
    raise ValueError(f'No benchmark matches "{benchmark_name}".'
                     f' Available benchmarks:\n{all_benchmark_names}')

  if target_device_name not in ALL_DEVICE_NAMES:
    raise ValueError(f'Target device "{target_device_name}" is not defined.'
                     f' Available device options:\n{ALL_DEVICE_NAMES}')

  if not no_download:
    _download_artifacts(benchmarks=benchmarks,
                        temp_dir=temp_dir,
                        verbose=verbose)

  _compile_models(benchmarks=benchmarks,
                  iree_compile_path=iree_compile_path,
                  temp_dir=temp_dir,
                  output_dir=output_dir,
                  target_device_name=target_device_name,
                  verbose=verbose)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
