#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#   PYTHON: Python interpreter, default: /usr/bin/python3
#   OOBI_VENV_DIR: path to create Python virtualenv, default:
#     iree-pjrt-benchmarks.venv
#   OOBI_TARGET_DEVICE: target benchmark device, can also be specified the first
#     argument.
#   OOBI_OUTPUT: path to output benchmark results, can also be specified the
#     second argument.
#   CUDA_SDK_DIR: path to the CUDA SDK directory.
#
# Example usage:
# ./benchmark_all.sh a2-highgpu-1g /tmp/results.json

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="${OOBI_VENV_DIR:-iree-pjrt-benchmarks.venv}"
PYTHON="${PYTHON:-/usr/bin/python3}"
CUDA_SDK_DIR="${CUDA_SDK_DIR:-/usr/local/cuda}"

TARGET_DEVICE="${1:-${OOBI_TARGET_DEVICE}}"
OUTPUT_PATH="${2:-${OOBI_OUTPUT}}"

# Create a subdirectory for all repos to be cloned into.
rm -rf github
mkdir github
pushd github

# Create virtual environment.
"${PYTHON}" -m venv "${VENV_DIR}" || echo "Could not create venv."
source "${VENV_DIR}/bin/activate" || echo "Could not activate venv"
python -m pip install --upgrade pip || echo "Could not upgrade pip"

VENV_DIR_PATH="$(realpath ${VENV_DIR})"

export IREE_COMPILER_OPTIONS="--iree-hal-target-backends=llvm-cpu --iree-flow-enable-data-tiling --iree-llvmcpu-enable-microkernels --iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false"

# Build pjrt plugin.
git clone -b enable-microkernels https://github.com/openxla/openxla-pjrt-plugin.git
pushd openxla-pjrt-plugin

python ./sync_deps.py
python -m pip install -U -r requirements.txt
python -m pip install -U requests

if [[ ! -d "${CUDA_SDK_DIR}" ]]; then
  CUDA_SDK_DIR=${HOME?}/.iree_cuda_deps
  ../iree/build_tools/docker/context/fetch_cuda_deps.sh ${CUDA_SDK_DIR?}
fi

python ./configure.py --cc=clang --cxx=clang++ --cuda-sdk-dir=$CUDA_SDK_DIR

source .env.sh

echo "IREE_PJRT_COMPILER_LIB_PATH: ${IREE_PJRT_COMPILER_LIB_PATH}"
echo "PJRT_NAMES_AND_LIBRARY_PATHS: ${PJRT_NAMES_AND_LIBRARY_PATHS}"
echo "IREE_CUDA_DEPS_DIR: ${IREE_CUDA_DEPS_DIR}"

# Build.
bazel build iree/integrations/pjrt/...

popd
popd

declare -a GPU_BENCHMARK_NAMES=(
  "models/RESNET50_FP32_JAX_.+"
  "models/BERT_LARGE_FP32_JAX_.+"
  "models/T5_LARGE_FP32_JAX_.+"
  "models/T5_4CG_LARGE_FP32_JAX_.+"
  "models/GPT2LMHEAD_FP32_JAX_.+"
)

declare -a CPU_BENCHMARK_NAMES=(
  "models/RESNET50_FP32_JAX_.+_BATCH1/.+"
  "models/RESNET50_FP32_JAX_.+_BATCH64/.+"
  "models/RESNET50_FP32_JAX_.+_BATCH128/.+"
  "models/BERT_LARGE_FP32_JAX_.+_BATCH1/.+"
  "models/BERT_LARGE_FP32_JAX_.+_BATCH32/.+"
  "models/BERT_LARGE_FP32_JAX_.+_BATCH64/.+"
  "models/T5_LARGE_FP32_JAX_.+_BATCH1/.+"
  "models/T5_LARGE_FP32_JAX_.+_BATCH16/.+"
  "models/T5_LARGE_FP32_JAX_.+_BATCH32/.+"
  "models/T5_4CG_LARGE_FP32_JAX_.+_BATCH1/.+"
  "models/T5_4CG_LARGE_FP32_JAX_.+_BATCH16/.+"
  "models/T5_4CG_LARGE_FP32_JAX_.+_BATCH32/.+"
  "models/GPT2LMHEAD_FP32_JAX_.+_BATCH1/.+"
  "models/GPT2LMHEAD_FP32_JAX_.+_BATCH64/.+"
  "models/GPT2LMHEAD_FP32_JAX_.+_BATCH128/.+"
)

if [ "${TARGET_DEVICE}" = "a2-highgpu-1g" ]; then
  BENCHMARK_NAMES=("${GPU_BENCHMARK_NAMES[@]}")
  ITERATIONS=50
  DEVICE="iree_cuda"
elif [ "${TARGET_DEVICE}" = "c2-standard-16" ]; then
  BENCHMARK_NAMES=("${CPU_BENCHMARK_NAMES[@]}")
  ITERATIONS=10
  DEVICE="iree_cpu"
elif [ "${TARGET_DEVICE}" = "c2-standard-60" ]; then
  BENCHMARK_NAMES=("${CPU_BENCHMARK_NAMES[@]}")
  ITERATIONS=3
  DEVICE="iree_cpu"
else
  echo "Unsupported target device ${TARGET_DEVICE}."
  exit 1
fi

"${TD}/../scripts/create_results_json.sh" "${OUTPUT_PATH}"

python -m pip install --upgrade flax
python -m pip install --upgrade transformers
python -m pip install --upgrade pillow

PYTHON_VERSION="$(python --version | sed -e "s/^Python \(.*\)\.\(.*\)\..*$/\1\.\2/g")"

pip show transformers

# `transformers` is not yet up to date with JAX so we need to hack the source to make it compatible.
BERT_SOURCE_PATH="${VENV_DIR_PATH}/lib/python${PYTHON_VERSION}/site-packages/transformers/models/bert/modeling_flax_bert.py"
sed -i -e "s/: Optional\[jnp.DeviceArray\] = None/=None/g" "${BERT_SOURCE_PATH}"

T5_SOURCE_PATH="${VENV_DIR_PATH}/lib/python${PYTHON_VERSION}/site-packages/transformers/models/t5/modeling_flax_t5.py"
sed -i -e "s/: Optional\[jnp.DeviceArray\] = None/=None/g" "${T5_SOURCE_PATH}"

for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
  JAX_PLATFORMS="${DEVICE}" "${TD}/run_benchmarks.py" \
    --benchmark_name="${benchmark_name}" \
    --output="${OUTPUT_PATH}" \
    --iterations="${ITERATIONS}" \
    --compiler="iree" \
    --verbose
done
