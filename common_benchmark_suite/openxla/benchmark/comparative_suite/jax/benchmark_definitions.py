# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

from openxla.benchmark import def_types, unique_ids
from openxla.benchmark.devices import gcp_devices
from openxla.benchmark.comparative_suite import utils
from openxla.benchmark.comparative_suite.jax import model_definitions, test_data_definitions

T5_LARGE_FP32_JAX_512XI32_A2_HIGHGPU_1G = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_FP32_JAX_512XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32_BATCHES,
    target_device=gcp_devices.GCP_A2_HIGHGPU_1G,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)
T5_LARGE_FP32_JAX_512XI32_C2_STANDARD_16 = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_FP32_JAX_512XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32_BATCHES,
    target_device=gcp_devices.GCP_C2_STANDARD_16,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)
T5_LARGE_4CG_FP32_JAX_512XI32_A2_HIGHGPU_1G = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_4CG_FP32_JAX_512XI32_BATCHES,
    # TODO(mariecwhite): For now we use existing data defitions. Add correct
    # artifacts once artifact generation pipeline is implemented.
    batch_inputs=test_data_definitions.
    INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32_BATCHES,
    target_device=gcp_devices.GCP_A2_HIGHGPU_1G,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)
T5_LARGE_4CG_FP32_JAX_512XI32_C2_STANDARD_16 = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_4CG_FP32_JAX_512XI32_BATCHES,
    # TODO(mariecwhite): For now we use existing data defitions. Add correct
    # artifacts once artifact generation pipeline is implemented.
    batch_inputs=test_data_definitions.
    INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32_BATCHES,
    target_device=gcp_devices.GCP_C2_STANDARD_16,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)

ALL_BENCHMARKS = list(
    itertools.chain(
        T5_LARGE_FP32_JAX_512XI32_A2_HIGHGPU_1G.values(),
        T5_LARGE_4CG_FP32_JAX_512XI32_A2_HIGHGPU_1G.values(),
        T5_LARGE_FP32_JAX_512XI32_C2_STANDARD_16.values(),
        T5_LARGE_4CG_FP32_JAX_512XI32_C2_STANDARD_16.values(),
    ))
