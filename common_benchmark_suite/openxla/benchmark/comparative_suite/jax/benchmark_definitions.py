# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

from openxla.benchmark.devices import gcp_devices
from openxla.benchmark.comparative_suite import utils
from openxla.benchmark.comparative_suite.jax import model_definitions, test_data_definitions

T5_LARGE_FP32_JAX_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_FP32_JAX_512XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_T5_LARGE_FP32_JAX_512XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)
T5_LARGE_FP16_JAX_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_FP16_JAX_512XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_T5_LARGE_FP16_JAX_512XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_T5_LARGE_FP16_JAX_512X1024XF16_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)
T5_LARGE_BF16_JAX_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_BF16_JAX_512XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_T5_LARGE_BF16_JAX_512XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_T5_LARGE_BF16_JAX_512X1024XBF16_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)
T5_4CG_LARGE_FP32_JAX_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_4CG_LARGE_FP32_JAX_512XI32_BATCHES,
    # TODO(mariecwhite): For now we use existing data definitions. Add correct
    # artifacts once artifact generation pipeline is implemented.
    batch_inputs=test_data_definitions.
    INPUT_DATA_T5_LARGE_FP32_JAX_512XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)

BERT_LARGE_FP32_JAX_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_FP32_JAX_384XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_BERT_LARGE_FP32_JAX_384XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_BERT_LARGE_FP32_JAX_384X1024XF32_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)
BERT_LARGE_FP16_JAX_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_FP16_JAX_384XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_BERT_LARGE_FP16_JAX_384XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_BERT_LARGE_FP16_JAX_384X1024XF16_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)
BERT_LARGE_BF16_JAX_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_BF16_JAX_384XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_BERT_LARGE_BF16_JAX_384XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_BERT_LARGE_BF16_JAX_384X1024XBF16_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)

RESNET50_FP32_JAX_3X224X224XF32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.RESNET50_FP32_JAX_3X224X224XF32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_RESNET50_FP32_JAX_3X224X224XF32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_RESNET50_FP32_JAX_2048X7X7XF32_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)
RESNET50_FP16_JAX_3X224X224XF16_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.RESNET50_FP16_JAX_3X224X224XF16_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_RESNET50_FP16_JAX_3X224X224XF16_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_RESNET50_FP16_JAX_2048X7X7XF16_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)
RESNET50_BF16_JAX_3X224X224XBF16_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.RESNET50_BF16_JAX_3X224X224XBF16_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_RESNET50_BF16_JAX_3X224X224XBF16_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_RESNET50_BF16_JAX_2048X7X7XBF16_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)

ALL_BENCHMARKS = list(
    itertools.chain(
        T5_LARGE_FP32_JAX_512XI32_CASES.values(),
        T5_LARGE_FP16_JAX_512XI32_CASES.values(),
        T5_LARGE_BF16_JAX_512XI32_CASES.values(),
        T5_4CG_LARGE_FP32_JAX_512XI32_CASES.values(),
        BERT_LARGE_FP32_JAX_384XI32_CASES.values(),
        BERT_LARGE_FP16_JAX_384XI32_CASES.values(),
        BERT_LARGE_BF16_JAX_384XI32_CASES.values(),
        RESNET50_FP32_JAX_3X224X224XF32_CASES.values(),
        RESNET50_FP16_JAX_3X224X224XF16_CASES.values(),
        RESNET50_BF16_JAX_3X224X224XBF16_CASES.values(),
    ))
