# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

from openxla.benchmark.comparative_suite import utils
from openxla.benchmark.comparative_suite.tf import model_definitions, test_data_definitions

T5_LARGE_FP32_TF_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_FP32_TF_512XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_T5_LARGE_FP32_TF_512XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_T5_LARGE_FP32_TF_512X1024XF32_BATCHES,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)

BERT_LARGE_FP32_TF_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_FP32_TF_384XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_BERT_LARGE_FP32_TF_384XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_BERT_LARGE_FP32_TF_384X1024XF32_BATCHES,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)

RESNET50_FP32_TF_3X224X224XF32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.RESNET50_FP32_TF_3X224X224XF32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_RESNET50_FP32_TF_3X224X224XF32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_RESNET50_FP32_TF_2048X7X7XF32_BATCHES,
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)

EFFICIENTNETB7_FP32_TF_600X600X3XF32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_EFFICIENTNETB7_FP32_TF_1000XF32_BATCHES,
    batch_sizes=[1, 64, 128],
)

ALL_BENCHMARKS = list(
    itertools.chain(
        T5_LARGE_FP32_TF_512XI32_CASES.values(),
        BERT_LARGE_FP32_TF_384XI32_CASES.values(),
        RESNET50_FP32_TF_3X224X224XF32_CASES.values(),
        EFFICIENTNETB7_FP32_TF_600X600X3XF32_CASES.values(),
    ))
