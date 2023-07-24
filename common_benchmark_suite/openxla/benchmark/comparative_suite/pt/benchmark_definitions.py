# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

from openxla.benchmark.comparative_suite import utils
from openxla.benchmark.comparative_suite.pt import model_definitions

# Example benchmarks.
EXAMPLE_FP32_PT_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.EXAMPLE_FP32_PT_BATCHES,
    # See check_tensor_outputs in comparative_benchmark/utils.py for the
    # available parameters.
    verify_parameters={
        "absolute_tolerance": 0.1,
    },
    batch_sizes=[1, 16, 32],
)

BERT_LARGE_FP32_PT_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_FP32_PT_384XI32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)
BERT_LARGE_FP16_PT_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_FP16_PT_384XI32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)
RESNET50_FP32_PT_3X224X224XF32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.RESNET50_FP32_PT_3X224X224XF32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)
RESNET50_FP16_PT_3X224X224XF16_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.RESNET50_FP16_PT_3X224X224XF16_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)

ALL_BENCHMARKS = list(
    itertools.chain(
        EXAMPLE_FP32_PT_CASES.values(),
        BERT_LARGE_FP32_PT_384XI32_CASES.values(),
        BERT_LARGE_FP16_PT_384XI32_CASES.values(),
        RESNET50_FP32_PT_3X224X224XF32_CASES.values(),
        RESNET50_FP16_PT_3X224X224XF16_CASES.values(),
    ))
