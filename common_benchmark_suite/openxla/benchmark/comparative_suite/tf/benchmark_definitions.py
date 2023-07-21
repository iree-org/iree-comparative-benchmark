# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

from openxla.benchmark.comparative_suite import utils
from openxla.benchmark.comparative_suite.tf import model_definitions

T5_LARGE_FP32_TF_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_FP32_TF_512XI32_BATCHES,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)

BERT_LARGE_FP32_TF_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_FP32_TF_384XI32_BATCHES,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)

RESNET50_FP32_TF_224X224X3XF32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.RESNET50_FP32_TF_224X224X3XF32_BATCHES,
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)

ALL_BENCHMARKS = list(
    itertools.chain(
        T5_LARGE_FP32_TF_512XI32_CASES.values(),
        BERT_LARGE_FP32_TF_384XI32_CASES.values(),
        RESNET50_FP32_TF_224X224X3XF32_CASES.values(),
    ))
