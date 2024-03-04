# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

from openxla.benchmark import def_types, testdata
from openxla.benchmark.comparative_suite import utils
from openxla.benchmark.comparative_suite.jax import model_definitions

T5_LARGE_FP32_JAX_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_FP32_JAX_512XI32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)
T5_LARGE_FP16_JAX_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_FP16_JAX_512XI32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)
T5_LARGE_BF16_JAX_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_LARGE_BF16_JAX_512XI32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)
T5_4CG_LARGE_FP32_JAX_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.T5_4CG_LARGE_FP32_JAX_512XI32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 16, 24, 32, 48],
)

BERT_LARGE_FP32_JAX_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_FP32_JAX_384XI32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)
BERT_LARGE_FP16_JAX_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_FP16_JAX_384XI32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)
BERT_LARGE_BF16_JAX_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_BF16_JAX_384XI32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)

RESNET50_FP32_JAX_3X224X224XF32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.RESNET50_FP32_JAX_3X224X224XF32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)
RESNET50_FP16_JAX_3X224X224XF16_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.RESNET50_FP16_JAX_3X224X224XF16_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)
RESNET50_BF16_JAX_3X224X224XBF16_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.RESNET50_BF16_JAX_3X224X224XBF16_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)

GPT2LMHEAD_FP32_JAX_512XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.GPT2LMHEAD_FP32_JAX_512XI32_BATCHES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    batch_sizes=[1, 64, 128],
)

T5_4CG_SMALL_FP32_JAX_1X128XI32_GEN_CASES = utils.build_gen_benchmark_cases(
    model_dict=model_definitions.T5_4CG_SMALL_FP32_JAX_1X128XI32_GENS,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    gen_sizes=[16, 32, 64, 128, 256],
)

BERT_BASE_FP32_JAX_I32_INPUT_SEQUENCE_CASES = utils.build_input_sequence_benchmark_cases(
    model_dict=model_definitions.BERT_BASE_FP32_JAX_I32_INPUT_SEQUENCES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    input_sequence_lengths=[8, 32, 64, 128, 256, 512],
)

BERT_BASE_FP16_JAX_I32_INPUT_SEQUENCE_CASES = utils.build_input_sequence_benchmark_cases(
    model_dict=model_definitions.BERT_BASE_FP16_JAX_I32_INPUT_SEQUENCES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    input_sequence_lengths=[8, 32, 64, 128, 256, 512],
)

BERT_BASE_BF16_JAX_I32_INPUT_SEQUENCE_CASES = utils.build_input_sequence_benchmark_cases(
    model_dict=model_definitions.BERT_BASE_BF16_JAX_I32_INPUT_SEQUENCES,
    verify_parameters={
        "absolute_tolerance": 0.5,
    },
    input_sequence_lengths=[8, 32, 64, 128, 256, 512],
)

T5_SMALL_FP32_JAX_1X128XI32_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.T5_SMALL_FP32_JAX_1X128XI32,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

VIT_CLASSIFICATION_JAX_3X224X224XF32_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.T5_SMALL_FP32_JAX_1X128XI32,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

GPT2LMHEAD_PIPELINE_JAX_1X4XI32_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.VIT_CLASSIFICATION_JAX_3X224X224XF32,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

SD_PIPELINE_FP32_JAX_64XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.SD_PIPELINE_FP32_JAX_64XI32_BATCHES,
    verify_parameters={"absolute_tolerance": 0.5},
    batch_sizes=[1, 8],
)

SD_PIPELINE_FP16_JAX_64XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.SD_PIPELINE_FP16_JAX_64XI32_BATCHES,
    verify_parameters={"absolute_tolerance": 0.5},
    batch_sizes=[1, 8],
)

SD_PIPELINE_BF16_JAX_64XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.SD_PIPELINE_BF16_JAX_64XI32_BATCHES,
    verify_parameters={"absolute_tolerance": 0.5},
    batch_sizes=[1, 8],
)

GEMMA2BIT_GREEDY_FP32_JAX_1X1024XI32_256XI32_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.GEMMA2BIT_GREEDY_FP32_JAX_1X1024XI32_256XI32,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

GEMMA2BIT_GREEDY_BF16_JAX_1X1024XI32_256XI32_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.GEMMA2BIT_GREEDY_BF16_JAX_1X1024XI32_256XI32,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

GEMMA2BIT_GREEDY_FP16_JAX_1X1024XI32_256XI32_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.GEMMA2BIT_GREEDY_FP16_JAX_1X1024XI32_256XI32,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

DOT_PRODUCT_JAX_1X256X2048XI8I8_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.DOT_PRODUCT_JAX_1X256X2048XI8I8,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

DOT_PRODUCT_JAX_1X256X2048XI8I4_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.DOT_PRODUCT_JAX_1X256X2048XI8I4,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

DOT_PRODUCT_JAX_256X256X2048XI8I8_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.DOT_PRODUCT_JAX_256X256X2048XI8I8,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

DOT_PRODUCT_JAX_256X256X2048XI8I4_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.DOT_PRODUCT_JAX_256X256X2048XI8I4,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
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
        GPT2LMHEAD_FP32_JAX_512XI32_CASES.values(),
        # Models with different gen sizes.
        T5_4CG_SMALL_FP32_JAX_1X128XI32_GEN_CASES.values(),
        # Models with different input sequences.
        BERT_BASE_FP32_JAX_I32_INPUT_SEQUENCE_CASES.values(),
        BERT_BASE_FP16_JAX_I32_INPUT_SEQUENCE_CASES.values(),
        BERT_BASE_BF16_JAX_I32_INPUT_SEQUENCE_CASES.values(),
        # Pipelines.
        SD_PIPELINE_FP32_JAX_64XI32_CASES.values(),
        SD_PIPELINE_FP16_JAX_64XI32_CASES.values(),
        SD_PIPELINE_BF16_JAX_64XI32_CASES.values(),
    )) + [
        GPT2LMHEAD_PIPELINE_JAX_1X4XI32_CASE,
        T5_SMALL_FP32_JAX_1X128XI32_CASE,
        GPT2LMHEAD_PIPELINE_JAX_1X4XI32_CASE,
        GEMMA2BIT_GREEDY_FP32_JAX_1X1024XI32_256XI32_CASE,
        GEMMA2BIT_GREEDY_BF16_JAX_1X1024XI32_256XI32_CASE,
        GEMMA2BIT_GREEDY_FP16_JAX_1X1024XI32_256XI32_CASE,
        DOT_PRODUCT_JAX_1X256X2048XI8I8_CASE,
        DOT_PRODUCT_JAX_1X256X2048XI8I4_CASE,
        DOT_PRODUCT_JAX_256X256X2048XI8I8_CASE,
        DOT_PRODUCT_JAX_256X256X2048XI8I4_CASE,
    ]
