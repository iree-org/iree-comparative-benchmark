# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

from openxla.benchmark import def_types, testdata
from openxla.benchmark.comparative_suite import utils
from openxla.benchmark.comparative_suite.tflite import model_definitions

BERT_BASE_FP32_TFLITE_I32_INPUT_SEQUENCE_CASES = utils.build_input_sequence_benchmark_cases(
    model_dict=model_definitions.BERT_BASE_FP32_TFLITE_I32_INPUT_SEQUENCES,
    verify_parameters={"absolute_tolerance": 0.5},
    input_sequence_lengths=[8, 32, 64, 128, 256, 512],
)

BERT_BASE_FP16_TFLITE_I32_INPUT_SEQUENCE_CASES = utils.build_input_sequence_benchmark_cases(
    model_dict=model_definitions.BERT_BASE_FP16_TFLITE_I32_INPUT_SEQUENCES,
    verify_parameters={"absolute_tolerance": 0.5},
    input_sequence_lengths=[8, 32, 64, 128, 256, 512],
)

BERT_BASE_DYN_QUANT_TFLITE_I32_INPUT_SEQUENCES_CASES = utils.build_input_sequence_benchmark_cases(
    model_dict=model_definitions.BERT_BASE_DYN_QUANT_TFLITE_I32_INPUT_SEQUENCES,
    verify_parameters={"absolute_tolerance": 2.0},
    input_sequence_lengths=[8, 32, 64, 128, 256, 512],
)

BERT_BASE_INT8_TFLITE_I32_INPUT_SEQUENCES_CASES = utils.build_input_sequence_benchmark_cases(
    model_dict=model_definitions.BERT_BASE_INT8_TFLITE_I32_INPUT_SEQUENCES,
    verify_parameters={"absolute_tolerance": 2.0},
    input_sequence_lengths=[8, 32, 64, 128, 256, 512],
)

VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

VIT_CLASSIFICATION_FP16_TFLITE_3X224X224XF32_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.VIT_CLASSIFICATION_FP16_TFLITE_3X224X224XF32,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 0.5},
)

VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 2.0},
)

VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8_CASE = def_types.BenchmarkCase.build(
    model=model_definitions.VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8,
    input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
    verify_parameters={"absolute_tolerance": 2.0},
)

ALL_BENCHMARKS = list(
    itertools.chain(
        BERT_BASE_FP32_TFLITE_I32_INPUT_SEQUENCE_CASES.values(),
        BERT_BASE_FP16_TFLITE_I32_INPUT_SEQUENCE_CASES.values(),
        BERT_BASE_DYN_QUANT_TFLITE_I32_INPUT_SEQUENCES_CASES.values(),
        BERT_BASE_INT8_TFLITE_I32_INPUT_SEQUENCES_CASES.values(),
    )) + [
        VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32_CASE,
        VIT_CLASSIFICATION_FP16_TFLITE_3X224X224XF32_CASE,
        VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32_CASE,
        VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8_CASE,
    ]
