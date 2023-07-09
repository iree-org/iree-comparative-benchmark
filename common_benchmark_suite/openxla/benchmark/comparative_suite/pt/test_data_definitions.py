# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import string

from openxla.benchmark.comparative_suite import utils
from openxla.benchmark import def_types, unique_ids

# Bert large inputs.
INPUT_DATA_BERT_LARGE_FP32_PT_384XI32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    id=utils.BATCH_ID(unique_ids.INPUT_DATA_BERT_LARGE_FP32_PT_384XI32),
    name=utils.BATCH_NAME("INPUT_DATA_BERT_LARGE_FP32_PT_384XI32"),
    tags=["input-data", "seqlen384", utils.BATCH_TAG],
    source_info="Original text: 'a photo of a cat'.",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
            utils.ModelTestDataArtifactTemplate(
                data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
                data_parameters={
                    "tensor_names": ["input_ids", "attention_mask"],
                    "tensor_dimensions": [
                        utils.BATCH_TENSOR_DIMS("384xi32"),
                        utils.BATCH_TENSOR_DIMS("384xi32"),
                    ],
                },
                source_url=string.Template(
                    "https://storage.googleapis.com/iree-model-artifacts/pt/pt_models_20230401.795_1680469670/BERT_LARGE_FP32_PT_384XI32_BATCH${batch_size}/inputs_npy.tgz"
                ),
            )
    })

INPUT_DATA_BERT_LARGE_FP16_PT_384XI32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    id=utils.BATCH_ID(unique_ids.INPUT_DATA_BERT_LARGE_FP16_PT_384XI32),
    name=utils.BATCH_NAME("INPUT_DATA_BERT_LARGE_FP16_PT_384XI32"),
    tags=["input-data", "seqlen384", utils.BATCH_TAG],
    source_info="Original text: 'a photo of a cat'.",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
            utils.ModelTestDataArtifactTemplate(
                data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
                data_parameters={
                    "tensor_names": ["input_ids", "attention_mask"],
                    "tensor_dimensions": [
                        utils.BATCH_TENSOR_DIMS("384xi32"),
                        utils.BATCH_TENSOR_DIMS("384xi32"),
                    ],
                },
                source_url=string.Template(
                    "https://storage.googleapis.com/iree-model-artifacts/pt/pt_models_20230522.846_1684830698/BERT_LARGE_FP16_PT_384XI32_BATCH${batch_size}/inputs_npy.tgz"
                ),
            )
    })

INPUT_DATA_BERT_LARGE_FP32_PT_384XI32_BATCHES = utils.build_batch_model_test_data(
    template=INPUT_DATA_BERT_LARGE_FP32_PT_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])
INPUT_DATA_BERT_LARGE_FP16_PT_384XI32_BATCHES = utils.build_batch_model_test_data(
    template=INPUT_DATA_BERT_LARGE_FP16_PT_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

# Bert large outputs.
OUTPUT_DATA_BERT_LARGE_FP32_PT_384X1024XF32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    id=utils.BATCH_ID(unique_ids.OUTPUT_DATA_BERT_LARGE_FP32_PT_384X1024XF32),
    name=utils.BATCH_NAME("OUTPUT_DATA_BERT_LARGE_FP32_PT_384X1024XF32"),
    tags=["output-data", utils.BATCH_TAG],
    source_info="",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
            utils.ModelTestDataArtifactTemplate(
                data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
                data_parameters={
                    "tensor_names": ["output_0"],
                    "tensor_dimensions": [
                        utils.BATCH_TENSOR_DIMS("384x1024xf32")
                    ],
                },
                verify_parameters={
                    "absolute_tolerance": 0.5,
                },
                source_url=string.Template(
                    "https://storage.googleapis.com/iree-model-artifacts/pt/pt_models_20230401.795_1680469670/BERT_LARGE_FP32_PT_384XI32_BATCH${batch_size}/outputs_npy.tgz"
                ))
    })

OUTPUT_DATA_BERT_LARGE_FP16_PT_384X1024XF16_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    id=utils.BATCH_ID(unique_ids.OUTPUT_DATA_BERT_LARGE_FP16_PT_384X1024XF16),
    name=utils.BATCH_NAME("OUTPUT_DATA_BERT_LARGE_FP16_PT_384X1024XF16"),
    tags=["output-data", utils.BATCH_TAG],
    source_info="",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
            utils.ModelTestDataArtifactTemplate(
                data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
                data_parameters={
                    "tensor_names": ["output_0"],
                    "tensor_dimensions": [
                        utils.BATCH_TENSOR_DIMS("384x1024xf16")
                    ],
                },
                verify_parameters={
                    "absolute_tolerance": 0.5,
                },
                source_url=string.Template(
                    "https://storage.googleapis.com/iree-model-artifacts/pt/pt_models_20230522.846_1684830698/BERT_LARGE_FP16_PT_384XI32_BATCH${batch_size}/outputs_npy.tgz"
                ))
    })

OUTPUT_DATA_BERT_LARGE_FP32_PT_384X1024XF32_BATCHES = utils.build_batch_model_test_data(
    template=OUTPUT_DATA_BERT_LARGE_FP32_PT_384X1024XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])
OUTPUT_DATA_BERT_LARGE_FP16_PT_384X1024XF16_BATCHES = utils.build_batch_model_test_data(
    template=OUTPUT_DATA_BERT_LARGE_FP16_PT_384X1024XF16_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])
