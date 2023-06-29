# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import string

from openxla.benchmark.comparative_suite import utils
from openxla.benchmark import def_types, unique_ids

# T5 large inputs.
INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    id=utils.BATCH_ID(unique_ids.INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32),
    name=utils.BATCH_NAME("INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32"),
    tags=["input-data", "seqlen512", utils.BATCH_TAG],
    source_info=
    "Original text: 'Studies have been shown that owning a dog is good for you'",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
            utils.ModelTestDataArtifactTemplate(
                data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
                data_parameters={
                    "tensor_names": ["input_ids", "decoder_input_ids"],
                    "tensor_dimensions": [
                        utils.BATCH_TENSOR_DIMS("512xi32"),
                        utils.BATCH_TENSOR_DIMS("512xi32"),
                    ]
                },
                verify_parameters={},
                source_url=string.Template(
                    "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752/T5_LARGE_FP32_JAX_512XI32_BATCH${batch_size}/input_npy.tgz"
                ))
    })
INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32_BATCHES = utils.build_batch_model_test_data(
    template=INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])

# T5 large outputs.
OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    id=utils.BATCH_ID(unique_ids.OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32),
    name=utils.BATCH_NAME("OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32"),
    tags=["output-data", utils.BATCH_TAG],
    source_info="",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
            utils.ModelTestDataArtifactTemplate(
                data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
                data_parameters={
                    "tensor_names": ["output_0"],
                    "tensor_dimensions": [
                        utils.BATCH_TENSOR_DIMS("512x1024xi32"),
                    ]
                },
                verify_parameters={
                    "absolute_tolerance": 0.5,
                },
                source_url=string.Template(
                    "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752/T5_LARGE_FP32_JAX_512XI32_BATCH${batch_size}/output_npy.tgz"
                ))
    })
OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32_BATCHES = utils.build_batch_model_test_data(
    template=OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])
