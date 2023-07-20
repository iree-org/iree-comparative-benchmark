# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import string

from openxla.benchmark.comparative_suite import utils
from openxla.benchmark import def_types

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.13.0rc2_1688540251/"

# T5-Large inputs.
INPUT_DATA_T5_LARGE_FP32_TF_512XI32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    name=utils.BATCH_NAME("INPUT_DATA_T5_LARGE_FP32_TF_512XI32"),
    tags=["input-data", "seqlen512", utils.BATCH_TAG],
    source_info=
    "Original text: 'Studies have been shown that owning a dog is good for you'",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
            utils.ModelTestDataArtifactTemplate(
                data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
                data_parameters={
                    "tensor_names": [
                        "serving_default_input_ids",
                        "serving_default_decoder_input_ids"
                    ],
                    "tensor_dimensions": [
                        utils.BATCH_TENSOR_DIMS("512xi32"),
                        utils.BATCH_TENSOR_DIMS("512xi32"),
                    ]
                },
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "T5_LARGE_FP32_TF_512XI32_BATCH${batch_size}/inputs_npy.tgz"
                ),
            )
    })
INPUT_DATA_T5_LARGE_FP32_TF_512XI32_BATCHES = utils.build_batch_model_test_data(
    template=INPUT_DATA_T5_LARGE_FP32_TF_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])

# T5-Large outputs.
OUTPUT_DATA_T5_LARGE_FP32_TF_512X1024XF32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    name=utils.BATCH_NAME("OUTPUT_DATA_T5_LARGE_FP32_TF_512X1024XF32"),
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
                    PARENT_GCS_DIR +
                    "T5_LARGE_FP32_TF_512XI32_BATCH${batch_size}/outputs_npy.tgz"
                ),
            )
    })
OUTPUT_DATA_T5_LARGE_FP32_TF_512X1024XF32_BATCHES = utils.build_batch_model_test_data(
    template=OUTPUT_DATA_T5_LARGE_FP32_TF_512X1024XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])

# Bert-Large inputs.
INPUT_DATA_BERT_LARGE_FP32_TF_384XI32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    name=utils.BATCH_NAME("INPUT_DATA_BERT_LARGE_FP32_TF_384XI32"),
    tags=["input-data", "seqlen384", utils.BATCH_TAG],
    source_info="Original text: 'a photo of a cat'.",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
            utils.ModelTestDataArtifactTemplate(
                data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
                data_parameters={
                    "tensor_names": [
                        "serving_default_input_ids",
                        "serving_default_attention_mask"
                    ],
                    "tensor_dimensions": [
                        utils.BATCH_TENSOR_DIMS("384xi32"),
                        utils.BATCH_TENSOR_DIMS("384xi32"),
                    ],
                },
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "BERT_LARGE_FP32_TF_384XI32_BATCH${batch_size}/inputs_npy.tgz"
                ),
            )
    })
INPUT_DATA_BERT_LARGE_FP32_TF_384XI32_BATCHES = utils.build_batch_model_test_data(
    template=INPUT_DATA_BERT_LARGE_FP32_TF_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

# Bert-Large outputs.
OUTPUT_DATA_BERT_LARGE_FP32_TF_384X1024XF32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    name=utils.BATCH_NAME("OUTPUT_DATA_BERT_LARGE_FP32_TF_384X1024XF32"),
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
                    PARENT_GCS_DIR +
                    "BERT_LARGE_FP32_TF_384XI32_BATCH${batch_size}/outputs_npy.tgz"
                ))
    })
OUTPUT_DATA_BERT_LARGE_FP32_TF_384X1024XF32_BATCHES = utils.build_batch_model_test_data(
    template=OUTPUT_DATA_BERT_LARGE_FP32_TF_384X1024XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

# ResNet50 inputs.
INPUT_DATA_RESNET50_FP32_TF_224X224X3XF32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    name=utils.BATCH_NAME("INPUT_DATA_RESNET50_FP32_TF_224X224X3XF32"),
    tags=["input-data", "imagenet", utils.BATCH_TAG],
    source_info=
    "Original image: https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
            utils.ModelTestDataArtifactTemplate(
                data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
                data_parameters={
                    "tensor_names": ["pixel_values"],
                    "tensor_dimensions": [
                        utils.BATCH_TENSOR_DIMS("3x224x224xf32")
                    ],
                },
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "RESNET50_FP32_TF_224X224X3XF32_BATCH${batch_size}/inputs_npy.tgz"
                ))
    })
INPUT_DATA_RESNET50_FP32_TF_224X224X3XF32_BATCHES = utils.build_batch_model_test_data(
    template=INPUT_DATA_RESNET50_FP32_TF_224X224X3XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

# ResNet50 outputs.
OUTPUT_DATA_RESNET50_FP32_TF_2048X7X7XF32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    name=utils.BATCH_NAME("OUTPUT_DATA_RESNET50_FP32_TF_2048X7X7XF32"),
    tags=["input-data", "imagenet", utils.BATCH_TAG],
    source_info="",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
            utils.ModelTestDataArtifactTemplate(
                data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
                data_parameters={
                    "tensor_names": ["output_0"],
                    "tensor_dimensions": [
                        utils.BATCH_TENSOR_DIMS("2048x7x7xf32")
                    ],
                },
                verify_parameters={
                    "absolute_tolerance": 0.5,
                },
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "RESNET50_FP32_TF_224X224X3XF32_BATCH${batch_size}/outputs_npy.tgz"
                ))
    })

OUTPUT_DATA_RESNET50_FP32_TF_2048X7X7XF32_BATCHES = utils.build_batch_model_test_data(
    template=OUTPUT_DATA_RESNET50_FP32_TF_2048X7X7XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

# EfficientNetB7 inputs.
EFFICIENTNET_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.13.0_1689819439/"

INPUT_DATA_EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    id=utils.BATCH_ID(unique_ids.INPUT_DATA_EFFICIENTNETB7_FP32_TF_600X600X3XF32),
    name=utils.BATCH_NAME("INPUT_DATA_EFFICIENTNETB7_FP32_TF_600X600X3XF32"),
    tags=["input-data", "imagenet", utils.BATCH_TAG],
    source_info=
    "Original image: https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
          utils.ModelTestDataArtifactTemplate(
              data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
              data_parameters={
                  "tensor_names": ["pixel_values"],
                  "tensor_dimensions": [
                      utils.BATCH_TENSOR_DIMS("600x600x3xf32")
                  ],
              },
              source_url=string.Template(
                  EFFICIENTNET_GCS_DIR +
                  "EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCH${batch_size}/inputs_npy.tgz"
              ))
    })
INPUT_DATA_EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCHES = utils.build_batch_model_test_data(
    template=INPUT_DATA_EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 64, 128])

# EfficientNetB7 outputs.
OUTPUT_DATA_EFFICIENTNETB7_FP32_TF_1000XF32_BATCH_TEMPLATE = utils.ModelTestDataTemplate(
    id=utils.BATCH_ID(unique_ids.OUTPUT_DATA_EFFICIENTNETB7_FP32_TF_1000XF32),
    name=utils.BATCH_NAME("OUTPUT_DATA_EFFICIENTNETB7_FP32_TF_1000XF32"),
    tags=["input-data", "imagenet", utils.BATCH_TAG],
    source_info="",
    artifacts={
        def_types.ModelTestDataFormat.NUMPY_TENSORS:
          utils.ModelTestDataArtifactTemplate(
              data_format=def_types.ModelTestDataFormat.NUMPY_TENSORS,
              data_parameters={
                  "tensor_names": ["output_0"],
                  "tensor_dimensions": [
                      utils.BATCH_TENSOR_DIMS("1000xf32")
                  ],
              },
              verify_parameters={
                  "absolute_tolerance": 0.5,
              },
              source_url=string.Template(
                  EFFICIENTNET_GCS_DIR +
                  "EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCH${batch_size}/outputs_npy.tgz"
              ))
    })
OUTPUT_DATA_EFFICIENTNETB7_FP32_TF_1000XF32_BATCHES = utils.build_batch_model_test_data(
    template=OUTPUT_DATA_EFFICIENTNETB7_FP32_TF_1000XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 64, 128])
