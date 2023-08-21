# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import string

from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite import utils

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/"
ARTIFACTS_DIR_URL_TEMPLATE = string.Template(PARENT_GCS_DIR + "${name}")

# T5-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model
T5_TF_IMPL = def_types.ModelImplementation(
    name="T5_TF",
    tags=["fp32", "transformer-encoder", "transformer-decoder", "t5"],
    framework_type=def_types.ModelFrameworkType.TF_V2,
    module_path=f"{utils.MODELS_MODULE_PATH}.tf.t5.t5_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model",
)
T5_LARGE_FP32_TF_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("T5_LARGE_FP32_TF_512XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=T5_TF_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "t5-large",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TF_SAVEDMODEL_V2,
    ],
)
T5_LARGE_FP32_TF_512XI32_BATCHES = utils.build_batch_models(
    template=T5_LARGE_FP32_TF_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])

# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_TF_IMPL = def_types.ModelImplementation(
    name="BERT_TF",
    tags=["transformer-encoder", "bert"],
    framework_type=def_types.ModelFrameworkType.TF_V2,
    module_path=f"{utils.MODELS_MODULE_PATH}.tf.bert.bert_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
)
BERT_LARGE_FP32_TF_384XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("BERT_LARGE_FP32_TF_384XI32"),
    tags=[utils.BATCH_TAG, "fp32"],
    model_impl=BERT_TF_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "seq_len": 384,
        "model_name": "bert-large-uncased",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TF_SAVEDMODEL_V2,
    ],
)
BERT_LARGE_FP32_TF_384XI32_BATCHES = utils.build_batch_models(
    template=BERT_LARGE_FP32_TF_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

# ResNet50 models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/resnet#transformers.TFResNetModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50
RESNET_TF_IMPL = def_types.ModelImplementation(
    name="RESNET_TF",
    tags=["cnn", "resnet"],
    framework_type=def_types.ModelFrameworkType.TF_V2,
    module_path=f"{utils.MODELS_MODULE_PATH}.tf.resnet.resnet_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/resnet#transformers.TFResNetModel",
)
RESNET50_FP32_TF_224X224X3XF32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("RESNET50_FP32_TF_224X224X3XF32"),
    tags=[utils.BATCH_TAG, "fp32"],
    model_impl=RESNET_TF_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "microsoft/resnet-50",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TF_SAVEDMODEL_V2,
    ],
)
RESNET50_FP32_TF_224X224X3XF32_BATCHES = utils.build_batch_models(
    template=RESNET50_FP32_TF_224X224X3XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

# EfficientNet models.
# Model implementation from https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet.
EFFICIENTNET_TF_IMPL = def_types.ModelImplementation(
    name="EFFICIENTNET_TF",
    tags=["cnn", "efficientnet"],
    framework_type=def_types.ModelFrameworkType.TF_V2,
    module_path=f"{utils.MODELS_MODULE_PATH}.tf.efficientnet.efficientnet_model",
    source_info=
    "https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet",
)
EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("EFFICIENTNETB7_FP32_TF_600X600X3XF32"),
    tags=[utils.BATCH_TAG, "fp32"],
    model_impl=EFFICIENTNET_TF_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "efficientnetb7",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TF_SAVEDMODEL_V2,
    ],
)

EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCHES = utils.build_batch_models(
    template=EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 64, 128])

ALL_MODELS = list(
    itertools.chain(
        T5_LARGE_FP32_TF_512XI32_BATCHES.values(),
        BERT_LARGE_FP32_TF_384XI32_BATCHES.values(),
        RESNET50_FP32_TF_224X224X3XF32_BATCHES.values(),
        EFFICIENTNETB7_FP32_TF_600X600X3XF32_BATCHES.values(),
    ))
