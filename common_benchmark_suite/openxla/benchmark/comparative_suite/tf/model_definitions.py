# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from openxla.benchmark import def_types, unique_ids
from openxla.benchmark.comparative_suite import utils

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1685428719/"

T5_TF_IMPL = def_types.ModelImplementation(
    id=unique_ids.MODEL_IMPL_T5_TF,
    name="T5_TF",
    tags=["fp32", "transformer-encoder", "transformer-decoder", "t5"],
    framework_type=def_types.ModelFrameworkType.TF_V2,
    module_path=f"{utils.MODELS_MODULE_PATH}.tf.t5.t5_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model",
)

T5_LARGE_FP32_TF_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_T5_LARGE_FP32_TF),
    name=utils.BATCH_NAME("T5_LARGE_FP32_TF_512XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=T5_TF_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "t5-large",
    },
    artifacts={
        # TODO(#12): Add artifacts once artifact generation pipeline is implemented.
    },
)
T5_LARGE_FP32_TF_512XI32_BATCHES = utils.build_batch_models(
    template=T5_LARGE_FP32_TF_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])

# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_TF_IMPL = def_types.ModelImplementation(
    id=unique_ids.MODEL_IMPL_BERT_TF,
    name="BERT_TF",
    tags=["transformer-encoder", "bert"],
    framework_type=def_types.ModelFrameworkType.TF_V2,
    module_path=f"{utils.MODELS_MODULE_PATH}.tf.bert.bert_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
)

BERT_LARGE_FP32_TF_384XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32),
    name=utils.BATCH_NAME("BERT_LARGE_FP32_TF_384XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=BERT_TF_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "seq_len": 384,
        "model_name": "bert-large-uncased",
    },
    artifacts={
        # TODO(#12): Add artifacts once artifact generation pipeline is implemented.
    })

BERT_LARGE_FP32_TF_384XI32_BATCHES = utils.build_batch_models(
    template=BERT_LARGE_FP32_TF_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])
