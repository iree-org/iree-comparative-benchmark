# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from openxla.benchmark import def_types, unique_ids
from openxla.benchmark.comparative_suite import utils

# Bert models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_PT_IMPL = def_types.ModelImplementation(
    id=unique_ids.MODEL_IMPL_BERT_PT,
    name="BERT_PT",
    tags=["transformer-encoder", "bert"],
    framework_type=def_types.ModelFrameworkType.PYTORCH,
    module_path=f"{utils.MODELS_MODULE_PATH}.pt.bert.bert_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
)

BERT_LARGE_FP32_PT_384XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_BERT_LARGE_FP32_PT_384XI32),
    name=utils.BATCH_NAME("BERT_LARGE_FP32_PT_384XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=BERT_PT_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "seq_len": 384,
        "model_name": "bert-large-uncased",
    },
    artifacts={
        # TODO(#12): Add artifacts once artifact generation pipeline is implemented.
    })

BERT_LARGE_FP16_PT_384XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_BERT_LARGE_FP16_PT_384XI32),
    name=utils.BATCH_NAME("BERT_LARGE_FP16_PT_384XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=BERT_PT_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp16",
        "seq_len": 384,
        "model_name": "bert-large-uncased",
    },
    artifacts={
        # TODO(#12): Add artifacts once artifact generation pipeline is implemented.
    })

BERT_LARGE_FP32_PT_384XI32_BATCHES = utils.build_batch_models(
    template=BERT_LARGE_FP32_PT_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

BERT_LARGE_FP16_PT_384XI32_BATCHES = utils.build_batch_models(
    template=BERT_LARGE_FP16_PT_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])
