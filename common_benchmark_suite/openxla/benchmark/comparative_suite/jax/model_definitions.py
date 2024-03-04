# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import string

from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite import utils

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.25_1709787220"
ARTIFACTS_DIR_URL_TEMPLATE = string.Template(PARENT_GCS_DIR + "/${name}")

T5_JAX_IMPL = def_types.ModelImplementation(
    name="T5_JAX",
    tags=["transformer-encoder", "transformer-decoder", "t5"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.t5.t5_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5Model",
)

T5_LARGE_FP32_JAX_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("T5_LARGE_FP32_JAX_512XI32"),
    tags=[utils.BATCH_TAG, "fp32"],
    model_impl=T5_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "t5-large",
        "seq_len": 512,
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
)
T5_LARGE_FP16_JAX_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("T5_LARGE_FP16_JAX_512XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=T5_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp16",
        "model_name": "t5-large",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
)
T5_LARGE_BF16_JAX_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("T5_LARGE_BF16_JAX_512XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=T5_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "bf16",
        "model_name": "t5-large",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
)
# TODO(#54): Template should also support data types so we don't need to define
# for each data types.
T5_LARGE_FP32_JAX_512XI32_BATCHES = utils.build_batch_models(
    template=T5_LARGE_FP32_JAX_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])
T5_LARGE_FP16_JAX_512XI32_BATCHES = utils.build_batch_models(
    template=T5_LARGE_FP16_JAX_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])
T5_LARGE_BF16_JAX_512XI32_BATCHES = utils.build_batch_models(
    template=T5_LARGE_BF16_JAX_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])

T5_SMALL_FP32_JAX_1X128XI32 = def_types.Model(
    name="T5_SMALL_FP32_JAX_1X128XI32",
    tags=["fp32", "batch-1"],
    model_impl=T5_JAX_IMPL,
    model_parameters={
        "batch_size": 1,
        "data_type": "fp32",
        "model_name": "t5-small",
        "seq_len": 128
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
    artifacts_dir_url=f"{PARENT_GCS_DIR}/T5_SMALL_FP32_JAX_1X128XI32",
)

T5_4CG_JAX_IMPL = def_types.ModelImplementation(
    name="T5_4CG_JAX",
    tags=[
        "transformer-encoder", "transformer-decoder", "t5", "auto-regressive"
    ],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=
    f"{utils.MODELS_MODULE_PATH}.jax.t5.t5_for_conditional_generation",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5ForConditionalGeneration",
)

T5_4CG_LARGE_FP32_JAX_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("T5_4CG_LARGE_FP32_JAX_512XI32"),
    tags=[utils.BATCH_TAG, "fp32"],
    model_impl=T5_4CG_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "t5-large",
        "seq_len": 512
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
)
T5_4CG_LARGE_FP32_JAX_512XI32_BATCHES = utils.build_batch_models(
    template=T5_4CG_LARGE_FP32_JAX_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48])

T5_4CG_SMALL_FP32_JAX_1X128XI32_GEN_TEMPLATE = utils.ModelTemplate(
    name=utils.GEN_NAME("T5_4CG_SMALL_FP32_JAX_1X128XI32"),
    tags=["fp32", "batch-1", utils.GEN_TAG],
    model_impl=T5_4CG_JAX_IMPL,
    model_parameters={
        "batch_size": 1,
        "data_type": "fp32",
        "model_name": "t5-small",
        "seq_len": 128,
        "max_new_tokens": utils.GEN_SIZE_PARAM,
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
)
T5_4CG_SMALL_FP32_JAX_1X128XI32_GENS = utils.build_gen_models(
    template=T5_4CG_SMALL_FP32_JAX_1X128XI32_GEN_TEMPLATE,
    gen_sizes=[16, 32, 64, 128, 256])


# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_JAX_IMPL = def_types.ModelImplementation(
    name="BERT_JAX",
    tags=["transformer-encoder", "bert"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.bert.bert_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
)

BERT_LARGE_FP32_JAX_384XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("BERT_LARGE_FP32_JAX_384XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=BERT_JAX_IMPL,
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
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
        def_types.ModelArtifactType.TFLITE_FP16,
        def_types.ModelArtifactType.TFLITE_DYNAMIC_RANGE_QUANT,
        def_types.ModelArtifactType.TFLITE_INT8,
    ],
)

BERT_LARGE_FP16_JAX_384XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("BERT_LARGE_FP16_JAX_384XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=BERT_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp16",
        "seq_len": 384,
        "model_name": "bert-large-uncased",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
    ],
)

BERT_LARGE_BF16_JAX_384XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("BERT_LARGE_BF16_JAX_384XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=BERT_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "bf16",
        "seq_len": 384,
        "model_name": "bert-large-uncased",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCHES = utils.build_batch_models(
    template=BERT_LARGE_FP32_JAX_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

BERT_LARGE_FP16_JAX_384XI32_BATCHES = utils.build_batch_models(
    template=BERT_LARGE_FP16_JAX_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

BERT_LARGE_BF16_JAX_384XI32_BATCHES = utils.build_batch_models(
    template=BERT_LARGE_BF16_JAX_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

BERT_BASE_FP32_JAX_I32_INPUT_SEQUENCE_TEMPLATE = utils.ModelTemplate(
    name=utils.SEQ_LEN_NAME("BERT_BASE_FP32_JAX_I32"),
    tags=[utils.SEQ_LEN_TAG],
    model_impl=BERT_JAX_IMPL,
    model_parameters={
        "batch_size": 1,
        "data_type": "fp32",
        "seq_len": utils.SEQ_LEN_PARAM,
        "model_name": "bert-base-uncased",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
        def_types.ModelArtifactType.TFLITE_FP16,
        def_types.ModelArtifactType.TFLITE_DYNAMIC_RANGE_QUANT,
        def_types.ModelArtifactType.TFLITE_INT8,
    ],
)
BERT_BASE_FP32_JAX_I32_INPUT_SEQUENCES = utils.build_input_sequence_models(
    template=BERT_BASE_FP32_JAX_I32_INPUT_SEQUENCE_TEMPLATE,
    input_sequence_lengths=[8, 32, 64, 128, 256, 512])

BERT_BASE_FP16_JAX_I32_INPUT_SEQUENCE_TEMPLATE = utils.ModelTemplate(
    name=utils.SEQ_LEN_NAME("BERT_BASE_FP16_JAX_I32"),
    tags=[utils.SEQ_LEN_TAG],
    model_impl=BERT_JAX_IMPL,
    model_parameters={
        "batch_size": 1,
        "data_type": "fp16",
        "seq_len": utils.SEQ_LEN_PARAM,
        "model_name": "bert-base-uncased",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
    ],
)
BERT_BASE_FP16_JAX_I32_INPUT_SEQUENCES = utils.build_input_sequence_models(
    template=BERT_BASE_FP16_JAX_I32_INPUT_SEQUENCE_TEMPLATE,
    input_sequence_lengths=[8, 32, 64, 128, 256, 512])

BERT_BASE_BF16_JAX_I32_INPUT_SEQUENCE_TEMPLATE = utils.ModelTemplate(
    name=utils.SEQ_LEN_NAME("BERT_BASE_BF16_JAX_I32"),
    tags=[utils.SEQ_LEN_TAG],
    model_impl=BERT_JAX_IMPL,
    model_parameters={
        "batch_size": 1,
        "data_type": "bf16",
        "seq_len": utils.SEQ_LEN_PARAM,
        "model_name": "bert-base-uncased",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
    ],
)
BERT_BASE_BF16_JAX_I32_INPUT_SEQUENCES = utils.build_input_sequence_models(
    template=BERT_BASE_BF16_JAX_I32_INPUT_SEQUENCE_TEMPLATE,
    input_sequence_lengths=[8, 32, 64, 128, 256, 512])


# ResNet models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/resnet#transformers.FlaxResNetModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50

RESNET_JAX_IMPL = def_types.ModelImplementation(
    name="RESNET_JAX",
    tags=["cnn", "resnet"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.resnet.resnet_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/resnet#transformers.FlaxResNetModel",
)

RESNET50_FP32_JAX_3X224X224XF32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("RESNET50_FP32_JAX_3X224X224XF32"),
    tags=[utils.BATCH_TAG],
    model_impl=RESNET_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "microsoft/resnet-50",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
        def_types.ModelArtifactType.TFLITE_FP16,
        def_types.ModelArtifactType.TFLITE_DYNAMIC_RANGE_QUANT,
        def_types.ModelArtifactType.TFLITE_INT8,
    ],
)

RESNET50_FP16_JAX_3X224X224XF16_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("RESNET50_FP16_JAX_3X224X224XF16"),
    tags=[utils.BATCH_TAG],
    model_impl=RESNET_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp16",
        "model_name": "microsoft/resnet-50",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
    ],
)

RESNET50_BF16_JAX_3X224X224XBF16_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("RESNET50_BF16_JAX_3X224X224XBF16"),
    tags=[utils.BATCH_TAG],
    model_impl=RESNET_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "bf16",
        "model_name": "microsoft/resnet-50",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
    ],
)

RESNET50_FP32_JAX_3X224X224XF32_BATCHES = utils.build_batch_models(
    template=RESNET50_FP32_JAX_3X224X224XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

RESNET50_FP16_JAX_3X224X224XF16_BATCHES = utils.build_batch_models(
    template=RESNET50_FP16_JAX_3X224X224XF16_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

RESNET50_BF16_JAX_3X224X224XBF16_BATCHES = utils.build_batch_models(
    template=RESNET50_BF16_JAX_3X224X224XBF16_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

# GPT2 models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.FlaxGPT2LMHeadModel.
GPT2LMHEAD_JAX_IMPL = def_types.ModelImplementation(
    name="GPT2LMHEAD_JAX",
    tags=["cnn", "gpt2lmhead"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.gpt2.gpt2lmhead_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.FlaxGPT2LMHeadModel",
)

GPT2LMHEAD_FP32_JAX_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("GPT2LMHEAD_FP32_JAX_512XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=GPT2LMHEAD_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "gpt2",
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
        def_types.ModelArtifactType.TFLITE_FP16,
        def_types.ModelArtifactType.TFLITE_DYNAMIC_RANGE_QUANT,
        def_types.ModelArtifactType.TFLITE_INT8,
    ],
)

GPT2LMHEAD_FP32_JAX_512XI32_BATCHES = utils.build_batch_models(
    template=GPT2LMHEAD_FP32_JAX_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 64, 128])

GPT2LMHEAD_PIPELINE_JAX_IMPL = def_types.ModelImplementation(
    name="GPT2_PIPELINE_JAX",
    tags=["transformer-decoder", "gpt2", "ggml", "pipeline"],
    framework_type=def_types.ModelFrameworkType.TF_V2,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.gpt2.gpt2_pipeline",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.FlaxGPT2LMHeadModel",
)

GPT2LMHEAD_PIPELINE_JAX_1X4XI32 = def_types.Model(
    name="GPT2LMHEAD_PIPELINE_JAX_1X4XI32",
    tags=["fp32", "batch-1"],
    model_impl=GPT2LMHEAD_PIPELINE_JAX_IMPL,
    model_parameters=dict(batch_size=1, data_type="fp32", model_name="gpt2"),
    artifacts_dir_url=f"{PARENT_GCS_DIR}/GPT2LMHEAD_PIPELINE_JAX_1X4XI32",
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
)

# Stable Diffusion models.
# Model implementation from https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.FlaxStableDiffusionPipeline.
SD_PIPELINE_JAX_IMPL = def_types.ModelImplementation(
    name="SD_PIPELINE_JAX",
    tags=["stable-diffusion", "pipeline"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=
    f"{utils.MODELS_MODULE_PATH}.jax.stable_diffusion.stable_diffusion_pipeline",
    source_info=
    "hhttps://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.FlaxStableDiffusionPipeline",
)

SD_PIPELINE_FP32_JAX_64XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("SD_PIPELINE_FP32_JAX_64XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=SD_PIPELINE_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "runwayml/stable-diffusion-v1-5",
        "seq_len": 64,
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
)

SD_PIPELINE_FP32_JAX_64XI32_BATCHES = utils.build_batch_models(
    template=SD_PIPELINE_FP32_JAX_64XI32_BATCH_TEMPLATE, batch_sizes=[1, 8])

SD_PIPELINE_FP16_JAX_64XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("SD_PIPELINE_FP16_JAX_64XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=SD_PIPELINE_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp16",
        "model_name": "runwayml/stable-diffusion-v1-5",
        "seq_len": 64,
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
)

SD_PIPELINE_FP16_JAX_64XI32_BATCHES = utils.build_batch_models(
    template=SD_PIPELINE_FP16_JAX_64XI32_BATCH_TEMPLATE, batch_sizes=[1, 8])

SD_PIPELINE_BF16_JAX_64XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    name=utils.BATCH_NAME("SD_PIPELINE_BF16_JAX_64XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=SD_PIPELINE_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "bf16",
        "model_name": "runwayml/stable-diffusion-v1-5",
        "seq_len": 64,
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
)

SD_PIPELINE_BF16_JAX_64XI32_BATCHES = utils.build_batch_models(
    template=SD_PIPELINE_BF16_JAX_64XI32_BATCH_TEMPLATE, batch_sizes=[1, 8])

# ViT models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/vit#transformers.TFViTForImageClassification.
VIT_CLASSIFICATION_JAX_IMPL = def_types.ModelImplementation(
    name="VIT_CLASSIFICATION_JAX",
    tags=["vision-transformer"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.vit.vit_for_classification",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/vit#transformers.FlaxViTForImageClassification",
)

VIT_CLASSIFICATION_JAX_3X224X224XF32 = def_types.Model(
    name="VIT_CLASSIFICATION_JAX_3X224X224XF32",
    tags=["fp32", "batch-1"],
    model_impl=VIT_CLASSIFICATION_JAX_IMPL,
    model_parameters={
        "batch_size": 1,
        "model_name": "google/vit-base-patch16-224",
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
        def_types.ModelArtifactType.TFLITE_FP16,
        def_types.ModelArtifactType.TFLITE_DYNAMIC_RANGE_QUANT,
        def_types.ModelArtifactType.TFLITE_INT8,
    ],
    artifacts_dir_url=f"{PARENT_GCS_DIR}/VIT_CLASSIFICATION_JAX_3X224X224XF32",
)

# Gemma models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/gemma#transformers.FlaxGemmaForCausalLM.
GEMMA_PIPELINE_JAX_IMPL = def_types.ModelImplementation(
    name="GEMMA_PIPELINE_JAX",
    tags=["gemma", "pipeline"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.gemma.gemma_pipeline",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/gemma#transformers.FlaxGemmaForCausalLM",
)

GEMMA2BIT_GREEDY_FP32_JAX_1X1024XI32_256XI32 = def_types.Model(
    name="GEMMA2BIT_GREEDY_FP32_JAX_1X1024XI32_256XI32",
    tags=["fp32", "batch-1", "seqlen-1024"],
    model_impl=GEMMA_PIPELINE_JAX_IMPL,
    model_parameters={
        "batch_size": 1,
        "data_type": "fp32",
        "seq_len": 1024,
        "max_new_tokens": 256,
        "model_name": "google/gemma-2b-it",
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
    artifacts_dir_url=
    f"{PARENT_GCS_DIR}/GEMMA2BIT_GREEDY_FP32_JAX_1X1024XI32_256XI32",
)

GEMMA2BIT_GREEDY_BF16_JAX_1X1024XI32_256XI32 = def_types.Model(
    name="GEMMA2BIT_GREEDY_BF16_JAX_1X1024XI32_256XI32",
    tags=["bf16", "batch-1", "seqlen-1024"],
    model_impl=GEMMA_PIPELINE_JAX_IMPL,
    model_parameters={
        "batch_size": 1,
        "data_type": "bf16",
        "seq_len": 1024,
        "max_new_tokens": 256,
        "model_name": "google/gemma-2b-it",
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
    artifacts_dir_url=
    f"{PARENT_GCS_DIR}/GEMMA2BIT_GREEDY_BF16_JAX_1X1024XI32_256XI32",
)

GEMMA2BIT_GREEDY_FP16_JAX_1X1024XI32_256XI32 = def_types.Model(
    name="GEMMA2BIT_GREEDY_FP16_JAX_1X1024XI32_256XI32",
    tags=["fp16", "batch-1", "seqlen-1024"],
    model_impl=GEMMA_PIPELINE_JAX_IMPL,
    model_parameters={
        "batch_size": 1,
        "data_type": "fp16",
        "seq_len": 1024,
        "max_new_tokens": 256,
        "model_name": "google/gemma-2b-it",
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
    artifacts_dir_url=
    f"{PARENT_GCS_DIR}/GEMMA2BIT_GREEDY_FP16_JAX_1X1024XI32_256XI32",
)

# Dotprod microbenchmarks.
DOT_PRODUCT_JAX_IMPL = def_types.ModelImplementation(
    name="DOT_PRODUCT_JAX",
    tags=["microbenchmark"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.dotprod.dot_product",
    source_info="",
)

DOT_PRODUCT_JAX_1X256X2048XI8I8 = def_types.Model(
    name="DOT_PRODUCT_JAX_1X256X2048XI8I8",
    tags=["i8i8i32"],
    model_impl=DOT_PRODUCT_JAX_IMPL,
    model_parameters={
        "model_name": "dotprod",
        "lhs_shape": (1, 256),
        "lhs_type": "int8",
        "rhs_shape": (256, 2048),
        "rhs_type": "int8",
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.LINALG_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
    artifacts_dir_url=f"{PARENT_GCS_DIR}/DOT_PRODUCT_JAX_1X256X2048XI8I8",
)

DOT_PRODUCT_JAX_1X256X2048XI8I4 = def_types.Model(
    name="DOT_PRODUCT_JAX_1X256X2048XI8I4",
    tags=["i8i4i32"],
    model_impl=DOT_PRODUCT_JAX_IMPL,
    model_parameters={
        "model_name": "dotprod",
        "lhs_shape": (1, 256),
        "lhs_type": "int8",
        "rhs_shape": (256, 2048),
        "rhs_type": "int4",
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.LINALG_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
    artifacts_dir_url=f"{PARENT_GCS_DIR}/DOT_PRODUCT_JAX_1X256X2048XI8I4",
)

DOT_PRODUCT_JAX_1X256X2048XF32F32 = def_types.Model(
    name="DOT_PRODUCT_JAX_1X256X2048XF32F32",
    tags=["f32f32f32"],
    model_impl=DOT_PRODUCT_JAX_IMPL,
    model_parameters={
        "model_name": "dotprod",
        "lhs_shape": (1, 256),
        "lhs_type": "fp32",
        "rhs_shape": (256, 2048),
        "rhs_type": "fp32",
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.LINALG_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
        def_types.ModelArtifactType.TFLITE_FP16,
        def_types.ModelArtifactType.TFLITE_DYNAMIC_RANGE_QUANT,
        def_types.ModelArtifactType.TFLITE_INT8,
    ],
    artifacts_dir_url=f"{PARENT_GCS_DIR}/DOT_PRODUCT_JAX_1X256X2048XF32F32",
)

DOT_PRODUCT_JAX_256X256X2048XI8I8 = def_types.Model(
    name="DOT_PRODUCT_JAX_256X256X2048XI8I8",
    tags=["i8i8i32"],
    model_impl=DOT_PRODUCT_JAX_IMPL,
    model_parameters={
        "model_name": "dotprod",
        "lhs_shape": (256, 256),
        "lhs_type": "int8",
        "rhs_shape": (256, 2048),
        "rhs_type": "int8",
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.LINALG_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
    artifacts_dir_url=f"{PARENT_GCS_DIR}/DOT_PRODUCT_JAX_256X256X2048XI8I8",
)

DOT_PRODUCT_JAX_256X256X2048XI8I4 = def_types.Model(
    name="DOT_PRODUCT_JAX_256X256X2048XI8I4",
    tags=["i8i4i32"],
    model_impl=DOT_PRODUCT_JAX_IMPL,
    model_parameters={
        "model_name": "dotprod",
        "lhs_shape": (256, 256),
        "lhs_type": "int8",
        "rhs_shape": (256, 2048),
        "rhs_type": "int4",
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.LINALG_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
    ],
    artifacts_dir_url=f"{PARENT_GCS_DIR}/DOT_PRODUCT_JAX_256X256X2048XI8I4",
)

DOT_PRODUCT_JAX_256X256X2048XF32F32 = def_types.Model(
    name="DOT_PRODUCT_JAX_256X256X2048XF32F32",
    tags=["f32f32f32"],
    model_impl=DOT_PRODUCT_JAX_IMPL,
    model_parameters={
        "model_name": "dotprod",
        "lhs_shape": (1, 256),
        "lhs_type": "fp32",
        "rhs_shape": (256, 2048),
        "rhs_type": "fp32",
    },
    exported_model_types=[
        def_types.ModelArtifactType.STABLEHLO_MLIR,
        def_types.ModelArtifactType.LINALG_MLIR,
        def_types.ModelArtifactType.XLA_HLO_DUMP,
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO,
        def_types.ModelArtifactType.TFLITE_FP16,
        def_types.ModelArtifactType.TFLITE_DYNAMIC_RANGE_QUANT,
        def_types.ModelArtifactType.TFLITE_INT8,
    ],
    artifacts_dir_url=f"{PARENT_GCS_DIR}/DOT_PRODUCT_JAX_256X256X2048XF32F32",
)

ALL_MODELS = list(
    itertools.chain(
        # Models with different batch sizes.
        T5_LARGE_FP32_JAX_512XI32_BATCHES.values(),
        T5_LARGE_FP16_JAX_512XI32_BATCHES.values(),
        T5_LARGE_BF16_JAX_512XI32_BATCHES.values(),
        T5_4CG_LARGE_FP32_JAX_512XI32_BATCHES.values(),
        BERT_LARGE_FP32_JAX_384XI32_BATCHES.values(),
        BERT_LARGE_FP16_JAX_384XI32_BATCHES.values(),
        BERT_LARGE_BF16_JAX_384XI32_BATCHES.values(),
        RESNET50_FP32_JAX_3X224X224XF32_BATCHES.values(),
        RESNET50_FP16_JAX_3X224X224XF16_BATCHES.values(),
        RESNET50_BF16_JAX_3X224X224XBF16_BATCHES.values(),
        GPT2LMHEAD_FP32_JAX_512XI32_BATCHES.values(),
        # Models with different gen sizes.
        T5_4CG_SMALL_FP32_JAX_1X128XI32_GENS.values(),
        # Models with different input sequences.
        BERT_BASE_FP32_JAX_I32_INPUT_SEQUENCES.values(),
        BERT_BASE_FP16_JAX_I32_INPUT_SEQUENCES.values(),
        BERT_BASE_BF16_JAX_I32_INPUT_SEQUENCES.values(),
        # Pipelines.
        SD_PIPELINE_FP32_JAX_64XI32_BATCHES.values(),
        SD_PIPELINE_FP16_JAX_64XI32_BATCHES.values(),
        SD_PIPELINE_BF16_JAX_64XI32_BATCHES.values(),
    )) + [
        GPT2LMHEAD_PIPELINE_JAX_1X4XI32, T5_SMALL_FP32_JAX_1X128XI32,
        VIT_CLASSIFICATION_JAX_3X224X224XF32,
        GEMMA2BIT_GREEDY_FP32_JAX_1X1024XI32_256XI32,
        GEMMA2BIT_GREEDY_BF16_JAX_1X1024XI32_256XI32,
        GEMMA2BIT_GREEDY_FP16_JAX_1X1024XI32_256XI32,
        DOT_PRODUCT_JAX_1X256X2048XI8I8, DOT_PRODUCT_JAX_1X256X2048XI8I4,
        DOT_PRODUCT_JAX_256X256X2048XI8I8, DOT_PRODUCT_JAX_256X256X2048XI8I4,
        DOT_PRODUCT_JAX_1X256X2048XF32F32, DOT_PRODUCT_JAX_256X256X2048XF32F32
    ]
