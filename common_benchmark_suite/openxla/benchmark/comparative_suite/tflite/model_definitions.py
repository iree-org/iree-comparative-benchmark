# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import string

from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite import utils

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/tflite/tflite_models_1703027804"

ARTIFACTS_DIR_URL_TEMPLATE = string.Template(PARENT_GCS_DIR + "/${name}")

TFLITE_MODEL_IMPL = def_types.ModelImplementation(
    name="TFLITE_MODEL_IMPL",
    tags=["tflite"],
    framework_type=def_types.ModelFrameworkType.TFLITE,
    module_path=f"{utils.MODELS_MODULE_PATH}.tflite.tflite_model",
)

JAX_MODELS_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.23_1702848353"

BERT_BASE_FP32_TFLITE_I32_INPUT_SEQUENCE_TEMPLATE = utils.ModelTemplate(
    name=utils.SEQ_LEN_NAME("BERT_BASE_FP32_TFLITE_I32"),
    tags=[utils.SEQ_LEN_TAG],
    model_impl=TFLITE_MODEL_IMPL,
    model_parameters={
        "model_uri":
            string.Template(
                JAX_MODELS_GCS_DIR +
                "/BERT_BASE_FP32_JAX_I32_SEQLEN${seq_len}/model_fp32.tflite"),
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TOSA_MLIR,
    ],
)

BERT_BASE_FP32_TFLITE_I32_INPUT_SEQUENCES = utils.build_input_sequence_models(
    template=BERT_BASE_FP32_TFLITE_I32_INPUT_SEQUENCE_TEMPLATE,
    input_sequence_lengths=[8, 32, 64, 128, 256, 512])

BERT_BASE_FP16_TFLITE_I32_INPUT_SEQUENCE_TEMPLATE = utils.ModelTemplate(
    name=utils.SEQ_LEN_NAME("BERT_BASE_FP16_TFLITE_I32"),
    tags=[utils.SEQ_LEN_TAG],
    model_impl=TFLITE_MODEL_IMPL,
    model_parameters={
        "model_uri":
            string.Template(
                JAX_MODELS_GCS_DIR +
                "/BERT_BASE_FP32_JAX_I32_SEQLEN${seq_len}/model_fp16.tflite"),
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TOSA_MLIR,
    ],
)

BERT_BASE_FP16_TFLITE_I32_INPUT_SEQUENCES = utils.build_input_sequence_models(
    template=BERT_BASE_FP16_TFLITE_I32_INPUT_SEQUENCE_TEMPLATE,
    input_sequence_lengths=[8, 32, 64, 128, 256, 512])

BERT_BASE_DYN_QUANT_TFLITE_I32_INPUT_SEQUENCE_TEMPLATE = utils.ModelTemplate(
    name=utils.SEQ_LEN_NAME("BERT_BASE_DYN_QUANT_TFLITE_I32"),
    tags=[utils.SEQ_LEN_TAG],
    model_impl=TFLITE_MODEL_IMPL,
    model_parameters={
        "model_uri":
            string.Template(
                JAX_MODELS_GCS_DIR +
                "/BERT_BASE_FP32_JAX_I32_SEQLEN${seq_len}/model_dynamic_range_quant.tflite"
            ),
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.TFLITE_DYNAMIC_RANGE_QUANT,
        def_types.ModelArtifactType.TOSA_MLIR,
    ],
)

BERT_BASE_DYN_QUANT_TFLITE_I32_INPUT_SEQUENCES = utils.build_input_sequence_models(
    template=BERT_BASE_DYN_QUANT_TFLITE_I32_INPUT_SEQUENCE_TEMPLATE,
    input_sequence_lengths=[8, 32, 64, 128, 256, 512])

BERT_BASE_INT8_TFLITE_I32_INPUT_SEQUENCE_TEMPLATE = utils.ModelTemplate(
    name=utils.SEQ_LEN_NAME("BERT_BASE_INT8_TFLITE_I32"),
    tags=[utils.SEQ_LEN_TAG],
    model_impl=TFLITE_MODEL_IMPL,
    model_parameters={
        "model_uri":
            string.Template(
                JAX_MODELS_GCS_DIR +
                "/BERT_BASE_FP32_JAX_I32_SEQLEN${seq_len}/model_int8.tflite"),
    },
    artifacts_dir_url=ARTIFACTS_DIR_URL_TEMPLATE,
    exported_model_types=[
        def_types.ModelArtifactType.TFLITE_INT8,
        def_types.ModelArtifactType.TOSA_MLIR,
    ],
)

BERT_BASE_INT8_TFLITE_I32_INPUT_SEQUENCES = utils.build_input_sequence_models(
    template=BERT_BASE_INT8_TFLITE_I32_INPUT_SEQUENCE_TEMPLATE,
    input_sequence_lengths=[8, 32, 64, 128, 256, 512])

VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32 = def_types.Model(
    name="VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32",
    tags=["fp32", "batch-1"],
    model_impl=TFLITE_MODEL_IMPL,
    model_parameters={
        "model_uri":
            f"{JAX_MODELS_GCS_DIR}/VIT_CLASSIFICATION_JAX_3X224X224XF32/model_fp32.tflite",
    },
    artifacts_dir_url=
    f"{PARENT_GCS_DIR}/VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32",
    exported_model_types=[
        def_types.ModelArtifactType.TFLITE_FP32,
        def_types.ModelArtifactType.TOSA_MLIR,
    ],
)

VIT_CLASSIFICATION_FP16_TFLITE_3X224X224XF32 = def_types.Model(
    name="VIT_CLASSIFICATION_FP16_TFLITE_3X224X224XF32",
    tags=["fp16", "batch-1"],
    model_impl=TFLITE_MODEL_IMPL,
    model_parameters={
        "model_uri":
            f"{JAX_MODELS_GCS_DIR}/VIT_CLASSIFICATION_JAX_3X224X224XF32/model_fp16.tflite",
    },
    artifacts_dir_url=
    f"{PARENT_GCS_DIR}/VIT_CLASSIFICATION_FP16_TFLITE_3X224X224XF32",
    exported_model_types=[
        def_types.ModelArtifactType.TFLITE_FP16,
        def_types.ModelArtifactType.TOSA_MLIR,
    ],
)

VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32 = def_types.Model(
    name="VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32",
    tags=["dyn-quant", "batch-1"],
    model_impl=TFLITE_MODEL_IMPL,
    model_parameters={
        "model_uri":
            f"{JAX_MODELS_GCS_DIR}/VIT_CLASSIFICATION_JAX_3X224X224XF32/model_dynamic_range_quant.tflite",
    },
    artifacts_dir_url=
    f"{PARENT_GCS_DIR}/VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32",
    exported_model_types=[
        def_types.ModelArtifactType.TFLITE_DYNAMIC_RANGE_QUANT,
        def_types.ModelArtifactType.TOSA_MLIR,
    ],
)

VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8 = def_types.Model(
    name="VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8",
    tags=["int8", "batch-1"],
    model_impl=TFLITE_MODEL_IMPL,
    model_parameters={
        "model_uri":
            f"{JAX_MODELS_GCS_DIR}/VIT_CLASSIFICATION_JAX_3X224X224XF32/model_int8.tflite",
    },
    artifacts_dir_url=
    f"{PARENT_GCS_DIR}/VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8",
    exported_model_types=[
        def_types.ModelArtifactType.TFLITE_INT8,
        def_types.ModelArtifactType.TOSA_MLIR,
    ],
)

ALL_MODELS = list(
    itertools.chain(
        BERT_BASE_FP32_TFLITE_I32_INPUT_SEQUENCES.values(),
        BERT_BASE_FP16_TFLITE_I32_INPUT_SEQUENCES.values(),
        BERT_BASE_DYN_QUANT_TFLITE_I32_INPUT_SEQUENCES.values(),
        BERT_BASE_INT8_TFLITE_I32_INPUT_SEQUENCES.values(),
    )) + [
        VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32,
        VIT_CLASSIFICATION_FP16_TFLITE_3X224X224XF32,
        VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32,
        VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8,
    ]
