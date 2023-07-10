# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import string

from openxla.benchmark import def_types, unique_ids
from openxla.benchmark.comparative_suite import utils

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.13_1688607404/"

T5_JAX_IMPL = def_types.ModelImplementation(
    id=unique_ids.MODEL_IMPL_T5_JAX,
    name="T5_JAX",
    tags=["transformer-encoder", "transformer-decoder", "t5"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.t5.t5_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5Model",
)

T5_LARGE_FP32_JAX_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32),
    name=utils.BATCH_NAME("T5_LARGE_FP32_JAX_512XI32"),
    tags=[utils.BATCH_TAG, "fp32"],
    model_impl=T5_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "t5-large",
        "seq_len": 512,
    },
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "T5_LARGE_FP32_JAX_512XI32_BATCH${batch_size}/stablehlo.mlirbc"
                ),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "T5_LARGE_FP32_JAX_512XI32_BATCH${batch_size}/xla_hlo_before_optimizations.txt"
                ),
            ),
    },
)
T5_LARGE_FP16_JAX_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_T5_LARGE_FP16_JAX_512XI32),
    name=utils.BATCH_NAME("T5_LARGE_FP16_JAX_512XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=T5_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp16",
        "model_name": "t5-large",
    },
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "T5_LARGE_FP16_JAX_512XI32_BATCH${batch_size}/stablehlo.mlirbc"
                ),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "T5_LARGE_FP16_JAX_512XI32_BATCH${batch_size}/xla_hlo_before_optimizations.txt"
                ),
            ),
    },
)
T5_LARGE_BF16_JAX_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_T5_LARGE_BF16_JAX_512XI32),
    name=utils.BATCH_NAME("T5_LARGE_BF16_JAX_512XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=T5_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "bf16",
        "model_name": "t5-large",
    },
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "T5_LARGE_BF16_JAX_512XI32_BATCH${batch_size}/stablehlo.mlirbc"
                ),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "T5_LARGE_BF16_JAX_512XI32_BATCH${batch_size}/xla_hlo_before_optimizations.txt"
                ),
            ),
    },
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

T5_4CG_JAX_IMPL = def_types.ModelImplementation(
    id=unique_ids.MODEL_IMPL_T5_4CG_JAX,
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
    id=utils.BATCH_ID(unique_ids.MODEL_T5_4CG_LARGE_FP32_JAX),
    name=utils.BATCH_NAME("T5_4CG_LARGE_FP32_JAX_512XI32"),
    tags=[utils.BATCH_TAG, "fp32"],
    model_impl=T5_4CG_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "t5-large",
        "seq_len": 512
    },
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "T5_4CG_LARGE_FP32_JAX_512XI32_BATCH${batch_size}/stablehlo.mlirbc"
                ),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "T5_4CG_LARGE_FP32_JAX_512XI32_BATCH${batch_size}/xla_hlo_before_optimizations.txt"
                ),
            ),
    },
)
T5_4CG_LARGE_FP32_JAX_512XI32_BATCHES = utils.build_batch_models(
    template=T5_4CG_LARGE_FP32_JAX_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48])

# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_JAX_IMPL = def_types.ModelImplementation(
    id=unique_ids.MODEL_IMPL_BERT_JAX,
    name="BERT_JAX",
    tags=["transformer-encoder", "bert"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.bert.bert_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
)

BERT_LARGE_FP32_JAX_384XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32),
    name=utils.BATCH_NAME("BERT_LARGE_FP32_JAX_384XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=BERT_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "seq_len": 384,
        "model_name": "bert-large-uncased",
    },
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "BERT_LARGE_FP32_JAX_384XI32_BATCH${batch_size}/stablehlo.mlirbc"
                ),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "BERT_LARGE_FP32_JAX_384XI32_BATCH${batch_size}/xla_hlo_before_optimizations.txt"
                ),
            ),
    })

BERT_LARGE_FP16_JAX_384XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_BERT_LARGE_FP16_JAX_384XI32),
    name=utils.BATCH_NAME("BERT_LARGE_FP16_JAX_384XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=BERT_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp16",
        "seq_len": 384,
        "model_name": "bert-large-uncased",
    },
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "BERT_LARGE_FP16_JAX_384XI32_BATCH${batch_size}/stablehlo.mlirbc"
                ),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "BERT_LARGE_FP16_JAX_384XI32_BATCH${batch_size}/xla_hlo_before_optimizations.txt"
                ),
            ),
    })

BERT_LARGE_BF16_JAX_384XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_BERT_LARGE_BF16_JAX_384XI32),
    name=utils.BATCH_NAME("BERT_LARGE_BF16_JAX_384XI32"),
    tags=[utils.BATCH_TAG],
    model_impl=BERT_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "bf16",
        "seq_len": 384,
        "model_name": "bert-large-uncased",
    },
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "BERT_LARGE_BF16_JAX_384XI32_BATCH${batch_size}/stablehlo.mlirbc"
                ),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "BERT_LARGE_BF16_JAX_384XI32_BATCH${batch_size}/xla_hlo_before_optimizations.txt"
                ),
            ),
    })

BERT_LARGE_FP32_JAX_384XI32_BATCHES = utils.build_batch_models(
    template=BERT_LARGE_FP32_JAX_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

BERT_LARGE_FP16_JAX_384XI32_BATCHES = utils.build_batch_models(
    template=BERT_LARGE_FP16_JAX_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

BERT_LARGE_BF16_JAX_384XI32_BATCHES = utils.build_batch_models(
    template=BERT_LARGE_BF16_JAX_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

# ResNet models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/resnet#transformers.FlaxResNetModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50

RESNET_JAX_IMPL = def_types.ModelImplementation(
    id=unique_ids.MODEL_IMPL_RESNET_JAX,
    name="RESNET_JAX",
    tags=["cnn", "resnet"],
    framework_type=def_types.ModelFrameworkType.JAX,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax.resnet.resnet_model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/resnet#transformers.FlaxResNetModel",
)

RESNET50_FP32_JAX_3X224X224XF32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32),
    name=utils.BATCH_NAME("RESNET50_FP32_JAX_3X224X224XF32"),
    tags=[utils.BATCH_TAG],
    model_impl=RESNET_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp32",
        "model_name": "microsoft/resnet-50",
    },
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "RESNET50_FP32_JAX_3X224X224XF32_BATCH${batch_size}/stablehlo.mlirbc"
                ),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "RESNET50_FP32_JAX_3X224X224XF32_BATCH${batch_size}/xla_hlo_before_optimizations.txt"
                ),
            ),
    })

RESNET50_FP16_JAX_3X224X224XF16_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_RESNET50_FP16_JAX_3X224X224XF16),
    name=utils.BATCH_NAME("RESNET50_FP16_JAX_3X224X224XF16"),
    tags=[utils.BATCH_TAG],
    model_impl=RESNET_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "fp16",
        "model_name": "microsoft/resnet-50",
    },
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "RESNET50_FP16_JAX_3X224X224XF16_BATCH${batch_size}/stablehlo.mlirbc"
                ),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "RESNET50_FP16_JAX_3X224X224XF16_BATCH${batch_size}/xla_hlo_before_optimizations.txt"
                ),
            ),
    })

RESNET50_BF16_JAX_3X224X224XBF16_BATCH_TEMPLATE = utils.ModelTemplate(
    id=utils.BATCH_ID(unique_ids.MODEL_RESNET50_BF16_JAX_3X224X224XBF16),
    name=utils.BATCH_NAME("RESNET50_BF16_JAX_3X224X224XBF16"),
    tags=[utils.BATCH_TAG],
    model_impl=RESNET_JAX_IMPL,
    model_parameters={
        "batch_size": utils.BATCH_SIZE_PARAM,
        "data_type": "bf16",
        "model_name": "microsoft/resnet-50",
    },
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "RESNET50_BF16_JAX_3X224X224XBF16_BATCH${batch_size}/stablehlo.mlirbc"
                ),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "RESNET50_BF16_JAX_3X224X224XBF16_BATCH${batch_size}/xla_hlo_before_optimizations.txt"
                ),
            ),
    })

RESNET50_FP32_JAX_3X224X224XF32_BATCHES = utils.build_batch_models(
    template=RESNET50_FP32_JAX_3X224X224XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

RESNET50_FP16_JAX_3X224X224XF16_BATCHES = utils.build_batch_models(
    template=RESNET50_FP16_JAX_3X224X224XF16_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

RESNET50_BF16_JAX_3X224X224XBF16_BATCHES = utils.build_batch_models(
    template=RESNET50_BF16_JAX_3X224X224XBF16_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

ALL_MODELS = list(
    itertools.chain(
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
    ))
