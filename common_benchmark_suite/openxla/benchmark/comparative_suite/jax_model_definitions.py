# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import string

from openxla.benchmark import def_types, unique_ids
from openxla.benchmark.comparative_suite import utils

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752"

# Constants and functions help build batch templates.
BATCH_ID = lambda model_id: string.Template(model_id + "-batch${batch_size}")
BATCH_NAME = lambda name: string.Template(name + "_BATCH${batch_size}")
BATCH_TAG = string.Template("batch-${batch_size}")
BATCH_SIZE_PARAM = string.Template("${batch_size}")

T5_LARGE_FP32_JAX_IMPL = def_types.ModelImplementation(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX,
    name="T5_LARGE_FP32_JAX",
    tags=["fp32", "transformer-encoder", "transformer-decoder", "t5"],
    framework_type=def_types.ModelFrameworkType.JAX,
    data_type=def_types.ModelDataType.FP32,
    module_path=f"{utils.MODELS_MODULE_PATH}.jax_models.t5_large.model",
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5Model",
)

T5_LARGE_FP32_JAX_512XI32_BATCH_TEMPLATE = utils.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_T5_LARGE_FP32_JAX),
    name=BATCH_NAME("T5_LARGE_FP32_JAX_512XI32"),
    tags=["batch-1"],
    model_impl=T5_LARGE_FP32_JAX_IMPL,
    model_parameters={"batch_size": BATCH_SIZE_PARAM},
    artifacts={
        def_types.ModelArtifactType.STABLEHLO:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.STABLEHLO,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "/T5_LARGE/batch_${batch_size}/stablehlo.mlirbc"),
            ),
        def_types.ModelArtifactType.XLA_HLO_DUMP:
            utils.ModelArtifactTemplate(
                artifact_type=def_types.ModelArtifactType.XLA_HLO_DUMP,
                source_url=string.Template(
                    PARENT_GCS_DIR +
                    "/T5_LARGE/batch_${batch_size}/hlo/jit_forward.before_optimizations.txt"
                ),
            ),
    },
)
T5_LARGE_FP32_JAX_512XI32_BATCHES = utils.build_batch_models(
    template=T5_LARGE_FP32_JAX_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])
