# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""List of unique IDs in the benchmark suite.

Each ID should start with a UUID generated from uuid.uuid4().
"""

################################################################################
# T5 large models                                                              #
################################################################################
MODEL_T5_LARGE = "173c7180-bad4-4b91-8423-4beeb13d2b0a-MODEL_T5_LARGE"
MODEL_T5_LARGE_FP32 = f"{MODEL_T5_LARGE}-fp32"
# JAX T5 models and test data.
MODEL_T5_LARGE_FP32_JAX = f"{MODEL_T5_LARGE_FP32}-JAX"
INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32 = "4552b15b-8d90-498c-9282-f6553ae4db38"
OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32 = "e245bd97-a1fa-4f35-b04b-92664c4f49db"

# Note: `4CG` is short for `ForConditionalGeneration`.
MODEL_T5_LARGE_4CG = "1dd120bd-fa12-4d26-9f84-92a8c8f50d1e"
MODEL_T5_LARGE_4CG_FP32 = f"{MODEL_T5_LARGE_4CG}-fp32"
# JAX T5ForConditionalGeneration models and test data.
MODEL_T5_LARGE_4CG_FP32_JAX = f"{MODEL_T5_LARGE_4CG_FP32}-JAX"
INPUT_DATA_T5_LARGE_4CG_SEQLEN512_I32 = "49de56f6-a3d2-4b25-abc1-0a922f0affec"
OUTPUT_DATA_T5_LARGE_4CG_FP32_JAX_512X1024XF32 = "49eb591d-4d91-4b16-94bf-7d01cda3826b"

################################################################################
# Device IDs                                                                   #
################################################################################
DEVICE_SPEC_GCP_C2_STANDARD_16 = "9a4804f1-b1b9-46cd-b251-7f16a655f782"
DEVICE_SPEC_GCP_A2_HIGHGPU_1G = "78c56b95-2d7d-44b5-b5fd-8e47aa961108"
