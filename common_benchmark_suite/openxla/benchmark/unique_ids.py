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
# JAX T5 large models and test data.
MODEL_T5_LARGE_FP32_JAX = f"{MODEL_T5_LARGE_FP32}-JAX"
INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32 = "4552b15b-8d90-498c-9282-f6553ae4db38"
OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32 = "e245bd97-a1fa-4f35-b04b-92664c4f49db"
