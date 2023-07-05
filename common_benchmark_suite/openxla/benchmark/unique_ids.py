# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""List of unique IDs in the benchmark suite.

Each ID should start with a UUID generated from uuid.uuid4().
"""

################################################################################
# T5 models                                                              #
################################################################################
MODEL_IMPL_T5 = "173c7180-bad4-4b91-8423-4beeb13d2b0a-MODEL_IMPL"
MODEL_T5_LARGE = "173c7180-bad4-4b91-8423-4beeb13d2b0a-MODEL_T5_LARGE"
MODEL_T5_LARGE_FP32 = f"{MODEL_T5_LARGE}-fp32"
MODEL_T5_LARGE_FP16 = f"{MODEL_T5_LARGE}-fp16"
MODEL_T5_LARGE_BF16 = f"{MODEL_T5_LARGE}-bf16"

# JAX T5 models and test data.
MODEL_IMPL_T5_JAX = f"{MODEL_IMPL_T5}-JAX"

MODEL_T5_LARGE_FP32_JAX = f"{MODEL_T5_LARGE_FP32}-JAX"
MODEL_T5_LARGE_FP16_JAX = f"{MODEL_T5_LARGE_FP16}-JAX"
MODEL_T5_LARGE_BF16_JAX = f"{MODEL_T5_LARGE_BF16}-JAX"

MODEL_T5_LARGE_FP32_JAX_512XI32 = f"{MODEL_T5_LARGE_FP32_JAX}-512xi32"
MODEL_T5_LARGE_FP16_JAX_512XI32 = f"{MODEL_T5_LARGE_FP16_JAX}-512xi32"
MODEL_T5_LARGE_BF16_JAX_512XI32 = f"{MODEL_T5_LARGE_BF16_JAX}-512xi32"

INPUT_DATA_T5_LARGE_JAX_512 = "4552b15b-8d90-498c-9282-f6553ae4db38"
INPUT_DATA_T5_LARGE_FP32_JAX_512XI32 = f"{INPUT_DATA_T5_LARGE_JAX_512}-fp32"
INPUT_DATA_T5_LARGE_FP16_JAX_512XI32 = f"{INPUT_DATA_T5_LARGE_JAX_512}-fp16"
INPUT_DATA_T5_LARGE_BF16_JAX_512XI32 = f"{INPUT_DATA_T5_LARGE_JAX_512}-bf16"
OUTPUT_DATA_T5_LARGE_JAX_512X1024 = "e245bd97-a1fa-4f35-b04b-92664c4f49db"
OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32 = f"{OUTPUT_DATA_T5_LARGE_JAX_512X1024}-fp32"
OUTPUT_DATA_T5_LARGE_FP16_JAX_512X1024XF16 = f"{OUTPUT_DATA_T5_LARGE_JAX_512X1024}-fp16"
OUTPUT_DATA_T5_LARGE_BF16_JAX_512X1024XBF16 = f"{OUTPUT_DATA_T5_LARGE_JAX_512X1024}-bf16"

# Note: `4CG` is short for `ForConditionalGeneration`.
MODEL_T5_LARGE_4CG = "1dd120bd-fa12-4d26-9f84-92a8c8f50d1e"
MODEL_T5_LARGE_4CG_FP32 = f"{MODEL_T5_LARGE_4CG}-fp32"
# JAX T5ForConditionalGeneration models and test data.
MODEL_T5_LARGE_4CG_FP32_JAX = f"{MODEL_T5_LARGE_4CG_FP32}-JAX"
INPUT_DATA_T5_LARGE_4CG_SEQLEN512_I32 = "49de56f6-a3d2-4b25-abc1-0a922f0affec"
OUTPUT_DATA_T5_LARGE_4CG_FP32_JAX_512X1024XF32 = "49eb591d-4d91-4b16-94bf-7d01cda3826b"

################################################################################
# Bert models                                                                  #
################################################################################
MODEL_BERT = "47cb0d3a-5eb7-41c7-9d7c-97aae7023ecf"
MODEL_BERT_LARGE = f"{MODEL_BERT}-MODEL_BERT_LARGE"
MODEL_BERT_LARGE_FP32 = f"{MODEL_BERT_LARGE}-fp32"
MODEL_BERT_LARGE_FP16 = f"{MODEL_BERT_LARGE}-fp16"
MODEL_BERT_LARGE_BF16 = f"{MODEL_BERT_LARGE}-bf16"

# JAX Bert models
MODEL_IMPL_BERT_JAX = f"{MODEL_BERT}-MODEL_IMPL-JAX"

MODEL_BERT_LARGE_FP32_JAX = f"{MODEL_BERT_LARGE_FP32}-JAX"
MODEL_BERT_LARGE_FP32_JAX_384XI32 = f"{MODEL_BERT_LARGE_FP32_JAX}-384xi32"

MODEL_BERT_LARGE_FP16_JAX = f"{MODEL_BERT_LARGE_FP16}-JAX"
MODEL_BERT_LARGE_FP16_JAX_384XI32 = f"{MODEL_BERT_LARGE_FP16_JAX}-384xi32"

MODEL_BERT_LARGE_BF16_JAX = f"{MODEL_BERT_LARGE_BF16}-JAX"
MODEL_BERT_LARGE_BF16_JAX_384XI32 = f"{MODEL_BERT_LARGE_BF16_JAX}-384xi32"

INPUT_DATA_BERT_LARGE_JAX_384XI32 = "4ba707a4-1de7-4bc8-a9f6-b40b04af503d"
INPUT_DATA_BERT_LARGE_FP32_JAX_384XI32 = f"{INPUT_DATA_BERT_LARGE_JAX_384XI32}-fp32"
INPUT_DATA_BERT_LARGE_FP16_JAX_384XI32 = f"{INPUT_DATA_BERT_LARGE_JAX_384XI32}-fp16"
INPUT_DATA_BERT_LARGE_BF16_JAX_384XI32 = f"{INPUT_DATA_BERT_LARGE_JAX_384XI32}-bf16"
OUTPUT_DATA_BERT_LARGE_JAX_384X1024 = "dce6c4a4-0fc3-43d6-bb01-b818331cdbd3"
OUTPUT_DATA_BERT_LARGE_FP32_JAX_384X1024XF32 = f"{OUTPUT_DATA_BERT_LARGE_JAX_384X1024}-fp32"
OUTPUT_DATA_BERT_LARGE_FP16_JAX_384X1024XF16 = f"{OUTPUT_DATA_BERT_LARGE_JAX_384X1024}-fp16"
OUTPUT_DATA_BERT_LARGE_BF16_JAX_384X1024XBF16 = f"{OUTPUT_DATA_BERT_LARGE_JAX_384X1024}-bf16"

################################################################################
# Device IDs                                                                   #
################################################################################
DEVICE_SPEC_GCP_C2_STANDARD_16 = "9a4804f1-b1b9-46cd-b251-7f16a655f782"
DEVICE_SPEC_GCP_A2_HIGHGPU_1G = "78c56b95-2d7d-44b5-b5fd-8e47aa961108"
