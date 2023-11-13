# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image that can run https://github.com/nod-ai/convperf.

FROM gcr.io/iree-oss/openxla-benchmark/base@sha256:0f98f0d27199bcb40f8f76fc628a12f19b87b6f2f7d270a8f5ad9265f06effec

######## OpenMP ########
RUN apt-get update \
    && apt-get install -y libomp-14-dev
##############
