# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image that can run https://github.com/nod-ai/convperf.

FROM gcr.io/iree-oss/openxla-benchmark/base-python3.10@sha256:e19b4743fe06d0d779fb8a47e5d37112e1a25319dce8e1f381d73a99ed29dac2

######## OpenMP ########
RUN apt-get update \
    && apt-get install -y libomp-14-dev
##############
