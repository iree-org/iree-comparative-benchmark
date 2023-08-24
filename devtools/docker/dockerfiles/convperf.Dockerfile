# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image that can run https://github.com/nod-ai/convperf.

FROM gcr.io/iree-oss/openxla-benchmark/base@sha256:1bf3e319465ec8fb465baae3f6ba9a5b09cb84a5349a675c671a552fc77f2251

######## OpenMP ########
RUN apt-get update \
    && apt-get install -y libomp-14-dev
##############
