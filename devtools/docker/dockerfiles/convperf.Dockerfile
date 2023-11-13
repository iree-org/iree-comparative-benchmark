# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image that can run https://github.com/nod-ai/convperf.

FROM gcr.io/iree-oss/openxla-benchmark/base@sha256:2dbee52eaa63e62137682f0eda701ac4cf59b8e16395daa757f6e1906b52dd82

######## OpenMP ########
RUN apt-get update \
    && apt-get install -y libomp-14-dev
##############
