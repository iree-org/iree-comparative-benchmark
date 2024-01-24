# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image that can run https://github.com/nod-ai/convperf.

FROM gcr.io/iree-oss/openxla-benchmark/base-python3.10@sha256:245a074284cfed5de60cf06a153e3bcd9a9c42702b6bb66a39bb47ef23b61669

######## OpenMP ########
RUN apt-get update \
    && apt-get install -y libomp-14-dev
##############
