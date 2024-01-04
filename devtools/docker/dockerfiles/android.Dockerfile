# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling towards Android.

FROM gcr.io/iree-oss/openxla-benchmark/base-python3.11@sha256:fe125eee60a701732ee61aa23e8bb1ae9d48ef0846a4ea78f607b00f7cbc43ec

######## Android NDK ########
ARG NDK_VERSION=r25c
WORKDIR /install-ndk

ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"
ENV ANDROID_NDK_API_LEVEL "25"

RUN wget -q "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip" \
    && unzip -q "android-ndk-${NDK_VERSION}-linux.zip" -d /usr/src/  \
    && rm -rf /install-ndk

WORKDIR /
