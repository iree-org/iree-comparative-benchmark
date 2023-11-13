# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling towards Android.

FROM gcr.io/iree-oss/openxla-benchmark/base@sha256:2dbee52eaa63e62137682f0eda701ac4cf59b8e16395daa757f6e1906b52dd82

ARG NDK_VERSION=r26b
WORKDIR /install-ndk

ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"

RUN wget -q "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip" \
    && unzip -q "android-ndk-${NDK_VERSION}-linux.zip" -d /usr/src/  \
    && rm -rf /install-ndk

WORKDIR /
