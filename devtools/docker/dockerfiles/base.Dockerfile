# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image that includes base packages.

# Ubuntu 22.04.
FROM ubuntu@sha256:817cfe4672284dcbfee885b1a66094fd907630d610cab329114d036716be49ba

######## Base ########
RUN apt-get update \
  && apt-get install -y \
    git \
    unzip \
    wget \
    curl \
    gnupg2 \
    python3-numpy \
    cmake \
    ninja-build \
    clang \
    lld \
    numactl

######## Python ########
WORKDIR /install-python

ARG PYTHON_VERSION=3.10

COPY devtools/docker/context/python_build_requirements.txt devtools/docker/context/install_python_deps.sh ./
RUN ./install_python_deps.sh "${PYTHON_VERSION}" \
  && apt-get -y install python-is-python3 \
  && rm -rf /install-python

ENV PYTHON_BIN /usr/bin/python3

WORKDIR /

######## Bazel ########
WORKDIR /install-bazel
COPY devtools/docker/context/install_bazel.sh devtools/docker/context/.bazelversion ./
RUN ./install_bazel.sh && rm -rf /install-bazel
WORKDIR /
