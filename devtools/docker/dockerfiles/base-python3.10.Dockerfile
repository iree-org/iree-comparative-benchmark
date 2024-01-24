# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling towards Android.

FROM gcr.io/iree-oss/openxla-benchmark/base@sha256:e05581d117fd00c31fcd3055ac43862d227a19335434c35be6a7b82411b06d3d

######## Python ########
WORKDIR /install-python

ARG PYTHON_VERSION=3.10

COPY devtools/docker/context/python_build_requirements.txt devtools/docker/context/install_python_deps.sh ./
RUN ./install_python_deps.sh "${PYTHON_VERSION}" \
  && apt-get -y install python-is-python3 \
  && rm -rf /install-python

ENV PYTHON_BIN /usr/bin/python3
ENV TF_PYTHON_VERSION "${PYTHON_VERSION}"

WORKDIR /
