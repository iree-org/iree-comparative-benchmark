# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from openxla.benchmark import def_types

HOST_GPU = def_types.DeviceSpec(
    name="host-gpu",
    host_type="host",
    host_model="unknown",
    host_environment="unknown",
    accelerator_type="gpu",
    accelerator_model="unknown",
    accelerator_architecture="unknown",
    accelerator_attributes={},
)

HOST_CPU = def_types.DeviceSpec(
    name="host-cpu",
    host_type="host",
    host_model="unknown",
    host_environment="unknown",
    accelerator_type="cpu",
    accelerator_model="unknown",
    accelerator_architecture="unknown",
    accelerator_attributes={},
)

ALL_DEVICES = [HOST_GPU, HOST_CPU]
