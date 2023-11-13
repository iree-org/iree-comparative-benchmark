# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from openxla.benchmark import def_types

MOBILE_PIXEL_6_PRO = def_types.DeviceSpec(
    name="pixel-6-pro",
    host_type="mobile",
    host_model="pixel-6-pro",
    host_environment="android",
    accelerator_type="cpu",
    accelerator_model="armv8.2-a",
    accelerator_architecture="armv8.2-a",
    accelerator_attributes={
        "num_of_cores": 8,
    },
)

MOBILE_PIXEL_8_PRO = def_types.DeviceSpec(
    name="pixel-8-pro",
    host_type="mobile",
    host_model="pixel-8-pro",
    host_environment="android",
    accelerator_type="cpu",
    accelerator_model="armv9-a",
    accelerator_architecture="armv9-a",
    accelerator_attributes={
        "num_of_cores": 9,
    },
)

ALL_DEVICES = [MOBILE_PIXEL_6_PRO, MOBILE_PIXEL_8_PRO]
