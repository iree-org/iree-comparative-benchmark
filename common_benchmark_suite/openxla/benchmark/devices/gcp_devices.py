# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from openxla.benchmark import def_types, unique_ids

GCP_A2_HIGHGPU_1G = def_types.DeviceSpec(
    id=unique_ids.DEVICE_SPEC_GCP_A2_HIGHGPU_1G,
    name="a2-highgpu-1g",
    host_type="gcp",
    host_model="a2-highgpu-1g",
    host_environment="linux-x86_64",
    accelerator_type="gpu",
    accelerator_model="nvidia-a100",
    accelerator_architecture="ampere",
    accelerator_attributes={
        "num_of_gpus": 1,
    },
)

GCP_C2_STANDARD_16 = def_types.DeviceSpec(
    id=unique_ids.DEVICE_SPEC_GCP_C2_STANDARD_16,
    name="c2-standard-16",
    host_type="gcp",
    host_model="c2-standard-16",
    host_environment="linux-x86_64",
    accelerator_type="cpu",
    accelerator_model="intel-cascadelake",
    accelerator_architecture="x86_64-cascadelake",
    accelerator_attributes={
        "num_of_cores": 16,
    },
)

ALL_DEVICES = [GCP_A2_HIGHGPU_1G, GCP_C2_STANDARD_16]
