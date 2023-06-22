## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines protocols for a subset of types from `google.cloud.storage`.
   Check out `local_storage` for an alternative implementation of these types.
"""

from typing import Protocol, IO, Sequence, runtime_checkable


@runtime_checkable
class Blob(Protocol):

  @property
  def name(self):
    ...

  def open(self) -> IO:
    ...

  @property
  def bucket(self) -> "Bucket":
    ...


@runtime_checkable
class Bucket(Protocol):

  def list_blobs(self) -> Sequence[Blob]:
    ...

  def blob(self, filepath: str) -> Blob:
    ...


@runtime_checkable
class Client(Protocol):

  def get_bucket(self, name: str) -> Bucket:
    ...
