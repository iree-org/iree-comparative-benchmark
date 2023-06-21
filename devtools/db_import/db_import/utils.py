## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypeVar, Callable, Iterable

T = TypeVar("T")
R = TypeVar("R")


def first_no_except(function: Callable[[T], R], iterable: Iterable[T]) -> R:
  """ Returns `function(item)` for the first `item` from `iterable` that didn't raise an exception.

  - If all items raise an exception the exception raised by `function(last_item)` is raised.
  - If the iterable yields no items, `None` is returned.
  """
  last_exception = None

  for el in iterable:
    try:
      return function(el)
    except Exception as e:
      last_exception = e

  if last_exception:
    raise last_exception
