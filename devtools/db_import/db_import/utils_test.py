#!/usr/bin/env python
## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from db_import.utils import first_no_except


class TestFirstNoExcept(unittest.TestCase):

  def test_empty(self):

    def raise_immediately(arg):
      raise RuntimeError("")

    self.assertEqual(first_no_except(raise_immediately, []), None)

  def test_failure(self):

    def raise_immediately(arg):
      raise RuntimeError("")

    with self.assertRaises(RuntimeError):
      first_no_except(raise_immediately, [1])

  def test_success(self):

    def raise_when_smaller_5(arg):
      raise_when_smaller_5.number_of_calls += 1
      if arg < 5:
        raise RuntimeError()

      return arg

    raise_when_smaller_5.number_of_calls = 0

    self.assertEqual(first_no_except(raise_when_smaller_5, range(10)), 5)
    self.assertEqual(raise_when_smaller_5.number_of_calls, 6)
