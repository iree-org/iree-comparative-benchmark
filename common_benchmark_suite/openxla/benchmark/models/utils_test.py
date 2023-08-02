# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from openxla.benchmark.models import utils


class UtilsTest(unittest.TestCase):

  def test_canonicalize_to_tuple_with_single_value(self):

    result = utils.canonicalize_to_tuple("test")

    self.assertEqual(result, ("test",))

  def test_canonicalize_to_tuple_with_tuple(self):

    result = utils.canonicalize_to_tuple(("a", "b"))

    self.assertEqual(result, ("a", "b"))


if __name__ == "__main__":
  unittest.main()
