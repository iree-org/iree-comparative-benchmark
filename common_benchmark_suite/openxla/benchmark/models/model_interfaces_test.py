# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test the uses of model interfaces."""

from typing import List, Tuple
import unittest

from openxla.benchmark.models import model_interfaces


class InferenceModelTest(unittest.TestCase):

  def test_protocol_with_single_input(self):

    class TestModel(model_interfaces.InferenceModel):

      def generate_default_inputs(self) -> str:
        return "abc"

      def preprocess(self, raw_text: str) -> List[int]:
        return [ord(c) for c in raw_text]

      def forward(self, processed_input: List[int]) -> List[int]:
        return [num + 1 for num in processed_input]

      def postprocess(self, output: List[int]) -> str:
        return "".join(chr(num) for num in output)

    model: model_interfaces.InferenceModel = TestModel()

    raw_input = model.generate_default_inputs()
    processed_input = model.preprocess(raw_input)
    output = model.forward(processed_input)
    processed_output = model.postprocess(output)

    self.assertEqual(raw_input, "abc")
    self.assertEqual(processed_input, [97, 98, 99])
    self.assertEqual(output, [98, 99, 100])
    self.assertEqual(processed_output, "bcd")

  def test_protocol_with_tuple_inputs(self):

    class TestModel(model_interfaces.InferenceModel):

      def generate_default_inputs(self) -> Tuple[str, str]:
        return ("abc", "123")

      def preprocess(self, raw_first: str,
                     raw_second: str) -> Tuple[List[int], List[int]]:
        return ([ord(c) for c in raw_first], [ord(c) for c in raw_second])

      def forward(self, first: List[int],
                  second: List[int]) -> Tuple[List[int], List[int]]:
        return (second, first)

      def postprocess(self, first_out: List[int], second_out: List[int]) -> str:
        return "".join(chr(num) for num in (first_out + second_out))

    model: model_interfaces.InferenceModel = TestModel()

    raw_inputs = model.generate_default_inputs()
    processed_inputs = model.preprocess(*raw_inputs)
    outputs = model.forward(*processed_inputs)
    processed_output = model.postprocess(*outputs)

    self.assertEqual(raw_inputs, ("abc", "123"))
    self.assertEqual(processed_inputs, ([97, 98, 99], [49, 50, 51]))
    self.assertEqual(outputs, ([49, 50, 51], [97, 98, 99]))
    self.assertEqual(processed_output, "123abc")


if __name__ == "__main__":
  unittest.main()
