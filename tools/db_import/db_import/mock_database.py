## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class MockTable:
  def __init__(self, name : str):
    self.name : str = name

  @property
  def table_id(self):
    return self.name.split('.')[1]

  @property
  def dataset_id(self):
    return self.name.split('.')[0]


class MockClient:

  def get_table(self, name : str):
    return MockTable(name)

  def query(self):
    raise RuntimeError("Not supported")

  def insert_rows(self, table, rows : list):
    print(rows)
