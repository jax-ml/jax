# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the

import numpy as np


class HashableArray:
  __slots__ = ["val"]
  val: np.ndarray

  def __init__(self, val):
    self.val = np.array(val, copy=True)
    self.val.setflags(write=False)

  def __repr__(self):
    return f"HashableArray({self.val!r})"

  def __str__(self):
    return f"HashableArray({self.val})"

  def __hash__(self):
    return hash((self.val.shape, self.val.dtype, self.val.tobytes()))

  def __eq__(self, other):
    return isinstance(other, HashableArray) and np.array_equal(
        self.val, other.val
    )
