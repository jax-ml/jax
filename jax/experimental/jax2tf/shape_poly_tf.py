# Copyright 2021 Google LLC
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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Specialized shape polymorphism support for jax2tf.

See the shape_poly.py module documentation, the jax2tf.convert docstring, and the
[README](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
"""
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf  # type: ignore[import]

from jax.experimental.jax2tf import shape_poly

TfVal = Any

class DimExprTfVal(shape_poly.DimExpr):
  """Express dimensions using tf.shape and the TF arguments."""
  def __init__(self, tfval: TfVal):
    super(DimExprTfVal, self).__init__(tfval)

  @classmethod
  def for_arg(cls, arg: TfVal) -> Sequence[shape_poly.DimExpr]:
    tf_shape = tf.shape(arg)
    return tuple(DimExprTfVal(tf_shape[i]) for i in range(len(np.shape(arg))))

  def is_known_constant(self) -> Optional[int]:
    # When under TF eager, the dimension expressions should be constants.
    # Under TF graph, they will not be.
    try:
      return self.raw.numpy()
    except AttributeError as e:
      assert str(e).find("numpy") > 0, e
      return None

  def add(self, other: Union[shape_poly.DimExpr, int]) -> shape_poly.DimExpr:
    if isinstance(other, shape_poly.DimExpr):
      other = other.raw  # type: ignore[assignment]
    return DimExprTfVal(tf.math.add(self.raw, other))

  def subtract(self, other: Union[shape_poly.DimExpr, int]) -> shape_poly.DimExpr:
    if isinstance(other, shape_poly.DimExpr):
      other = other.raw  # type: ignore[assignment]
    return DimExprTfVal(tf.math.subtract(self.raw, other))

  def multiply(self, other: Union[shape_poly.DimExpr, int]) -> shape_poly.DimExpr:
    if isinstance(other, shape_poly.DimExpr):
      other = other.raw  # type: ignore[assignment]
    return DimExprTfVal(tf.math.multiply(self.raw, other))

  def divmod(self, factor: int) -> Tuple[shape_poly.DimExpr, shape_poly.DimExpr]:
    dividend = DimExprTfVal(tf.math.floordiv(self.raw, factor)) if factor != 1 else self
    mod = DimExprTfVal(tf.math.floormod(self.raw, factor))
    return dividend, mod
