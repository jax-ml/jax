# Copyright 2026 The JAX Authors.
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

"""Tests for experimental searchsorted primitive."""

from typing import Callable, NamedTuple, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
from jax._src import config
from jax._src.numpy import hijax
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax._src import test_util as jtu

config.parse_flags_with_absl()

SHAPES = ((), (4,), (2, 3))
DTYPES = (np.int32, np.float32)

class OpRecord(NamedTuple):
  name: str
  hijax_op: Callable[[ArrayLike, ArrayLike], Array]
  np_op: Callable[[np.ndarray, np.ndarray], np.ndarray]
  shapes: Sequence[tuple[int, ...]]
  dtypes: Sequence[DTypeLike]


BINARY_OPERATORS = [
  OpRecord("multiply", hijax.multiply, np.multiply, SHAPES, DTYPES),
  OpRecord("add", hijax.add, np.add, SHAPES, DTYPES),
  OpRecord("subtract", hijax.subtract, np.subtract, SHAPES, DTYPES),
]


class NumpyHijaxOpTest(jtu.JaxTestCase):
  @parameterized.named_parameters(
      dict(testcase_name=f"_{op.name}_{dtype.__name__}{list(shape)}",
           hijax_op=op.hijax_op, np_op=op.np_op, shape=shape, dtype=dtype)
      for op in BINARY_OPERATORS
      for shape in op.shapes
      for dtype in op.dtypes)
  def test_binary_op(self, hijax_op, np_op, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape, dtype))
    self._CheckAgainstNumpy(np_op, hijax_op, args_maker)
    self._CompileAndCheck(hijax_op, args_maker)

  @parameterized.named_parameters(
      dict(testcase_name=f"_{op.name}_{dtype.__name__}{list(shape)}",
           hijax_op=op.hijax_op, shape=shape, dtype=dtype)
      for op in BINARY_OPERATORS
      for shape in op.shapes
      for dtype in [np.float32])
  def test_binary_op_grads(self, hijax_op, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args = (rng(shape, dtype), rng(shape, dtype))
    jtu.check_grads(hijax_op, args, 2, atol=1E-2, rtol=1E-2)

  @parameterized.named_parameters(
      dict(testcase_name=f"_{op.name}_{dtype.__name__}{list(shape)}",
           hijax_op=op.hijax_op, np_op=op.np_op, shape=shape, dtype=dtype)
      for op in BINARY_OPERATORS
      for shape in op.shapes if shape != ()
      for dtype in op.dtypes)
  def test_binary_op_batching(self, hijax_op, np_op, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape, dtype))
    self._CheckAgainstNumpy(np_op, jax.vmap(hijax_op), args_maker)
    self._CompileAndCheck(jax.vmap(hijax_op), args_maker)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
