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

import inspect
import re
from typing import Callable, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import lax
from jax._src import test_util as jtu
from jax.config import config
import numpy as np

config.parse_flags_with_absl()
FLAGS = config.FLAGS

class JitPolyTest(jtu.JaxTestCase):
  def setUp(self):
    self.prev_enable_mlir = config.jax_enable_mlir
    config.update("jax_enable_mlir", True)

  def tearDown(self):
    config.update("jax_enable_mlir", self.prev_enable_mlir)

  def run_static(self, f: Callable, args: Sequence):
    jf = jax.jit(f, backend="iree")
    print(f"[{self._testMethodName}]: with static shapes:", jf.lower(*args).compiler_ir(dialect="mhlo"))
    ys = jf(*args)
    self.assertAllClose(ys, f(*args))

  def run_poly(self, f: Callable, args: Sequence,
               polymorphic_shapes: Sequence[str]):
    jfp = jax.jit(f, experimental_polymorphic_shapes=polymorphic_shapes,
                  backend="iree")
    print(f"[{self._testMethodName}]: with polymorphic shapes:", jfp.lower(*args).compiler_ir(dialect="mhlo"))
    yp = jfp(*args)
    self.assertAllClose(yp, f(*args))

  def run_static_and_poly(self, f: Callable, args: Sequence,
                          polymorphic_shapes: Sequence[str]):
    self.run_static(f, args)
    self.run_poly(f, args, polymorphic_shapes)

  def test_sin(self):
    def f(x):  # f32[h, w] -> f32[h, w]
      return jnp.sin(x)

    x = np.ones((3, 5), dtype=np.float32)
    self.run_static_and_poly(f, [x], ["h, w"])

  def test_nested_sin(self):
    def nested_f(x):  # f32[h, v] -> f32[h, v]
      # A nested call that needs shape variables
      return jnp.sin(x)

    def f(x):  # f32[h, w] -> f32[h, w]
      return jnp.sin(x) + jax.jit(nested_f)(x)
    x = np.ones((3, 5), dtype=np.float32)
    self.run_static_and_poly(f, [x], ["h, w"])

  def test_transpose(self):
    def f(x):  # f32[h, w] -> f32[w, h]
      return x.T

    x = np.ones((3, 5), dtype=np.float32)
    self.run_static_and_poly(f, [x], ["h, w"])

  def test_matmul(self):
    def f(x):  # f32[w, w] -> f32[w, w]
      return jnp.matmul(x, x)

    x = np.ones((5, 5), dtype=np.float32)
    self.run_static_and_poly(f, [x], ["w, w"])

  def test_matmul_shape_error(self):
    def f(x):  # f32[h, w] -> error
      return jnp.matmul(x, x)

    x = np.ones((5, 5), dtype=np.float32)
    with self.assertRaisesRegex(TypeError,
                                re.escape("dot_general requires contracting dimensions to have the same shape, got [w] and [h]")):
      self.run_static_and_poly(f, [x], ["h, w"])

  def test_concat(self):
    def f(x):  # f32[h, w] -> f32[h, 2 * w]
      return jnp.concatenate([x, jnp.sin(x)], axis=1)

    x = np.ones((3, 5), dtype=np.float32)
    self.run_static_and_poly(f, [x], ["h, w"])

  def test_reshape(self):
    def f(x):  # f32[h, w] -> f32[h * w]
      return jnp.reshape(x, (-1,))
    x = np.ones((2, 5), dtype=np.float32)
    self.run_static_and_poly(f, [x], ["h, w"])

  def test_nested_reshape(self):
    def nested_f(x):  # f32[h, v] -> f32[h, v
      # A nested call that needs to compute with shapes
      return jnp.arange(x.shape[0] * x.shape[1], dtype=x.dtype).reshape(x.shape)

    def f(x):  # f32[h, w] -> f32[h, w]
      return jnp.sin(x) + jax.jit(nested_f)(x)
    x = np.ones((3, 5), dtype=np.float32)
    self.run_static_and_poly(f, [x], ["h, w"])

  def test_cond(self):
    def f(x):  # f32[w, w] -> f32[w, w]
      return lax.cond(True,
                      lambda x: jnp.sin(x),
                      lambda x: jnp.matmul(x, x), x)
    x = np.ones((5, 5), dtype=np.float32)
    self.run_static_and_poly(f, [x], ["w, w"])

  def test_arange(self):
    def f(x):  # f32[w] -> f32[w]
      return jnp.arange(x.shape[0], dtype=x.dtype) + x
    x = np.ones((5,), dtype=np.float32)
    self.run_static_and_poly(f, [x], ["w"])

  def test_broadcast(self):
    def f(x):  # f32[w] -> f32[w, w]
      return jnp.broadcast_to(x, (x.shape[0], x.shape[0]))
    x = np.ones((5,), dtype=np.float32)
    self.run_static_and_poly(f, [x], ["w"])

  def test_stack(self):
    def f(x):
      return jnp.stack([jnp.sin(x), jnp.cos(x)])

    x = np.ones((5,), dtype=np.float32)
    self.run_static_and_poly(f, [x], "w")
