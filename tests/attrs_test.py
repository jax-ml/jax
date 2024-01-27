# Copyright 2024 The JAX Authors.
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

from __future__ import annotations

from dataclasses import dataclass

from absl.testing import absltest
from absl.testing import parameterized

import jax

from jax._src import config
from jax._src import test_util as jtu
from jax._src.util import safe_zip, safe_map

from jax.experimental.attrs import jax_setattr, jax_getattr

config.parse_flags_with_absl()

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

@dataclass
class Thing:
  x: float

class AttrsTest(jtu.JaxTestCase):

  @parameterized.parameters([True, False])
  def test_basic(self, jit: bool):
    thing = Thing(1.0)

    def double_it() -> None:
      cur_x = jax_getattr(thing, "x")
      jax_setattr(thing, "x", cur_x * 2)

    if jit:
      double_it = jax.jit(double_it)

    self.assertEqual(thing.x, 1.0)
    double_it()
    self.assertEqual(thing.x, 2.0)
    double_it()
    self.assertEqual(thing.x, 4.0)
    double_it()
    self.assertEqual(thing.x, 8.0)
    double_it()
    self.assertEqual(thing.x, 16.0)

  def test_nesting_basic(self):
    thing = Thing(1.0)

    @jax.jit
    @jax.jit
    def double_it() -> None:
      cur_x = jax_getattr(thing, "x")
      jax_setattr(thing, "x", cur_x * 2)

    self.assertEqual(thing.x, 1.0)
    double_it()
    self.assertEqual(thing.x, 2.0)
    double_it()
    self.assertEqual(thing.x, 4.0)
    double_it()
    self.assertEqual(thing.x, 8.0)
    double_it()
    self.assertEqual(thing.x, 16.0)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
