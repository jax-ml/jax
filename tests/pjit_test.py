# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from jax import test_util as jtu
from jax.api import pjit
from jax.interpreters.parallel import psum

from jax.config import config
config.parse_flags_with_absl()


class PmapTest(jtu.JaxTestCase):

  @jtu.skip_on_devices("gpu")
  def testBasic(self):
    f = lambda x: x - psum(x, 'i')
    x = onp.arange(8., dtype=onp.float32).reshape(4, 2)
    f = pjit(f, axis_name='i', in_axes=0, out_axes=0, mesh_axis=0)
    ans = f(x)
    expected = x - x.sum(0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("gpu")
  def testTupleOutput(self):
    f = lambda x: (x - psum(x, 'i'),)
    x = onp.arange(8., dtype=onp.float32).reshape(4, 2)
    f = pjit(f, axis_name='i', in_axes=0, out_axes=0, mesh_axis=0)
    ans = f(x)
    expected = (x - x.sum(0),)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("gpu")
  def testTupleInput(self):
    f = lambda x: x[0] - psum(x[0], 'i')
    x = onp.arange(8., dtype=onp.float32).reshape(4, 2)
    f = pjit(f, axis_name='i', in_axes=0, out_axes=0, mesh_axis=0)
    ans = f((x,))
    expected = x - x.sum(0)
    self.assertAllClose(ans, expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
