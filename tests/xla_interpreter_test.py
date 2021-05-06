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

import unittest

from absl.testing import absltest

from jax import test_util as jtu
from jax._src import api
from jax.interpreters import xla


class XlaInterpreterTest(jtu.JaxTestCase):

  @unittest.skipIf(not xla._ALLOW_ARG_PRUNING, "Test requires jaxlib 0.1.66")
  def test_prune_jit_args(self):
    def f(*args):
      return args[0]

    closed_jaxpr = api.make_jaxpr(f)(*range(10))
    pruned_jaxpr, kept_const_idx, kept_var_idx = xla._prune_unused_inputs(
        closed_jaxpr.jaxpr)
    assert len(pruned_jaxpr.invars) == 1
    assert kept_const_idx == set()
    assert kept_var_idx == {0}


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
