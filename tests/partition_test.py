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

import functools

from absl.testing import absltest

from jax import api
from jax import core
from jax import test_util as jtu
from jax.interpreters import xla
from jax.lax import lax
from jax.lib import xla_bridge as xb
import numpy as onp

from jax.config import config
config.parse_flags_with_absl()


class PartitionTest(jtu.JaxTestCase):

  def test_jaxpr_to_xla(self):
    new_var = core.gensym('')
    a = new_var()
    b = new_var()
    c = new_var()

    jaxpr = core.Jaxpr(
        constvars=(),
        freevars=(),
        invars=(a, b),
        outvars=(c,),
        eqns=(core.JaxprEqn(
            invars=(a, b),
            outvars=(c,),
            primitive=lax.add_p,
            bound_subjaxprs=None,
            params={},
            partition_id=1),))

    cb = xb.make_computation_builder('xla_computation')
    xla_args = [
        cb.ParameterFromNumpy(onp.array((), dtype=onp.float32)),
        cb.ParameterFromNumpy(onp.array((), dtype=onp.float32))
    ]
    outs = xla.jaxpr_subcomp(cb, jaxpr, 'cpu', None, (), (), *xla_args)
    computation = cb.Build(*outs)

    self.assertIn('sharding={maximal device=1}', computation.GetHloText())


if __name__ == '__main__':
  absltest.main()
