# Copyright 2020 Google LLC
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

import functools

from absl.testing import absltest

from jax import abstract_arrays
from jax import api
from jax import core
from jax import test_util as jtu
from jax.config import config
from jax.interpreters import partition
from jax.interpreters import xla
from jax.lax import lax
from jax.lib import xla_bridge as xb
import numpy as onp


config.parse_flags_with_absl()


class PartitionTest(jtu.JaxTestCase):

  def test_jaxpr_to_xla(self):
    new_var = core.gensym('')
    a = new_var()
    b = new_var()
    c = new_var()
    d = new_var()
    e = new_var()

    partition_put_eqn0 = core.JaxprEqn(
        invars=(a,), outvars=(c,), primitive=partition.partition_put_p,
        bound_subjaxpr=None, params={'partition_id': 1})
    partition_put_eqn1 = core.JaxprEqn(
        invars=(b,), outvars=(d,), primitive=partition.partition_put_p,
        bound_subjaxpr=None, params={'partition_id': 1})
    add_eqn = core.JaxprEqn(
        invars=(c, d), outvars=(e,), primitive=lax.add_p,
        bound_subjaxpr=None, params={})

    partition_jaxpr = core.Jaxpr(
        constvars=(),
        invars=(a, b),
        outvars=(e,),
        eqns=(partition_put_eqn0, partition_put_eqn1, add_eqn))

    partition_typed_jaxpr = core.TypedJaxpr(
        jaxpr=partition_jaxpr,
        literals=(),
        in_avals=[abstract_arrays.ShapedArray((), onp.float32)] * 2,
        out_avals=[abstract_arrays.ShapedArray((), onp.float32)])

    jaxpr = core.Jaxpr(
        constvars=(),
        invars=(a, b),
        outvars=(c,),
        eqns=(core.JaxprEqn(
            invars=(a, b),
            outvars=(c,),
            primitive=partition.partition_p,
            bound_subjaxpr=None,
            params={'jaxpr': partition_typed_jaxpr, 'partition_ids': [0, 1]}),))

    cb = xb.make_computation_builder('xla_computation')
    xla_args = [
        cb.ParameterFromNumpy(onp.array((), dtype=onp.float32)),
        cb.ParameterFromNumpy(onp.array((), dtype=onp.float32))
    ]
    outs = xla.jaxpr_subcomp(cb, jaxpr, 'cpu', None, (), '', *xla_args)
    computation = cb.Build(*outs)

    self.assertIn('sharding={maximal device=1}', computation.GetHloText())

  def test_jaxpr(self):
    @functools.partial(partition.partition, num_partitions=2)
    def f(x):
      y = x * x
      z = 38. + partition.partition_put(y, 1)
      return z

    jaxpr = api.make_jaxpr(f)(2.)
    self.assertEqual(
        '{ lambda  ; a.\n'
        '  let b = partition[ jaxpr={ lambda  ; a.\n'
        '                             let b = mul a a\n'
        '                                 c = partition_put[ partition_id=1 ] b\n'
        '                                 d = add c 38.0\n'
        '                             in [d] }\n'
        '                     partition_ids=(0, 1) ] a\n'
        '  in [b] }', str(jaxpr))

  def test_xla_computation(self):
    @functools.partial(partition.partition, num_partitions=2)
    def f(x):
      y = x * x
      z = 38. + partition.partition_put(y, 1)
      return z

    computation = api.xla_computation(f)(2.)
    self.assertIn('sharding={maximal device=0}', computation.GetHloText())
    self.assertIn('sharding={maximal device=1}', computation.GetHloText())

  def test_simple(self):
    if xb.device_count() < 2:
      self.skipTest('requires two devices')

    @functools.partial(partition.partition, num_partitions=2)
    def f(x):
      y = x * x
      z = 38. + partition.partition_put(y, 1)
      return z

    self.assertEqual(42., f(2.))
    self.assertEqual(xb.local_devices()[1], f(2.).device_buffer.device())

  def test_jit(self):
    if xb.device_count() < 2:
      self.skipTest('requires two devices')

    @functools.partial(partition.partition, num_partitions=2)
    def f(x):
      y = x * x
      z = 38. + partition.partition_put(y, 1)
      return z

    self.assertEqual(42., api.jit(f)(2.))
    self.assertEqual(xb.local_devices()[1], f(2.).device_buffer.device())

  def test_pmap(self):
    if xb.device_count() < 2:
      self.skipTest('requires two devices')

    @functools.partial(partition.partition, num_partitions=2)
    def f(x):
      y = x * x
      z = 38. + partition.partition_put(y, 1)
      return z

    with self.assertRaises(NotImplementedError):
      api.pmap(f)(onp.array([2.]))


if __name__ == '__main__':
  absltest.main()
