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

import dataclasses
import functools
from typing import Any
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.lib import jaxlib_extension_version, xla_version
from jax.experimental import donation_recovery as jax_dr
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()


@dataclasses.dataclass(frozen=True)
class _MockRawFuture:
  py_args: tuple[tuple[Any, ...], dict[str, Any]]
  recovered_by_id: dict[int, Any] | None

  def is_ready(self) -> bool:
    return True

  def get(self, leaves: list[Any]) -> list[Any] | None:
    if self.recovered_by_id is None:
      return None
    return [self.recovered_by_id.get(id(leaf), leaf) for leaf in leaves]


@dataclasses.dataclass(frozen=True)
class _CustomContainer:
  arr: Any
  val: str


jax.tree_util.register_pytree_node(
    _CustomContainer,
    lambda c: ((c.arr,), c.val),
    lambda aux, children: _CustomContainer(children[0], aux),
)


@unittest.skipIf(
    jaxlib_extension_version < 527 or xla_version < 91,
    'Requires jaxlib_extension_version >= 527 and xla_version >= 91',
)
class DonationRecoveryTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if xla_bridge.backend_pjrt_c_api_version() is not None:
      self.skipTest('Donation recovery is not supported via PJRT C API.')

  @parameterized.named_parameters(
      ('basic', 'basic'),
      ('closed_over_constants', 'closed_over_constants'),
      ('unused_donated_arg', 'unused_donated_arg'),
      ('kwargs_only', 'kwargs_only'),
  )
  def test_successful_execution_returns_none(self, case_name: str):
    x = jnp.ones((3,), dtype=jnp.float32)
    if case_name == 'basic':
      f = jax.jit(lambda x: x * 2, donate_argnames=('x',))
      call_fn = lambda: f(x)
      expected = np.ones((3,), dtype=np.float32) * 2
    elif case_name == 'closed_over_constants':
      const_arr = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)
      f = jax.jit(lambda x: x + const_arr, donate_argnames=('x',))
      call_fn = lambda: f(x)
      expected = np.array([11.0, 21.0, 31.0], dtype=np.float32)
    elif case_name == 'unused_donated_arg':
      unused = jnp.ones((3,), dtype=jnp.float32) * 5
      f = jax.jit(lambda x, unused: x * 3, donate_argnames=('x', 'unused'))
      call_fn = lambda: f(x, unused)
      expected = np.ones((3,), dtype=np.float32) * 3
    else:
      f = jax.jit(lambda *, a, b: a + b, donate_argnames=('a',))
      call_fn = lambda: f(a=x, b=20)
      expected = np.ones((3,), dtype=np.float32) * 21

    with jax_dr.DonationRecovery() as rec:
      y = call_fn()
      future = rec.future()

    y.block_until_ready()
    self.assertArraysEqual(y, expected)
    self.assertIsNone(future.get())
    self.assertTrue(x.is_deleted())

  def test_scope_reentrance_rejected(self):
    with jax_dr.DonationRecovery():
      with self.assertRaisesRegex(ValueError, 'Nested DonationRecovery scopes'):
        with jax_dr.DonationRecovery():
          pass

  def test_multiple_jit_calls_in_scope_rejected(self):
    @jax.jit(donate_argnames=('x',))
    def f(x):
      return x + 1

    x1 = jnp.ones((2, 2))
    x2 = jnp.ones((2, 2))
    with jax_dr.DonationRecovery():
      f(x1)
      with self.assertRaisesRegex(
          ValueError, 'can only wrap at most one JIT execution'
      ):
        f(x2)
    # Ensure second argument was not deleted because error was raised before
    # dispatch.
    self.assertFalse(x2.is_deleted())

  def test_future_without_execution_raises(self):
    with jax_dr.DonationRecovery() as rec:
      with self.assertRaisesRegex(RuntimeError, 'No JIT execution occurred'):
        rec.future()
    with self.assertRaisesRegex(RuntimeError, 'No JIT execution occurred'):
      rec.future()

  def test_pytree_and_static_args_preservation(self):
    @functools.partial(
        jax.jit, static_argnums=(1,), donate_argnames=('d', 'kw_tree')
    )
    def f(d, static_scale, nd_tuple, *, kw_tree, kw_nd):
      return (
          d['nested']['val'] * static_scale + nd_tuple[0],
          kw_tree['list'][0] + nd_tuple[1],
          kw_tree['list'][1] + kw_nd,
      )

    d = {'nested': {'val': jnp.ones((2,), dtype=jnp.float32)}}
    nd_tuple = (
        jnp.ones((2,), dtype=jnp.float32) * 2,
        jnp.ones((2,), dtype=jnp.float32) * 3,
    )
    kw_tree = {
        'list': [
            jnp.ones((2,), dtype=jnp.float32) * 4,
            jnp.ones((2,), dtype=jnp.float32) * 5,
        ]
    }
    kw_nd = jnp.ones((2,), dtype=jnp.float32) * 6

    with jax_dr.DonationRecovery() as rec:
      out1, out2, out3 = f(d, 5, nd_tuple, kw_tree=kw_tree, kw_nd=kw_nd)
      future = rec.future()

    # 1 * 5 + 2 = 7; 4 + 3 = 7; 5 + 6 = 11
    out1.block_until_ready()
    self.assertArraysEqual(out1, np.ones((2,), dtype=np.float32) * 7)
    self.assertArraysEqual(out2, np.ones((2,), dtype=np.float32) * 7)
    self.assertArraysEqual(out3, np.ones((2,), dtype=np.float32) * 11)
    self.assertIsNone(future.get())
    self.assertTrue(d['nested']['val'].is_deleted())
    self.assertTrue(kw_tree['list'][0].is_deleted())
    self.assertTrue(kw_tree['list'][1].is_deleted())
    self.assertFalse(nd_tuple[0].is_deleted())
    self.assertFalse(nd_tuple[1].is_deleted())
    self.assertFalse(kw_nd.is_deleted())

  def test_aot_compiled_execution(self):
    def f(tree, *, scale):
      return tree['x'] + scale

    x = jnp.ones((2, 3), dtype=jnp.float32)
    scale = jnp.ones((2, 3), dtype=jnp.float32) * 10.0
    compiled = (
        jax.jit(f, donate_argnames=('tree',))
        .lower({'x': x}, scale=scale)
        .compile()
    )

    with jax_dr.DonationRecovery() as rec:
      out = compiled({'x': x}, scale=scale)
      future = rec.future()

    self.assertEqual(
        future._raw_future.py_args, (({'x': x},), {'scale': scale})
    )
    out.block_until_ready()
    self.assertArraysEqual(out, np.ones((2, 3), dtype=np.float32) * 11.0)
    self.assertIsNone(future.get())
    self.assertTrue(x.is_deleted())
    self.assertFalse(scale.is_deleted())

  @parameterized.named_parameters(
      ('nested_dict_and_kwargs', 'nested_dict'),
      ('custom_pytree_node', 'custom_pytree'),
      ('prng_key_array', 'prng_key'),
  )
  def test_recovery_future_pytree_reconstruction(self, case_name: str):
    if case_name == 'nested_dict':
      orig_a = jnp.ones((2,), dtype=jnp.float32)
      orig_b = jnp.ones((2,), dtype=jnp.float32) * 2
      orig_c = jnp.ones((2,), dtype=jnp.float32) * 3
      rec_a = jnp.ones((2,), dtype=jnp.float32) * 100
      rec_c = jnp.ones((2,), dtype=jnp.float32) * 200
      py_args = (({'k': orig_a}, 42, orig_b), {'kw': {'nested': orig_c}})
      recovered_by_id = {id(orig_a): rec_a, id(orig_c): rec_c}
    elif case_name == 'custom_pytree':
      orig_arr = jnp.ones((3,), dtype=jnp.float32)
      rec_arr = jnp.ones((3,), dtype=jnp.float32) * 50
      py_args = ((_CustomContainer(orig_arr, 'aux_info'),), {})
      recovered_by_id = {id(orig_arr): rec_arr}
    else:
      orig_key = jax.random.key(7)
      rec_key = jax.random.key(42)
      py_args = (({'rng': orig_key},), {})
      recovered_by_id = {
          id(jax.random.key_data(orig_key)): jax.random.key_data(rec_key)
      }

    future = jax_dr.RecoveryFuture(_MockRawFuture(py_args, recovered_by_id))
    self.assertTrue(future.is_ready())
    rec_args, rec_kwargs = future.get()

    if case_name == 'nested_dict':
      self.assertArraysEqual(
          rec_args[0]['k'], np.ones((2,), dtype=np.float32) * 100
      )
      self.assertEqual(rec_args[1], 42)
      self.assertArraysEqual(rec_args[2], np.ones((2,), dtype=np.float32) * 2)
      self.assertArraysEqual(
          rec_kwargs['kw']['nested'], np.ones((2,), dtype=np.float32) * 200
      )
    elif case_name == 'custom_pytree':
      self.assertIsInstance(rec_args[0], _CustomContainer)
      self.assertEqual(rec_args[0].val, 'aux_info')
      self.assertArraysEqual(
          rec_args[0].arr, np.ones((3,), dtype=np.float32) * 50
      )
      self.assertEmpty(rec_kwargs)
    else:
      self.assertTrue(
          jax.dtypes.issubdtype(rec_args[0]['rng'].dtype, jax.dtypes.prng_key)
      )
      self.assertArraysEqual(
          jax.random.key_data(rec_args[0]['rng']),
          jax.random.key_data(jax.random.key(42)),
      )
      self.assertEmpty(rec_kwargs)

  def test_unrecoverable_failure_raises(self):
    def fail_cb(x):
      raise ValueError('Unrecoverable hardware error')

    f = jax.jit(
        lambda x: jax.pure_callback(fail_cb, x, x), donate_argnames=('x',)
    )
    x = jnp.ones((3,), dtype=jnp.float32)
    with jax_dr.DonationRecovery() as rec:
      _ = f(x)
      future = rec.future()
    with self.assertRaisesRegex(
        jax.errors.JaxRuntimeError, 'Unrecoverable hardware error'
    ):
      future.get()
    self.assertTrue(x.is_deleted())


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
