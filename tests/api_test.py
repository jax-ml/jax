# Copyright 2018 The JAX Authors.
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

import collections
import collections.abc
import concurrent.futures
from contextlib import contextmanager
import copy
import dataclasses
import enum
import functools
from functools import partial
import gc
import importlib
import inspect
import itertools as it
import operator
import operator as op
import os
import re
import subprocess
import sys
import traceback
import types
from typing import NamedTuple
import unittest
import weakref

from absl import logging
from absl.testing import absltest, parameterized
import jax
from jax import device_put, float0, grad, hessian, jacfwd, jacrev, jit
from jax import lax
from jax import tree_util
from jax._src import api, api_util, dtypes, lib
from jax._src import array
from jax._src import config
from jax._src import core
from jax._src import linear_util as lu
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src import debugging
from jax._src import pjit as pjit_lib
from jax._src import sharding_impls
from jax._src.ad_checkpoint import saved_residuals
from jax._src.interpreters import ad as ad_internal
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.compilation_cache import is_persistent_cache_enabled
from jax._src.lib import _jax
import jax._src.util as jax_util
from jax.ad_checkpoint import checkpoint_name, checkpoint as new_checkpoint
from jax.errors import (UnexpectedTracerError, TracerIntegerConversionError,
                        ConcretizationTypeError, TracerBoolConversionError)
from jax.experimental import pjit
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import xla
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

config.parse_flags_with_absl()


def _check_instance(self, x):
  self.assertIsInstance(x, array.ArrayImpl)


class JitTest(jtu.BufferDonationTestCase):
  """Shared tests between the Python and the C++ jax,jit implementations.

  Because the Python implementation supports more features, we need to have the
  Python tests that extend the C++ tests (and not the other way around).
  """

  def test_jit_repr(self):
    def my_function():
      return
    jitted = jit(my_function)
    self.assertEqual(repr(jitted), f"<PjitFunction of {repr(my_function)}>")

  def test_fun_name(self):
    def my_function():
      return

    with self.subTest("function"):
      jitted = jit(my_function)
      self.assertEqual(
          jitted.__getstate__()["function_name"], my_function.__name__
      )
    with self.subTest("default_partial"):
      my_partial = partial(my_function)
      jitted = jit(my_partial)
      self.assertEqual(
          jitted.__getstate__()["function_name"], my_function.__name__
      )
    with self.subTest("nested_default_partial"):
      my_partial = partial(partial(my_function))
      jitted = jit(my_partial)
      self.assertEqual(
          jitted.__getstate__()["function_name"], my_function.__name__
      )
    with self.subTest("named_partial"):
      my_partial = partial(my_function)
      my_partial.__name__ = "my_partial"
      jitted = jit(my_partial)
      self.assertEqual(
          jitted.__getstate__()["function_name"], my_partial.__name__
      )
    with self.subTest("lambda"):
      jitted = jit(lambda: my_function())
      self.assertEqual(jitted.__getstate__()["function_name"], "<lambda>")

  def test_jit_repr_errors(self):
    class Callable:
      def __call__(self): pass
      def __repr__(self):
        raise ValueError("invalid repr")

    # repr succeeds when underlying function repr fails.
    jitted = jit(Callable())
    self.assertEqual(repr(jitted), "<PjitFunction>")

    # repr succeeds when object is malformed.
    del jitted.__wrapped__
    self.assertEqual(repr(jitted), "<PjitFunction>")

  def test_jit_of_noncallable(self):
    self.assertRaisesRegex(TypeError, "Expected a callable value.*",
                           lambda: jit(3))

  def test_jit_of_generator(self):

    def gen(x):
      yield x

    self.assertRaisesRegex(TypeError,
                           "Expected a function, got a generator function.*",
                           lambda: jit(gen))

  @parameterized.parameters([
      # Integer support
      (1, 2, 3, 4, 5),
      # Numpy array support
      (
          np.asarray(1, np.int32),
          np.asarray(2, np.int32),
          np.asarray(3, np.int32),
          np.asarray(4, np.int32),
          np.asarray(5, np.int32),
      ),
  ])
  def test_jit_static_args(self, one, two, three, four, five):
    side = []

    def f(x, y, z, flag=False, flag2=False):
      del flag2  # unused
      assert flag
      side.append(None)
      return 100 * x + 10 * y + z

    f1 = jit(f, static_argnums=(3, 4))
    assert f1(one, two, three, True, False) == 123
    assert len(side) == 1
    assert f1(one, two, three, True, False) == 123
    assert len(side) == 1  # Obvious cache hit.
    assert f1(two, one, three, True, False) == 213
    assert len(side) == 1  # Should cache hit because same signature.
    assert f1(two, one, three, True, True) == 213
    assert len(side) == 2

    side[:] = []
    f2 = jit(f, static_argnums=(0, 2, 3, 4))
    assert f2(1, 2, 3, True, False) == 123
    assert len(side) == 1
    assert f2(1, 3, 3, True, False) == 133
    assert len(side) == 1
    assert f2(2, 2, 3, True, False) == 223
    assert len(side) == 2
    assert f2(2, 4, 3, True, False) == 243
    assert len(side) == 2
    assert f2(2, 4, 3, True, True) == 243
    assert len(side) == 3
    assert f2(2, 5, 3, True, True) == 253
    assert len(side) == 3

  def test_static_args_equality(self):
    class A():

      def __hash__(self):
        return 1

      def __eq__(self, other):
        return isinstance(other, A)

    side = []
    def f(x, static_arg):
      del static_arg
      side.append(None)
      return x * 100

    f1 = jit(f, static_argnums=(1,))

    self.assertEqual(f1(1, A()), 100)
    self.assertLen(side, 1)
    self.assertEqual(f1(1, A()), 100)
    self.assertLen(side, 1)
    f1_cpp = getattr(f1, "_cpp_jitted_f", f1)
    self.assertEqual(f1_cpp._cache_size(), 1)

  @parameterized.parameters([
      (1, 2, 3),
      (
          np.asarray(1, np.int32),
          np.asarray(2, np.int32),
          np.asarray(3, np.int32),
      ),
  ])
  def test_jit_kwargs(self, one, two, three):
    side = []
    # For the CPP jit, we need to clear the cache to prevent cache hits between
    # parameterized tests.
    if hasattr(jit, "cache_clear"):
      jit.cache_clear()

    def f(x, y, z):
      side.append(None)
      return 100 * x + 10 * y + z.astype(y.dtype)

    f = jit(f)
    assert f(one, two, three) == 123
    assert len(side) == 1
    assert f(one, two, three) == 123
    assert len(side) == 1

    assert f(one, two, z=three) == 123
    assert len(side) == 2  # actually recompiles from kwarg
    assert f(one, two, z=three) == 123
    assert len(side) == 2  # but should still cache

    f(one, two, z=np.zeros(3))  # doesn't crash
    if config.enable_x64.value:
      # In the above call, three is of a new type (int64), thus it should
      # trigger a new compilation.
      assert len(side) == 3

  def test_jit_device(self):
    device = jax.devices()[-1]
    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      x = jit(lambda x: x, device=device)(3.)
    _check_instance(self, x)
    self.assertEqual(x.devices(), {device})

  @parameterized.named_parameters(
      ('jit', jax.jit),
      ('pjit', pjit.pjit),
  )
  @jtu.skip_on_devices("cpu")
  def test_jit_default_device(self, module):
    if jax.device_count() == 1:
      raise unittest.SkipTest("Test requires multiple devices")

    system_default_devices = jnp.add(1, 1).devices()
    self.assertLen(system_default_devices, 1)
    system_default_device = list(system_default_devices)[0]
    test_device = jax.devices()[-1]
    self.assertNotEqual(system_default_device, test_device)

    f = module(lambda x: x + 1)
    self.assertEqual(f(1).devices(), system_default_devices)

    with jax.default_device(test_device):
      self.assertEqual(jnp.add(1, 1).devices(), {test_device})
      self.assertEqual(f(1).devices(), {test_device})

    self.assertEqual(jnp.add(1, 1).devices(), system_default_devices)
    self.assertEqual(f(1).devices(), system_default_devices)

    with jax.default_device(test_device):
      # Explicit `device` or `backend` argument to jit overrides default_device
      with jtu.ignore_warning(category=DeprecationWarning,
                              message="backend and device argument"):
        self.assertEqual(
            module(f, device=system_default_device)(1).devices(),
            system_default_devices)
        out = module(f, backend="cpu")(1)
      self.assertEqual(next(iter(out.devices())).platform, "cpu")

      # Sticky input device overrides default_device
      sticky = jax.device_put(1, system_default_device)
      self.assertEqual(jnp.add(sticky, 1).devices(), system_default_devices)
      self.assertEqual(f(sticky).devices(), system_default_devices)

      # Test nested default_devices
      with jax.default_device(system_default_device):
        self.assertEqual(f(1).devices(), system_default_devices)
      self.assertEqual(f(1).devices(), {test_device})

    # Test a few more non-default_device calls for good luck
    self.assertEqual(jnp.add(1, 1).devices(), system_default_devices)
    self.assertEqual(f(sticky).devices(), system_default_devices)
    self.assertEqual(f(1).devices(), system_default_devices)

  def test_jit_default_platform(self):
    with jax.default_device("cpu"):
      result = jax.jit(lambda x: x + 1)(1)
    self.assertEqual(result.device.platform, "cpu")
    self.assertEqual(result.device, jax.local_devices(backend="cpu")[0])

    result = jax.jit(lambda x: x + 1)(1)
    self.assertEqual(result.device.platform, jax.default_backend())
    self.assertEqual(result.device, jax.local_devices()[0])

  def test_complex_support(self):
    self.assertEqual(jit(lambda x: x + 1)(1 + 1j), 2 + 1j)

  @parameterized.parameters("static_argnums", "donate_argnums")
  def test_jit_argnums_overflow_error(self, argnum_type: str):
    def f(a, b, c):
      ...

    def g(a, /, b, *, c):
      ...

    def h(a, *args):
      ...

    def i():
      ...

    # Simplest cases
    jit(f, **{argnum_type: (0, 1)})
    jit(g, **{argnum_type: (0, 1)})
    jit(f, **{argnum_type: (0, 1, -3)})

    # Out of bounds without *args
    with self.assertRaises(ValueError):
      jit(f, **{argnum_type: (0, 1, 3)})

    with self.assertRaises(ValueError):
      jit(f, **{argnum_type: (0, 1, -4)})

    with self.assertRaises(ValueError):
      jit(g, **{argnum_type: (0, 1, 3)})

    with self.assertRaises(ValueError):
      jit(g, **{argnum_type: (0, 1, -3)})

    # Out of bounds with *args
    jit(h, **{argnum_type: (0, 999)})
    jit(h, **{argnum_type: (0, -999)})

    # No positional arguments
    jit(i, static_argnums=())
    jit(i)

  @parameterized.parameters("static_argnames", "donate_argnames")
  def test_jit_argnames_validation(self, argnum_type: str):
    def f(a, b, c):
      ...

    def g(a, b, **kwargs):
      ...

    def h(a, /, b, c, *args, **kwargs):
      ...

    # Simplest case
    jit(f, **{argnum_type: ("b", "c")})

    # Undefined arg without **kwargs
    with self.assertRaises(ValueError):
      jit(f, **{argnum_type: ("b", "c", "not_defined")})

    # Undefined arg with **kwargs
    jit(g, **{argnum_type: ("a", "b", "not_defined")})

    jit(h, **{argnum_type: ("b", "c")})
    jit(h, **{argnum_type: ("b", "c", "not_defined")})

    # Positional only
    with self.assertRaises(ValueError):
      jit(h, **{argnum_type: ("a", "c")})

    # Var positional
    with self.assertRaises(ValueError):
      jit(h, **{argnum_type: ("args", "c")})

  def test_jit_with_many_args_works(self):

    @jit
    def f(args_list):
      return sum(args_list)

    self.assertEqual(f(list(range(500))), sum(range(500)))

  # Jit and Donate arguments

  def test_donate_argnames_signature_fail(self):
    inp = np.arange(4)
    with self.assertRaisesRegex(
        ValueError,
        "Getting the signature of function.*failed. Pass donate_argnums "
        "instead of donate_argnames."):
      jax.jit(np.dot, donate_argnames='a')(inp, inp)

  @parameterized.named_parameters(
      ("argnums", "donate_argnums", (0, 1)),
      ("argnames", "donate_argnames", ('x', 'y')),
  )
  def test_jit_donate_warning_raised(self, argnum_type, argnum_val):
    x = jnp.array([1.0, 2.0], jnp.float32)
    y = jnp.array([1, 2], jnp.int32)
    f = jit(lambda x, y: x.sum() + jnp.float32(y.sum()),
                 **{argnum_type: argnum_val})
    with self.assertWarnsRegex(UserWarning, "Some donated buffers were not usable"):
      f(x, y)

  @parameterized.named_parameters(
      ("argnums", "donate_argnums", 0),
      ("argnames", "donate_argnames", 'x'),
  )
  @jtu.device_supports_buffer_donation()
  def test_jit_donate_invalidates_input(self, argnum_type, argnum_val):
    # We can't just use `lambda x: x` because JAX simplifies this away to an
    # empty XLA computation.
    move = jit(lambda x: x + x - x, **{argnum_type: argnum_val})
    x = jnp.ones([])
    y = move(x)
    self.assertDeleted(x)
    self.assertEqual(y, 1.)

  @parameterized.named_parameters(
      ("donate_argnums", "donate_argnums", (2, 3)),
      ("donate_argnames", "donate_argnames", ('c', 'd')),
  )
  @jtu.device_supports_buffer_donation()
  def test_jit_donate_static_argnums(self, argnum_type, argnum_val):
    jit_fun = jit(
        lambda a, b, c, d: ((a + b + c), (a + b + d)),
        static_argnums=(0, 1),
        **{argnum_type: argnum_val})

    c = jax.device_put(jnp.array([2., 2.]))
    d = jax.device_put(jnp.array([1., 1., 1., 1.]))
    e, f = jit_fun(1, 2, c, d)
    np.testing.assert_allclose(e, jnp.array([5., 5.]))
    np.testing.assert_allclose(f, jnp.array([4., 4., 4., 4.]))
    self.assertDeleted(c)
    self.assertDeleted(d)

  @jtu.device_supports_buffer_donation()
  def test_jit_donate_argnames_kwargs_static_argnums(self):
    jit_fun = jit(
        lambda a, b, c, d, e: ((a + b + c), (a + b + d), (a + b + e)),
        static_argnums=(0, 1),
        donate_argnames=('d', 'e'))

    c = jax.device_put(jnp.array([2., 2.]))
    d = jax.device_put(jnp.array([1., 1., 1., 1.]))
    e = jax.device_put(jnp.array([3., 3., 3., 3.]))
    f, g, h = jit_fun(1, 2, c, d=d, e=e)
    np.testing.assert_allclose(f, jnp.array([5., 5.]))
    np.testing.assert_allclose(g, jnp.array([4., 4., 4., 4.]))
    np.testing.assert_allclose(h, jnp.array([6., 6., 6., 6.]))
    self.assertNotDeleted(c)
    self.assertDeleted(d)
    self.assertDeleted(e)

  def test_device_put_aliasing(self):
    arr = jax.device_put(np.arange(8), jax.devices()[0])
    out = jax.device_put(arr, may_alias=True, donate=False)
    self.assertEqual(id(arr), id(out))

    out = jax.device_put(arr, may_alias=False, donate=False)
    self.assertNotEqual(id(arr), id(out))

    with self.assertRaisesRegex(
        ValueError, "may_alias and donate cannot be True at the same time."):
      jax.device_put(arr, may_alias=True, donate=True)

    out = jax.device_put(arr,
                         jax.sharding.SingleDeviceSharding(jax.devices()[0]),
                         may_alias=True, donate=False)
    self.assertEqual(id(arr), id(out))

    out = jax.device_put(arr,
                         jax.sharding.SingleDeviceSharding(jax.devices()[0]),
                         may_alias=False, donate=False)
    self.assertNotEqual(id(arr), id(out))

  def test_device_put_aliasing_with_diff_compatible_sharding(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices")

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:2]).reshape((2, 1)), ("x", "y")
    )
    x = jax.device_put(
        np.arange(16).reshape((4, 4)),
        jax.NamedSharding(mesh, P("x", None)),
    )
    expanded_mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:2]).reshape((1, 2, 1)), ("replicas", "x", "y")
    )
    dst_sharding = jax.NamedSharding(expanded_mesh, P("x", None))
    # No transfer should happen because the array is aliased to compatible
    # sharding that only has a mesh with an additional dimension of size 1.
    with jax.transfer_guard_device_to_device("disallow_explicit"):
      res = jax.device_put(x, dst_sharding, may_alias=True)
    self.assertEqual(dst_sharding, res.sharding)

  @parameterized.named_parameters(
      ("argnums", "donate_argnums", 0),
      ("argnames", "donate_argnames", 'x'),
  )
  @jtu.device_supports_buffer_donation()
  def test_jit_donate_weak_type(self, argnum_type, argnum_val):
    # input has weak-type, output does not have weak-type
    move = jit(lambda x: x.astype(int), **{argnum_type: argnum_val})
    x = jnp.broadcast_to(2, (3,))
    move(x)
    self.assertDeleted(x)

  @parameterized.named_parameters(
      ("argnums", "donate_argnums", (0,)),
      ("argnames", "donate_argnames", ('array',)),
  )
  def test_jnp_array_copy(self, argnum_type, argnum_val):
    # https://github.com/jax-ml/jax/issues/3412

    @partial(jit, **{argnum_type: argnum_val})
    def _test(array):
      return array.at[0].set(77)

    x = jnp.asarray([0, 1])
    x_copy = jnp.array(x, copy=True)
    with jtu.ignore_warning():
      _test(x)  # donation

    # Gives: RuntimeError: Invalid argument: CopyToHostAsync() called on invalid buffer.
    print(x_copy)  # doesn't crash

  @jtu.device_supports_buffer_donation()
  def test_specify_donate_argnums_and_argnames(self):
    @partial(jax.jit, donate_argnums=0, donate_argnames=('inp2', 'inp3'))
    def f(inp1, inp2, inp3):
      return inp1 * 2, inp2 * 2, inp3 * 2

    x = jnp.ones((2, 5)) * 4
    y = jnp.ones((2, 5)) * 2
    z = jnp.ones((2, 5))

    f(x, inp2=y, inp3=z)
    self.assertDeleted(x)
    self.assertDeleted(y)
    self.assertDeleted(z)

  def test_resolve_argnums_signature_fail(self):
    api_util.resolve_argnums(int, None, None, None, None, None)  # doesn't crash

  @jtu.device_supports_buffer_donation()
  def test_donate_argnames_with_args(self):
    @partial(jax.jit, donate_argnames='inp1')
    def f(inp1):
      return inp1 * 2

    x = jax.device_put(jnp.ones((2, 5)) * 4, jax.devices()[0])
    f(x)
    self.assertDeleted(x)

  @jtu.device_supports_buffer_donation()
  def test_donate_argnums_with_kwargs(self):
    @partial(jax.jit, donate_argnums=0)
    def f(inp1):
      return inp1 * 2

    x = jax.device_put(jnp.ones((2, 5)) * 4, jax.devices()[0])
    f(inp1=x)
    self.assertDeleted(x)

  def test_donate_args_info_aot(self):
    def fn(x, y):
      return jax.tree.map(lambda i: i * 2, x), y * 2

    x = jax.device_put({"A": np.array(1.0), "B": np.array(2.0)},
                       jax.devices()[0])
    y = jax.device_put(np.array(3.0), jax.devices()[0])

    f = jax.jit(fn, donate_argnums=1)
    lowered = f.lower(x, y)
    args_info = lowered.args_info[0]
    # x is not donated.
    self.assertFalse(args_info[0]['A'].donated)
    self.assertFalse(args_info[0]['B'].donated)
    # y is donated.
    self.assertTrue(args_info[1].donated)

    g = jax.jit(fn, donate_argnums=0)
    lowered = g.lower(x, y)
    args_info = lowered.args_info[0]
    # x is donated.
    self.assertTrue(args_info[0]['A'].donated)
    self.assertTrue(args_info[0]['B'].donated)
    # y is not donated.
    self.assertFalse(args_info[1].donated)

  def test_double_donation(self):
    def add(x, y):
      return x + y

    f = jax.jit(add, donate_argnums=(0,))
    x = jnp.zeros((10,), jnp.float32)

    with self.assertRaises(RuntimeError):
      result = f(x, x)
      result.block_until_ready()

  @parameterized.named_parameters(
      ('argnames', {'donate_argnames': ('z', 'y')}),
      ('argnums', {'donate_argnums': (0, 1)})
  )
  def test_dict_donation(self, jit_kwargs):
    @partial(jax.jit, **jit_kwargs)
    def f(z, y, x):
      return z, y, x

    z = {'c': 3.}
    y = {'b': 2.}
    x = {'a': 1.}

    _, kwargs_info = f.lower(z=z, y=y, x=x).args_info
    self.assertTrue(kwargs_info['z']['c'].donated)
    self.assertTrue(kwargs_info['y']['b'].donated)
    self.assertFalse(kwargs_info['x']['a'].donated)

  @parameterized.named_parameters(
      ('argnames', {'donate_argnames': ('z', 'y')}),
      ('argnums', {'donate_argnums': (0, 1)})
  )
  def test_dict_donation_args_kwargs(self, jit_kwargs):
    @partial(jax.jit, **jit_kwargs)
    def f(z, y, x):
      return z, y, x

    z = {'c': 3.}
    y = {'b': 2.}
    x = {'a': 1.}

    args_info, kwargs_info = f.lower(z, y=y, x=x).args_info
    self.assertTrue(args_info[0]['c'].donated)
    self.assertTrue(kwargs_info['y']['b'].donated)
    self.assertFalse(kwargs_info['x']['a'].donated)

  def test_intersecting_static_and_donate_argnames(self):
    with self.assertRaisesRegex(
        ValueError, "static_argnames and donate_argnames cannot intersect"):
      jax.jit(lambda x: x, static_argnames='x', donate_argnames='x')

  def test_jit_global_cache(self):
    def f(x):
      assert python_should_be_executing
      return x

    python_should_be_executing = True
    jit(f)(2)
    python_should_be_executing = False
    jit(f)(3)

  @jtu.thread_unsafe_test()  # GC effects aren't predictable with threads
  def test_jit_cache_clear(self):
    @jit
    def f(x, y):
      return x + y

    client = jax.devices()[0].client
    gc.collect()
    num_live_initial = len(client.live_executables())
    f(1, 2).block_until_ready()
    gc.collect()
    num_live = len(client.live_executables())
    self.assertEqual(num_live_initial + 1, num_live)
    f.clear_cache()
    gc.collect()
    num_live = len(client.live_executables())
    self.assertEqual(num_live_initial, num_live)

  def test_jit_shallow_copy(self):
    def f(x):
      return copy.copy(x)
    jit(f)(1)

  def test_jit_deep_copy(self):
    def f(x):
      return copy.deepcopy(x)
    jit(f)(1)

  def test_disable_jit(self):
    effects = []

    @jit
    def f(x):
      effects.append(1)
      return x

    with api.disable_jit():
      f(2)
      f(2)
    assert len(effects) == 2

    f(2)
    f(2)
    assert len(effects) == 3

  def test_static_argnum_on_method(self):

    class A:

      @functools.partial(jit, static_argnums=(0,))
      def my_func_jit(self, x):
        return x+2

    A().my_func_jit(3)

  def test_static_argnum_on_static_method_is_not_supported(self):
    with self.assertRaisesRegex(TypeError, "Expected a callable value"):

      class A:

        @functools.partial(jit, static_argnums=(0,))
        @classmethod
        def my_classmethod_jit(cls, x):
          return x+2

  def test_staticmethod_is_not_supported(self):
    with self.assertRaisesRegex(TypeError,
                                "staticmethod arguments are not supported"):

      class A:

        @functools.partial(jit)
        @staticmethod
        def my_staticmethod_jit(x):
          return x + 2

  def test_concurrent_jit(self):
    @jit
    def f(x):
      return x + x - 3.

    xs = [self.rng().randn(i) for i in range(10)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(partial(f, x)) for x in xs]
      ys = [f.result() for f in futures]
    for x, y in zip(xs, ys):
      self.assertAllClose(x * 2 - 3., y)

  def test_trivial_computations(self):
    x = jnp.array([1, 2, 3])
    y = jit(lambda x: x)(x)
    self.assertNotEqual(x.unsafe_buffer_pointer(), y.unsafe_buffer_pointer())

    z1, z2 = jit(lambda x: (x, x))(x)
    self.assertNotEqual(z1.unsafe_buffer_pointer(), z2.unsafe_buffer_pointer())

    x1, x2 = jnp.array([1, 2]), jnp.array([2, 3])
    z1, z2, z3 = jit(lambda x, y: (y, 1, x))(x1, x2)
    self.assertNotEqual(z1.unsafe_buffer_pointer(), x2.unsafe_buffer_pointer())
    self.assertNotEqual(z3.unsafe_buffer_pointer(), x1.unsafe_buffer_pointer())
    self.assertEqual(z2, 1)

  def test_print_token_buffer_error(self):
    token = jax.lax.create_token()
    with self.assertRaisesRegex(
        RuntimeError, "Cannot convert a token-shape buffer to a numpy array."
    ):
      token._buf._value

  def test_trivial_computations_with_tokens(self):
    @jit
    def noop(arr, token):
      return arr, token

    arr = jnp.ones(10)
    token = jax.lax.create_token()
    _, out_token = noop(arr, token)

    self.assertIsInstance(token, core.Token)
    self.assertIsInstance(out_token, core.Token)
    # Different token objects.
    self.assertIsNot(token, out_token)

  def test_jit_bad_input(self):
    def f(x):
      return x

    err_str = ("Error interpreting argument to .* as an abstract array. The problematic "
               "value is of type .* and was passed to the function at path x.")
    with self.assertRaisesRegex(TypeError, err_str):
      jit(f)("foo")

    # Jax type objects aren't valid data arguments.
    with self.assertRaisesRegex(TypeError, err_str):
      jit(f)(jnp.int32)

  def test_jit_masked_array(self):
    x = np.ma.array([1, 2, 3], mask=[True, False, True])
    f = jit(lambda x: x)
    with self.assertRaisesRegex(ValueError, "numpy masked arrays are not supported"):
      f(x)

  def test_jit_on_all_devices(self):
    # Verifies we can run the same computation on every device present, even
    # if they are, for example, different models of GPU.
    data = self.rng().rand(1000).astype(np.float32)
    f = jit(jnp.negative)
    for device in jax.local_devices():
      x = device_put(data, device=device)
      np.testing.assert_array_equal(-data, f(x))

  def test_jit_nested_donate_ignored(self):
    jit_fun = jit(lambda x: jit(lambda y: y**2, donate_argnums=0)(x))
    a = jax.device_put(jnp.array(1))

    # NOTE(mattjj): stopped raising error here and instead just ignored
    # with self.assertRaisesRegex(ValueError, "nested.*not supported"):
    #   jit_fun(a)

    jit_fun(a)  # doesn't crash

  def test_jit_reference_dropping(self):
    x = jnp.ones(10)
    f = (lambda x: lambda: x)(x)  # reference to x in f's closure
    g = jit(f)
    x = weakref.ref(x)      # no more strong ref to x in this scope
    assert x() is not None  # x is still around
    f()                     # f runs
    g()                     # g runs
    g()                     # g runs a second time
    del f                   # delete the raw callable
    assert x() is not None  # x is still around
    g()                     # g still runs
    del g                   # no more references to x
    assert x() is None      # x is gone

  def test_jit_of_nonweakreferenceable_function(self):
    class CallableWithSlots:
      __slots__ = []
      def __call__(self, x):
        return x + 1

    c = CallableWithSlots()
    with self.assertRaisesRegex(TypeError, "cannot create weak reference.*"):
      weakref.ref(c)
    # Building a jit object does not crash.
    f = jit(c)
    with self.assertRaisesRegex(TypeError, "cannot create weak reference.*"):
      # Calling the jit object will fail, but not because of the C++ JIT. The
      # Python-level jit cache requires weak reference support.
      f(3)

  def test_jit_raises_on_first_invocation_on_non_hashable_static_argnum(self):
    f = lambda x, y: x + 3
    jitted_f = jit(f, static_argnums=(1,))

    msg = "Non-hashable static arguments are not supported"
    with self.assertRaisesRegex(ValueError, msg):
      jitted_f(1, np.asarray(1))

  def test_cpp_jit_raises_on_non_hashable_static_argnum(self):
    f = lambda x, y: x + 3
    jitted_f = jit(f, static_argnums=[1])

    jitted_f(1, 1)

    msg = "Non-hashable static arguments are not supported"

    with self.assertRaisesRegex(ValueError, msg):
      jitted_f(1, np.asarray(1))

    class HashableWithoutEq:

      def __hash__(self):
        return 1

      def __eq__(self, other):
        raise NotImplementedError(
            "A Python error is as is, without stack trace")

    with self.assertRaisesRegex(
        ValueError,
        re.escape("static arguments should be comparable using __eq__")):
      jitted_f(1, HashableWithoutEq())
      # __eq__ would only be called if we might have a cache hit. Call the
      # function a second time with exactly the same arguments to make sure that
      # we could.
      jitted_f(1, HashableWithoutEq())

  def test_cpp_jit_raises_other_exceptions_when_hashing_fails(self):
    class A:
      def __hash__(self):
        raise ValueError

    f = jax.jit(lambda x: x + 1, static_argnums=(0,))
    a = A()
    with self.assertRaisesRegex(ValueError, '^$'):  # no extra message
      f(a)

  def test_cpp_jitted_function_returns_PyBuffer(self):
    jitted_f = jit(lambda a: a + 1)
    jitted_f(1)
    out = jitted_f(2)
    self.assertIsInstance(out.sharding, jax.sharding.SingleDeviceSharding)
    self.assertIsInstance(out, array.ArrayImpl)

  @parameterized.named_parameters(
      ('jit', jax.jit),
      ('pjit', pjit.pjit)
  )
  @jtu.skip_on_devices("cpu")
  def test_explicit_backend(self, module):
    f = lambda x: x + 1
    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      jitted_f = module(f, backend=jtu.device_under_test())
      jitted_f_cpu = module(f, backend="cpu")

    result = jitted_f(1.)
    result_cpu = jitted_f_cpu(1.)
    self.assertEqual(list(result.devices())[0].platform, jtu.device_under_test())
    self.assertEqual(list(result_cpu.devices())[0].platform, "cpu")

  @parameterized.named_parameters(
      ('jit', jax.jit),
      ('pjit', pjit.pjit)
  )
  @jtu.skip_on_devices("cpu")
  def test_device_to_device_copy_between_backends(self, module):
    # b/186624243
    f = lambda x: x + 1
    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      jitted_f = module(f, backend=jtu.device_under_test())
      jitted_f_cpu = module(f, backend="cpu")

    x = np.arange(30).reshape(1, 10, 3)
    result = jitted_f(x)
    result_cpu = jitted_f_cpu(result)
    result_2 = jitted_f(result_cpu)
    result_cpu_2 = jitted_f_cpu(result_2)
    self.assertAllClose(result_2, x + 3)
    self.assertAllClose(result_cpu_2, x + 4)

  @jtu.skip_on_devices("cpu")
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def test_mismatched_nested_backends(self):
    @partial(jax.jit, backend=jtu.device_under_test())
    def f(x):
      return jax.jit(lambda x: x + 1, backend="cpu")(x)

    msg = 'Received incompatible devices for jitted computation'
    with self.assertRaisesRegex(ValueError, msg):
      f(1.)

  @jax.legacy_prng_key('allow')
  def test_omnistaging(self):
    # See https://github.com/jax-ml/jax/issues/5206

    # TODO(frostig): remove `wrap` once we always enable_custom_prng
    def wrap(arr):
      arr = np.array(arr, dtype=np.uint32)
      if config.enable_custom_prng.value:
        return jax.random.wrap_key_data(arr)
      else:
        return arr

    key_list = [None]

    def init():
      key, subkey = jax.random.split(key_list[0])
      key_list[0] = key
      return jax.random.normal(subkey, ())

    key_list[0] = wrap([2384771982, 3928867769])
    init()
    jit(init)()
    self.assertIsInstance(key_list[0], core.Tracer)
    del key_list[0]

  def test_jit_wrapped_attributes(self):
    def f(x: int) -> int:
      """docstring of f."""
      return x + 1
    f.some_value = 4
    jf = jit(f)
    for attr in ["doc", "name", "module", "qualname", "annotations"]:
      self.assertEqual(
        {attr: getattr(f, f"__{attr}__")},
        {attr: getattr(jf, f"__{attr}__")})
    self.assertEqual(f.some_value, jf.some_value)

  def test_jit_python_builtin(self):
    x = jnp.array([1, 2])
    expected = x + 1
    jit_add = jit(operator.add, static_argnums=(1,))
    actual = jit_add(x, 1)
    self.assertArraysEqual(expected, actual)

  def test_infer_argnums_and_argnames(self):
    def f(x, y=1):
      pass

    sig = inspect.signature(f)

    argnums, argnames = api_util.infer_argnums_and_argnames(
        sig, argnums=None, argnames=None)
    assert argnums == ()
    assert argnames == ()

    argnums, argnames = api_util.infer_argnums_and_argnames(
        sig, argnums=0, argnames=None)
    assert argnums == (0,)
    assert argnames == ('x',)

    argnums, argnames = api_util.infer_argnums_and_argnames(
        sig, argnums=None, argnames='y')
    assert argnums == (1,)
    assert argnames == ('y',)

    argnums, argnames = api_util.infer_argnums_and_argnames(
        sig, argnums=0, argnames='y')  # no validation
    assert argnums == (0,)
    assert argnames == ('y',)

    def g(x, y, *args):
      pass

    sig = inspect.signature(g)

    argnums, argnames = api_util.infer_argnums_and_argnames(
        sig, argnums=(1, 2), argnames=None)
    assert argnums == (1, 2)
    assert argnames == ('y',)

    def h(x, y, **kwargs):
      pass

    sig = inspect.signature(h)

    argnums, argnames = api_util.infer_argnums_and_argnames(
        sig, argnums=None, argnames=('foo', 'bar'))
    assert argnums == ()
    assert argnames == ('foo', 'bar')

  def test_jit_with_static_argnames(self):

    def f(x):
      assert x == 'foo'
      return 1

    f_nums = jit(f, static_argnums=0)
    assert f_nums('foo') == 1
    assert f_nums(x='foo') == 1

    f_names = jit(f, static_argnames='x')
    assert f_names('foo') == 1
    assert f_names(x='foo') == 1

  def test_new_static_argnum_on_keyword_arguments(self):
    f = jit(lambda x: x, static_argnums=0)
    y = f(x=4)
    assert y == 4

  def test_new_static_argnum_with_default_arguments(self):
    f = jit(lambda x=4: x, static_argnums=0)
    y = f()
    assert y == 4

  def test_jit_with_mismatched_static_argnames(self):
    x_is_tracer, y_is_tracer = False, False
    def f(x, y):
      assert isinstance(x, core.Tracer) == x_is_tracer
      assert isinstance(y, core.Tracer) == y_is_tracer
      return 1

    # If both static_argnums and static_argnames are provided, they are allowed
    # to disagree and `jit` will respect the user's choices.
    f_nums = jit(f, static_argnums=1, static_argnames=())
    x_is_tracer, y_is_tracer = True, False
    assert f_nums(2, 'foo') == 1
    x_is_tracer, y_is_tracer = True, True
    assert f_nums(1, y=2) == 1

    f_names = jit(f, static_argnums=(), static_argnames='y')
    x_is_tracer, y_is_tracer = True, True
    assert f_names(2, 3) == 1
    x_is_tracer, y_is_tracer = True, False
    assert f_names(1, y='foo') == 1

    f_mixed = jit(f, static_argnums=(1,), static_argnames='x')
    x_is_tracer, y_is_tracer = True, False
    assert f_mixed(2, 'foo') == 1
    x_is_tracer, y_is_tracer = True, True
    assert f_mixed(1, y=3) == 1
    x_is_tracer, y_is_tracer = False, True
    assert f_mixed(x='foo', y=3) == 1

  # TODO(zhangqiaorjc): Test pruning constants after DCE pass prunes primitive
  # applications.
  @parameterized.parameters(2, 3, 4)
  def test_jit_with_pruned_args(self, num_args):
    def f(*args):
      used = np.array(2)
      return args[1] + used
    f_pruned = jit(f)
    args = range(num_args)
    with jtu.count_device_put() as count:
      np.testing.assert_allclose(f_pruned(*args), 3)
    self.assertEqual(count(), 1)

  def testBuffersAreFreedPromptly(self):
    # Regression test for a bug where garbage collection was delayed too long
    # for NumPy buffers that are aliased zero-copy by the runtime.
    @jit
    def f(x):
      return x + 1

    refs = []
    x = np.ones((10000,), np.float32)
    for step in range(1000):
      x = f(x)
      refs.append(weakref.ref(x))
      x = np.asarray(x)

    # We expect most of the input buffers to have been garbage
    # collected in parallel with the execution. We can't call
    # block_until_ready() here because it would force a garbage collection.
    live_refs = len([ref for ref in refs if ref() is not None])
    self.assertLessEqual(live_refs, 100)

  def test_jit_lower_compile(self):
    def f(x):
      return jnp.sqrt(x ** 2) + 1.

    f_jit = jit(f)
    lowered = f_jit.lower(1.)
    compiled = lowered.compile()
    self.assertAllClose(compiled(1.), 2.)
    self.assertEqual(lowered.in_avals, compiled.in_avals)
    expected_dtype = np.float64 if config.enable_x64.value else np.float32
    for obj in [lowered, compiled]:
      self.assertEqual(
          obj.in_avals,
          ((core.ShapedArray([], expected_dtype, weak_type=True),), {}))
      self.assertEqual(obj.in_tree, jax.tree.flatten(((0,), {}))[1])

  def test_jit_lower_duck_typing(self):
    f_jit = jit(lambda x: 2 * x)
    f_low = f_jit.lower(jax.ShapeDtypeStruct((), 'float32'))  # doesn't crash
    f_exe = f_low.compile()
    self.assertAllClose(f_exe(jnp.float32(1.)), jnp.float32(2.))

  def test_jit_lower_compile_in_tree_mismatch(self):
    def f(x):
      return jnp.sqrt(x ** 2) + 1.

    f_jit = jit(f)
    f_low = f_jit.lower(1.)
    f_exe = f_low.compile()
    self.assertRaisesRegex(
        TypeError,
        'Function compiled with input pytree does not match the input pytree it'
        ' was called with',
        lambda: f_exe([1.]))

  def test_jit_lower_compile_trivial(self):
    def f(x): return x
    out = jit(f).lower(1.).compile()(4.)
    self.assertAllClose(out, 4.)

  def test_jit_lower_compile_sharding_computation(self):
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    def f(x): return jax.lax.with_sharding_constraint(x, s)
    out = jit(f).lower(1.).compile()(4.)
    self.assertAllClose(out, 4.)

  def test_jit_lower_compile_trivial_in_tree_mismatch(self):
    def f(x): return x
    f_exe = jit(f).lower(1.).compile()
    self.assertRaisesRegex(
        TypeError,
        "Function compiled with input pytree does not match the input pytree it"
        " was called with",
        lambda: f_exe([4.0]),
    )

  def test_jit_lower_compile_arg_type_mismatch(self):
    def f(x):
      return jnp.sqrt(x ** 2) + 1.

    x = jnp.array(1, dtype=int)
    x_f32 = x.astype(jnp.float32)
    x_i32 = x.astype(jnp.int32)
    f_exe = jit(f).lower(x_f32).compile()
    self.assertRaisesRegex(
        TypeError,
        r"Argument types differ .*"
        r"The mismatches are:\n"
        r"Argument 'x' compiled with.*float32.*and called with.*int32.*",
        lambda: f_exe(x_i32))

  def test_jit_lower_compile_multi_arg(self):
    def f(*args):
      x, *_ = args
      return jnp.sqrt(x ** 2) + 1.
    f_exe = jit(f).lower(1., 1.).compile()
    self.assertAllClose(f_exe(1., 1.), 2.)

  def test_jit_lower_compile_trivial_multi_arg(self):
    def f(*args):
      x, *_ = args
      return x
    f_exe = jit(f).lower(1., 1.).compile()
    self.assertAllClose(f_exe(1., 1.), 1.)

  def test_jit_lower_donate_argnums_available(self):
    def f(*args):
      x, *_ = args
      return x + 4.
    f_low = jit(f, donate_argnums=(0,)).lower(1., 1.)
    f_com = f_low.compile()
    f_low.donate_argnums == f_com.donate_argnums == (0,)

  def test_jit_lower_compile_vmap(self):
    f = jit(lambda x: x + 4).lower(1.).compile()
    def err():
      return jax.vmap(lambda x: f(x) + 2)(jnp.ones(3))
    self.assertRaisesRegex(
        TypeError,
        "Cannot apply JAX transformations to a function lowered and compiled "
        "for a particular signature. Detected .*BatchTracer",
        err)

  def test_jit_lower_as_text(self):
    f = jit(lambda x: x + 4).lower(1.)
    self.assertIsInstance(f.as_text(), str)
    self.assertIsInstance(f.as_text(dialect='hlo'), str)
    self.assertIsInstance(f.as_text(dialect="stablehlo"), str)

  def test_jit_lower_compiler_ir(self):
    f = jit(lambda x: x + 4).lower(1.)
    self.assertIsNotNone(f.compiler_ir())
    self.assertIsNotNone(f.compiler_ir(dialect='hlo'))
    self.assertIsNotNone(f.compiler_ir(dialect="stablehlo"))

  def test_jit_lower_trivial_compiler_ir(self):
    f = jit(lambda x: x).lower(1.)
    self.assertIsNotNone(f.compiler_ir())
    self.assertIsNotNone(f.compiler_ir(dialect='hlo'))
    self.assertIsNotNone(f.compiler_ir(dialect="stablehlo"))

  def test_jit_replica_attributes(self):
    hlo = jit(lambda x: x + 4).lower(1.).as_text("stablehlo")
    self.assertIn("mhlo.num_partitions = 1", hlo)
    self.assertIn("mhlo.num_replicas = 1", hlo)

  def test_jit_lower_no_pruning(self):
    compiled = jit(lambda x, y: x + y).lower(1., 2.).compile()
    self.assertEqual(compiled._executable._kept_var_idx, {0, 1})
    self.assertLen(compiled._executable.in_avals, 2)

    compiled = jit(lambda x, y: x).lower(1., 2.).compile()
    self.assertEqual(compiled._executable._kept_var_idx, {0})
    self.assertLen(compiled._executable.in_avals, 1)

    compiled = jit(lambda x, y: x, keep_unused=True).lower(
        1., 2.).compile()
    self.assertEqual(compiled._executable._kept_var_idx, {0, 1})
    self.assertLen(compiled._executable.in_avals, 2)
    # Also works with jax.jit
    jitted_f = jit(lambda x, y: x, keep_unused=True)
    with jtu.count_pjit_cpp_cache_miss() as count:
      _ = jitted_f(1, 2)
    self.assertEqual(count(), 1)

  def test_jit_lower_compile_compiler_ir(self):
    f = jit(lambda x: x + 4).lower(1.).compile()
    self.assertIsNotNone(f.runtime_executable())

  def test_jit_lower_trivial_compile_compiler_ir(self):
    f = jit(lambda x: x).lower(1.).compile()
    self.assertIsNotNone(f.runtime_executable())

  def test_jit_lower_compile_as_text(self):
    f = jit(lambda x: x).lower(1.).compile()
    g = jit(lambda x: x + 4).lower(1.).compile()
    self.assertIsInstance(f.as_text(), (str, type(None)))
    self.assertIsInstance(g.as_text(), (str, type(None)))

  def test_jit_lower_cost_analysis(self):
    # TODO(b/261771737): add support for uncompiled cost analysis in C API.
    if "PJRT C API" in xla_bridge.get_backend().platform_version:
      raise unittest.SkipTest("C API does not support uncompiled cost analysis")
    f = jit(lambda x: x).lower(1.)
    g = jit(lambda x: x + 4).lower(1.)
    f.cost_analysis()  # doesn't raise
    g.cost_analysis()  # doesn't raise

  def test_jit_lower_compile_cost_analysis(self):
    f = jit(lambda x: x).lower(1.).compile()
    g = jit(lambda x: x + 4).lower(1.).compile()
    self.assertIsNotNone(f.cost_analysis())
    self.assertIsNotNone(g.cost_analysis())

  def test_jit_lower_compile_memory_analysis(self):
    f = jit(lambda x: x).lower(1.).compile()
    g = jit(lambda x: x + 4).lower(1.).compile()
    f.memory_analysis()  # doesn't raise
    g.memory_analysis()  # doesn't raise

  def test_jit_lower_compile_executable(self):
    f = jit(lambda x: x).lower(1.).compile()
    g = jit(lambda x: x + 4).lower(1.).compile()
    self.assertIsNotNone(f.runtime_executable())
    self.assertIsNotNone(g.runtime_executable())

  def test_jit_lower_compile_with_compiler_options(self):
    def f(x):
      return jnp.sqrt(x ** 2) + 1.

    f_jit = jit(f)
    lowered = f_jit.lower(1.)
    lowered.compile(  # doesn't crash
        compiler_options={
            "xla_embed_ir_in_executable": True,
            "xla_dump_max_hlo_modules": 200,
            "xla_gpu_auto_spmd_partitioning_memory_budget_ratio": 0.5,
        }
    )

  def test_compile_options_jit(self):
    def f(x):
      return jnp.sqrt(x ** 2) + 1.

    f_jit = jit(
        f,
        compiler_options={
            "xla_embed_ir_in_executable": True,
            "xla_dump_max_hlo_modules": 200,
            "xla_gpu_auto_spmd_partitioning_memory_budget_ratio": 0.5,
        })(1.0)  # doesn't crash.

  def test_exec_time_optimization_effort_compiler_option(self):
    def f(x):
      return jnp.sqrt(x ** 2) + 1.

    f_jit = jit(
        f,
        compiler_options={
            "exec_time_optimization_effort": 0.0,
        })(1.0)  # doesn't crash.

    with self.assertRaisesRegex(_jax.XlaRuntimeError, "No such"):
      f_jit = jit(
          f,
          compiler_options={
              "exec_time_compilation_effort": 0.0,
          })(1.0)

  def test_optimization_level_compiler_option(self):
    def f(x):
      return jnp.sqrt(x**2) + 1.0

    f_jit = jit(
        f,
        compiler_options={
            "optimization_level": config.EffortLevel.O1.value,
        },
    )(
        1.0
    )  # doesn't crash.

  def test_memory_fitting_level_compiler_option(self):
    def f(x):
      return jnp.sqrt(x**2) + 1.0

    f_jit = jit(
        f,
        compiler_options={
            "memory_fitting_level": config.EffortLevel.O0.value,
        },
    )(
        1.0
    )  # doesn't crash.

  def test_jit_lower_compile_with_compiler_options_invalid(self):
    def f(x):
      return jnp.sqrt(x ** 2) + 1.

    f_jit = jit(f)
    lowered = f_jit.lower(1.)

    self.assertRaisesRegex(
        _jax.XlaRuntimeError, "No such compile option: 'invalid_key'",
        lambda: lowered.compile(
            compiler_options={"invalid_key": "invalid_value"}))

    self.assertRaisesRegex(
        _jax.XlaRuntimeError, "is not a valid bool value.",
        lambda: lowered.compile(
            compiler_options={"xla_embed_ir_in_executable": "invalid_value"}))

  def test_jit_compile_with_compiler_options_multiple(self):
    def f(x):
      return jnp.sqrt(x ** 2) + 1.

    with jtu.count_jit_compilation_cache_miss() as count:
      jit(f, compiler_options={"xla_embed_ir_in_executable": True})(1.)
      jit(f, compiler_options={"xla_embed_ir_in_executable": False})(1.)
    self.assertEqual(count(), 2)

    # We should still error on invalid options after some valid compiles
    with self.assertRaisesRegex(
        _jax.XlaRuntimeError, "No such compile option: 'invalid_key'"):
      jit(f, compiler_options={"invalid_key": "invalid_value"})(1.)

  def test_lower_compile_with_compiler_options_multiple(self):
    def f(x):
      return jnp.sqrt(x ** 2) + 1.

    f_jit = jit(f)
    lowered = f_jit.lower(1.)

    l1 = lowered.compile()
    l2 = lowered.compile(
        compiler_options={"xla_embed_ir_in_executable": True})
    l3 = lowered.compile(
        compiler_options={"xla_embed_ir_in_executable": False})

    # Ideally we could test that these objects are different only in
    # that they respect the different options. Object identity is a
    # heuristic proxy for that.
    self.assertTrue(l1 is not l2)
    self.assertTrue(l1 is not l3)
    self.assertTrue(l2 is not l3)

    # We should still error on invalid options after some valid compiles
    self.assertRaisesRegex(
        _jax.XlaRuntimeError, "No such compile option: 'invalid_key'",
        lambda: lowered.compile(
            compiler_options={"invalid_key": "invalid_value"}))

  def test_jit_enum_as_dict_keys_fails(self):
    class E(enum.Enum):
      A = 0
      B = 1

    @jit
    def f(d) -> float:
      return d[E.A]

    with self.assertRaisesRegex(
        (TypeError, ValueError),
        "('<' not supported|Comparator raised exception).*"):
      f({E.A: 1.0, E.B: 2.0})

  def test_jit_static_argnums_requires_type_equality(self):
    # See: https://github.com/jax-ml/jax/pull/9311
    @partial(jit, static_argnums=(0,))
    def f(k):
      assert python_should_be_executing
      return k

    # Values of 'x' that compare as equal but have different types do not lead
    # to cache hits.
    for x in [1, True, 1.0]:
      python_should_be_executing = True
      self.assertEqual(x, f(x))
      python_should_be_executing = False
      self.assertEqual(x, f(x))

  def test_caches_depend_on_axis_env(self):
    # https://github.com/jax-ml/jax/issues/9187
    f = lambda: lax.axis_size("i")
    g = jax.jit(f)
    expected = jax.vmap(f, axis_name="i", axis_size=2, out_axes=None)()
    ans = jax.vmap(g, axis_name="i", axis_size=2, out_axes=None)()
    self.assertEqual(ans, expected)

    # This second call to g could erroneously get a cache hit.
    expected = jax.vmap(f, axis_name="i", axis_size=3, out_axes=None)()
    ans = jax.vmap(g, axis_name="i", axis_size=3, out_axes=None)()
    self.assertEqual(ans, expected)

  # Since stackless, the vmap(f) version gets compiled a second time
  @unittest.skip
  def test_caches_dont_depend_on_unnamed_axis_env(self):
    # https://github.com/jax-ml/jax/issues/9187
    f = jax.jit(lambda: jnp.sin(1))
    expected = f()
    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      ans = jax.vmap(f, axis_size=2, out_axes=None)()
    self.assertEqual(count(), 0)  # no compiles
    self.assertArraysAllClose(ans, expected, check_dtypes=True)

  def test_cache_key_defaults(self):
    # https://github.com/jax-ml/jax/discussions/11875
    f = jit(lambda x: (x ** 2).sum())
    self.assertEqual(f._cache_size(), 0)
    x = jnp.arange(5.0)
    for _ in range(3):
      _ = f(x)
    self.assertEqual(f._cache_size(), 1)

  def test_jit_nan_times_zero(self):
    # https://github.com/jax-ml/jax/issues/4780
    def f(x):
      return 1 + x * 0
    self.assertAllClose(f(np.nan), np.nan)
    self.assertAllClose(jit(f)(np.nan), np.nan)

  def test_no_tracing(self):
    @jax.jit
    def f(x):
      return x

    x = jnp.arange(3)
    y = jnp.arange(4)

    _ = f(x)  # no crash

    with self.assertRaisesRegex(RuntimeError, 'no_tracing'):
      with jax.no_tracing():
        _ = f(y)  # crash!


class APITest(jtu.JaxTestCase):

  def test_grad_item(self):
    def f(x):
      if x.astype(bool).item():
        return x ** 2
      else:
        return x
    out = jax.grad(f)(2.0)
    self.assertEqual(out, 4)

  def test_jit_item(self):
    def f(x):
      return x.item()
    x = jnp.array(1.0)
    self.assertEqual(f(x), x)
    with self.assertRaisesRegex(core.ConcretizationTypeError, "Abstract tracer value"):
      jax.jit(f)(x)

  @parameterized.named_parameters(
      ('grad', jax.grad),
      ('jacfwd', jax.jacfwd),
      ('jacref', jax.jacrev),
  )
  def test_grad_wrap(self, transform):
    # Ensures that transforms wrap transformed functions with the correct signature.

    @partial(jit, static_argnames=['flag'])
    @transform
    def my_function(x, flag):
      return x if flag else jnp.zeros_like(x)

    self.assertEqual(my_function(1.0, False), 0.0)
    self.assertEqual(my_function(1.0, True), 1.0)

  def test_grad_bad_input(self):
    def f(x):
      return x

    self.assertRaisesRegex(
        TypeError, ".* 'foo' of type <.*'str'> is not a valid JAX type",
        lambda: grad(f)("foo"))

  def test_grad_argnums(self):
    def f(x, y, z, flag=False):
      assert flag
      return 1.0 * x + 2.0 * y + 3.0 * z

    assert grad(f)(1.0, 1.0, 1.0, flag=True) == 1.0
    assert grad(f, argnums=1)(1.0, 1.0, 1.0, flag=True) == 2.0
    assert grad(f, argnums=(2, 0))(1.0, 1.0, 1.0, flag=True) == (3.0, 1.0)

  def test_value_and_grad_argnums(self):
    def f(x, y, z, flag=False):
      assert flag
      return 1.0 * x + 2.0 * y + 3.0 * z

    y = f(1.0, 1.0, 1.0, flag=True)
    assert api.value_and_grad(f)(1.0, 1.0, 1.0, flag=True) == (y, 1.0)
    assert api.value_and_grad(f, argnums=1)(1.0, 1.0, 1.0, flag=True) == (y, 2.0)
    assert api.value_and_grad(f, argnums=(2, 0))(1.0, 1.0, 1.0, flag=True) == (y, (3.0, 1.0))

  @jtu.thread_unsafe_test()  # Concurrent cache eviction means we may retrace.
  def test_grad_of_jit(self):
    side = []

    @jit
    def f(x):
      side.append(None)
      return x * x

    assert grad(f)(1.0) == 2.0
    assert len(side) == 1
    assert grad(f)(2.0) == 4.0
    assert len(side) == 1

  @jtu.thread_unsafe_test()  # Concurrent ache eviction means we may retrace.
  def test_jit_of_grad(self):
    side = []

    @jit
    def f(x):
      side.append(None)
      return x * x

    g = jit(grad(f))
    assert g(1.0) == 2.0
    assert len(side) == 1
    assert g(2.0) == 4.0
    assert len(side) == 1

  @jtu.thread_unsafe_test()  # Concurrent ache eviction means we may retrace.
  def test_fwd_and_bwd(self):
    def f(x, W):
        return x @ W

    x = W = cot_out = jnp.ones((4,4))
    expected_y, f_vjp = api.vjp(f, x, W)
    expected_cot_x, expected_cot_W = f_vjp(cot_out)

    fwd, bwd = api.fwd_and_bwd(f, argnums=(0,1))
    y, residuals = fwd(x, W)
    cot_x, cot_W = bwd(residuals, cot_out)

    self.assertArraysAllClose(y, expected_y)
    self.assertArraysAllClose(cot_x, expected_cot_x)
    self.assertArraysAllClose(cot_W, expected_cot_W)

    with jax.no_tracing():
      y, residuals = fwd(x, W)
      cot_x, cot_W = bwd(residuals, cot_out)  # no recompilation

  @parameterized.named_parameters(
      {"testcase_name": f"_{transform.__name__}", "transform": transform}
      for transform in [grad, jacfwd, jacrev])
  def test_ad_weak_types(self, transform):
    out = transform(lambda x: x)(1.0)
    self.assertTrue(dtypes.is_weakly_typed(out))

  def test_bad_input(self):
    def f(x):
      return x

    with self.assertRaisesRegex(TypeError, ".* 'foo' of type <.*'str'> is not a valid JAX type"):
      grad(f)("foo")


    err_str = ("Error interpreting argument to .* as an abstract array. The problematic "
               "value is of type .* and was passed to the function at path x.")
    with self.assertRaisesRegex(TypeError, err_str):
      jit(f)("foo")

  def test_grad_tuple_output(self):
    jtu.check_raises(lambda: grad(lambda x: (x,x))(1.0), TypeError,
                     "Gradient only defined for scalar-output functions. ")

  def test_grad_unit_output(self):
    jtu.check_raises(lambda: grad(lambda x: ())(np.zeros(3)), TypeError,
                     "Gradient only defined for scalar-output functions. ")

  def test_grad_nonscalar_output(self):
    jtu.check_raises(lambda: grad(lambda x: x)(np.zeros(3)), TypeError,
                     "Gradient only defined for scalar-output functions. ")

  def test_unwrapped_numpy(self):
    def f(x):
      return np.exp(x)

    with self.assertRaisesRegex(Exception, "The numpy.ndarray conversion .*"):
      grad(f)(np.zeros(3))

  def test_binop_mismatch(self):
    def f(x, y):
      return x + y

    jtu.check_raises(
        lambda: f(jnp.zeros(3), jnp.zeros(4)),
        TypeError,
        "add got incompatible shapes for broadcasting: (3,), (4,).")

    jtu.check_raises(
        lambda: grad(f)(np.zeros(3), np.zeros(4)),
        TypeError,
        "add got incompatible shapes for broadcasting: (3,), (4,).")

  def test_dot_mismatch(self):
    def f(x, y):
      return jnp.dot(x, y)

    self.assertRaisesRegex(
        TypeError, ("dot_general requires contracting dimensions to have "
                    "the same shape, got \\(3L?,\\) and \\(4L?,\\)."),
      lambda: grad(f)(np.zeros(3), np.zeros(4)))

  def test_abstract_error_message(self):
    for castfun in [float, complex, int]:
      def f(x):
        return castfun(x)

      self.assertRaisesRegex(
          TypeError,
          f"[Tt]ry using `x.astype\\({castfun.__name__}\\)`",
          lambda: jit(f)(1.0))

  def test_switch_value_jit(self):
    def f(x):
      y = x > 0
      if y:
        return x
      else:
        return -x

    assert grad(f)(1.0) == 1.0
    assert grad(f)(-1.0) == -1.0
    with self.assertRaisesRegex(core.ConcretizationTypeError,
                                "Attempted boolean conversion"):
      jit(f)(1)

  def test_list_index_err(self):
    L = [1, 2, 3]
    def f(n):
      return L[n]

    assert jit(f, static_argnums=(0,))(0) == L[0]
    self.assertRaisesRegex(
        TypeError,
        r"The __index__\(\) method was called on traced array.*",
        lambda: jit(f)(0))

  def test_range_err(self):
    def f(x, n):
      for i in range(n):
        x = x + i
      return x

    assert jit(f, static_argnums=(1,))(0, 5) == 10
    self.assertRaisesRegex(
        TypeError,
        r"The __index__\(\) method was called on traced array.*",
        lambda: jit(f)(0, 5))

  def test_cast_int(self):
    f = lambda x: int(x)
    self.assertRaisesRegex(
        TypeError,
        "('(?:JaxprTracer|DynamicJaxprTracer)' object cannot be interpreted as an integer"
        "|Abstract tracer value encountered where concrete value is expected.*)", lambda: jit(f)(0))

  def test_casts(self):
    for castfun in [hex, oct]:
      f = lambda x: castfun(x)
      self.assertRaisesRegex(
          TypeError,
          r"The __index__\(\) method was called on traced array.*", lambda: jit(f)(0))

  def test_unimplemented_interpreter_rules(self):
    foo_p = core.Primitive('foo')
    def foo(x):
      return foo_p.bind(x)

    jtu.check_raises(lambda: foo(1.0), NotImplementedError,
                     "Evaluation rule for 'foo' not implemented")

    jtu.check_raises(lambda: jit(foo)(1.0), NotImplementedError,
                     "Abstract evaluation for 'foo' not implemented")

    jtu.check_raises(lambda: grad(foo)(1.0), NotImplementedError,
                     "Differentiation rule for 'foo' not implemented")

    foo_p.def_abstract_eval(lambda x: x)

    jtu.check_raises_regexp(lambda: jit(foo)(1.0), NotImplementedError,
                     ".* rule for primitive 'foo' not found.*")

    foo_p.def_impl(lambda x: x)
    ad.defjvp(foo_p, lambda g, x: foo(g))

    jtu.check_raises(lambda: grad(foo)(1.0), NotImplementedError,
                     "Transpose rule (for reverse-mode differentiation) for 'foo' not implemented")

  def test_wrong_output_abstract_eval(self):
    foo_p = core.Primitive('foo')
    def foo(x):
      return foo_p.bind(x)
    foo_p.def_abstract_eval(lambda x: [x]) # Shouldn't return a list.

    foo_p.def_impl(lambda x: x)
    jitted = jit(lambda x: foo(x))
    jtu.check_raises(lambda: jitted(1.0), ValueError,
                     "foo.abstract_eval() method should return a tuple or")

    foo2_p = core.Primitive('foo2')
    foo2_p.multiple_results = True
    def foo2(x):
      return foo2_p.bind(x),

    foo2_p.def_abstract_eval(lambda x: x) # Should return a list.
    foo2_p.def_impl(lambda x: [x])
    jitted = jit(lambda x: foo2(x))
    jtu.check_raises(lambda: jitted(1.0), ValueError,
                     "foo2.abstract_eval() method should return a tuple or")

  def test_is_subclass(self):
    self.assertFalse(issubclass(np.ndarray, jax.Array))

  def test_is_instance(self):
    def f(x):
      self.assertIsInstance(x, jax.Array)
      self.assertNotIsInstance(x, np.ndarray)
      return x + 2
    jit(f)(3)
    jax.vmap(f)(np.arange(3))

  def test_device_put_and_get(self):
    x = np.arange(12.).reshape((3, 4)).astype("float32")
    dx = api.device_put(x)
    _check_instance(self, dx)
    self.assertIsInstance(dx, jax.Array)
    self.assertNotIsInstance(dx, np.ndarray)
    x2 = api.device_get(dx)
    self.assertNotIsInstance(x2, jax.Array)
    self.assertIsInstance(x2, np.ndarray)
    assert np.all(x == x2)

    y = [x, (2 * x, 3 * x)]
    dy = api.device_put(y)
    y2 = api.device_get(dy)
    self.assertIsInstance(y2, list)
    self.assertIsInstance(y2[0], np.ndarray)
    assert np.all(y2[0] == x)
    self.assertIsInstance(y2[1], tuple)
    self.assertIsInstance(y2[1][0], np.ndarray)
    assert np.all(y2[1][0] == 2 * x)
    self.assertIsInstance(y2[1][1], np.ndarray)
    assert np.all(y2[1][1] == 3 * x)

  def test_device_put_sharding(self):
    mesh = jax.sharding.Mesh(jax.devices(), ('x',))
    s = jax.NamedSharding(mesh, P('x'))
    x = jnp.arange(len(jax.devices()))

    y = jax.device_put(x, s)
    self.assertEqual(y.sharding, s)
    self.assertArraysAllClose(y, x)

    # this might hit a special fast path
    z = jax.device_put(y, s)
    self.assertEqual(z.sharding, s)
    self.assertArraysAllClose(z, x)
    self.assertIs(z, y)  # no copy

    w = jax.device_put(z)
    self.assertIs(w, z)

    u = jax.device_put(y, jax.devices()[0])
    self.assertArraysAllClose(u, y)
    self.assertEqual(u.devices(), {jax.devices()[0]})

  def test_device_put_sharding_tree(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices")

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]).reshape((2, 1)),
                             ("x", "y"))
    s1 = jax.NamedSharding(mesh, P("x"))
    s2 = jax.NamedSharding(mesh, P("y"))
    s3 = jax.NamedSharding(mesh, P("x", "y"))

    x = jnp.arange(2)
    y = jnp.arange(2) + 10
    z = (jnp.arange(2) + 100).reshape((2, 1))

    out = jax.device_put((x, (y, z)), device=(s1, (s2, s3)))
    self.assertEqual(out[0].sharding, s1)
    self.assertEqual(out[1][0].sharding, s2)
    self.assertEqual(out[1][1].sharding, s3)

    self.assertArraysAllClose(out[0], x)
    self.assertArraysAllClose(out[1][0], y)
    self.assertArraysAllClose(out[1][1], z)

  def test_device_put_sharding_tree_prefix(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices")

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]).reshape((2, 1)), ("x", "y"))
    s1 = jax.sharding.NamedSharding(mesh, P("x"))
    s2 = jax.sharding.NamedSharding(mesh, P("y"))

    x = jnp.arange(2)
    y = jnp.arange(2) + 10
    z = jnp.arange(2) + 100

    out = jax.device_put((x, (y, z)), device=(s1, s2))
    self.assertEqual(out[0].sharding, s1)
    self.assertEqual(out[1][0].sharding, s2)
    self.assertEqual(out[1][1].sharding, s2)

    self.assertArraysAllClose(out[0], x)
    self.assertArraysAllClose(out[1][0], y)
    self.assertArraysAllClose(out[1][1], z)

  def test_device_put_sharding_mismatched_tree_same_leaf_count(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices")

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]).reshape((2, 1)), ("x", "y"))
    s1 = jax.sharding.NamedSharding(mesh, P("x"))
    s2 = jax.sharding.NamedSharding(mesh, P("y"))

    x = jnp.arange(2)
    y = jnp.arange(2) + 10
    z = jnp.arange(2) + 100

    with self.assertRaisesRegex(
        ValueError,
        "device_put device specification must be a tree prefix of the "
        r"corresponding value, got specification \(\(NamedSharding\(.*\), "
        r"NamedSharding\(.*\)\), NamedSharding\(.*\)\) for value tree "
        r"PyTreeDef\(\(\*, \(\*, \*\)\)\)."
    ):
      jax.device_put((x, (y, z)), device=((s1, s2), s2))

  def test_device_put_sharding_mismatched_tree_different_leaf_count(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices")

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]).reshape((2, 1)), ("x", "y"))
    s1 = jax.sharding.NamedSharding(mesh, P("x"))
    s2 = jax.sharding.NamedSharding(mesh, P("y"))

    x = jnp.arange(2)
    y = jnp.arange(2) + 10
    z = jnp.arange(2) + 100

    with self.assertRaisesRegex(
        ValueError,
        "device_put device specification must be a tree prefix of the "
        r"corresponding value, got specification \(NamedSharding\(.*\), "
        r"NamedSharding\(.*\)\) for value tree PyTreeDef\(\(\*, \*, \*\)\)."
    ):
      jax.device_put((x, y, z), device=(s1, s2))

  def test_internal_device_put_with_device(self):
    # Hitting the cache for a single-device jitted execution while using a numpy
    # array calls internal `DevicePutWithDevice`.
    f = jax.jit(lambda x: x + 1)
    f(np.arange(8))

    with jtu.count_internal_device_puts() as counts:
      f(np.arange(8))
    self.assertEqual(counts(), {"device_put_with_device": 1})

  def test_internal_device_put_fully_replicated(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices")

    # Creating an array from a numpy array with a fully-replicated sharding
    # calls internal `DevicePutWithSharding`, taking the fully-replicated sub
    # case.
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), "x")
    sharding = jax.NamedSharding(mesh, P())

    with jtu.count_internal_device_puts() as counts:
      jax.device_put(np.arange(8), sharding)
    self.assertEqual(
        counts(),
        {"device_put_with_sharding": 1, "device_put_fully_replicated": 1},
    )

  def test_internal_device_put_batched(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices")

    # Creating an array from a numpy array with a non-fully-replicated sharding
    # calls internal `DevicePutWithSharding`, performing batched creation of a
    # multi-shard array.
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), "x")
    sharding = jax.NamedSharding(mesh, P("x"))

    with jtu.count_internal_device_puts() as counts:
      jax.device_put(np.arange(8), sharding)
    self.assertEqual(
        counts(), {"device_put_with_sharding": 1, "device_put_batched": 1}
    )

  def test_internal_device_put_assembled(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices")

    # Creating an array from per-device JAX arrays calls internal
    # `DevicePutWithSharding`, performing per-shard array adoption followed by
    # assembly.
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), "x")
    sharding = jax.NamedSharding(mesh, P("x"))

    arr = np.arange(8)
    per_device_arrs = {
        # Use uncommitted arrays that are not aligned with the destination
        # sharding so that we trigger `BatchedDevicePut`.
        sharding_impls.hashed_index(index): jnp.array(arr[index])
        for _, index in sharding.devices_indices_map(arr.shape).items()
    }
    data_callback = lambda index: per_device_arrs[
        sharding_impls.hashed_index(index)
    ]
    with jtu.count_internal_device_puts() as counts:
      jax.make_array_from_callback(arr.shape, sharding, data_callback)
    self.assertEqual(
        counts(), {"device_put_with_sharding": 1, "device_put_assembled": 1}
    )

  def test_device_put_custom_type_not_accepting_none_leaves(self):

    class CustomNode(list):
      pass

    def unflatten(unused_aux_data, children):
      self.assertIsNotNone(children[0])
      return CustomNode(children)

    tree_util.register_pytree_node(CustomNode, lambda x: (x, None), unflatten)
    jax.device_put(CustomNode([0.1]))

  def test_vmap_inconsistent_sizes_constructs_proper_error_message(self):
    def f(x1, x2, g):
      return g(x1, x2)

    with self.assertRaisesRegex(
        ValueError,
        "vmap got inconsistent sizes for array axes to be mapped:"
    ):
      jax.vmap(f, (0, 0, None))(jnp.ones(2), jnp.ones(3), jnp.add)

  def test_vmap_inconsistent_sizes_constructs_proper_error_message_kwargs(self):
    # regression test for https://github.com/jax-ml/jax/issues/24406
    def f(x1, x2, a3):
      return x1 + x2 + a3

    with self.assertRaisesRegex(
      ValueError,
      "vmap got inconsistent sizes for array axes to be mapped:\n"
      r"  \* most axes \(2 of them\) had size 2, e.g. axis 0 of argument x1 of type float32\[2\];\n"
      r"  \* one axis had size 1: axis 0 of kwargs\['a3'\] of type float32\[1\]",
    ):
      jax.vmap(f)(
        jnp.ones(2, dtype=jnp.float32),
        a3=jnp.ones(1, dtype=jnp.float32),
        x2=jnp.ones(2, dtype=jnp.float32)
      )

  def test_vmap_inconsistent_sizes_constructs_proper_error_message_starargs(self):
    # regression test for https://github.com/jax-ml/jax/issues/26908
    def f(x, *args):
      return x - functools.reduce(jnp.add, args)

    with self.assertRaisesRegex(
      ValueError,
      "vmap got inconsistent sizes for array axes to be mapped:"
    ):
      jax.vmap(f)(jnp.ones(4), jnp.ones(2), jnp.ones(2))

  def test_vmap_sentinel(self):

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class Foo:
      x: jax.Array

      def __init__(self, x):
        nonlocal saw_sentinel
        if x is jax._src.api_util.SENTINEL:
          saw_sentinel += 1
        self.x = x

    x = jnp.arange(10)

    # assert that sentinel is seen once for vmap in_axes
    saw_sentinel = 0
    jax.vmap(lambda f: f.x)(Foo(x))
    self.assertEqual(saw_sentinel, 1)

    # assert that sentinel is seen once for vmap out_axes
    saw_sentinel = 0
    jax.vmap(Foo)(x)
    self.assertEqual(saw_sentinel, 1)

    # assert that sentinel is seen twice with vmap in_axes and out_axes
    saw_sentinel = 0
    jax.vmap(lambda f: Foo(f.x + 1))(Foo(x))
    self.assertEqual(saw_sentinel, 2)


  def test_device_get_scalar(self):
    x = np.arange(12.).reshape((3, 4)).astype("float32")
    x = api.device_put(x)
    _check_instance(self, x)
    self.assertIsInstance(x.sharding, jax.sharding.SingleDeviceSharding)
    for s in x.addressable_shards:
      self.assertArraysEqual(s.data, x)
      self.assertEqual(s.replica_id, 0)
      self.assertEqual(s.index, (slice(None), slice(None)))
    y = [x, 2]
    y2 = api.device_get(y)
    self.assertIsInstance(y2, list)
    self.assertIsInstance(y2[0], np.ndarray)
    assert np.all(y2[0] == x)
    self.assertIsInstance(y2[1], int)
    self.assertEqual(y2[1], 2)

  @parameterized.parameters([(3,)], [(2, 0)])
  def test_device_put_across_devices(self, shape):
    if len(jax.local_devices()) < 2:
      raise unittest.SkipTest("this test requires multiple devices")
    d1, d2 = jax.local_devices()[:2]
    data = self.rng().randn(*shape).astype(np.float32)
    x = api.device_put(data, device=d1)
    self.assertEqual(x.devices(), {d1})

    y = api.device_put(x, device=d2)
    self.assertEqual(y.devices(), {d2})

    np.testing.assert_array_equal(data, np.array(y))
    # Make sure these don't crash
    api.device_put(x)
    api.device_put(y)

  @jtu.skip_on_devices("cpu")
  def test_device_put_across_platforms(self):
    default_device = jax.devices()[0]
    cpu_device = jax.devices("cpu")[0]

    np_arr = np.array([1,2,3])
    scalar = 1
    device_arr = jnp.array([1,2,3])
    assert device_arr.devices() == {default_device}

    for val in [np_arr, device_arr, scalar]:
      x = api.device_put(val, device=cpu_device)
      self.assertEqual(x.devices(), {cpu_device})

  def test_device_put_on_single_device_donated_buffer_fails(self):
    @partial(jax.jit, donate_argnums=0)
    def f(inp1):
      return inp1 * 2

    x = jnp.zeros((10,), jnp.float32)
    f(x)

    with self.assertRaises(RuntimeError):
      result = jax.device_put(x, jax.devices()[0])
      result.block_until_ready()

    with self.assertRaises(RuntimeError):
      result = jax.device_put(x, jax.devices()[-1])
      result.block_until_ready()

  def test_device_put_on_multi_device_donated_buffer_fails(self):
    @partial(jax.jit, donate_argnums=0)
    def f(inp1):
      return inp1 * 2

    mesh1 = jax.sharding.Mesh(jax.devices(), ("x",))
    s1 = jax.NamedSharding(mesh1, P("x"))

    mesh2 = jax.sharding.Mesh(tuple(reversed(jax.devices())), ("x",))
    s2 = jax.NamedSharding(mesh2, P("x"))

    x = jax.device_put(np.arange(len(jax.devices()), dtype=jnp.float32), s1)
    f(x)

    with self.assertRaises(RuntimeError):
      result = jax.device_put(x, s1)
      result.block_until_ready()

    with self.assertRaises(RuntimeError):
      result = jax.device_put(x, s2)
      result.block_until_ready()


  @jax.default_matmul_precision("float32")
  def test_jacobian(self):
    R = self.rng().randn
    A = R(4, 3)
    x = R(3)

    f = lambda x: jnp.dot(A, x)
    assert np.allclose(jacfwd(f)(x), A)
    assert np.allclose(jacrev(f)(x), A)

    f = lambda x: jnp.tanh(jnp.dot(A, x))
    assert np.allclose(jacfwd(f)(x), jacrev(f)(x))

  @jax.default_matmul_precision("float32")
  def test_hessian(self):
    R = self.rng().randn
    A = R(4, 4)
    x = R(4)

    f = lambda x: jnp.dot(x, jnp.dot(A, x))
    assert np.allclose(hessian(f)(x), A + A.T)

  @jax.default_matmul_precision("float32")
  def test_hessian_holomorphic(self):
    R = self.rng().randn
    A = R(4, 4)
    x = R(4).astype('complex64') * (1 + 2j)

    f = lambda x: jnp.dot(x, jnp.dot(A.astype(x.dtype), x))
    assert np.allclose(hessian(f, holomorphic=True)(x), A + A.T)

  @jax.default_matmul_precision("float32")
  def test_hessian_aux(self):
    R = self.rng().randn
    A = R(4, 4)
    x = R(4)

    f = lambda x: (jnp.dot(x, jnp.dot(A, x)), x)
    h, aux = hessian(f, has_aux=True)(x)
    assert np.allclose(h, A + A.T)
    assert np.allclose(aux, x)

  def test_std_basis(self):
    basis = api._std_basis(jnp.zeros(3))
    assert getattr(basis, "shape", None) == (3, 3)
    assert np.allclose(basis, np.eye(3))

    basis = api._std_basis(jnp.zeros((3, 3)))
    assert getattr(basis, "shape", None) == (9, 3, 3)
    assert np.allclose(basis, np.eye(9).reshape(9, 3, 3))

    basis = api._std_basis([0., (jnp.zeros(3), jnp.zeros((3, 4)))])
    assert isinstance(basis, list) and len(basis) == 2
    assert getattr(basis[0], "shape", None) == (16,)
    assert isinstance(basis[1], tuple) and len(basis[1]) == 2
    assert getattr(basis[1][0], "shape", None) == (16, 3)
    assert getattr(basis[1][1], "shape", None) == (16, 3, 4)

  @jtu.skip_on_devices("tpu")
  def test_jacobian_on_pytrees(self):
    for jacfun in [jacfwd, jacrev]:
      ans = jacfun(lambda x, y: (x, y))(0., 1.)
      expected = (1., 0.)
      self.assertAllClose(ans, expected, check_dtypes=False)

      ans = jacfun(lambda x, y: (x, y), 1)(0., 1.)
      expected = (0., 1.)
      self.assertAllClose(ans, expected, check_dtypes=False)

      ans = jacfun(lambda x, y: (x, y), (0, 1))(0., 1.)
      expected = ((1., 0.),
                  (0., 1.),)
      self.assertAllClose(ans, expected, check_dtypes=False)

      ans = jacfun(lambda x: x[:2])((1., 2., 3.))
      expected = ((1., 0., 0.),
                  (0., 1., 0.))
      self.assertAllClose(ans, expected, check_dtypes=False)

      R = self.rng().randn
      x = jnp.array(R(2))
      y = jnp.array(R(3))
      ans = jacfun(lambda x, y: {'x': x, 'xy': jnp.outer(x, y)})(x, y)
      expected = {'x': np.eye(2),
                  'xy': np.kron(np.eye(2), y[:, None]).reshape(2, 3, 2)}
      self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("tpu")
  def test_hessian_on_pytrees(self):
    ans = hessian(lambda x: jnp.array(x)**2)((1., 2.))
    expected = ((np.array([2., 0.]), np.array([0., 0.])),
                (np.array([0., 0.]), np.array([0., 2.])))
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("tpu")
  def test_issue1372(self):
    def quad(x):
      return jnp.dot(x, x)

    def f(x, u):
      return quad(x) + quad(u)

    x, u = jnp.ones(5), jnp.ones(2)

    rev = jacrev
    fwd = jacfwd

    # Diagonal entries
    self.assertEqual(rev(rev(f, 0), 0)(x, u).shape, (5, 5))
    self.assertEqual(rev(fwd(f, 0), 0)(x, u).shape, (5, 5))
    self.assertEqual(fwd(rev(f, 0), 0)(x, u).shape, (5, 5))
    self.assertEqual(fwd(fwd(f, 0), 0)(x, u).shape, (5, 5))
    self.assertEqual(rev(rev(f, 1), 1)(x, u).shape, (2, 2))
    self.assertEqual(rev(fwd(f, 1), 1)(x, u).shape, (2, 2))
    self.assertEqual(fwd(rev(f, 1), 1)(x, u).shape, (2, 2))
    self.assertEqual(fwd(fwd(f, 1), 1)(x, u).shape, (2, 2))

    # Off-diagonal entries by reverse-mode on the outside
    self.assertEqual(rev(rev(f, 1), 0)(x, u).shape, (2, 5))
    self.assertEqual(rev(fwd(f, 1), 0)(x, u).shape, (2, 5))
    self.assertEqual(rev(rev(f, 0), 1)(x, u).shape, (5, 2))
    self.assertEqual(rev(fwd(f, 0), 1)(x, u).shape, (5, 2))

    # Off-diagonal entries by forward-mode on the outside
    self.assertEqual(fwd(rev(f, 1), 0)(x, u).shape, (2, 5))
    self.assertEqual(fwd(fwd(f, 1), 0)(x, u).shape, (2, 5))
    self.assertEqual(fwd(rev(f, 0), 1)(x, u).shape, (5, 2))
    self.assertEqual(fwd(fwd(f, 0), 1)(x, u).shape, (5, 2))

  def test_large_device_constant(self):
    ans = jit(lambda x: 2 * x)(jnp.ones(int(2e6)))  # doesn't crash
    self.assertAllClose(ans, np.ones(int(2e6)) * 2., check_dtypes=False)

  def test_grad_and_aux_basic(self):
    g, aux = grad(lambda x: (x**3, [x**2]), has_aux=True)(3.)
    self.assertAllClose(g, grad(lambda x: x**3)(3.))
    self.assertAllClose(aux, [9.], check_dtypes=False)

  def test_grad_and_aux_error(self):
    with self.assertRaisesRegex(TypeError, "two-element tuple"):
      grad(lambda x: (1, 2, 3), has_aux=True)(1.)

    with self.assertRaisesRegex(TypeError, "two-element tuple"):
      grad(lambda x: x, has_aux=True)(1.)

    with self.assertRaisesRegex(TypeError, "two-element tuple"):
      grad(lambda x: (x,), has_aux=True)(1.)

  def test_grad_and_aux_nested(self):
    def f(x):
      g, aux = grad(lambda x: (x**3, [x**3]), has_aux=True)(x)
      return aux[0]

    f2 = lambda x: x**3

    self.assertEqual(grad(f)(4.), grad(f2)(4.))
    self.assertEqual(jit(grad(f))(4.), grad(f2)(4.))
    self.assertEqual(jit(grad(jit(f)))(4.), grad(f2)(4.))

    def f(x):
      g, aux = grad(lambda x: (x**3, [x**3]), has_aux=True)(x)
      return aux[0] * jnp.sin(x)

    f2 = lambda x: x**3 * jnp.sin(x)

    self.assertEqual(grad(f)(4.), grad(f2)(4.))
    self.assertEqual(jit(grad(f))(4.), grad(f2)(4.))
    self.assertEqual(jit(grad(jit(f)))(4.), grad(f2)(4.))

  def test_grad_and_aux_constant(self):
    g, aux = grad(lambda x: (x**3, [4.]), has_aux=True)(4.)
    self.assertEqual(g, grad(lambda x: x**3)(4.))
    self.assertEqual(aux, [4.])

    g, aux = grad(lambda x: (x**3, [x**2, 4.]), has_aux=True)(4.)
    self.assertEqual(g, grad(lambda x: x**3)(4.))
    self.assertEqual(aux, [4.**2, 4.])

  def test_grad_and_aux_no_tracers(self):
    # see https://github.com/jax-ml/jax/issues/1950
    def f(x):
      aux = dict(identity=x, p1=x+1)
      return x ** 2, aux

    _, aux = jax.grad(f, has_aux=True)(3.)
    self.assertIsInstance(aux, dict)
    for val in aux.values():
      self.assertNotIsInstance(val, core.Tracer)

  def test_jacfwd_and_aux_basic(self):
    jac, aux = jacfwd(lambda x: (x**3, [x**2]), has_aux=True)(3.)
    self.assertAllClose(jac, jacfwd(lambda x: x**3)(3.))
    self.assertAllClose(aux, [9.], check_dtypes=False)

  def test_jacrev_and_aux_basic(self):
    jac, aux = jacrev(lambda x: (x**3, [x**2]), has_aux=True)(3.)
    self.assertAllClose(jac, jacrev(lambda x: x**3)(3.))
    self.assertAllClose(aux, [9.], check_dtypes=False)

  def test_jacfwd_and_aux_nested(self):
    def f(x):
      jac, aux = jacfwd(lambda x: (x**3, [x**3]), has_aux=True)(x)
      return aux[0]

    f2 = lambda x: x**3

    self.assertEqual(jacfwd(f)(4.), jacfwd(f2)(4.))
    self.assertEqual(jit(jacfwd(f))(4.), jacfwd(f2)(4.))
    self.assertEqual(jit(jacfwd(jit(f)))(4.), jacfwd(f2)(4.))

    def f(x):
      jac, aux = jacfwd(lambda x: (x**3, [x**3]), has_aux=True)(x)
      return aux[0] * jnp.sin(x)

    f2 = lambda x: x**3 * jnp.sin(x)

    self.assertEqual(jacfwd(f)(4.), jacfwd(f2)(4.))
    self.assertEqual(jit(jacfwd(f))(4.), jacfwd(f2)(4.))
    self.assertEqual(jit(jacfwd(jit(f)))(4.), jacfwd(f2)(4.))

  def test_jacrev_and_aux_nested(self):
    def f(x):
      jac, aux = jacrev(lambda x: (x**3, [x**3]), has_aux=True)(x)
      return aux[0]

    f2 = lambda x: x**3

    self.assertEqual(jacrev(f)(4.), jacrev(f2)(4.))
    self.assertEqual(jit(jacrev(f))(4.), jacrev(f2)(4.))
    self.assertEqual(jit(jacrev(jit(f)))(4.), jacrev(f2)(4.))

    def f(x):
      jac, aux = jacrev(lambda x: (x**3, [x**3]), has_aux=True)(x)
      return aux[0] * jnp.sin(x)

    f2 = lambda x: x**3 * jnp.sin(x)

    self.assertEqual(jacrev(f)(4.), jacrev(f2)(4.))
    self.assertEqual(jit(jacrev(f))(4.), jacrev(f2)(4.))
    self.assertEqual(jit(jacrev(jit(f)))(4.), jacrev(f2)(4.))

  def test_jvp_and_aux_basic(self):
    fun = lambda x: (x**3, [x**2])
    primals, tangents, aux = api.jvp(fun, (3.,), (4.,), has_aux=True)
    expected_primals, expected_tangents = api.jvp(lambda x: x**3, (3.,), (4.,))
    self.assertAllClose(primals, expected_primals, check_dtypes=True)
    self.assertAllClose(tangents, expected_tangents, check_dtypes=True)
    self.assertEqual(aux, [3.**2])

  def test_jvp_mismatched_arguments(self):
    self.assertRaisesRegex(
      TypeError,
      ("primal and tangent arguments to jax.jvp must have the same tree "
       "structure"),
      lambda: api.jvp(lambda x, y: x * y, (np.float32(2),), ()))
    # If primals and tangents must both be tuples or both lists
    self.assertRaisesRegex(
      TypeError,
      ("primal and tangent arguments to jax.jvp must have the same tree "
       "structure"),
      lambda: api.jvp(lambda x, y: x * y, (np.float32(2),), [np.float32(2)]))
    self.assertRaisesRegex(
      TypeError,
      "primal and tangent arguments to jax.jvp do not match.",
      lambda: api.jvp(lambda x: -x, (np.float16(2),), (np.float32(4),)))
    # If primals and tangents are not of the same shape then raise error
    fun = lambda x: x+1
    with self.assertRaisesRegex(
      ValueError, "jvp called with different primal and tangent shapes"):
      api.jvp(fun, (jnp.array([1.,2.,3.]),), (jnp.array([1.,2.,3.,4.]),))
    with self.assertRaisesRegex(
      ValueError, "jvp called with different primal and tangent shapes"):
      api.jvp(fun, (jnp.float32(10.),), (jnp.array([1.,2.,3.], dtype=jnp.float32),))
    with self.assertRaisesRegex(
      ValueError, "jvp called with different primal and tangent shapes"):
      api.jvp(fun, (jnp.array([1.,2.,3.], dtype=jnp.float32),), (jnp.float32(20.),))
    with self.assertRaisesRegex(
      ValueError, "jvp called with different primal and tangent shapes"):
      api.jvp(fun, (jnp.array([1.,2.,3.]),), (20.,))

  def test_jvp_non_tuple_arguments(self):
    def f(x, y): return x + y
    self.assertRaisesRegex(
        TypeError,
        "primal and tangent arguments to jax.jvp must be tuples or lists; found float and tuple.",
        lambda: api.jvp(f, 0., (1.,)))
    self.assertRaisesRegex(
        TypeError,
        "primal and tangent arguments to jax.jvp must be tuples or lists; found tuple and ndarray.",
        lambda: api.jvp(f, (0.,), np.array([1., 2.])))

  def test_vjp_mismatched_arguments(self):
    _, pullback = api.vjp(lambda x, y: x * y, np.float32(3), np.float32(4))
    self.assertRaisesRegex(
      ValueError, "unexpected tree structure",
      lambda: pullback((np.float32(7), np.float32(100))))
    self.assertRaisesRegex(
      ValueError, "unexpected JAX type",
      lambda: pullback(np.float16(42)))

  def test_vjp_bad_cotangent_shape(self):
    x = np.ones((2, 5), dtype=np.float32)
    y = np.ones((5, 3), dtype=np.float32)
    def f_jax(x, y):
      return jnp.matmul(x, y)
    res, pullback = jax.vjp(f_jax, x, y)
    with self.assertRaisesRegex(ValueError, "unexpected JAX type"):
      pullback(np.ones((2, 4), dtype=np.float32))

  def test_jvp_jit_cached(self):
    """Bug in caching in presence of JVP and JIT."""

    def func(x):
      def inner(y):
        return y * x

      # Must have two calls to the inner jit (the second one hits the cache)
      res1 = api.jit(inner)(4.)
      res2 = api.jit(inner)(5.)
      return res1 + res2

    self.assertAllClose((45., 9.), api.jvp(func, (5.,), (1.,)))

  def test_linear_transpose_abstract(self):
    x = types.SimpleNamespace(shape=(3,), dtype=np.dtype(np.float32))
    y = jnp.arange(3, dtype=np.float32)
    transpose_fun = api.linear_transpose(lambda x: 2 * x, x)
    z, = transpose_fun(y)
    self.assertArraysEqual(2 * y, z, check_dtypes=True)

  def test_linear_transpose_integer(self):
    f = lambda x: 2 * x
    transpose = api.linear_transpose(f, 1)
    actual, = transpose(3)
    expected = 6
    self.assertEqual(actual, expected)

  def test_linear_transpose_dce(self):
    # https://github.com/jax-ml/jax/issues/15660
    f = jit(lambda x: (2 * x, x > 0))
    g = lambda x: f(x)[0]
    api.linear_transpose(g, 1.)(1.)

  def test_linear_transpose_error(self):
    with self.assertRaisesRegex(
        TypeError, "linear_transpose only supports"):
      api.linear_transpose(lambda x: 2. * x, 1)
    transpose_fun = api.linear_transpose(lambda x: [x, x], 1.0)
    with self.assertRaisesRegex(TypeError, "cotangent tree does not match"):
      transpose_fun(1.0)

    transpose_fun = api.linear_transpose(lambda x: jnp.stack([x, x]), 1.0)
    with self.assertRaisesRegex(TypeError, "cotangent type does not match"):
      transpose_fun(1.0)

    transpose_fun = api.linear_transpose(lambda x: 1j * x, 1.0)
    with self.assertRaisesRegex(TypeError, "cotangent type does not match"):
      transpose_fun(1.0)

    transpose_fun = api.linear_transpose(lambda x: x, 1.0)
    with self.assertRaisesRegex(TypeError, "cotangent type does not match"):
      transpose_fun(1j)

  def test_linear_transpose_complex(self):
    f = lambda x: (1 + 2j) * x
    transpose = api.linear_transpose(f, 1j)
    actual, = transpose(3 + 4j)
    expected = -5 + 10j
    self.assertEqual(actual, expected)

  def test_linear_transpose_zeros(self):
    f = lambda x: x[0]
    transpose = api.linear_transpose(f, [1., 2.])
    actual, = transpose(3.)
    expected = [3., 0.]
    self.assertEqual(actual, expected)

  def test_complex_grad_raises_error(self):
    self.assertRaises(TypeError, lambda: grad(lambda x: jnp.sin(x))(1 + 2j))

  def test_holomorphic_grad(self):
    out = grad(lambda x: jnp.sin(x), holomorphic=True)(1 + 2j)
    expected = 2.0327230070196656 - 3.0518977991518j
    self.assertAllClose(out, expected, check_dtypes=False)

  def test_nonholomorphic_grad(self):
    zs = 0.5j * np.arange(5) + np.arange(5)

    def f(z):
      return jnp.sum(jnp.cos(jnp.abs(z)))

    ans = grad(f)(zs)
    expected = np.array([ 0.          + 0.j,
                          -0.80430663 + 0.40215331j,
                          -0.70368982 + 0.35184491j,
                           0.1886467  - 0.09432335j,
                           0.86873727 - 0.43436864j])
    self.assertAllClose(ans, expected, check_dtypes=False,
                        atol=jtu.default_gradient_tolerance,
                        rtol=jtu.default_gradient_tolerance)

  def test_complex_output_jacrev_raises_error(self):
    self.assertRaises(TypeError, lambda: jacrev(lambda x: jnp.sin(x))(1 + 2j))

  def test_nonholomorphic_jacrev(self):
    # code based on https://github.com/jax-ml/jax/issues/603
    zs = 0.5j * np.arange(5) + np.arange(5)

    def f(z):
      return jnp.cos(jnp.linalg.norm(2 * z))

    ans = jacrev(f)(zs)
    expected = grad(f)(zs)
    self.assertAllClose(ans, expected)

  @jax.numpy_dtype_promotion('standard')  # Test explicitly exercises implicit dtype promotion.
  def test_heterogeneous_jacfwd(self):
    # See https://github.com/jax-ml/jax/issues/7157
    # See https://github.com/jax-ml/jax/issues/7780
    x = np.array([2.0], dtype=np.float16)
    y = np.array([3.0], dtype=np.float32)
    a = (x, y)

    def f(tup):
      jtu._check_dtypes_match(tup, a)
      x, y = tup
      return x, y, x + y

    actual = jacfwd(f)(a)
    desired = ((np.array(1., dtype=np.float16), np.array(0., dtype=np.float16)),
               (np.array(0., dtype=np.float32), np.array(1., dtype=np.float32)),
               (np.array(1., dtype=np.float32), np.array(1., dtype=np.float32)))
    jtu._check_dtypes_match(actual, desired)
    jtu.check_eq(actual, desired)

  @jax.numpy_dtype_promotion('standard')  # Test explicitly exercises implicit dtype promotion.
  def test_heterogeneous_jacrev(self):
    # See https://github.com/jax-ml/jax/issues/7157
    # See https://github.com/jax-ml/jax/issues/7780
    x = np.array([2.0], dtype=np.float16)
    y = np.array([3.0], dtype=np.float32)
    a = (x, y)

    def f(tup):
      jtu._check_dtypes_match(tup, a)
      x, y = tup
      return x, y, x + y

    actual = jacrev(f)(a)
    desired = ((np.array(1., dtype=np.float16), np.array(0., dtype=np.float32)),
               (np.array(0., dtype=np.float16), np.array(1., dtype=np.float32)),
               (np.array(1., dtype=np.float16), np.array(1., dtype=np.float32)))
    jtu._check_dtypes_match(actual, desired)
    jtu.check_eq(actual, desired)

  def test_heterogeneous_grad(self):
    # See https://github.com/jax-ml/jax/issues/7157
    x = np.array(1.0+1j)
    y = np.array(2.0)
    a = (x, y)

    def f(tup):
      jtu._check_dtypes_match(tup, a)
      x, y = tup
      return jnp.square(jnp.abs(x)) + y

    actual = grad(f)(a)
    desired = (np.array(2 - 2j), np.array(1.))
    jtu._check_dtypes_match(actual, desired)
    jtu.check_eq(actual, desired)

  def test_complex_input_jacfwd_raises_error(self):
    self.assertRaises(TypeError, lambda: jacfwd(lambda x: jnp.sin(x))(1 + 2j))

  def test_legacy_devicearray_repr(self):
    dx = device_put(3.)
    str(dx.item())  # doesn't crash

  def test_devicearray_repr(self):
    x = device_put(jnp.zeros(3))
    _check_instance(self, x)
    repr(x)  # doesn't crash

    x = device_put(jnp.full(3, 1 + 1j))
    _check_instance(self, x)
    repr(x)  # doesn't crash

  def test_devicearray_delete(self):
    x = device_put(1.)
    x.delete()
    self.assertRaisesRegex(RuntimeError, "Array has been deleted.",
                           lambda: repr(x))

  def test_devicearray_block_until_ready(self):
    x = device_put(1.)
    y = x.block_until_ready()
    # Tests mostly that block_until_ready() does not produce an error.
    self.assertTrue(y is x)

  def test_block_until_ready_function(self):
    # Just tests that we don't error...
    pytree = (device_put(1.), np.ones(3))
    pytree = jax.block_until_ready(pytree)
    self.assertAllClose(pytree[0], jnp.array(1.), check_dtypes=False)
    self.assertAllClose(pytree[1], np.ones(3), check_dtypes=False)

  def test_block_until_ready_numpy_arrays(self):
    pytree = (np.ones(1), np.ones(2))
    pytree = jax.block_until_ready(pytree)
    self.assertAllClose(pytree[0], np.ones(1), check_dtypes=False)
    self.assertAllClose(pytree[1], np.ones(2), check_dtypes=False)

  def test_block_until_ready_mixed(self):
    pytree = (device_put(1.), device_put(2.), np.ones(3), 4)
    pytree = jax.block_until_ready(pytree)
    self.assertAllClose(pytree[0], jnp.array(1.), check_dtypes=False)
    self.assertAllClose(pytree[1], jnp.array(2.), check_dtypes=False)
    self.assertAllClose(pytree[2], np.ones(3), check_dtypes=False)
    self.assertEqual(pytree[3], 4)

  def test_copy_to_host_async(self):
    x = device_put(1.)
    y = jax.copy_to_host_async(x)
    # Tests mostly that copy_to_host_async() does not produce an error.
    self.assertIs(y, x)
    self.assertEqual(np.asarray(y), 1.)

  def test_copy_to_host_async_non_array(self):
    # Just tests that we don't error...
    o = object()
    mock_array = unittest.mock.Mock()
    mock_array.copy_to_host_async.return_value = None
    x = [o, 1, 2, 3, mock_array]
    y = jax.copy_to_host_async(x)
    self.assertIs(y, x)
    self.assertEqual(y, [o, 1, 2, 3, mock_array])
    mock_array.copy_to_host_async.assert_called_once()

  def test_copy_to_host_async_does_not_hide_attribute_error(self):
    x = unittest.mock.Mock()
    x.copy_to_host_async.side_effect = AttributeError("foo")
    with self.assertRaisesRegex(AttributeError, "foo"):
      jax.copy_to_host_async(x)

  @jtu.thread_unsafe_test()  # Weakref destruction seems unpredictable with threads
  def test_devicearray_weakref_friendly(self):
    x = device_put(1.)
    y = weakref.ref(x)
    self.assertEqual(y(), 1.)
    del x
    self.assertIsNone(y())

  def test_namedtuple_transparency(self):
    # See https://github.com/jax-ml/jax/issues/446
    Point = collections.namedtuple("Point", ["x", "y"])

    def f(pt):
      return jnp.sqrt(pt.x ** 2 + pt.y ** 2)

    pt = Point(1., 2.)

    f(pt)  # doesn't crash
    g = api.grad(f)(pt)
    self.assertIsInstance(g, Point)

    f_jit = api.jit(f)
    self.assertAllClose(f(pt), f_jit(pt), check_dtypes=False)

  def test_namedtuple_subclass_transparency(self):
    # See https://github.com/jax-ml/jax/issues/806
    Point = collections.namedtuple("Point", ["x", "y"])

    class ZeroPoint(Point):
      def is_zero(self):
        return (self.x == 0) and (self.y == 0)

    pt = ZeroPoint(0., 0.)

    def f(pt):
      return 0. if pt.is_zero() else jnp.sqrt(pt.x ** 2 + pt.y ** 2)

    f(pt)  # doesn't crash
    _ = api.grad(f)(pt)
    self.assertIsInstance(pt, ZeroPoint)

  @parameterized.parameters(1, 2, 3)
  def test_shape_dtype_struct(self, i):
    s = api.ShapeDtypeStruct(shape=(i, 2, 3), dtype=jnp.float32)
    self.assertEqual(s.shape, (i, 2, 3))
    self.assertEqual(s.dtype, jnp.float32)
    self.assertEqual(s.ndim, 3)
    self.assertEqual(s.size, i * 2 * 3)
    self.assertLen(s, i)
    for f in (str, repr):
      self.assertEqual(
          f(s), f"ShapeDtypeStruct(shape=({i}, 2, 3), dtype=float32)")

  def test_shape_dtype_struct_scalar(self):
    s = api.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
    self.assertEmpty(s.shape)
    self.assertEqual(s.size, 1)
    self.assertEqual(s.ndim, 0)
    with self.assertRaisesRegex(TypeError, "len[(][)] of unsized object"):
      _ = len(s)

  def test_shape_dtype_struct_hash(self):
    s1 = api.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.float32)
    s2 = api.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.float32)
    s3 = api.ShapeDtypeStruct(shape=(2, 4), dtype=jnp.float32)
    self.assertEqual(hash(s1), hash(s2))
    self.assertNotEqual(hash(s1), hash(s3))

  def test_shape_dtype_struct_invalid_shape(self):
    with self.assertRaisesRegex(TypeError, "'int' object is not iterable"):
      api.ShapeDtypeStruct(shape=4, dtype='float32')

  def test_shape_dtype_struct_dtype_none(self):
    with self.assertRaisesRegex(ValueError, "dtype must be specified"):
      api.ShapeDtypeStruct(shape=(), dtype=None)

  def test_eval_shape(self):
    def fun(x, y):
      return jnp.tanh(jnp.dot(x, y) + 3.)

    x = jnp.ones((2, 3))
    y = jnp.ones((3, 4))
    out_shape = api.eval_shape(fun, x, y)

    self.assertEqual(out_shape.shape, (2, 4))

  def test_eval_shape_constants(self):
    def fun():
      x = jnp.ones((2, 3))
      y = jnp.ones((3, 4))
      return jnp.tanh(jnp.dot(x, y) + 3.)

    out_shape = api.eval_shape(fun)

    self.assertEqual(out_shape.shape, (2, 4))

  def test_eval_shape_tuple_unpacking(self):
    def fun(x, y):
      a, b = x
      return a + b + y

    x = (jnp.ones(2), jnp.ones(2))
    y = 3.
    out_shape = api.eval_shape(fun, x, y)

    self.assertEqual(out_shape.shape, (2,))

  def test_eval_shape_tuple_itemgetting(self):
    def fun(x, y):
      return x[0] + x[1] + y

    x = (jnp.ones(2), jnp.ones(2))
    y = 3.
    out_shape = api.eval_shape(fun, x, y)

    self.assertEqual(out_shape.shape, (2,))

  def test_eval_shape_output_dict(self):
    def fun(x, y):
      return {'hi': x[0] + x[1] + y}

    x = (jnp.ones(2), jnp.ones(2))
    y = 3.
    out_shape = api.eval_shape(fun, x, y)
    out_shape = jax.tree.map(np.shape, out_shape)

    self.assertEqual(out_shape, {'hi': (2,)})

  def test_eval_shape_shape_error(self):
    def fun(x, y):
      return jnp.tanh(jnp.dot(x, y) + 3.)

    x = jnp.ones((3, 3))
    y = jnp.ones((4, 4))

    self.assertRaises(TypeError, lambda: api.eval_shape(fun, x, y))

  def test_eval_shape_trace_cache_share(self):
    def f(x):
      return x

    inp = np.arange(8)

    with jtu.count_jit_tracing_cache_miss() as count:
      jax.eval_shape(f, inp)
      jax.jit(f)(inp)

    self.assertEqual(count(), 1)

  @jtu.thread_unsafe_test()  # jit cache misses aren't thread safe
  def test_jit_infer_params_cache(self):
    def f(x):
      return x

    f_jit = jax.jit(f)

    def g(x):
      x = f_jit(x)  # noqa: F821
      x = f_jit(x)  # noqa: F821
      return x

    g_jit = jax.jit(g)

    inp = np.arange(8)
    with jtu.count_jit_infer_params_cache_miss() as count:
      g_jit(inp)

    self.assertDictEqual(count, {f: 1, g: 1})
    cache_size = pjit_lib._infer_params_cached.cache_info().currsize
    del count, f, f_jit, g, g_jit
    # Cache should only keep a weak reference to f and g.
    self.assertLess(pjit_lib._infer_params_cached.cache_info().currsize,
                    cache_size, msg=pjit_lib._infer_params_cached.cache_keys())

  def test_eval_shape_out_shardings(self):
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    @partial(jax.jit, out_shardings=s)
    def f(x):
      return x * 2

    inp = np.arange(8)
    out = f.eval_shape(inp)
    self.assertEqual(out.sharding, s)
    self.assertEqual(out.shape, (inp * 2).shape)

  def test_eval_shape_duck_typing(self):
    def fun(A, b, x):
      return jnp.dot(A, x) + b

    class MyArgArray:
      def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = np.dtype(dtype)

    A = MyArgArray((3, 4), jnp.float32)
    b = MyArgArray((1, 5), jnp.float32)
    x = MyArgArray((4, 5), jnp.float32)
    out_shape = api.eval_shape(fun, A, b, x)

    self.assertEqual(out_shape.shape, (3, 5))

  def test_eval_shape_duck_typing2(self):
    # https://github.com/jax-ml/jax/issues/5683
    class EasyDict(dict):
      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    x = EasyDict(shape=(3,), dtype=np.dtype('float32'))
    out_shape = api.eval_shape(lambda x: x, x)  # doesn't crash
    self.assertEqual(out_shape.shape, (3,))

  def test_issue_871(self):
    T = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
    x = jnp.array([1, 2, 3])
    msg = ("linearized function called on tangent values inconsistent with "
           "the original primal values")

    y, f_jvp = api.linearize(jnp.sum, x)
    with self.assertRaisesRegex(ValueError, msg):
      f_jvp(T)

    y, f_jvp = api.linearize(api.jit(jnp.sum), x)
    with self.assertRaisesRegex(ValueError, msg):
      f_jvp(T)

  def test_grad_of_int_errors(self):
    # Errors without allow_int=True
    dfn = grad(lambda x: x ** 2)
    self.assertRaisesRegex(
      TypeError,
      (r"grad requires real- or complex-valued inputs \(input dtype that is a "
       r"sub-dtype of np.inexact\), but got int.*."),
      lambda: dfn(3))

  def test_jvp_of_int_identity(self):
    primals = (1,)
    tangents = (np.zeros(shape=(), dtype=float0),)

    _, out = api.jvp(lambda x: x, primals, tangents)
    self.assertEqual(out, np.zeros(shape=(), dtype=float0))

  def test_jvp_of_int_add(self):
    primals = (2,)
    tangents = (np.zeros(shape=(), dtype=float0),)

    _, out_tangent = api.jvp(lambda x: x+1, primals, tangents)
    self.assertEqual(out_tangent, np.zeros(shape=(), dtype=float0))

  def test_jit_jvp_of_int(self):
    primals = (2,)
    tangents = (np.zeros(shape=(), dtype=float0),)

    _, out_tangent = api.jvp(jax.jit(lambda x: x+1), primals, tangents)
    self.assertEqual(out_tangent, np.zeros(shape=(), dtype=float0))

  def test_jvp_of_convert_element_type(self):
    fun = lambda x: x.astype(np.int32) + 1
    primal, tangent = jax.jvp(fun, (2.,), (1.,))
    self.assertAllClose(primal, np.int32(3))
    self.assertEqual(tangent, np.zeros((), dtype=float0))

  def test_vjp_of_int_index(self):
    primal, fn_vjp = api.vjp(lambda x, i: x[i], np.ones(2)*2, 1)
    tangent_x, tangent_i = fn_vjp(1.)
    self.assertEqual(primal, 2.)
    self.assertAllClose(tangent_x, jnp.array([0., 1.]))
    self.assertEqual(tangent_i, np.zeros(shape=(), dtype=float0))

  def test_vjp_of_int_shapes(self):
    out, fn_vjp = api.vjp(
        lambda x: lax.reshape(x, (2, 2)), np.ones((4, 1), dtype=int))
    tangent, = fn_vjp(np.zeros((2, 2), dtypes.float0))
    self.assertArraysEqual(tangent, np.zeros(shape=(4, 1), dtype=float0))

  def test_jit_vjp_of_int(self):
    primal, fn_vjp = api.vjp(lambda x, y: x+y, 2, 1)
    tangent_x, tangent_i = jax.jit(fn_vjp)(np.zeros((), dtypes.float0))
    self.assertEqual(primal, 3)
    self.assertEqual(tangent_x, np.zeros(shape=(), dtype=float0))
    self.assertEqual(tangent_i, np.zeros(shape=(), dtype=float0))

  def test_vjp_of_int_fulllike(self):
    # Regression test for tangent and cotangent mismatch in convert_element_type
    # transpose rule wrt a ConstVar
    f = lax.full_like
    out, vjp = api.vjp(f, jnp.zeros((2, 2)), 1)
    self.assertAllClose(out, jnp.ones((2, 2)))
    tangent_x, tangent_y = vjp(out)
    self.assertAllClose(tangent_x, jnp.zeros((2, 2)))
    self.assertEqual(tangent_y, np.zeros(shape=(), dtype=float0))

  def test_grad_of_int(self):
    # Need real-valued output, but testing integer input.
    out = api.grad(lambda x: x+0., allow_int=True)(1)
    self.assertEqual(out, np.zeros(shape=(), dtype=float0))

  def test_grad_of_bool(self):
    def cond(pred):
      return lax.cond(pred, lambda _: 1., lambda _: 2., 1.)
    value, grd = api.value_and_grad(cond, allow_int=True)(True)
    self.assertEqual(value, 1.)
    self.assertEqual(grd, np.zeros(shape=(), dtype=float0))

  def test_grad_of_int_index(self):
    grad_x, grad_i = api.grad(lambda x, i: x[i], argnums=(0, 1),
                              allow_int=True)(np.ones(2), 1)
    self.assertAllClose(grad_x, jnp.array([0., 1.]))
    self.assertEqual(grad_i, np.zeros(shape=(), dtype=float0))

  def test_jit_grad_of_int(self):
    grad_f = api.grad(lambda x, i: x[i], argnums=(0, 1), allow_int=True)
    grad_x, grad_i = jax.jit(grad_f)(np.ones(2), 1)
    self.assertAllClose(grad_x, jnp.array([0., 1.]))
    self.assertEqual(grad_i, np.zeros(shape=(), dtype=float0))

  def test_float0_reshape(self):
    # dtype-agnostic operations are supported
    float0_array = jax.grad(lambda x: jnp.sum(x+0.),
                            allow_int=True)(np.ones((2, 4), dtype=int))

    self.assertArraysEqual(float0_array.reshape((4, 2)),
                           np.zeros((4, 2), dtype=float0))
    self.assertArraysEqual(float0_array.transpose(),
                           np.zeros((4, 2), dtype=float0))

  def test_float0_error(self):
    # float0 is incompatible with other dtypes
    float0_array = jax.grad(lambda x: x+0., allow_int=True)(1)
    error_text = "float0s do not support any operations by design"

    with self.assertRaisesRegex(TypeError, error_text):
      # dispatch via Array
      _ = float0_array + jnp.zeros(())

    with self.assertRaisesRegex(TypeError, error_text):
      # dispatch via lax
      _ = lax.add(float0_array, jnp.zeros(()))

  def test_grad_complex_result_errors(self):
    dfn = grad(lambda x: x ** 2 + 1j)
    self.assertRaisesRegex(
      TypeError,
      (r"grad requires real-valued outputs \(output dtype that is a "
       r"sub-dtype of np.floating\), but got complex.*"),
      lambda: dfn(3.))

  def test_holomorphic_grad_of_float_errors(self):
    dfn = grad(lambda x: x ** 2, holomorphic=True)
    self.assertRaisesRegex(
      TypeError,
      (r"grad with holomorphic=True requires inputs with complex dtype, "
       r"but got float.*"),
      lambda: dfn(3.))

  def test_holomorphic_jacrev_of_float_errors(self):
    dfn = jacrev(lambda x: x ** 2, holomorphic=True)
    self.assertRaisesRegex(
      TypeError,
      (r"jacrev with holomorphic=True requires inputs with complex dtype, "
       r"but got float.*"),
      lambda: dfn(3.))

  def test_holomorphic_jacfwd_of_float_errors(self):
    dfn = jacfwd(lambda x: x ** 2, holomorphic=True)
    self.assertRaisesRegex(
      TypeError,
      (r"jacfwd with holomorphic=True requires inputs with complex dtype, "
       r"but got float.*"),
      lambda: dfn(3.))

  def test_jacfwd_of_complex_errors(self):
    dfn = jacfwd(lambda x: x ** 2)
    self.assertRaisesRegex(
      TypeError,
      (r"jacfwd requires real-valued inputs \(input dtype that is a "
       r"sub-dtype of np.floating\), but got complex.*"),
      lambda: dfn(3. + 1j))

  def test_compiler_ir(self):
    # TODO(phawkins): merge these tests with the `xla_computation` tests.
    def e(x):
      return jnp.sin(jnp.cos(x))
    hlo = api.jit(e).lower(2.).compiler_ir(dialect="hlo").as_hlo_text()
    self.assertIn(' cosine', hlo)
    self.assertIn(' sine', hlo)
    stablehlo = str(api.jit(e).lower(2.).compiler_ir(dialect="stablehlo"))
    self.assertIn("stablehlo.cosine", stablehlo)
    self.assertIn("stablehlo.sine", stablehlo)

  def test_concurrent_device_get_and_put(self):
    def f(x):
      for _ in range(100):
        y = jax.device_put(x)
        x = jax.device_get(y)
      return x

    xs = [self.rng().randn(i) for i in range(10)]
    # Make sure JAX backend is initialised on the main thread since some JAX
    # backends install signal handlers.
    jax.device_put(0)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(partial(f, x)) for x in xs]
      ys = [f.result() for f in futures]
    for x, y in zip(xs, ys):
      self.assertAllClose(x, y)

  def test_dtype_from_builtin_types(self):
    for dtype in [bool, int, float, complex]:
      with self.assertNoWarnings():
        x = jnp.array(0, dtype=dtype)
      self.assertEqual(x.dtype, dtypes.canonicalize_dtype(dtype))

  def test_dtype_warning(self):
    # cf. issue #1230
    if config.enable_x64.value:
      raise unittest.SkipTest("test only applies when x64 is disabled")

    def check_warning(warn, nowarn):
      with self.assertWarnsRegex(UserWarning, "Explicitly requested dtype"):
        warn()
      with self.assertNoWarnings():
        nowarn()

    check_warning(lambda: jnp.array([1, 2, 3], dtype="float64"),
                  lambda: jnp.array([1, 2, 3], dtype="float32"))
    check_warning(lambda: jnp.array([1, 2, 3], dtype="float64"),
                  lambda: jnp.array([1, 2, 3], dtype=float))
    check_warning(lambda: jnp.ones(3, dtype=np.float64),
                  lambda: jnp.ones(3))
    check_warning(lambda: jnp.ones(3, dtype=np.float64),
                  lambda: jnp.ones(3, dtype=float))
    check_warning(lambda: jnp.ones_like(3, dtype=np.int64),
                  lambda: jnp.ones_like(3, dtype=np.int32))
    check_warning(lambda: jnp.zeros(3, dtype="int64"),
                  lambda: jnp.zeros(3, dtype="int32"))
    check_warning(lambda: jnp.zeros_like(3, dtype="float64"),
                  lambda: jnp.zeros_like(3, dtype="float32"))
    check_warning(lambda: jnp.full((2, 3), 1, dtype="int64"),
                  lambda: jnp.full((2, 3), 1))
    check_warning(lambda: jnp.ones(3).astype("float64"),
                  lambda: jnp.ones(3).astype("float32"))
    check_warning(lambda: jnp.eye(3, dtype=np.float64),
                  lambda: jnp.eye(3))
    check_warning(lambda: jnp.arange(3, dtype=np.float64),
                  lambda: jnp.arange(3, dtype=np.float32))
    check_warning(lambda: jnp.linspace(0, 3, dtype=np.float64),
                  lambda: jnp.linspace(0, 3, dtype=np.float32))
    check_warning(lambda: jnp.tri(2, dtype="float64"),
                  lambda: jnp.tri(2, dtype="float32"))
    check_warning(lambda: jnp.arange(1).astype("float64"),
                  lambda: jnp.arange(1).astype(float))
    check_warning(lambda: jnp.arange(1.0).astype("int64"),
                  lambda: jnp.arange(1.0).astype(int))

  def test_error_for_invalid_dtype(self):
    err_str = ("Error interpreting argument to .* as an abstract array. The problematic "
               r"value is of type .* and was passed to the function at path args\[1\].")
    with jax.enable_checks(False):
      with self.assertRaisesRegex(TypeError, err_str):
        lax.add(jnp.array(7), np.array("hello"))
    # TODO(dougalm): re-enable checks at the beginning of `bind`. We just
    # need to know which arguments to a generic primitive are ordinary operands vs functions.
    # with jax.enable_checks(True):
    #   with self.assertRaises(AssertionError):
    #     lax.add(jnp.array(7), np.array("hello"))

  def test_vmap_preserves_docstr(self):
    def superfun(a):
      """Does things with stuff."""

    self.assertRegex(api.vmap(superfun).__doc__, "\n".join([
        "Vectorized version of superfun.*",
        "",
        "Original documentation:",
        "",
        superfun.__doc__,
    ]))

  def test_vmap_in_axes_list(self):
    # https://github.com/jax-ml/jax/issues/2367
    dictionary = {'a': 5., 'b': jnp.ones(2)}
    x = jnp.zeros(3)
    y = jnp.arange(3.)

    def f(dct, x, y):
      return dct['a'] + dct['b'] + x + y

    out1 = api.vmap(f, (None, 0, 0))(dictionary, x, y)
    out2 = api.vmap(f, [None, 0, 0])(dictionary, x, y)
    self.assertAllClose(out1, out2)

  def test_vmap_in_axes_non_tuple_error(self):
    # https://github.com/jax-ml/jax/issues/18548
    with self.assertRaisesRegex(
        TypeError,
        re.escape("vmap in_axes must be an int, None, or a tuple of entries corresponding "
                  "to the positional arguments passed to the function, but got {'a': 0}.")):
      jax.vmap(lambda x: x['a'], in_axes={'a': 0})

  def test_vmap_in_axes_wrong_length_tuple_error(self):
    # https://github.com/jax-ml/jax/issues/18548
    with self.assertRaisesRegex(
        ValueError,
        re.escape("vmap in_axes must be an int, None, or a tuple of entries corresponding to the "
                  "positional arguments passed to the function, but got len(in_axes)=2, len(args)=1")):
      jax.vmap(lambda x: x['a'], in_axes=(0, {'a': 0}))({'a': jnp.zeros((3, 3))})

  def test_vmap_in_axes_tree_prefix_error(self):
    # https://github.com/jax-ml/jax/issues/795
    value_tree = jnp.ones(3)
    self.assertRaisesRegex(
        ValueError,
        "vmap in_axes specification must be a tree prefix of the corresponding "
        r"value, got specification \(\[0\],\) for value tree "
        + re.escape(f"{jax.tree.structure((value_tree,))}."),
        lambda: api.vmap(lambda x: x, in_axes=([0],))(value_tree)
    )

  def test_vmap_in_axes_leaf_types(self):
    with self.assertRaisesRegex(
        TypeError, r"vmap in_axes must be an int, None, or .*"):
      api.vmap(lambda x: x, in_axes=(jnp.array([1., 2.]),))(jnp.array([1., 2.]))

  def test_vmap_out_axes_leaf_types(self):
    with self.assertRaisesRegex(
        TypeError, r"vmap out_axes must be an int, None, or .*"):
      api.vmap(lambda x: x, out_axes=(jnp.array([1., 2.]),))(jnp.array([1., 2.]))

  def test_vmap_unbatched_object_passthrough_issue_183(self):
    # https://github.com/jax-ml/jax/issues/183
    fun = lambda f, x: f(x)
    vfun = api.vmap(fun, (None, 0))
    ans = vfun(lambda x: x + 1, jnp.arange(3))
    self.assertAllClose(ans, np.arange(1, 4), check_dtypes=False)

  def test_vmap_mismatched_keyword(self):
    # https://github.com/jax-ml/jax/issues/10193
    @jax.vmap
    def f(x, y):
      return x + y

    with self.assertRaisesRegex(
        ValueError,
        "vmap got inconsistent sizes for array axes to be mapped:\n"
        r"  \* one axis had size 1: axis 0 of argument x of type int32\[1\];"
        "\n"
        r"  \* one axis had size 2: axis 0 of kwargs\['y'\] of type int32\[2\]"):
      f(jnp.array([1], 'int32'), y=jnp.array([1, 2], 'int32'))

  def test_vmap_mismatched_axis_sizes_error_message_issue_705(self):
    # https://github.com/jax-ml/jax/issues/705
    def h(a, b):
      return jnp.sum(a) + jnp.sum(b)

    X = self.rng().randn(10, 4).astype('float32')
    U = self.rng().randn(10, 2).astype('float32')

    with self.assertRaisesRegex(
        ValueError,
        "vmap got inconsistent sizes for array axes to be mapped:\n"
        r"  \* one axis had size 10: axis 0 of argument a of type float32\[10,4\];""\n"
        r"  \* one axis had size 2: axis 1 of argument b of type float32\[10,2\]"):
      api.vmap(h, in_axes=(0, 1))(X, U)

    with self.assertRaisesRegex(
        ValueError,
        "vmap got inconsistent sizes for array axes to be mapped:\n"
        r"  \* most axes \(2 of them\) had size 10, e.g. axis 0 of argument x "
        r"of type float32\[10,4\];" "\n"
        r"  \* one axis had size 2: axis 1 of argument y of type float32\[10,2\]"):
      api.vmap(lambda x, y, z: None, in_axes=(0, 1, 0))(X, U, X)

    with self.assertRaisesRegex(
        ValueError,
        "vmap got inconsistent sizes for array axes to be mapped:\n"
        r"  \* most axes \(2 of them\) had size 2, e.g. axis 1 of argument b\[0\] "
        r"of type float32\[10,2\];" "\n"
        r"  \* one axis had size 10: axis 0 of argument a of type float32\[10,4\]"):
      api.vmap(h, in_axes=(0, 1))(X, [U, U])

    error = (r"vmap was requested to map its argument along axis 0, which "
             r"implies that its rank should be at least 1, but is only 0 "
             r"\(its shape is \(\)\)")
    with self.assertRaisesRegex(ValueError, error):
      # The mapped inputs cannot be scalars
      api.vmap(lambda x: x)(1.)

    with self.assertRaisesRegex(
        ValueError, "vmap must have at least one non-None value in in_axes"):
      # If the output is mapped, there must be a non-None in_axes
      api.vmap(lambda x: x, in_axes=None)(jnp.array([1., 2.]))

    error = (r"vmap was requested to map its argument along axis 1, which "
             r"implies that its rank should be at least 2, but is only 1 "
             r"\(its shape is \(2,\)\)")
    with self.assertRaisesRegex(ValueError, error):
      api.vmap(lambda x: x, in_axes=1)(jnp.array([1., 2.]))

    # Error is: TypeError: only integer scalar arrays can be converted to a scalar index
    with self.assertRaisesRegex(
        ValueError,
        "vmap out_axes specification must be a tree prefix of the "
        "corresponding value.*"):
      api.vmap(lambda x: x, in_axes=0, out_axes=(2, 3))(jnp.array([1., 2.]))

    with self.assertRaisesRegex(
        ValueError,
        r"vmap has mapped output \(axis_name='foo'\) but out_axes is None"):
      # If the output is mapped (user-named axis), then there must be some
      # out_axes specified.
      api.vmap(lambda x: x, out_axes=None, axis_name="foo")(jnp.array([1., 2.]))

    with self.assertRaisesRegex(
        ValueError,
        "at vmap out_axes"):
      # If the output is mapped (unnamed axis), then there must be some out_axes
      # specified.
      api.vmap(lambda x: x, out_axes=None)(jnp.array([1., 2.]))

  def test_vmap_structured_in_axes(self):

    A, B, C, D = 2, 3, 4, 5
    K = 6  # batch size
    x = np.ones((K, A, B))  # batch axis in different locations
    y = np.ones((B, K, C))
    z = np.ones((C, D, K))

    def foo(tree_arg):
      x, (y, z) = tree_arg
      return jnp.dot(x, jnp.dot(y, z))

    tree = (x, (y, z))
    vfoo = api.vmap(foo, in_axes=((0, (1, 2)),))
    self.assertEqual(vfoo(tree).shape, (6, 2, 5))

    Point = collections.namedtuple("Point", ["x", "y"])
    tree = (x, Point(y, z))
    vfoo = api.vmap(foo, in_axes=((0, Point(1, 2)),))
    self.assertEqual(vfoo(tree).shape, (6, 2, 5))

    def foo(tree_arg):
      x, dct = tree_arg
      y, z = dct['a'], dct['b']
      return jnp.dot(x, jnp.dot(y, z))

    tree = (x, {'a': y, 'b': z})
    vfoo = api.vmap(foo, in_axes=((0, {'a': 1, 'b': 2}),))
    self.assertEqual(vfoo(tree).shape, (6, 2, 5))

    tree = (x, collections.OrderedDict([('a', y), ('b', z)]))
    vfoo = api.vmap(
        foo, in_axes=((0, collections.OrderedDict([('a', 1), ('b', 2)])),))
    self.assertEqual(vfoo(tree).shape, (6, 2, 5))

  def test_vmap_in_axes_bool_error(self):
    # https://github.com/jax-ml/jax/issues/6372
    with self.assertRaisesRegex(TypeError, "must be an int"):
      api.vmap(lambda x: x, in_axes=False)(jnp.zeros(3))

  def test_pmap_in_axes_bool_error(self):
    # https://github.com/jax-ml/jax/issues/6372
    with self.assertRaisesRegex(TypeError, "must be an int"):
      api.pmap(lambda x: x, in_axes=False)(jnp.zeros(1))

  def test_vmap_empty_arguments(self):
    with self.assertRaisesRegex(
        ValueError,
        "vmap wrapped function must be passed at least one argument "
        r"containing an array, got empty \*args=\(\{\},\) and \*\*kwargs=\{\}"):
      api.vmap(lambda x: x)({})

  def test_pmap_empty_arguments(self):
    with self.assertRaisesRegex(
        ValueError,
        "pmap wrapped function must be passed at least one argument "
        r"containing an array, got empty \*args=\(\{\},\) and \*\*kwargs=\{\}"):
      api.pmap(lambda x: x)({})

  def test_pmap_global_cache(self):
    def f(x, y):
      return x, y

    x = np.ones((1, 1, 1))

    # All defaults
    with jtu.assert_num_jit_and_pmap_compilations(1):
      for _ in range(2):
        api.pmap(f)(x, x)

    # With axis name
    with jtu.assert_num_jit_and_pmap_compilations(1):
      for _ in range(2):
        api.pmap(f, 'i')(x, x)

    # With in_axes and out_axes
    for x_in, y_in, x_out, y_out in it.product(*((0, 1, 2) for _ in range(4))):
      with jtu.assert_num_jit_and_pmap_compilations(1):
        for _ in range(2):
          api.pmap(f, 'i', in_axes=(x_in, y_in), out_axes=(x_out, y_out))(x, x)

    # Forward-mode AD on the outside
    with jtu.assert_num_jit_and_pmap_compilations(1):
      for _ in range(2):
        api.jvp(api.pmap(f), (x, x), (x, x))

    # Reverse-mode AD on the outside. One compilation for forward, one for backward.
    with jtu.assert_num_jit_and_pmap_compilations(2):
      for _ in range(2):
        api.vjp(api.pmap(f), x, x)[1]((x, x))

  def test_device_array_repr(self):
    rep = jnp.ones(()) + 1.
    self.assertStartsWith(repr(rep), 'Array')

  def test_device_array_hash(self):
    rep = jnp.ones((1,)) + 1.
    _check_instance(self, rep)
    self.assertNotIsInstance(rep, collections.abc.Hashable)
    with self.assertRaisesRegex(TypeError, 'unhashable type'):
      hash(rep)

  def test_grad_without_enough_args_error_message(self):
    # https://github.com/jax-ml/jax/issues/1696
    def f(x, y): return x + y
    df = api.grad(f, argnums=0)
    self.assertRaisesRegex(
        TypeError,
        "differentiating with respect to argnums=0 requires at least 1 "
        "positional arguments to be passed by the caller, but got only 0 "
        "positional arguments.",
        lambda: partial(df, x=0.)(y=1.))

  def test_grad_object_array_error(self):
    x = np.array([1, 2, 3], dtype=object)
    with self.assertRaisesRegex(TypeError, ".*is not a valid JAX type"):
      jax.grad(lambda x: x)(x)

  @jtu.thread_unsafe_test()  # logging isn't thread-safe
  def test_jit_compilation_time_logging(self):
    @api.jit
    def f(x):
      return x * 2

    # make sure some initial warnings & cached operations already happen.
    f(jnp.ones(2))

    prev_level = logging.get_verbosity()
    try:
      logging.set_verbosity('DEBUG')
      with self.assertLogs(level=logging.DEBUG) as l:
        f(2.)
    finally:
      logging.set_verbosity(prev_level)
    self.assertGreaterEqual(len(l.output), 3)  # 3 lines
    self.assertTrue(any('Finished tracing' in line for line in l.output))
    self.assertTrue(any('Compiling f' in line for line in l.output))
    self.assertTrue(any('Finished XLA compilation' in line for line in l.output))

  def test_grad_of_jit_compilation_caching(self):
    if not hasattr(self, "assertLogs"):
      raise unittest.SkipTest("test requires assertLogs (python 3)")

    # make sure some initial warnings & cached operations already happen.
    api.grad(api.jit(lambda x: x))(1.0)

    @api.jit
    def f(x):
      return jnp.sin(x)

    prev_level = logging.get_verbosity()
    try:
      logging.set_verbosity('DEBUG')
      with self.assertLogs(level=logging.DEBUG) as l:
        ans1 = api.grad(f)(2.)
        ans2 = api.grad(f)(3.)
    finally:
      logging.set_verbosity(prev_level)
    self.assertGreaterEqual(len(l.output), 2 * 3)  # one for fwd, one for bwd, 3 lines each
    self.assertAllClose(ans1, np.cos(2.), check_dtypes=False)
    self.assertAllClose(ans2, np.cos(3.), check_dtypes=False)

  def test_grad_of_jit_compilation_caching2(self):
    # Like the above test, but instead of logging use our compile counters.

    # make sure some initial convert element type operations are pre-cached.
    api.grad(api.jit(lambda x: x))(1.0)

    @api.jit
    def f(x):
      return jnp.sin(x)

    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      _  = jax.grad(f)(3.)
    self.assertEqual(count(), 2)  # one for fwd, one for bwd

    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      _  = jax.grad(f)(3.)
      _  = jax.grad(f)(4.)
    self.assertEqual(count(), 0)  # cache hits on both fwd and bwd

  def test_grad_does_not_unflatten_tree_with_none(self):
    # https://github.com/jax-ml/jax/issues/7546
    class CustomNode(list):
      pass

    def unflatten(unused_aux_data, children):
      self.assertIsNotNone(children[0])
      return CustomNode(children)

    tree_util.register_pytree_node(CustomNode, lambda x: (x, None), unflatten)
    grad(lambda x: x[0])(CustomNode([0.]))

  def test_trivial_computations(self):
    x = jnp.array([1, 2, 3])
    y = api.jit(lambda x: x)(x)
    self.assertNotEqual(x.unsafe_buffer_pointer(), y.unsafe_buffer_pointer())

    z1, z2 = api.jit(lambda x: (x, x))(x)
    self.assertNotEqual(z1.unsafe_buffer_pointer(), z2.unsafe_buffer_pointer())

    x1, x2 = jnp.array([1, 2]), jnp.array([2, 3])
    z1, z2, z3 = api.jit(lambda x, y: (y, 1, x))(x1, x2)
    self.assertNotEqual(z1.unsafe_buffer_pointer(), x2.unsafe_buffer_pointer())
    self.assertNotEqual(z3.unsafe_buffer_pointer(), x1.unsafe_buffer_pointer())
    self.assertEqual(z2, 1)

  @jtu.thread_unsafe_test()  # monkey-patching mlir.jaxpr_subcomp isn't thread-safe
  def test_nested_jit_hoisting(self):
    @api.jit
    def f(x, y):
      z = 2 * x
      return y + z, 3

    @api.jit
    def g(x):
      return f(2, x)

    mlir_jaxpr_subcomp = mlir.jaxpr_subcomp

    jaxprs = []
    def mlir_jaxpr_subcomp_and_collect(c, jaxpr, *args, **kwargs):
      jaxprs.append(jaxpr)
      return mlir_jaxpr_subcomp(c, jaxpr, *args, **kwargs)

    try:
      mlir.jaxpr_subcomp = mlir_jaxpr_subcomp_and_collect
      ans = g(3)
    finally:
      mlir.jaxpr_subcomp = mlir_jaxpr_subcomp

    self.assertEqual(ans, (7, 3))
    self.assertLen(jaxprs, 2)
    outer_jaxpr, inner_jaxpr = jaxprs

    self.assertLen(outer_jaxpr.eqns, 1)
    prim_name = 'pjit'
    jaxpr_param = 'jaxpr'
    self.assertEqual(outer_jaxpr.eqns[0].primitive.name, f'{prim_name}')
    subjaxpr_1 = outer_jaxpr.eqns[0].params[f"{jaxpr_param}"]
    self.assertEqual(str(subjaxpr_1), str(inner_jaxpr))
    self.assertLen(inner_jaxpr.eqns, 2)
    self.assertEqual(inner_jaxpr.eqns[-2].primitive.name, 'mul')
    self.assertEqual(inner_jaxpr.eqns[-1].primitive.name, 'add')

  @jtu.thread_unsafe_test()  # count_primitive_compiles isn't thread-safe
  def test_primitive_compilation_cache(self):
    with jtu.count_primitive_compiles() as count:
      lax.add(1, 2)
      lax.add(2, 3)
    self.assertEqual(count(), 1)

  def test_arange_jit(self):
    # see https://github.com/jax-ml/jax/issues/553
    def fun(x):
      r = jnp.arange(x.shape[0])[x]
      return r

    jit(fun)(jnp.array([0, 1, 2], dtype=jnp.int32))  # doesn't crash

  def helper_save_tracer(self, x):
    self._saved_tracer = x
    return x

  def test_escaped_tracers_different_top_level_traces(self):
    api.jit(self.helper_save_tracer)(0.)
    with self.assertRaisesRegex(
        UnexpectedTracerError, "Encountered an unexpected tracer"):
      api.jit(lambda x: self._saved_tracer)(0.)

  def test_escaped_tracers_cant_lift_sublevels(self):
    api.jit(self.helper_save_tracer)(0.)
    with self.assertRaisesRegex(
        UnexpectedTracerError,
        re.compile(
          "Encountered an unexpected tracer",
          re.DOTALL)):
      api.jit(lambda x: x)(self._saved_tracer)

  @unittest.skip # TODO(dougalm): rethink what this should do under stackless
  def test_escaped_tracers_tracer_from_higher_level(self):
    api.grad(self.helper_save_tracer)(0.)
    with self.assertRaises(UnexpectedTracerError):
      api.grad(lambda x: x)(self._saved_tracer)

  def test_escaped_tracers_incompatible_sublevel(self):
    def func1(x):
      api.jit(self.helper_save_tracer)(0.)
      # Use the tracer
      return x + self._saved_tracer
    with self.assertRaisesRegex(
        UnexpectedTracerError,
        re.compile("Encountered an unexpected tracer",
                   re.DOTALL)):
      api.jit(func1)(2.)

  def test_escaped_tracers_cant_lift(self):
    def func1(x):
      api.grad(self.helper_save_tracer)(0.)
      return x + self._saved_tracer
    with self.assertRaisesRegex(
        UnexpectedTracerError,
        re.compile("unexpected tracer")):
      api.grad(func1)(2.)

  def test_escaped_tracers_not_among_input_tracers(self):
    def func1(x):
      api.grad(self.helper_save_tracer)(x)
      # Use the tracer
      return x + self._saved_tracer

    msg = "Encountered an unexpected tracer"
    with self.assertRaisesRegex(
        UnexpectedTracerError, re.compile(msg, re.DOTALL)):
      api.jit(func1)(2.0)

  def test_escaped_tracer_omnistaging(self):
    count = 1

    @jit
    def f():
      nonlocal count
      count = jnp.add(count, 1)
    f()  # leaked a tracer! but currently undetected

    def f(x, c):
      jnp.add(count, 1)
      return None, None

    @jit
    def g():
      lax.scan(f, None, None, length=2)

    with self.assertRaisesRegex(UnexpectedTracerError,
                                "was created on line"):
      g()

  def test_escaped_tracer_omnistaging_top_trace(self):
    count = 1

    def f(_, __):
      nonlocal count
      count = jnp.add(count, 1)
      return None, None

    lax.scan(f, None, None, length=2)  # leaked a tracer! (of level 1!)

    with self.assertRaisesRegex(UnexpectedTracerError,
                                "was created on line"):
      # The following call will try and raise the ones array to the count tracer
      # level, which is no longer live.
      jax.jit(jnp.add)(jnp.ones(()), count)


  def test_escaped_tracer_shape_dtype(self):
    with self.assertRaisesRegex(core.UnexpectedTracerError, r"int32\[4,3\]"):
      jax.jit(self.helper_save_tracer)(jnp.ones((4, 3), dtype=jnp.int32))
      _ = self._saved_tracer+1

  def test_pmap_static_kwarg_error_message(self):
    # https://github.com/jax-ml/jax/issues/3007
    def f(a, b):
      return a + b

    g = jax.pmap(f, static_broadcasted_argnums=(1,))

    msg = (r"pmapped function has static_broadcasted_argnums=\(1,\) but was "
           r"called with only 1 positional argument. All static broadcasted "
           r"arguments must be passed positionally.")
    with self.assertRaisesRegex(ValueError, msg):
      g(jnp.ones((1, 1)), b=1)

  def test_vmap_unmapped_last(self):
    @partial(jax.vmap, out_axes=-1)
    def f(x):
      return np.zeros((2,))
    f(np.zeros((5,)))

  # TODO(jakevdp): re-enable this if possible.
  @unittest.skipIf(True, "broken by convert_element_type change.")
  def test_xla_constant_dedup(self):
    y = np.array([7, 14], dtype=np.float32)
    def f(x):
      return x + y + y

    x = np.array([1, 2], dtype=np.float32)
    hlo_lines = jax.jit(f).lower(x).as_text('hlo').split('\n')
    hlo_lines = {s.strip() for s in hlo_lines}
    self.assertIn('constant.1 = f32[2]{0} constant({7, 14})', hlo_lines)
    self.assertNotIn('constant.2 = f32[2]{0} constant({7, 14})', hlo_lines)

  def test_eval_context(self):
    @jit
    def f():
      with core.eval_context():
        assert jnp.add(1, 1) == 2

    f()  # doesn't crash

  def test_linearize_aux(self):
    def fn(x):
      return x * 2 - 3, x > 0

    f, lin_fn, aux = api.linearize(fn, 3.4, has_aux=True)
    tang = lin_fn(5.)

    self.assertAllClose(f, 3.8)
    self.assertAllClose(tang, 10.)
    self.assertEqual(aux, True)

  def test_linearize_aval_error(self):
    # https://github.com/jax-ml/jax/issues/4622
    f = lambda x: x

    # these should not error
    _, f_jvp = api.linearize(f, 1.)
    f_jvp(1.)
    _, f_jvp = api.linearize(f, np.ones(2, np.int32))
    f_jvp(np.zeros(2, float0))

    # these should error
    _, f_jvp = api.linearize(f, 1.)
    with self.assertRaisesRegex(ValueError, "tangent values inconsistent"):
      f_jvp(1)
    _, f_jvp = api.linearize(f, np.ones(2, np.int32))
    with self.assertRaisesRegex(ValueError, "tangent values inconsistent"):
      f_jvp(np.ones(2, np.int32))

  def test_grad_of_token_consuming_primitive(self):
    # https://github.com/jax-ml/jax/issues/5463
    tokentest_p = core.Primitive("tokentest")
    tokentest_p.def_impl(partial(xla.apply_primitive, tokentest_p))
    tokentest_p.def_abstract_eval(lambda x, y: x)
    mlir.register_lowering(tokentest_p, lambda ctx, x, y: [x])
    ad.defjvp(tokentest_p, (lambda g, x, token: x), None)

    token = jax.lax.create_token(123)
    arr = jnp.ones((3, 2))
    res, vjp_fun = jax.vjp(lambda x: tokentest_p.bind(x, token), arr)
    # Should not crash.
    vjp_fun(arr)

  def test_jit_returning_token(self):
    x = jax.jit(jax.lax.create_token)(1.0)
    self.assertIsInstance(x, core.Token)

  def test_jit_capturing_token(self):
    tok = jax.lax.create_token()
    _, y = jax.jit(lambda x: (x + 2, tok))(7)
    self.assertIsInstance(y, core.Token)

  def test_leak_checker_catches_a_jit_leak(self):
    with jax.checking_leaks():
      lst = []

      @jit
      def f(x):
        lst.append(x)
        return x

      with self.assertRaisesRegex(Exception, r"Leaked"):
        f(3)

  def test_leak_checker_catches_a_pmap_leak(self):
    with jax.checking_leaks():
      lst = []

      @api.pmap
      def f(x):
        lst.append(x)
        return x

      with self.assertRaisesRegex(Exception, r"Leaked"):
        f(np.ones(1))

  def test_leak_checker_catches_a_grad_leak(self):
    with jax.checking_leaks():
      lst = []

      def f(x):
        lst.append(x)
        return x

      with self.assertRaisesRegex(Exception, r"Leaked trace"):
        api.grad(f)(3.)

  def test_leak_checker_avoids_false_positives(self):
    with jax.checking_leaks():
      api.vmap(lambda x: x)(np.arange(3.))  # doesn't crash

      @jit
      def f(x):
        return x
      f(3)  # doesn't crash
      api.vmap(f)(np.arange(3))  # doesn't crash
      api.grad(f)(3.)  # doesn't crash

      @api.pmap
      def f(x):
        return x
      f(np.ones(1))  # doesn't crash
      api.vmap(f)(np.ones((1, 1)))  # doesn't crash

  def test_leak_checker_catches_a_scan_leak(self):
    with jax.checking_leaks():
      lst = []

      to_scan = lambda c, x: (lst.append(c) or jnp.sin(c), None)

      with self.assertRaisesRegex(Exception, r"Leaked trace"):
        lax.scan(to_scan, 1., np.arange(3.))

  def test_leak_checker_avoids_false_positives_scan(self):
    with jax.checking_leaks():
      to_scan = lambda c, x: (jnp.sin(c), None)
      lax.scan(to_scan, 1., np.arange(3.))  # doesn't crash

  def test_leak_checker_avoids_false_positives_scan_jvp(self):
    with jax.checking_leaks():
      to_scan = lambda c, x: (c, None)

      def f(x):
        lax.scan(to_scan, x, None, length=1)
      api.jvp(f, (3.,), (1.,))  # doesn't crash

  def test_leak_checker_avoids_false_positives_scan_vmap(self):
    with jax.checking_leaks():
      to_scan = lambda c, _: (1., None)

      @api.vmap
      def f(x):
        lax.scan(to_scan, x, None, length=1)
      f(np.arange(5.))  # doesn't crash

  def test_leak_checker_avoids_false_positives_scan_vmap_2(self):
    with jax.checking_leaks():
      to_scan = lambda c, _: (c, None)

      @api.vmap
      def f(x):
        lax.scan(to_scan, x, None, length=1)
      f(np.arange(5.))  # doesn't crash

  def test_leak_checker_catches_a_sublevel_leak(self):
    with jax.checking_leaks():
      @jit
      def f(x):
        lst = []
        @jit
        def g(x):
          lst.append(x)
          return x

        x = g(x)
        return x

      msg = r'Leaked trace DynamicJaxprTrace'
      with self.assertRaisesRegex(Exception, f"{msg}"):
        f(3)

  def test_leak_checker_avoids_false_positive_custom_jvp(self):
    # see https://github.com/jax-ml/jax/issues/5636
    with jax.checking_leaks():
      @jax.custom_jvp
      def t(y):
        return y

      def t_jvp(p, t):
        pass

      t.defjvp(t_jvp)

      @jit
      def s(y):
        return t(y)
      s(3)  # doesn't crash

  def test_leak_checker_internal_error(self):
    def apply_fn(inp):
      fn = jax.checkpoint(lambda x: jax.nn.relu(1.0 * x))
      return jax.vjp(fn, inp)

    with jax.check_tracer_leaks():
      jax.jit(apply_fn)(1.0)  # don't crash

  def test_leak_checker_reference_chain(self):
    class A:
      def __init__(self, dct):
        self.dct = dct

    a = A({})
    x = jnp.arange(3)

    def sketch(x):
      def foo():
        return x
      a.dct['hi'] = [foo]
      return x

    # TODO(mattjj): full test msg below fails (harmlessly) on CI, investigate
    msg = (
        r"This BatchTracer with object id [0-9]+ was created on line:\n"
        r"  .*\n"
        r"<BatchTracer [0-9]+> is referred to by"
    )

    # msg = (
    #     r"This BatchTracer with object id [0-9]+ was created on line:\n"
    #     r"  .*\n"
    #     r"<BatchTracer [0-9]+> is referred to by <function [0-9]+> \(foo\) "
    #     r"closed-over variable x\n"
    #     r"<function [0-9]+> is referred to by <list [0-9]+>\[0\]\n"
    #     r"<list [0-9]+> is referred to by <dict [0-9]+>\['hi'\]\n"
    #     r"<dict [0-9]+> is referred to by <A [0-9]+>\.dct\n"
    # )

    with jax.check_tracer_leaks():
      with self.assertRaisesRegex(Exception, msg):
        jax.vmap(sketch)(x)

  def test_default_backend(self):
    first_local_device = jax.local_devices()[0]
    self.assertEqual(first_local_device.platform, jax.default_backend())

  @jtu.skip_on_devices("cpu")
  def test_default_device(self):
    system_default_devices = jnp.add(1, 1).devices()
    self.assertLen(system_default_devices, 1)
    test_device = jax.devices("cpu")[-1]

    # Sanity check creating array using system default device
    self.assertEqual(jnp.ones(1).devices(), system_default_devices)

    # Create array with default_device set
    with jax.default_device(test_device):
      # Hits cached primitive path
      self.assertEqual(jnp.ones(1).devices(), {test_device})
      # Uncached
      self.assertEqual(jnp.zeros((1, 2)).devices(), {test_device})

    # Test that we can reset to system default device
    self.assertEqual(jnp.ones(1).devices(), system_default_devices)

  def test_dunder_jax_array(self):
    # https://github.com/jax-ml/jax/pull/4725

    @partial(jax.tree_util.register_dataclass,
             data_fields=['jax_val'],
             meta_fields=[])
    class AlexArray:
      def __init__(self, jax_val):
        self.jax_val = jax_val
      def __jax_array__(self):
        return self.jax_val
      dtype = property(lambda self: self.jax_val.dtype)
      shape = property(lambda self: self.jax_val.shape)

    x = AlexArray(jnp.array([1., 2., 3.]))

    y = jax.jit(lambda x: x)(x)
    self.assertIsInstance(x, AlexArray)
    self.assertArraysEqual(jnp.asarray(x), jnp.asarray(y))

    y = jnp.sin(x)
    self.assertAllClose(y, jnp.sin(jnp.array([1., 2., 3.])))
    y = api.grad(api.jit(lambda x: jnp.sin(x).sum()))(x)
    self.assertIsInstance(y, AlexArray)
    self.assertAllClose(jnp.asarray(y), jnp.cos(jnp.array([1., 2., 3.])))

    x = AlexArray(jnp.array([[1., 2., 3.]]))
    y = api.pmap(jnp.sin)(x)
    self.assertAllClose(y, jnp.sin(jnp.array([[1., 2., 3.]])))

    x = jnp.array(1)
    a = AlexArray(x)
    for f in [jnp.isscalar, jnp.size, jnp.shape, jnp.dtype]:
      self.assertEqual(f(x), f(a))

    x = AlexArray(jnp.array(1))
    a1 = jnp.array(x)
    self.assertAllClose(1, a1)

    a2 = jnp.array(((x, x), [x, x]))
    self.assertAllClose(np.array(((1, 1), (1, 1))), a2)

  def test_dunder_jax_array_warnings(self):
    class AlexArray:
      def __init__(self, jax_val):
        self.jax_val = jax_val
      def __jax_array__(self):
        return self.jax_val

    f = jax.jit(lambda x: x)
    a = AlexArray(jnp.arange(4))
    msg = r"Triggering of __jax_array__\(\) during abstractification is deprecated."
    with self.assertDeprecationWarnsOrRaises('jax-abstract-dunder-array', msg):
      f(a)

  @jtu.thread_unsafe_test()  # count_jit_tracing_cache_miss() isn't thread-safe
  def test_eval_shape_weak_type(self):
    # https://github.com/jax-ml/jax/issues/23302
    arr = jax.numpy.array(1)

    def f(x):
      return jax.numpy.array(x)

    with jtu.count_jit_tracing_cache_miss() as count:
      jax.eval_shape(f, 1)
      out = jax.eval_shape(f, 1)

    self.assertEqual(count(), 1)
    self.assertTrue(out.weak_type)
    self.assertEqual(out.weak_type, arr.weak_type)

  def test_dunder_jax_array_bug(self):
    @jax.tree_util.register_pytree_node_class
    class A:
      x: jax.Array

      def __init__(self, x: jax.Array):
        self.x = x

      def tree_flatten(self):
        return ((self.x,), None)

      @classmethod
      def tree_unflatten(cls, _, children):
        x, = children
        return cls(x)

      def __jax_array__(self) -> jax.Array:
        return self.x

      ndim = property(operator.attrgetter('x.ndim'))
      dtype = property(operator.attrgetter('x.dtype'))
      shape = property(operator.attrgetter('x.shape'))

    a = A(jnp.ones((3, 3)))
    jnp.asarray(a)  # don't crash

    f = jax.jit(jnp.matmul)
    f(a, a)  # don't crash

  def test_constant_handler_mro(self):
    # https://github.com/jax-ml/jax/issues/6129

    class Foo(enum.IntEnum):
      bar = 1

    @api.pmap
    def f(_):
      return Foo.bar

    ans = f(jnp.arange(1))  # doesn't crash
    expected = jnp.arange(1) + 1
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters([
      {"testcase_name": f"{dtype.__name__}", "dtype": dtype}
      for dtype in jtu.dtypes.all])
  def test_constant_handlers(self, dtype):
    # https://github.com/jax-ml/jax/issues/9380
    @jax.jit
    def f():
      return jnp.exp(dtype(0))
    f()  # doesn't error

  def test_vmap_make_jaxpr_close_over_tracer(self):
    def run(inp):
      def f(x, y):
        return x + y
      g = lambda x: f(x, inp)
      jaxpr = jax.make_jaxpr(g)(1)
      return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1)

    jax.vmap(run)(jnp.arange(2))  # doesn't crash

  def test_large_python_ints(self):
    with self.assertRaises(OverflowError):
      jnp.multiply(2 ** 100, 3.)

    out = lax.convert_element_type(2 ** 100, jnp.float32)  # doesn't crash
    self.assertArraysEqual(out, np.float32(2 ** 100))

  def test_dot_precision_context_manager(self):
    x = jnp.zeros((2, 2))

    with jax.default_matmul_precision(None):
      jnp.dot(x, x)  # doesn't crash
      jaxpr = jax.make_jaxpr(jnp.dot)(x, x)
    # self.assertIn('precision=None', str(jaxpr))
    self.assertIs(jaxpr.jaxpr.eqns[0].params['precision'], None)

    with jax.default_matmul_precision("bfloat16"):
      x @ x  # doesn't crash
      jaxpr = jax.make_jaxpr(op.matmul)(x, x)
    self.assertIn('Precision.DEFAULT', str(jaxpr))

    with jax.default_matmul_precision("tensorfloat32"):
      jnp.dot(x, x)  # doesn't crash
      jaxpr = jax.make_jaxpr(jnp.dot)(x, x)
    self.assertIn('Precision.HIGH', str(jaxpr))

    with jax.default_matmul_precision("float32"):
      jnp.dot(x, x)  # doesn't crash
      jaxpr = jax.make_jaxpr(jnp.dot)(x, x)
    self.assertIn('Precision.HIGHEST', str(jaxpr))

    dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
    with jax.default_matmul_precision("tensorfloat32"):
      dot(x, x)  # doesn't crash
      jaxpr = jax.make_jaxpr(dot)(x, x)
    self.assertIn('Precision.HIGHEST', str(jaxpr))

  def test_dot_precision_flag(self):
    x = jnp.zeros((2, 2))

    with config.default_matmul_precision("tensorfloat32"):
      jnp.dot(x, x)  # doesn't crash
      jaxpr = jax.make_jaxpr(jnp.dot)(x, x)
    self.assertIn('Precision.HIGH', str(jaxpr))

    with config.default_matmul_precision("tensorfloat32"):
      jnp.dot(x, x)  # doesn't crash
      jaxpr = jax.make_jaxpr(jnp.dot)(x, x)
    self.assertIn('Precision.HIGH', str(jaxpr))

  @jtu.thread_unsafe_test()  # Updating global configs is not thread-safe.
  def test_dot_precision_forces_retrace(self):
    num_traces = 0

    def g(x):
      nonlocal num_traces
      num_traces += 1
      return jnp.dot(x, x)
    def f_cond(x):
      return lax.cond(True, g, g, x)

    @jax.jit
    def f_jit(x):
      nonlocal num_traces
      num_traces += 1
      return jnp.dot(x, x)

    for f in [f_jit, f_cond]:
      # Use _read() to read the flag value rather than threadlocal value.
      precision = config._read("jax_default_matmul_precision")
      try:
        num_traces = 0
        x = jnp.zeros((2, 2))
        f(x)
        self.assertEqual(num_traces, 1)
        f(x)
        self.assertEqual(num_traces, 1)
        with jax.default_matmul_precision("tensorfloat32"):
          f(x)
          self.assertEqual(num_traces, 2)
          config.update("jax_default_matmul_precision", "float32")
          f(x)
          self.assertGreaterEqual(num_traces, 2)
        nt = num_traces
        f(x)
        self.assertEqual(num_traces, nt + 1)
        f(x)
        self.assertEqual(num_traces, nt + 1)
      finally:
        config.update("jax_default_matmul_precision", precision)

  def test_backward_pass_ref_dropping(self):
    refs = []

    @jax.custom_vjp
    def f(x):
      return x
    def f_fwd(x):
      return x, None
    def f_rev(_, g):
      assert len(refs) != 2 or refs[0]() is None
      zero = np.zeros(())
      refs.append(weakref.ref(zero))
      return (zero,)
    f.defvjp(f_fwd, f_rev)

    api.grad(lambda x: f(f(f(x))))(1.)

  def test_jit_inline(self):
    @partial(api.jit, inline=False)
    def f(x):
      return x * 2

    jaxpr = api.make_jaxpr(f)(3)
    self.assertIn('pjit', str(jaxpr))

    @partial(api.jit, inline=True)
    def f(x):
      return x * 2

    jaxpr = api.make_jaxpr(f)(3)
    self.assertNotIn('pjit', str(jaxpr))

  # Repro for https://github.com/jax-ml/jax/issues/7229.
  def test_compute_with_large_transfer(self):
    def f(x, delta):
      return x + jnp.asarray(delta, x.dtype)

    # A large and potentially unaligned array to trigger non-zero-copy and
    # async device array copy.
    xs = self.rng().uniform(0., 1., size=(10, 131, 111, 3)).astype(np.float32)
    for x in xs:
      delta = self.rng().uniform(-0.5, 0.5, size=())
      jitted_f = api.jit(f)
      np.testing.assert_allclose(jitted_f(x, delta), f(x, delta))

  def test_vjp_fun_jit(self):
    # test that the function returned by vjp can be returned
    # from and passed to jitted functions
    f = lambda x: 2. * x

    @partial(jit, static_argnums=0)
    def linearize_vjp(f, x):
      _, vjp_fun = api.vjp(f, x)
      return vjp_fun

    linearized = linearize_vjp(f, 1.)
    actual = jit(lambda f, x: f(x))(linearized, 3.)
    expected = (6.,)
    self.assertEqual(actual, expected)

  def test_linearize_fun_jit(self):
    # test that the function returned by linearize can be returned
    # from and passed to jitted functions
    f = lambda x: 2. * x

    @partial(jit, static_argnums=0)
    def linearize(f, x):
      _, jvp_fun = api.linearize(f, x)
      return jvp_fun

    linearized = linearize(f, 1.)
    actual = jit(lambda f, x: f(x))(linearized, 3.)
    expected = 6.
    self.assertEqual(actual, expected)

  def test_linear_transpose_fun_jit(self):
    # test that the function returned by linear_transpose can be returned
    # from and passed to jitted functions
    f = lambda x: 2. * x

    @partial(jit, static_argnums=0)
    def transpose(f, x):
      return api.linear_transpose(f, x)

    transposed = transpose(f, 1.)
    actual = jit(lambda f, x: f(x))(transposed, 3.)
    expected = (6.,)
    self.assertEqual(actual, expected)

  def test_leaked_tracer_issue_7613(self):
    # from https://github.com/jax-ml/jax/issues/7613
    import numpy.random as npr

    def sigmoid(x):
      return 1. / (1. + jnp.exp(-x))

    x = jnp.ones((1, 50))
    A = jnp.array(npr.randn(50, 50), dtype=x.dtype)

    @jax.jit
    def loss(A, x):
      h = jax.nn.sigmoid(A * x)
      return jnp.sum((h - x)**2)

    with jax.checking_leaks():
      _ = jax.grad(loss)(A, x)  # doesn't crash

  def test_vmap_caching(self):
    # https://github.com/jax-ml/jax/issues/7621

    f = lambda x: jnp.square(x).mean()
    jf = jax.jit(f)
    x = jax.random.uniform(jax.random.key(0), shape=(8, 4))

    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      for _ in range(5):
        jax.hessian(jf)(x).block_until_ready()

      n = count()
      # The exact number of compilations may vary depending on the number of
      # jit decorators in the function above, but it should not grow after an
      # initial warmup phase.
      for _ in range(5):
        jax.hessian(jf)(x).block_until_ready()

    self.assertEqual(count(), n)

  def test_jnp_array_doesnt_device_put(self):
    with jtu.count_device_put() as count:
      api.make_jaxpr(lambda: jnp.array(3))()
    self.assertEqual(count(), 0)

  @jtu.thread_unsafe_test()  # Updating global configs is not thread-safe.
  def test_rank_promotion_forces_retrace(self):
    num_traces = 0

    def g(x):
      nonlocal num_traces
      num_traces += 1
      return x + x
    def f_cond(x):
      return lax.cond(True, g, g, x)

    @jax.jit
    def f_jit(x):
      nonlocal num_traces
      num_traces += 1
      return x + x

    for f in [f_jit, f_cond]:
      # Use _read() to read the flag value rather than threadlocal value.
      allow_promotion = jax.numpy_rank_promotion.get_global()
      try:
        config.update("jax_numpy_rank_promotion", "allow")
        num_traces = 0
        @jax.jit
        def f(x):
          nonlocal num_traces
          num_traces += 1
          return x + x
        x = jnp.zeros((2, 2))
        f(x)
        self.assertEqual(num_traces, 1)
        f(x)
        self.assertEqual(num_traces, 1)
        with jax.numpy_rank_promotion("warn"):
          f(x)
          self.assertEqual(num_traces, 2)
          config.update("jax_numpy_rank_promotion", "raise")
          f(x)
          self.assertGreaterEqual(num_traces, 2)
        nt = num_traces
        f(x)
        self.assertEqual(num_traces, nt)
        f(x)
        self.assertEqual(num_traces, nt)
      finally:
        config.update("jax_numpy_rank_promotion", allow_promotion)

  def test_grad_negative_argnums(self):
    def f(x, y):
      return x.sum() * y.sum()

    x = jax.random.normal(jax.random.key(0), (16, 16))
    y = jax.random.normal(jax.random.key(1), (16, 16))
    g = jax.grad(f, argnums=-1)
    g(x, y)  # doesn't crash

  def test_jit_negative_static_argnums(self):
    @partial(jax.jit, static_argnums=-1)
    def g(x, y):
      assert isinstance(y, int)
      return x * y
    for i in range(3):  # Loop verifies we exercise both Python and C++ dispatch
      self.assertEqual(2 * i, g(2, i), msg=i)

  def test_make_jaxpr_static_argnums_order(self):
    # https://github.com/jax-ml/jax/issues/28065
    def f(a, b, c):
      x = a + c
      y = b * c
      z = x - y
      return z

    for static_argnums in [(1, 0), (0, 1)]:
      val = jax.jit(f, static_argnums=static_argnums)(1, 2, 3)
      self.assertEqual(val, -2)
      jaxpr = jax.make_jaxpr(f, static_argnums=static_argnums)(1, 2, 3)
      self.assertEqual(jaxpr.eqns[0].invars[0].val, 1)
      self.assertEqual(jaxpr.eqns[1].invars[0].val, 2)

  def test_fastpath_cache_confusion(self):
    # https://github.com/jax-ml/jax/issues/12542
    @jax.jit
    def a(x):
      return ()

    @jax.jit
    def b(x):
      return a(x)

    @jax.jit
    def g(x):
      return x, x

    @jax.jit
    def h(x):
      return g(x)

    jaxpr = jax.make_jaxpr(h)(7)
    core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 7)

    b(8)  # don't crash

  def test_fastpath_cache_confusion2(self):
    @jax.jit
    def a():  # note nullary function, still staged out though
      return ()

    @jax.jit
    def b(x):
      return a()

    @jax.jit
    def g(x):
      return x, x

    @jax.jit
    def h(x):
      return g(x)

    jaxpr = jax.make_jaxpr(h)(7)
    core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 7)

    b(8)  # don't crash

  def test_vjp_multiple_arguments_error_message(self):
    # https://github.com/jax-ml/jax/issues/13099
    def foo(x):
      return (x, x)
    _, f_vjp = jax.vjp(foo, 1.0)
    with self.assertRaisesRegex(TypeError, "applied to foo"):
      f_vjp(1.0, 1.0)

  def test_make_jaxpr_weakref(self):
    class Foo(NamedTuple):
      x: int

      def __call__(self, y):
        return self.x + y

    jax.make_jaxpr(Foo(1))(3)  # don't crash

  def test_make_jaxpr_name(self):
    def foo(x, y, z):
      return x + y + z
    jfoo = jax.make_jaxpr(foo)
    self.assertEqual(jfoo.__name__, f"make_jaxpr({foo.__name__})")
    self.assertEqual(jfoo.__qualname__, f"make_jaxpr({foo.__qualname__})")
    self.assertEqual(jfoo.__module__, "jax")

  @jtu.thread_unsafe_test()  # Concurrent cache eviction means we may retrace
  def test_inner_jit_function_retracing(self):
    # https://github.com/jax-ml/jax/issues/7155
    inner_count = outer_count = 0

    @jax.jit
    def inner_fn(state):
      nonlocal inner_count
      inner_count += 1
      return 2*state

    @jax.jit
    def outer_fn(x):
      nonlocal outer_count
      outer_count += 1
      old_x = x
      for _ in range(10):
        x = inner_fn(x)
      x = x + old_x
      return x

    state = jnp.arange(5, dtype=jnp.uint32)
    inner_fn(state)
    outer_fn(state)

    self.assertEqual(inner_count, 1)
    self.assertEqual(outer_count, 1)

  def test_grad_conj_symbolic_zeros(self):
    # https://github.com/jax-ml/jax/issues/15400
    f = lambda x: jax.jit(lambda x, y: (x, y))(x, jax.lax.conj(x))[0]
    out = jax.grad(f)(3.0)  # doesn't crash
    self.assertAllClose(out, 1., check_dtypes=False)

  @jtu.thread_unsafe_test()
  def test_cache_clear_pmap(self):
    @jax.pmap
    def f(i):
      return i * 2

    f(np.arange(1, dtype='float32')).block_until_ready()
    self.assertEqual(f._cache_size, 1)
    jax.clear_caches()
    self.assertEqual(f._cache_size, 0)

  def test_invalid_value_device_put(self):
    with self.assertRaisesRegex(ValueError, r".*Received invalid value.*"):
      jax.device_put(jnp.arange(8), 'cpu')

  def test_num_cpu_devices_called_after_initialization(self):
    jax.devices()
    with self.assertRaisesRegex(
        RuntimeError,
        "jax_num_cpu_devices config should be updated before backends are "
        "initialized"):
      config.update('jax_num_cpu_devices', 2)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_clear_cache(self):
    @jax.jit
    def add(x):
      return x * 2

    inp = jnp.arange(8)

    with config.log_compiles(True):
      with self.assertLogs(level='WARNING') as cm:
        add(inp)
        jax.clear_caches()
        add(inp)
      tracing_add_count = 0
      for m in cm.output:
        if 'Finished tracing + transforming add for pjit' in m:
          tracing_add_count += 1
      self.assertEqual(tracing_add_count, 2)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_skip_internals(self):
    if is_persistent_cache_enabled():
      self.skipTest('With persistent cache, we see the cache misses')

    with config.explain_cache_misses(True):
      with self.assertNoLogs(level='WARNING'):
        for i in range(2):
          jnp.sin(jnp.arange(i + 1, dtype=np.float32))

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_first_miss(self):
    @jax.jit
    def f(x): return x
    x = jnp.float32(1.)

    expected_log_len = 1 if not is_persistent_cache_enabled() else 3
    # print on first miss, not on hit
    with config.explain_cache_misses(True):
      with self.assertLogs(level="WARNING") as cm:
        f(x)
        f(x)
    self.assertLen(cm.output, expected_log_len)
    msg = cm.output[0]
    self.assertIn("TRACING CACHE MISS", msg)
    self.assertIn("never seen function", msg)
    self.assertNotIn("explanation unavailable!", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_other_in_tree(self):
    @jax.jit
    def f(*args, **kwargs): return args[0]

    f(0., 1., y=(2., 2.1))

    with config.explain_cache_misses(True):
      with self.assertLogs(level="WARNING") as cm:
        # Same number of leaves but different trees
        f(0., (1., 1.1), y=2.)
    self.assertLen(cm.output, 1)
    msg = cm.output[0]
    self.assertIn("different input pytree", msg)
    self.assertNotIn("explanation unavailable!", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_other_arg_passed_as_kwarg(self):
    @jax.jit
    def f(x, y): return jnp.sin(x) + y

    f(0., 1.)

    # kwarg change
    with config.explain_cache_misses(True):
      with self.assertLogs(level="WARNING") as cm:
        f(0., y=1.)

    self.assertLen(cm.output, 1)
    msg = cm.output[0]
    self.assertIn("different number of args and kwargs, but same total number", msg)
    self.assertIn("now 1 args and kwargs with keys ['y']", msg)
    self.assertIn("before 1 args and kwargs with keys []", msg)
    self.assertNotIn("explanation unavailable!", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_other_static_argnums(self):
    @partial(jax.jit, static_argnums=(0, 2))
    def f(x, y, z):
      return y

    f(1., 2., "foo")

    with config.explain_cache_misses(True):
      with self.assertLogs(level="WARNING") as cm:
        f(1., 2., "bar")
    self.assertLen(cm.output, 1)
    msg = cm.output[0]
    self.assertIn("different value of static args", msg)
    self.assertIn("now 1.0, 'bar' and before 1.0, 'foo'", msg)
    self.assertNotIn("explanation unavailable!", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_other_static_argnames(self):
    @partial(jax.jit, static_argnames="foo")
    def f(*, foo):
      return 1

    f(foo="foo")

    with config.explain_cache_misses(True):
      with self.assertLogs(level="WARNING") as cm:
        f(foo="bar")
    self.assertLen(cm.output, 1)
    msg = cm.output[0]
    self.assertIn("different value of static kwargs", msg)
    self.assertIn("now {foo: 'bar'} and before {foo: 'foo'}", msg)
    self.assertNotIn('explanation unavailable!', msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_other_dtype(self):
    @jax.jit
    def f(x, y): return x
    f(np.float32(0), np.float32(1))

    with config.explain_cache_misses(True):
      with self.assertLogs(level='WARNING') as cm:
        f(np.float32(0), np.int32(1))
    self.assertLen(cm.output, 1)
    msg = cm.output[0]
    self.assertIn("different input types", msg)
    self.assertIn("at y, now i32[] and before f32[]", msg)
    self.assertNotIn("explanation unavailable!", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_other_weak_type(self):
    @jax.jit
    def f(x, y): return jnp.sin(x) + y

    y = jnp.arange(4, dtype="float32")
    f(jnp.float32(0.), y)
    # weak type change (assuming no x64)
    if config.enable_x64.value:
      self.skipTest("Work only for 32 bit mode")
    with config.explain_cache_misses(True):
      with self.assertLogs(level="WARNING") as cm:
        f(0., y)
    expected_log_len = 1 if not is_persistent_cache_enabled() else 3
    self.assertLen(cm.output, expected_log_len)
    msg = cm.output[0]
    self.assertIn("different input types", msg)
    self.assertIn("at x, now f32[]{weak_type=True} and before f32[]{weak_type=False}", msg)
    self.assertIn("https://docs.jax.dev/en/latest/type_promotion.html#weak-types", msg)
    self.assertNotIn("explanation unavailable!", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_other_shape(self):
    @jax.jit
    def f(x, y): return jnp.sin(x) + y
    f(np.float32(0), np.arange(1, dtype=np.float32))

    with config.explain_cache_misses(True):
      with self.assertLogs(level='WARNING') as cm:
        f(np.float32(0), np.arange(2, dtype=np.float32))
    expected_log_len = 1 if not is_persistent_cache_enabled() else 3
    self.assertLen(cm.output, expected_log_len)
    msg = cm.output[0]
    self.assertIn("different input types", msg)
    self.assertIn("at y, now f32[2] and before f32[1]", msg)
    self.assertNotIn("explanation unavailable!", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_other_shape_explain_closest(self):
    @jax.jit
    def f(x): return x
    f(np.ones((1, 2), dtype=np.float32))
    f(np.ones((10, 20, 30), dtype=np.float32))
    f(np.ones((1, 2, 3), dtype=np.float32))

    with config.explain_cache_misses(True):
      with self.assertLogs(level='WARNING') as cm:
        f(np.ones((10, 2, 30), dtype=np.float32))
    expected_log_len = 1 if not is_persistent_cache_enabled() else 3
    self.assertLen(cm.output, expected_log_len)
    msg = cm.output[0]
    self.assertIn("key with different input types", msg)
    self.assertIn("at x, now f32[10,2,30] and before f32[10,20,30]", msg)
    self.assertNotIn("explanation unavailable!", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_other_tracing_config(self):
    @jax.jit
    def f(x, y): return jnp.sin(x) + y

    f(0., 1.)
    # tracing config change
    with config.explain_cache_misses(True):
      with self.assertLogs(level="WARNING") as cm:
        with jax.numpy_rank_promotion("warn"):
          with jax.default_matmul_precision("high"):
            f(0., 1.)

    expected_log_len = 1 if not is_persistent_cache_enabled() else 3
    self.assertTrue(1 <= len(cm.output) <= expected_log_len)
    msg = cm.output[0]
    self.assertIn("key with different tracing context", msg)
    self.assertIn("now warn and before", msg)
    self.assertIn("now high and before", msg)
    self.assertNotIn("explanation unavailable!", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_multiple_changes(self):
    @jax.jit
    def f(x): return jnp.sin(x)

    call_1 = f(np.arange(4, dtype=np.float32))
    with jax.numpy_rank_promotion("warn"):
      call_2 = f(np.arange(8, dtype=np.float32))

    with config.explain_cache_misses(True):
      with self.assertLogs(level='WARNING') as cm:
        # Matches call_2 in shape but not context, and call_1 in context but
        # not in shape.
        f(np.arange(8, dtype=np.float32))

    self.assertLen(cm.output, 1)
    msg = cm.output[0]
    self.assertIn("key with different input types", msg)
    self.assertIn("at x, now f32[8] and before f32[4]", msg)
    self.assertIn("key with different tracing context", msg)
    self.assertNotIn("explanation unavailable!", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_new_function_in_loop(self):
    @jax.jit
    def f(x, y):
      return jnp.sin(x) * y['hi']

    x = jnp.float32(1.)

    with config.explain_cache_misses(True):
      with self.assertLogs(level='WARNING') as cm:
        for _ in range(2):
          jax.jit(lambda x: 2 * x)(3)
    if is_persistent_cache_enabled():
      # number of warnings depends on the backend
      self.assertTrue(4 <= len(cm.output) <= 6)
      msg = cm.output[3]
      self.assertIn('another function defined on the same line', msg)
    else:
      self.assertLen(cm.output, 2)
      _, msg = cm.output
      self.assertIn('another function defined on the same line', msg)

  def test_cache_miss_explanations_no_source_info(self):
    # ``operator.add`` is a built-in function and does not have source info.
    with config.explain_cache_misses(True):
      jax.jit(operator.add)(42, 24)

  @parameterized.named_parameters([
      {"testcase_name": f"{np.dtype(dtype)}", "dtype": dtype}
      for dtype in jtu.dtypes.custom_floats])
  def test_jit_custom_floats(self, dtype):
    f = lambda x: x + 1
    args_maker = lambda: [jnp.ones((), dtype=dtype)]
    self._CompileAndCheck(f, args_maker)

  def test_jvp_asarray_returns_array(self):
    # https://github.com/jax-ml/jax/issues/15676
    p, t = jax.jvp(jax.numpy.asarray, (1.,), (2.,))
    _check_instance(self, p)
    _check_instance(self, t)

  def test_scalar_conversion_errors(self):
    array_int = jnp.arange(10, dtype=int)
    scalar_float = jnp.float32(0)
    scalar_int = jnp.int32(0)
    empty_int = jnp.arange(0, dtype='int32')
    array1_float = jnp.arange(1, dtype='float32')

    assertIntError = partial(self.assertRaisesRegex, TypeError,
                             "Only integer scalar arrays can be converted to a scalar index.")
    for func in [operator.index, hex, oct]:
      assertIntError(func, array_int)
      assertIntError(func, empty_int)
      assertIntError(func, scalar_float)
      assertIntError(jax.jit(func), array_int)
      assertIntError(jax.jit(func), empty_int)
      assertIntError(jax.jit(func), scalar_float)
      self.assertRaises(TracerIntegerConversionError, jax.jit(func), scalar_int)
      _ = func(scalar_int)  # no error

    assertScalarError = partial(self.assertRaisesRegex, TypeError,
                                "Only scalar arrays can be converted to Python scalars.")
    for func in [int, float, complex]:
      assertScalarError(func, array_int)
      assertScalarError(jax.jit(func), array_int)
      self.assertRaises(ConcretizationTypeError, jax.jit(func), scalar_int)
      _ = func(scalar_int)  # no error
      assertScalarError(func, array1_float)

    assertEmptyBoolError = partial(
        self.assertRaisesRegex, ValueError,
        "The truth value of an empty array is ambiguous.")
    assertEmptyBoolError(bool, empty_int)
    assertEmptyBoolError(jax.jit(bool), empty_int)

    assertBoolError = partial(
        self.assertRaisesRegex, ValueError,
        "The truth value of an array with more than one element is ambiguous.")
    assertBoolError(bool, array_int)
    assertBoolError(jax.jit(bool), array_int)
    self.assertRaises(TracerBoolConversionError, jax.jit(bool), scalar_int)
    _ = bool(scalar_int)  # no error

  @jtu.run_on_devices('cpu')
  def test_asarray_no_copy_np(self):
    x = np.random.uniform(0, 1, (1000, 2000)).astype("float32")
    out = jnp.asarray(x)

    x_ptr = x.__array_interface__["data"][0]
    # This is because the PJRT CPU client shares memory if it is 16-byte aligned.
    if (x_ptr & 15) != 0:
      self.assertTrue(np.shares_memory(out, x))

  def test_mesh_creation_error_message(self):
    with self.assertRaisesRegex(ValueError, "ndim of its first argument"):
      jax.sharding.Mesh(jax.devices(), ("x", "y"))

  @jtu.thread_unsafe_test()  # weakref gc doesn't seem predictable
  def test_jit_boundmethod_reference_cycle(self):
    class A:
      def __init__(self):
        self._foo = jax.jit(self.foo)
      def foo(self):
        pass
    a = weakref.ref(A())
    gc.collect()
    assert a() is None

  def test_forwarding_bug(self):
    # Test for issue #20267.
    def f(x):

      @jax.jit
      def inner(a, x):
        return a, jnp.exp(x)

      return inner(0.0, x)[0]
    jax.grad(f)(1.)  # don't crash

  @parameterized.parameters(it.product(range(4), repeat=3))
  @jtu.run_on_devices("cpu")
  def test_jit_forwarding_correctness(self, seed, num_input_fwd, num_output_fwd):
    num_args = 3
    rng = np.random.RandomState(seed)
    in_perm = rng.permutation(num_args)
    out_perm = rng.permutation(num_args)

    @jax.jit
    def f(inputs):
      inputs = [inputs[i] for i in in_perm]
      outputs = inputs[:num_input_fwd] + [
          jnp.exp(inputs[i]) if i < num_output_fwd else jnp.sin(inputs[i])
          for i in range(num_args - num_input_fwd)]
      return [outputs[i] for i in out_perm]

    jtu.check_grads(f, (list(jnp.arange(float(num_args))),), order=1,
                    modes=['rev'], atol=1e-3, rtol=1e-3)

  @jtu.run_on_devices("cpu")
  def test_inner_jit_forwarding_happens(self):
    if not config.dynamic_shapes.value:
      self.skipTest("Only works for dynamic shapes")
    jaxpr = jax.make_jaxpr(lambda: jax.jit(lambda x: x)(3))()
    self.assertLen(jaxpr.jaxpr.outvars, 1)
    self.assertIsInstance(jaxpr.jaxpr.outvars[0], core.Literal)
    self.assertEqual(jaxpr.jaxpr.outvars[0].val, 3)

  @parameterized.parameters(range(8))
  @jtu.run_on_devices("cpu")
  def test_inner_jit_forwarding_correctness(self, num_input_fwd):
    if not config.dynamic_shapes.value:
      self.skipTest("Only works for dynamic shapes")
    num_args = 8
    rng = np.random.RandomState(0)

    @jax.jit
    def f(inputs):
      inputs = [inputs[i] for i in rng.permutation(num_args)]
      outputs = (inputs[:num_input_fwd] +
                 [jnp.sin(inputs[i]) for i in range(num_args - num_input_fwd)])
      return [outputs[i] for i in rng.permutation(num_args)]

    f2 = jax.jit(f)
    inputs = list(jnp.arange(float(num_args)))
    expected = f(inputs)
    ans = f2(inputs)
    for a, b in zip(ans, expected):
      self.assertAllClose(a, b)

  @unittest.skip # TODO(dougalm): figure out with Matt what to do with this feature
  def test_inner_jit_forwarded_consts_stay_const(self):
    out = jax.jit(lambda: int(jax.jit(lambda x: x)(3)))()  # don't crash
    self.assertEqual(out, 3)

  def test_lowering_platform_aot(self):
    @jax.jit
    def f(x):
      return x * 2

    f.trace(jnp.arange(8)).lower(lowering_platforms=('tpu',))  # doesn't crash

  def test_no_double_dots_in_error_message(self):
    @jax.jit
    def f(x):
      return 1 if x > 0 else 0

    with self.assertRaisesRegex(TracerBoolConversionError, r"with shape bool\[\]\.[^\.]"):
      f(0)

  def test_inlined_literals_with_error(self):
    @jax.jit
    def f():
      @partial(jax.jit, inline=True)
      def g():
        return jnp.sin(1.)
      if g() > 0:
        return 1.
      return 0.

    with self.assertRaisesRegex(TracerBoolConversionError, "Attempted boolean"):
      f()

  def test_inline_return_twice(self):
    # https://github.com/jax-ml/jax/issues/22944
    @jax.jit
    def add_one(x: int) -> int:
      return x + 1

    def add_one_and_dupe(x: int) -> tuple[int, int]:
      y = add_one(x)
      return (y, y)

    jit_add_one_dupe = jax.jit(add_one_and_dupe, inline=True)
    jax.eval_shape(jit_add_one_dupe, 0)  # don't crash

  def test_use_direct_linearize(self):

    def check_invariant_to_use_direct_linearize(f):
      with config.use_direct_linearize(False):
        ans1 = f()
      with config.use_direct_linearize(True):
        ans2 = f()

      self.assertEqual(ans1, ans2)

    def sin_of_sin(x):
      return lax.sin(jax.jit(lax.sin)(x))

    check_invariant_to_use_direct_linearize(lambda: jax.grad(sin_of_sin)(1.0))

  def test_deferred_primal_with_direct_linearize(self):
    def my_sin_lin(nzs, x):
      nz, = nzs
      return (my_sin_p.bind(x, accuracy=None), nz, x, lambda x, t: lax.mul(t, lax.cos(x)))

    my_sin_p = core.Primitive("my_sin_p")
    my_sin_p.def_impl(lax.sin)
    my_sin_p.def_abstract_eval(lambda x: x)
    ad_internal.primitive_linearizations[my_sin_p] = my_sin_lin

    with config.use_direct_linearize(True):
      jax.grad(my_sin_p.bind)(1.0)  # doesn't crash

  def test_ensure_compile_time_eval_no_leaks(self):
    # https://github.com/jax-ml/jax/issues/25847
    with jax.ensure_compile_time_eval():
      jnp.linalg.solve(jnp.eye(3), jnp.ones(3))  # doesn't crash

  def test_returned_non_jaxtype(self):

    class TestEnum(enum.Enum):
      A = enum.auto()

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class TestClass3:
      test_enum_field: TestEnum = dataclasses.field(metadata=dict(static=True))
      test_data_field: int

    def test_jax_function(test_class: TestClass3) -> TestEnum:
      return test_class.test_enum_field

    jitted_test_function = jax.jit(test_jax_function)
    with self.assertRaisesRegex(TypeError, "returned a value of type"):
        jitted_test_function(
            TestClass3(
                test_data_field=1,
                test_enum_field=TestEnum.A,
            )
        )


class RematTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  @jtu.thread_unsafe_test()  # monkey patches sin_p and cos_p
  def test_remat_basic(self, remat):
    @remat
    def g(x):
      return lax.sin(lax.sin(x)), 3.

    def f(x):
      x, _ = g(x)
      return x

    ans = f(2.)
    expected = np.sin(np.sin(2.))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans, f_lin = api.linearize(f, 2.)
    expected = np.sin(np.sin(2.))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = f_lin(3.)
    expected = np.cos(np.sin(2.)) * np.cos(2.) * 3.
    self.assertAllClose(ans, expected, check_dtypes=False)

    sin_calls = []
    cos_calls = []
    sin_impl = lax.sin_p.impl
    cos_impl = lax.cos_p.impl
    try:
      lax.sin_p.def_impl(lambda x, **kwargs: sin_calls.append(1) or sin_impl(x, **kwargs))
      lax.cos_p.def_impl(lambda x, **kwargs: cos_calls.append(1) or cos_impl(x, **kwargs))
      f_lin(3.)
    finally:
      lax.sin_p.def_impl(sin_impl)
      lax.cos_p.def_impl(cos_impl)
    self.assertEqual(len(sin_calls), 1)
    self.assertEqual(len(cos_calls), 2)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_remat_freevars(self, remat):
    def f1(x):
      y = 2 * jnp.sin(x)
      z = jnp.cos(x) * jnp.sin(y)
      return z

    def f2(x):
      y = 2 * jnp.sin(x)
      z = remat(lambda x: jnp.cos(x) * jnp.sin(y))(x)
      return z

    ans, f_lin = api.linearize(f2, 2.)
    expected, f_lin_expected = api.linearize(f1, 2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = f_lin(3.)
    expected = f_lin_expected(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @unittest.skip
  def test_remat_grad_python_control_flow_static_argnums(self):
    @partial(jax.remat, static_argnums=(0,))
    def g(x):
      with jax.ensure_compile_time_eval():
        x_pos = x > 0
      if x_pos:
        return lax.sin(x), 3.
      else:
        return lax.cos(x), 4.

    def f(x):
      x, _ = g(x)
      return x

    ans = f(2.)
    expected = np.sin(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(f)(2.)
    expected = np.cos(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @unittest.skip
  def test_remat_grad_python_control_flow_unhashable_static_argnums(self):
    @partial(jax.remat, static_argnums=(0,))
    def g(x):
      x = x.val
      with jax.ensure_compile_time_eval():
        x_pos = x > 0
      if x_pos:
        return lax.sin(x), 3.
      else:
        return lax.cos(x), 4.

    def f(x):
      x, _ = g(x)
      return x

    class A:
      def __init__(self, val):
        self.val = val
      def __hash__(self):
        raise TypeError

    ans = f(A(2.))
    expected = np.sin(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(lambda x: f(A(x)))(2.)
    expected = np.cos(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_remat_retracing(self):
    # This is *not* a very important behavior; remat doesn't need to provide
    # caching guarantees with the same importance as jit. But even so, in the
    # interest of not redoing tracing work (and thus make jax.remat more
    # feasible to use in eager mode), this test checks that we don't re-trace
    # the remat-decorated function.
    count = 0

    @jax.remat
    def g(x):
      nonlocal count
      count += 1
      return lax.sin(x), 3.

    def f(x):
      x, _ = g(x)
      return x

    for _ in range(10):
      y = f(2.)
      y.block_until_ready()
    self.assertEqual(count, 1)

  def test_remat_static_agnums_retracing(self):
    # This is *not* a super important behavior; remat doesn't need to provide
    # caching guarantees with the same importance as jit. But even so, in the
    # interest of not redoing tracing work (and thus make jax.remat more
    # feasible to use in eager mode), this test checks that we don't re-trace
    # the remat-decorated function *even with static_argnums*. See also the
    # above test, which doesn't check for static_argnums.
    count = 0

    @partial(jax.remat, static_argnums=(0,))
    def g(x):
      nonlocal count
      count += 1
      with jax.ensure_compile_time_eval():
        x_pos = x > 0
      if x_pos:
        return lax.sin(x), 3.
      else:
        return lax.cos(x), 4.

    def f(x):
      x, _ = g(x)
      return x

    for _ in range(10):
      y = f(2.)
      y.block_until_ready()
    self.assertEqual(count, 1)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_remat_jit(self, remat):
    @remat
    def g(x):
      return lax.sin(lax.sin(x))

    def f_(x):
      return g(x)
    f = api.jit(f_)

    ans = f(2.)
    expected = np.sin(np.sin(2.))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(f)(2.)
    expected = np.cos(np.sin(2.)) * np.cos(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jit(api.grad(f_))(2.)
    expected = np.cos(np.sin(2.)) * np.cos(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_remat_vmap(self, remat):
    @remat
    def g(x):
      return lax.sin(lax.sin(x))

    x = np.arange(3.)

    ans = api.vmap(g)(x)
    expected = np.sin(np.sin(x))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jacfwd(g)(x)
    expected = np.diag(np.cos(np.sin(x)) * np.cos(x))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jacrev(g)(x)
    expected = np.diag(np.cos(np.sin(x)) * np.cos(x))
    self.assertAllClose(ans, expected, check_dtypes=False)

    # Make sure that introducing constants in vmap works.
    constant_introducing_p = core.Primitive('introduce_constant')
    constant_introducing_p.def_abstract_eval(lambda x: x)
    def _constant_introducing_batcher(xs, ds):
      (x,), (d,) = xs, ds
      return (x + np.arange(x.size, dtype=x.dtype).reshape(x.shape)), d
    batching.primitive_batchers[constant_introducing_p] = _constant_introducing_batcher

    api.vmap(remat(constant_introducing_p.bind))(jnp.ones(20))

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_remat_vmap_not_leading_dim(self, remat):
    @remat
    def g(x):
      return lax.sin(lax.sin(x))

    x = np.arange(3 * 5.).reshape(3, 5)

    ans = api.vmap(g, 1, 0)(x)
    expected = np.sin(np.sin(x)).T
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_remat_higher_order_autodiff(self, remat):
    def f(x):
      return lax.cos(lax.sin(x))
    g = remat(f)

    ans = api.grad(api.grad(g))(3.)
    expected = api.grad(api.grad(f))(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_remat_scan(self, remat):
    to_scan = lambda c, x: (jnp.sin(c), None)

    def f_noremat(x):
      y, _ = lax.scan(to_scan, x, np.arange(3.))
      return y

    def f_yesremat(x):
      y, _ = lax.scan(remat(to_scan), x, np.arange(3.))
      return y

    ans = f_yesremat(4.)
    expected = f_noremat(4.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(f_yesremat)(4.)
    expected = api.grad(f_noremat)(4.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    jaxpr = api.make_jaxpr(api.linearize(f_yesremat, 4.)[1])(1.)
    scan_eqn, = jaxpr.jaxpr.eqns
    self.assertIn(' cos ', str(scan_eqn.params['jaxpr']))

    jaxpr = api.make_jaxpr(api.vjp(f_yesremat, 4.)[1])(1.)
    scan_eqn, = jaxpr.jaxpr.eqns
    self.assertIn(' cos ', str(scan_eqn.params['jaxpr']))

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  @jtu.thread_unsafe_test()  # monkey patches sin_p
  def test_remat_no_redundant_flops(self, remat):
    # see https://github.com/jax-ml/jax/pull/1749#issuecomment-558267584

    @api.jit
    def g(x):
      return f(2., x)

    @remat
    def f(x, y):
      return jnp.sin(x) * y

    # We swap out sin_p's impl rule to count how many times it's invoked
    called = []
    sin_impl = lax.sin_p.impl
    try:
      lax.sin_p.def_impl(lambda x, **kwargs: called.append(1) or sin_impl(x, **kwargs))
      api.grad(g)(3.)
    finally:
      lax.sin_p.def_impl(sin_impl)
    num_calls = len(called)
    self.assertLessEqual(num_calls, 1)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_remat_binomial_checkpointing(self, remat):
    def binom_checkpoint(funs):
      if len(funs) == 1:
        return funs[0]
      else:
        f1 = binom_checkpoint(funs[:len(funs)//2])
        f2 = binom_checkpoint(funs[len(funs)//2:])
        return remat(lambda x: f1(f2(x)))

    f1 = binom_checkpoint([jnp.sin, jnp.sin, jnp.sin, jnp.sin])
    f2 = lambda x: jnp.sin(jnp.sin(jnp.sin(jnp.sin(x))))
    x = 4.
    self.assertAllClose(f1(x), f2(x), check_dtypes=False)
    self.assertAllClose(api.grad(f1)(x), api.grad(f2)(x), check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_remat_symbolic_zeros(self, remat):
    # code from https://github.com/jax-ml/jax/issues/1907

    key = jax.random.key(0)
    key, split = jax.random.split(key)
    n = 5

    def func(D0):
      def shift(R, dR, **unused_kwargs):
        return R + dR

      def apply_fn(R):
        return D0 * R

      Rinit = jax.random.uniform(split, (n,3), minval=0.0, maxval=5.0,
                                 dtype=jnp.float32)

      def move(R,i):
        F = apply_fn(R)
        return shift(R, 0.001 * F), jnp.array([0.])

      move = remat(move)
      R, temp = lax.scan(move, Rinit, jnp.arange(2))
      return R[0, 0]

    api.grad(func)(5.0)  # doesn't crash

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_remat_jit2(self, remat):
    @api.jit
    def f(x):
      y = 2 * x

      @remat
      def g():
        return y

      return g()

    self.assertAllClose(f(3), 6, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_remat_nontrivial_env(self, remat):
    # simplified from https://github.com/jax-ml/jax/issues/2030

    @remat
    def foo(state, dt=0.5, c=1):
      u, u_t = state
      u_tt = c**2 * u
      u_t = u_t + u_tt * dt
      return (u, u_t)

    @partial(api.jit, static_argnums=(1,))
    def _multi_step(state, count, dt, c):
      f = lambda s, _: (foo(s, dt, c), _)
      return lax.scan(f, state, None, count)

    def multi_step(state, count, dt=1/jnp.sqrt(2), c=1):
      return _multi_step(state, count, dt, c)

    def loss(u0, target, steps, dt=1/jnp.sqrt(2), c=1):
      init = (u0, jnp.zeros_like(u0))
      (uf, _), _ = multi_step(init, steps, dt, c)
      return ((uf - target) ** 2).mean()

    target = jnp.zeros((128, 128))
    u0 = jnp.ones_like(target)
    loss(u0, target, 10)  # doesn't crash

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_remat_jit3(self, remat):
    # https://github.com/jax-ml/jax/issues/2180
    def f(w, x):
      a = jnp.dot(x, w)
      b = jnp.einsum("btd,bTd->btT", a, a)
      c = jnp.einsum("btT,btd->btd", b, a)
      return jnp.sum(c)

    w = jnp.ones([1, 1])
    x = jnp.ones([1, 1, 1])
    f = remat(f)
    api.grad(f)(w, x)  # doesn't crash

    @api.jit
    def mul(a, b):
      return a * b

    def f(w, x):
      a = mul(w, x)
      b = mul(a, a)
      return b

    w = 1.
    x = 1.
    f = remat(f)
    api.grad(f)(w, x)  # doesn't crash

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_remat_scan2(self, remat):
    # https://github.com/jax-ml/jax/issues/1963

    def scan_bug(x0):
      f = lambda x, _: (x + 1, None)
      def scanned_f(x, _):
        return lax.scan(f, x, xs=None, length=1)[0], None
      x, _ = remat(scanned_f)(x0, None)
      return x

    jax.grad(scan_bug)(1.0)  # doesn't crash

  def test_remat_jit_static_argnum_omnistaging(self):
    # https://github.com/jax-ml/jax/issues/2833
    # NOTE(mattjj): after #3370, this test doesn't actually call remat...
    def named_call(f):
      def named_f(*args):
        my_f = lambda: (f(*args),)
        f_ = lu.wrap_init(
            my_f, debug_info=api_util.debug_info("test_remat", my_f, args, {}))
        out, = core.call_p.bind(f_)
        return out
      return named_f

    def f(a_bool, y):
      if a_bool:
        return y + 1
      else:
        return y

    api.jit(named_call(f), static_argnums=0)(True, 1)  # no crash

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_remat_eval_counter(self, remat):
    # https://github.com/jax-ml/jax/issues/2737
    add_one_p = core.Primitive('add_one')
    add_one = add_one_p.bind

    num_evals = 0

    @contextmanager
    def assertEvals(n):
      start = num_evals
      yield
      assert num_evals - start == n

    def add_one_impl(x):
      nonlocal num_evals
      num_evals += 1
      return x + 1
    add_one_p.def_impl(add_one_impl)

    def add_one_jvp(pin, tin):
      pout = add_one(pin[0])
      return pout, pout * tin[0]
    ad.primitive_jvps[add_one_p] = add_one_jvp

    add_one_p.def_abstract_eval(lambda x: x)

    v = np.zeros((1,))

    f = remat(add_one)
    g = remat(lambda x: add_one(f(x)))

    # 2 calls needed to evaluate g
    with assertEvals(2):
      _, vjp = jax.vjp(g, v)
    # 2 calls made while transposing g, 1 call made while transposing f
    with assertEvals(3):
      vjp(v)

    @jax_util.curry
    def call(f, *args):
      my_f = lambda *args: [f(*args)]
      return core.call(
          lu.wrap_init(my_f,
                       debug_info=api_util.debug_info("test_remat", my_f,
                                                      args, {})),
          *args, name='foo')[0]

    f = call(add_one)
    g = remat(lambda x: add_one(f(x)))

    # 2 calls needed to evaluate g
    with assertEvals(2):
      _, vjp = jax.vjp(g, v)
    # 2 calls made while transposing g, no reevaluation for transposition of f
    with assertEvals(2):
      vjp(v)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_escaped_tracer_remat(self, remat):
    # b/169779185
    def f():
      seq = [jnp.zeros([])]
      def g():
        seq[0] += 1  # this is line 7 btw
        return seq[0]

      remat(g)()
      remat(lambda: g())()  # lambda defeats caching

    with self.assertRaisesRegex(UnexpectedTracerError, "global state"):
      api.jit(f)()

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_no_cse_widget_on_primals(self, remat):
    @remat
    def g(x):
      return lax.sin(lax.sin(x)), 3.

    def f(x):
      x, _ = g(x)
      return x

    text = jax.jit(f).lower(2.).as_text('hlo')
    self.assertNotIn('while', text)
    self.assertNotIn('conditional', text)
    self.assertNotIn('opt-barrier', text)

    text = jax.jit(grad(f)).lower(2.).as_text('hlo')
    self.assertTrue('while' in text or 'conditional' in text
                    or 'opt-barrier' in text)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_no_cse_widget_with_prevent_cse_false(self, remat):
    @partial(remat, prevent_cse=False)
    def g(x):
      return lax.sin(lax.sin(x)), 3.

    def f(x):
      x, _ = g(x)
      return x

    text = jax.jit(f).lower(2.).as_text('hlo')
    self.assertNotIn('while', text)
    self.assertNotIn('conditional', text)

    text = jax.jit(grad(f)).lower(2.).as_text('hlo')
    self.assertNotIn('while', text)
    self.assertNotIn('conditional', text)

  @parameterized.named_parameters(
      {"testcase_name": f"_{policy_name}_{remat_name}", "remat": remat,
       "policy": policy, "in_jaxpr2": in_jaxpr2, "not_in_jaxpr2": not_in_jaxpr2}
      for remat_name, remat in [
          ('old_remat', jax.remat),
          ('new_remat', new_checkpoint),
      ]
      for policy_name, policy, in_jaxpr2, not_in_jaxpr2 in [
          ('save_anything', lambda *_, **__: True, [], [' sin ', ' cos ']),
          ('save_nothing',  lambda *_, **__: False, [' sin ', ' cos '], []),
          ('save_sin',  lambda p, *_, **__: str(p) == 'sin', [' cos '], [' sin ']),
      ])
  def test_remat_custom_policy(self, remat, policy, in_jaxpr2, not_in_jaxpr2):
    for square in [lambda x: x * x, api.jit(lambda x: x * x)]:
      f = remat(lambda x: jnp.sin(square(jnp.sin(x))), policy=policy)
      y, f_lin = api.linearize(f, 1.)
      ydot = f_lin(2.)
      jaxpr_text = str(f_lin.func.args[0])
      for substr in in_jaxpr2:
        self.assertIn(substr, jaxpr_text)
      for substr in not_in_jaxpr2:
        self.assertNotIn(substr, jaxpr_text)
      y_expected, ydot_expected = api.jvp(lambda x: jnp.sin(square(jnp.sin(x))),
                                          [1.], [2.])
      self.assertAllClose(y, y_expected)
      self.assertAllClose(ydot, ydot_expected)
      jtu.check_grads(f, (3.,), order=2, modes=['fwd', 'rev'])

  @parameterized.named_parameters(
      {"testcase_name": f"_{remat_name}", "remat": remat}
      for remat_name, remat in [
          ('old_remat', jax.remat),
          ('new_remat', new_checkpoint),
      ])
  def test_remat_custom_policy_save_cos(self, remat):
    save_cos = lambda prim, *_, **__: str(prim) == 'cos'
    f = remat(lambda x: jnp.sin(jnp.sin(x)),  # different function
              policy=save_cos)
    _, f_lin = api.linearize(f, 1.)
    jaxpr_text = str(f_lin.func.args[0])
    self.assertNotIn(' sin ', jaxpr_text)
    self.assertNotIn(' cos ', jaxpr_text)
    jtu.check_grads(f, (3.,), order=2, modes=['fwd', 'rev'])

  @parameterized.named_parameters(
      {"testcase_name": f"_{remat_name}", "remat": remat}
      for remat_name, remat in [
          ('old_remat', jax.remat),
          ('new_remat', new_checkpoint),
      ])
  def test_remat_checkpoint_dots(self, remat):
    @partial(remat, policy=jax.checkpoint_policies.checkpoint_dots)
    def f(x):
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      return x

    _, f_lin = api.linearize(f, jnp.ones((2, 2)))
    jaxpr_text = str(f_lin.func.args[0])
    self.assertEqual(jaxpr_text.count(' sin '), 2)
    self.assertEqual(jaxpr_text.count(' dot_'), 6)
    jtu.check_grads(f, (jnp.ones((2, 2)),), order=2, modes=['fwd', 'rev'])

  @parameterized.named_parameters(
      {"testcase_name": f"_{remat_name}", "remat": remat}
      for remat_name, remat in [
          ('old_remat', jax.remat),
          ('new_remat', new_checkpoint),
      ])
  def test_remat_checkpoint_dots_with_no_batch_dims(self, remat):
    @partial(remat, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)
    def f(x):
      x = jnp.einsum('ij,jk->ik', x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.einsum('ij,jk->ik', x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.einsum('ij,jk->ik', x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      return x

    _, f_lin = api.linearize(f, jnp.ones((2, 2)))
    jaxpr_text = str(f_lin.func.args[0])
    self.assertEqual(jaxpr_text.count(' sin '), 2)
    self.assertEqual(jaxpr_text.count(' dot_general'), 6)
    jtu.check_grads(f, (jnp.ones((2, 2)),), order=2, modes=['fwd', 'rev'])

  @parameterized.named_parameters(
      {"testcase_name": f"_{remat_name}", "remat": remat}
      for remat_name, remat in [
          ('old_remat', jax.remat),
          ('new_remat', new_checkpoint),
      ])
  def test_remat_checkpoint_dots_with_no_batch_dims2(self, remat):
    @partial(remat, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)
    def f(x):
      x = jnp.einsum('nij,njk->nik', x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.einsum('nij,njk->nik', x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.einsum('nij,njk->nik', x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      return x

    _, f_lin = api.linearize(f, jnp.ones((3, 2, 2)))
    jaxpr_text = str(f_lin.func.args[0])
    self.assertEqual(jaxpr_text.count(' sin '), 2)
    self.assertEqual(jaxpr_text.count(' dot_general'), 9)
    jtu.check_grads(f, (jnp.ones((3, 2, 2)),), order=2, modes=['fwd', 'rev'])

  @parameterized.named_parameters(
      {"testcase_name": f"_{remat_name}", "remat": remat}
      for remat_name, remat in [
          ('old_remat', jax.remat),
          ('new_remat', new_checkpoint),
      ])
  def test_remat_checkpoint_dots_jit(self, remat):
    @api.jit
    @partial(remat, policy=jax.checkpoint_policies.checkpoint_dots)
    def f(x):
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x * 1e-3)
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x * 1e-3)
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x * 1e-3)
      return x

    _, f_lin = api.linearize(f, jnp.ones((2, 2)))
    jaxpr_text = str(f_lin.func.args[0])
    self.assertEqual(jaxpr_text.count(' sin '), 2)
    self.assertEqual(jaxpr_text.count(' dot_'), 6)
    jtu.check_grads(f, (jnp.ones((2, 2)),), order=2, modes=['fwd', 'rev'])

  def test_remat_checkpoint_dots_inside_scan(self):
    x = jnp.ones((5,))

    def f(W):
      @partial(jax.remat, policy=jax.checkpoint_policies.checkpoint_dots)
      def f(x):
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        return x

      def body(x, _): return f(x), None
      return lax.scan(body, x, None, length=2)[0]

    _, f_vjp = api.vjp(f, jnp.ones((5, 5)))
    jaxpr_text = str(f_vjp.args[0].func.args[1])

    # Two sine calls in the backward pass because while we don't save sines
    # within the (rematted) body function, we can save the scan carry, which
    # effectively saves one sine. Three cosines for the Jacobian coefficients.
    self.assertEqual(jaxpr_text.count(' sin '), 2)
    self.assertEqual(jaxpr_text.count(' cos '), 3)
    # Six calls to dot_general in the backward pass because we save the primal
    # matmuls and only compure the backward pass ones (two for each primal one).
    self.assertEqual(jaxpr_text.count(' dot_'), 6)

    jtu.check_grads(api.jit(f), (jnp.ones((5, 5)),), order=2,
                    modes=['fwd', 'rev'])

  def test_remat_custom_jvp_policy(self):
    @jax.custom_jvp
    def sin(x):
      return jnp.sin(x)
    def sin_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return sin(x), jnp.cos(x) * g
    sin.defjvp(sin_jvp)

    @partial(jax.remat, policy=jax.checkpoint_policies.checkpoint_dots)
    def f(x):
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = sin(x * 1e-3)
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = sin(x * 1e-3)
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = sin(x * 1e-3)
      return x

    jtu.check_grads(f, (3.,), order=2, modes=['fwd', 'rev'])

    def g(x):
      return lax.scan(lambda x, _: (f(x), None), x, None, length=2)[0]
    jtu.check_grads(g, (3.,), order=2, modes=['fwd', 'rev'])

  def test_remat_custom_vjp_policy(self):
    @jax.custom_vjp
    def sin(x):
      return jnp.sin(x)
    def sin_fwd(x):
      return sin(x), x
    def sin_bwd(x, y_bar):
      return (jnp.cos(x) * y_bar,)
    sin.defvjp(sin_fwd, sin_bwd)

    @partial(jax.remat, policy=jax.checkpoint_policies.checkpoint_dots)
    def f(x):
      @partial(api.named_call, name="dot")
      def dot2(y, z):
        return jnp.dot(x, jnp.dot(y, z, precision=lax.Precision.HIGHEST),
                       precision=lax.Precision.HIGHEST)

      x = dot2(x, x)
      x = sin(x * 1e-3)
      x = dot2(x, x)
      x = sin(x * 1e-3)
      x = dot2(x, x)
      x = sin(x * 1e-3)
      return x

    jtu.check_grads(f, (3.,), order=2, modes=['rev'])

    def g(x):
      return lax.scan(lambda x, _: (f(x), None), x, None, length=2)[0]
    jtu.check_grads(g, (3.,), order=2, modes=['rev'])

  @parameterized.named_parameters(
      {"testcase_name": f"_{remat_name}", "remat": remat}
      for remat_name, remat in [
          ('old_remat', jax.remat),
          ('new_remat', new_checkpoint),
      ])
  def test_remat_dropvar_policy(self, remat):
    def f(x):
      return x, x

    @partial(remat, policy=jax.checkpoint_policies.checkpoint_dots)
    def g(x):
      x = api.grad(lambda x: f(x)[0])(x)
      return x

    api.grad(g)(3.)

  def test_remat_custom_jvp_linear_policy(self):
    @jax.custom_jvp
    def sum(x):
      return jnp.sum(x, axis=0)
    @sum.defjvp
    def sum_jvp(primals, tangents):
      (x,), (xdot,) = primals, tangents
      return sum(x), sum(xdot)

    @partial(jax.remat, policy=jax.checkpoint_policies.checkpoint_dots)
    def f(x):
      return sum(x)
    jtu.check_grads(f, (jnp.ones(3),), order=2, modes=['fwd', 'rev'])

    def g(x):
      return lax.scan(lambda _, x: (None, f(x)), None, x)[1]
    jtu.check_grads(g, (jnp.ones((2, 3)),), order=2, modes=['fwd', 'rev'])

  def test_constants_not_hoisted(self):
    # The old implementation of remat worked by data dependence, and so
    # (potentially large) constants would not be rematerialized and could be
    # wastefully instantiated. This test checks that the newer remat
    # implementation avoids that. See https://github.com/jax-ml/jax/pull/8191.

    # no residuals from constants created inside jnp.einsum
    @partial(new_checkpoint, policy=lambda *_, **__: False)
    def f(x):
      return jnp.einsum('ii->i', x)
    res_avals = saved_residuals(f, jnp.ones((2, 2)))
    self.assertLen(res_avals, 0)

    # no residuals from jnp.zeros
    @partial(new_checkpoint, policy=lambda *_, **__: False)
    def f(x):
      return jnp.zeros_like(x) * x
    res_avals = saved_residuals(f, jnp.ones((2, 2)))
    self.assertLen(res_avals, 0)

    # no residuals from jnp.zeros, but input must be saved
    @partial(new_checkpoint, policy=lambda *_, **__: False)
    def f(x):
      return jnp.zeros_like(x) * jnp.sin(x)
    res_avals = saved_residuals(f, jnp.ones((2, 2)))
    self.assertLen(res_avals, 1)

  def test_name_saveable_input(self):
    @partial(jax.remat, policy=lambda p, *_, **__: 'mul' in str(p))
    def f(x):
      x = checkpoint_name(x * x, 'foo')
      x = x * x
      return x

    res = saved_residuals(f, 3.)
    self.assertStartsWith(res[1][1], "named 'foo'")

  def test_name_denylist(self):
    def f(x):
      y = checkpoint_name(jnp.multiply(2., 2.), 'y')
      z = checkpoint_name(jnp.multiply(2., 2.), 'z')
      w = checkpoint_name(jnp.multiply(2., 2.), 'w')
      u = jnp.multiply(2., 2.)
      return (((x * y) * z) * w) * u

    policy = jax.checkpoint_policies.save_any_names_but_these('y', 'z', 'w')
    res = saved_residuals(new_checkpoint(f, policy=policy), 1.)
    self.assertLen(res, 0)  # can't save anything

    policy = jax.checkpoint_policies.save_any_names_but_these('z', 'w')
    res = saved_residuals(new_checkpoint(f, policy=policy), 1.)
    self.assertLen(res, 1)  # can save only y

    policy = jax.checkpoint_policies.save_any_names_but_these('w')
    res = saved_residuals(new_checkpoint(f, policy=policy), 1.)
    self.assertLen(res, 2)  # can save y and z

    policy = jax.checkpoint_policies.save_any_names_but_these()
    res = saved_residuals(new_checkpoint(f, policy=policy), 1.)
    self.assertLen(res, 3)  # can save y, z, and w

  def test_name_allowlist(self):
    def f(x):
      y = checkpoint_name(jnp.multiply(2., 2.), 'y')
      z = checkpoint_name(jnp.multiply(2., 2.), 'z')
      w = checkpoint_name(jnp.multiply(2., 2.), 'w')
      u = jnp.multiply(2., 2.)
      return (((x * y) * z) * w) * u

    policy = jax.checkpoint_policies.save_only_these_names('y', 'z', 'w')
    res = saved_residuals(new_checkpoint(f, policy=policy), 1.)
    self.assertLen(res, 3)  # can save y, z, and w

    policy = jax.checkpoint_policies.save_only_these_names('z', 'w')
    res = saved_residuals(new_checkpoint(f, policy=policy), 1.)
    self.assertLen(res, 2)  # can save z and w

    policy = jax.checkpoint_policies.save_only_these_names('w')
    res = saved_residuals(new_checkpoint(f, policy=policy), 1.)
    self.assertLen(res, 1)  # can save w

    policy = jax.checkpoint_policies.save_only_these_names()
    res = saved_residuals(new_checkpoint(f, policy=policy), 1.)
    self.assertLen(res, 0)  # can't save anything!

  def test_saved_residuals_utility(self):
    def f(x, y):
      x1, x2 = x
      z = checkpoint_name(jnp.sin(3.), 'z')
      return z * ((x1 * x2) * y) * np.array([3.])

    res = saved_residuals(f, (2., 3.), y=4.)
    self.assertLen(res, 6)
    self.assertEqual(res[0][0].shape, (1,))
    self.assertEqual(res[0][1], "from a constant")
    self.assertEqual(res[1][0].shape, ())
    self.assertEqual(res[1][1], "from the argument x[0]")
    self.assertEqual(res[2][0].shape, ())
    self.assertEqual(res[2][1], "from the argument x[1]")
    self.assertEqual(res[3][0].shape, ())
    self.assertEqual(res[3][1], "from the argument y")
    self.assertEqual(res[4][0].shape, ())
    self.assertStartsWith(res[4][1], "named 'z'")
    self.assertEqual(res[5][0].shape, ())

  def test_saved_residuals_utility_jit(self):
    @jax.jit
    def f(x, y):
      x1, x2 = x
      z = checkpoint_name(jnp.sin(3.), 'z')
      return z * ((x1 * x2) * y) * np.array([3.])

    res = saved_residuals(f, (2., 3.), y=4.)
    self.assertLen(res, 6)
    self.assertEqual(res[0][0].shape, (1,))
    self.assertEqual(res[0][1], "from a constant")
    self.assertEqual(res[1][0].shape, ())
    self.assertEqual(res[1][1], "from the argument x[0]")
    self.assertEqual(res[2][0].shape, ())
    self.assertEqual(res[2][1], "from the argument x[1]")
    self.assertEqual(res[3][0].shape, ())
    self.assertEqual(res[3][1], "from the argument y")
    self.assertEqual(res[4][0].shape, ())
    self.assertStartsWith(res[4][1], "output of jitted function 'f'")
    self.assertEqual(res[5][0].shape, ())

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_policy', partial(jax.remat, policy=lambda *_, **__: False)),
          ('_new', partial(new_checkpoint, policy=lambda *_, **__: False)),
      ])
  def test_checkpoint_dropvars(self, remat):
    @remat
    def f(x):
      _, x = api.jit(lambda: (x, x))()
      return x

    _ = api.grad(f)(3.)  # doesn't crash

  def test_dce_keeps_eqns_with_used_outputs_but_no_used_inputs(self):
    @new_checkpoint
    def f(x):
      c = jax.jit(lambda: 3.)()
      return c * x

    _ = jax.grad(f)(3.)  # doesn't crash

  def test_linearize_caching(self):
    # https://github.com/jax-ml/jax/issues/9661
    identity = jax.checkpoint(jax.jit(lambda x: 2 * x))
    _, f_lin = jax.linearize(identity, 1.)
    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      for _ in range(20):
        f_lin(1.).block_until_ready()
    self.assertEqual(count(), 1)  # cached after first execution

  def test_vjp_caching(self):
    # https://github.com/jax-ml/jax/issues/9661
    identity = jax.checkpoint(jax.jit(lambda x: 2 * x))
    _, f_vjp = jax.vjp(identity, 1.)
    with jtu.count_pjit_cpp_cache_miss() as count:  # noqa: F841
      for _ in range(20):
        f_vjp(1.)[0].block_until_ready()
    self.assertEqual(count(), 2)  # fwd execute_trivial, backward_pass on bwd

  def test_vjp_caching_static_argnums(self):
    identity = jax.remat(lambda x, y: jax.jit(lambda x: 2 * x if y else x)(x),
                         static_argnums=(1,))
    _, f_vjp = jax.vjp(lambda x: identity(x, True), 1.)
    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      for _ in range(20):
        f_vjp(1.)[0].block_until_ready()
    self.assertEqual(count(), 2)  # fwd execute_trivial, backward_pass on bwd

  def test_fwd_caching(self):
    # see above test also
    identity = jax.checkpoint(jax.jit(lambda x: 2 * x))
    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      for _ in range(20):
        y, _ = jax.vjp(identity, 1.)
        y.block_until_ready()
    self.assertEqual(count(), 1)

  def test_fwd_caching_static_argnums(self):
    # see above test also
    identity = jax.checkpoint(jax.jit(lambda x: 2 * x), static_argnums=(0,))
    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      for _ in range(20):
        y = identity(1.)
        y.block_until_ready()
    self.assertEqual(count(), 1)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_remat_of_scan(self, remat):
    to_scan = lambda c, _: (jnp.sin(c), jnp.sin(c))
    f = lambda x: lax.scan(to_scan, x, None, length=3)
    jtu.check_grads(remat(f), (3.,), order=2, modes=['rev'])

    jaxpr = api.make_jaxpr(api.linearize(remat(f), 4.)[1])(1.)
    print("debug jaxpr: ", str(jaxpr))
    self.assertIn(' sin ', str(jaxpr))
    self.assertIn(' cos ', str(jaxpr))

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_const_in_jvp_scan(self, remat):
    @jax.custom_jvp
    def f(x):
      return x * jnp.arange(3.)
    @f.defjvp
    def f_jvp(primals, tangents):
      (x,), (xdot,) = primals, tangents
      return f(x), xdot * jnp.arange(3.)

    @remat
    def g(x):
      def body(c, _):
        return f(c), None
      y, _ = jax.lax.scan(body, x, None, length=1)
      return y.sum()

    jax.grad(g)(jnp.arange(3.))  # doesn't crash

  def test_remat_checkpoint_dots_outside_scan(self):
    # see also above test test_remat_checkpoint_dots_inside_scan
    x = jnp.ones((5,))

    @partial(new_checkpoint, policy=jax.checkpoint_policies.checkpoint_dots)
    def f(W):
      def f(x):
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        return x

      def body(x, _): return f(x), None
      return lax.scan(body, x, None, length=2)[0]

    _, f_vjp = api.vjp(f, jnp.ones((5, 5)))
    jaxpr = f_vjp.args[0].func.args[1]
    jaxpr_text = str(jaxpr)

    self.assertEqual(jaxpr_text.count(' sin '), 3)
    self.assertEqual(jaxpr_text.count(' cos '), 3)
    # Six calls to dot_general in the backward pass because we save the primal
    # matmuls and only compute the backward pass ones (two for each primal one).
    self.assertEqual(jaxpr_text.count(' dot_'), 6)

    jtu.check_grads(api.jit(f), (jnp.ones((5, 5)),), order=2,
                    modes=['fwd', 'rev'])

  def test_remat_of_scan_policy(self):
    save_cos = lambda prim, *_, **__: str(prim) == 'cos'
    to_scan = lambda c, _: (jnp.sin(c), jnp.sin(c))
    f = new_checkpoint(lambda x: lax.scan(to_scan, x, None, length=3),
                       policy=save_cos)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])

    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 0)

  def test_remat_of_scan_funky_custom_jvp(self):
    def scan_apply(f, x):
      y, _ = lax.scan(lambda x, _: (f(x), None), x, None, length=1)
      return y

    @jax.custom_jvp
    def sin(x):
      return jnp.sin(x)
    def sin_jvp(primals, tangents):
      x, = primals
      xdot, = tangents
      y, c = jax.jit(lambda: (jnp.sin(x), jnp.cos(x)))()
      ydot = c * xdot
      return y, ydot
    sin.defjvp(sin_jvp)

    save_cos = lambda prim, *_, **__: str(prim) == 'cos'
    f = new_checkpoint(partial(scan_apply, sin), policy=save_cos)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 0)

    save_sin = lambda prim, *_, **__: str(prim) == 'sin'
    f = new_checkpoint(partial(scan_apply, sin), policy=save_sin)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 1)

    f = new_checkpoint(partial(scan_apply, sin),
                       policy=jax.checkpoint_policies.everything_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 0)

    f = new_checkpoint(partial(scan_apply, sin),
                       policy=jax.checkpoint_policies.nothing_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 1)  # +1 b/c dce fixed point
    self.assertEqual(jaxpr_text.count(' cos '), 1)

    f = new_checkpoint(lambda x: scan_apply(sin, scan_apply(sin, x)),
                       policy=jax.checkpoint_policies.nothing_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 2)  # +1 b/c dce fixed point
    self.assertEqual(jaxpr_text.count(' cos '), 2)

  def test_remat_of_scan_funky_custom_jvp2(self):
    # Like the above test but instead of using jit inside custom_jvp, use scan.

    def scan_apply(f, x):
      y, _ = lax.scan(lambda x, _: (f(x), None), x, None, length=1)
      return y

    @jax.custom_jvp
    def sin(x):
      return jnp.sin(x)
    def sin_jvp(primals, tangents):
      x, = primals
      xdot, = tangents
      y, c = scan_apply(lambda xs: (jnp.sin(xs[0]), jnp.cos(xs[1])), (x, x))
      ydot = c * xdot
      return y, ydot
    sin.defjvp(sin_jvp)

    save_cos = lambda prim, *_, **__: str(prim) == 'cos'
    f = new_checkpoint(partial(scan_apply, sin), policy=save_cos)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 1)  # +1 b/c dce fixed point
    self.assertEqual(jaxpr_text.count(' cos '), 0)

    save_sin = lambda prim, *_, **__: str(prim) == 'sin'
    f = new_checkpoint(partial(scan_apply, sin), policy=save_sin)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 1)

    f = new_checkpoint(partial(scan_apply, sin),
                       policy=jax.checkpoint_policies.everything_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 0)

    f = new_checkpoint(partial(scan_apply, sin),
                       policy=jax.checkpoint_policies.nothing_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 1)  # +1 b/c dce fixed point
    self.assertEqual(jaxpr_text.count(' cos '), 1)

    f = new_checkpoint(lambda x: scan_apply(sin, scan_apply(sin, x)),
                       policy=jax.checkpoint_policies.nothing_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 2)  # +1 b/c dce fixed point
    self.assertEqual(jaxpr_text.count(' cos '), 2)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_remat_of_cond(self, remat):
    true_fn  = lambda c: (jnp.sin(c), jnp.sin(c))
    false_fn = lambda c: (jnp.sin(c), jnp.sin(c))
    f = lambda x: lax.cond(x > 0., true_fn, false_fn, x)
    jtu.check_grads(remat(f), (3.,), order=2, modes=['rev'])

    jaxpr = api.make_jaxpr(api.linearize(remat(f), 4.)[1])(1.)
    self.assertNotIn(' sin ', str(jaxpr))
    self.assertIn(' cos ', str(jaxpr))

    true_fn  = lambda c: jnp.sin(jnp.sin(c))
    false_fn = lambda c: c
    f = lambda x: lax.cond(x > 0., true_fn, false_fn, x)
    jtu.check_grads(remat(f), (3.,), order=2, modes=['rev'])

    jaxpr = api.make_jaxpr(api.linearize(remat(f), 4.)[1])(1.)
    self.assertIn(' sin ', str(jaxpr))
    self.assertIn(' cos ', str(jaxpr))

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_const_in_jvp_cond(self, remat):
    @jax.custom_jvp
    def f(x):
      return x * jnp.arange(3.)
    @f.defjvp
    def f_jvp(primals, tangents):
      (x,), (xdot,) = primals, tangents
      return f(x), xdot * jnp.arange(3.)

    @remat
    def g(x):
      y = jax.lax.cond(x.sum() > 0, f, lambda x: x, x)
      return y.sum()

    jax.grad(g)(jnp.arange(3.))  # doesn't crash

  def test_remat_checkpoint_dots_inside_cond(self):
    x = jnp.ones((5,))

    def f(W):
      @partial(jax.remat, policy=jax.checkpoint_policies.checkpoint_dots)
      def f(x):
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        return x

      return lax.cond(x.sum() > 0, f, lambda x: x, x)

    _, f_vjp = api.vjp(f, jnp.ones((5, 5)))
    jaxpr_text = str(f_vjp.args[0].func.args[1])

    self.assertEqual(jaxpr_text.count(' sin '), 2)
    self.assertEqual(jaxpr_text.count(' cos '), 3)
    # Five calls to dot_general in the backward pass because we have two for
    # each forward-pass dot, except for the first which only has one (as we are
    # differentiating with respect to only W and not x).
    self.assertEqual(jaxpr_text.count(' dot_'), 5)

    jtu.check_grads(api.jit(f), (jnp.ones((5, 5)),), order=2,
                    modes=['fwd', 'rev'])

  def test_remat_checkpoint_dots_outside_cond(self):
    # see also above test test_remat_checkpoint_dots_inside_cond
    # The behavior between the two tests is essentially identical, whereas for
    # scan different things are saved based on this difference in remat
    # placement (because of the carry).
    x = jnp.ones((5,))

    @partial(new_checkpoint, policy=jax.checkpoint_policies.checkpoint_dots)
    def f(W):
      def f(x):
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        x = jnp.sin(jnp.dot(x, W, precision=lax.Precision.HIGHEST))
        return x

      return lax.cond(x.sum() > 0, f, lambda x: x, x)

    _, f_vjp = api.vjp(f, jnp.ones((5, 5)))
    jaxpr = f_vjp.args[0].func.args[1]
    jaxpr_text = str(jaxpr)

    self.assertEqual(jaxpr_text.count(' sin '), 2)
    self.assertEqual(jaxpr_text.count(' cos '), 3)
    self.assertEqual(jaxpr_text.count(' dot_'), 5)

    jtu.check_grads(api.jit(f), (jnp.ones((5, 5)),), order=2,
                    modes=['fwd', 'rev'])

  def test_remat_of_cond_policy(self):
    save_cos = lambda prim, *_, **__: str(prim) == 'cos'
    f = new_checkpoint(lambda x: lax.cond(x > 0, jnp.sin, lambda x: x, x),
                       policy=save_cos)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])

    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 0)

  def test_remat_of_cond_funky_custom_jvp(self):
    def cond_apply(f, x):
      return lax.cond(x.sum() > -jnp.inf, f, lambda x: x, x)

    @jax.custom_jvp
    def sin(x):
      return jnp.sin(x)
    def sin_jvp(primals, tangents):
      x, = primals
      xdot, = tangents
      y, c = jax.jit(lambda: (jnp.sin(x), jnp.cos(x)))()
      ydot = c * xdot
      return y, ydot
    sin.defjvp(sin_jvp)

    save_cos = lambda prim, *_, **__: str(prim) == 'cos'
    f = new_checkpoint(partial(cond_apply, sin), policy=save_cos)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 0)

    save_sin = lambda prim, *_, **__: str(prim) == 'sin'
    f = new_checkpoint(partial(cond_apply, sin), policy=save_sin)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 1)

    f = new_checkpoint(partial(cond_apply, sin),
                       policy=jax.checkpoint_policies.everything_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 0)

    f = new_checkpoint(partial(cond_apply, sin),
                       policy=jax.checkpoint_policies.nothing_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 1)

    f = new_checkpoint(lambda x: cond_apply(sin, cond_apply(sin, x)),
                       policy=jax.checkpoint_policies.nothing_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 1)
    self.assertEqual(jaxpr_text.count(' cos '), 2)

  def test_remat_of_cond_funky_custom_jvp2(self):
    # Like the above test but instead of using jit inside custom_jvp, use cond.

    def cond_apply(f, x):
      return lax.cond(True, f, lambda x: x, x)

    @jax.custom_jvp
    def sin(x):
      return jnp.sin(x)
    def sin_jvp(primals, tangents):
      x, = primals
      xdot, = tangents
      y, c = cond_apply(lambda xs: (jnp.sin(xs[0]), jnp.cos(xs[1])), (x, x))
      ydot = c * xdot
      return y, ydot
    sin.defjvp(sin_jvp)

    save_cos = lambda prim, *_, **__: str(prim) == 'cos'
    f = new_checkpoint(partial(cond_apply, sin), policy=save_cos)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 0)

    save_sin = lambda prim, *_, **__: str(prim) == 'sin'
    f = new_checkpoint(partial(cond_apply, sin), policy=save_sin)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 1)

    f = new_checkpoint(partial(cond_apply, sin),
                       policy=jax.checkpoint_policies.everything_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 0)

    f = new_checkpoint(partial(cond_apply, sin),
                       policy=jax.checkpoint_policies.nothing_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 1)

    f = new_checkpoint(lambda x: cond_apply(sin, cond_apply(sin, x)),
                       policy=jax.checkpoint_policies.nothing_saveable)
    jtu.check_grads(f, (3.,), order=2, modes=['rev'])
    jaxpr = api.make_jaxpr(api.linearize(f, 4.)[1])(1.)
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 1)
    self.assertEqual(jaxpr_text.count(' cos '), 2)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_remat_of_while_loop(self, remat):
    def cond_fn(carry):
      i, _ = carry
      return i < 3
    def body_fn(carry):
      i, x = carry
      return i + 1, jnp.sin(x)
    def f(x):
      _, y = lax.while_loop(cond_fn, body_fn, (0, x))
      return y

    _, f_lin = jax.linearize(remat(f), 3.)
    y_dot = f_lin(1.0)
    expected = jax.grad(lambda x: jnp.sin(jnp.sin(jnp.sin(x))))(3.)
    self.assertArraysAllClose(y_dot, expected, check_dtypes=False)

    jaxpr = api.make_jaxpr(jax.linearize(remat(f), 4.)[1])(1.)
    self.assertIn(' sin ', str(jaxpr))
    self.assertIn(' cos ', str(jaxpr))

  def test_remat_of_while_loop_policy(self):
    def cond_fn(carry):
      i, _ = carry
      return i < 3
    def body_fn(carry):
      i, x = carry
      return i + 1, jnp.sin(x)
    def f(x):
      _, y = lax.while_loop(cond_fn, body_fn, (0, x))
      return y

    # even with a policy, we can't save residuals (w/o dynamic shapes)!
    save_cos = lambda prim, *_, **__: str(prim) == 'cos'
    g = new_checkpoint(f, policy=save_cos)
    jaxpr = api.make_jaxpr(jax.linearize(g, 4.)[1])(1.)
    self.assertIn(' sin ', str(jaxpr))
    self.assertIn(' cos ', str(jaxpr))

  @jtu.thread_unsafe_test()  # logging isn't thread-safe
  def test_remat_residual_logging(self):
    def f(x):
      x = jnp.sin(x)
      x = jnp.cos(x.sum())
      return x

    x = jnp.arange(3.)

    f1 = jax.remat(f)
    f2 = jax.remat(f, policy=lambda *_, **__: True)
    f3 = jax.remat(f, policy=lambda p, *_, **__: str(p) == 'cos')

    prev_level = logging.get_verbosity()
    try:
      logging.set_verbosity('DEBUG')
      with self.assertLogs(level=logging.DEBUG) as l:
        jax.grad(f1)(x)
    finally:
      logging.set_verbosity(prev_level)
    self.assertTrue(any('remat-decorated function saving inputs with shapes:'
                        in line for line in l.output))
    self.assertFalse(any('intermediates' in line for line in l.output))

    prev_level = logging.get_verbosity()
    try:
      logging.set_verbosity('DEBUG')
      with self.assertLogs(level=logging.DEBUG) as l:
        jax.grad(f2)(x)
    finally:
      logging.set_verbosity(prev_level)
    self.assertFalse(any('saving inputs' in line for line in l.output))
    self.assertTrue(any('remat-decorated function saving these intermediates:'
                        in line for line in l.output))
    self.assertTrue(any(' sin ' in line for line in l.output))
    self.assertTrue(any(' cos ' in line for line in l.output))

    prev_level = logging.get_verbosity()
    try:
      logging.set_verbosity('DEBUG')
      with self.assertLogs(level=logging.DEBUG) as l:
        jax.grad(f3)(x)
    finally:
      logging.set_verbosity(prev_level)
    self.assertTrue(any('remat-decorated function saving inputs with shapes:'
                        in line for line in l.output))
    self.assertTrue(any('and saving these intermediates:'
                        in line for line in l.output))
    self.assertFalse(any(' sin ' in line for line in l.output))
    self.assertTrue(any(' cos ' in line for line in l.output))

  def test_excess_precision_hell(self):
    finfo = jnp.finfo('bfloat16')
    eps = finfo.eps

    @jax.custom_vjp
    def dot(x):
      return jnp.dot(x, x)
    def dot_fwd(x):
      return dot(x), None
    def dot_bwd(_, g):
      return g,
    dot.defvjp(dot_fwd, dot_bwd)

    @jax.custom_vjp
    def foo(x):
      return jnp.float32(1.) * x.astype('float32')
    def foo_fwd(x):
      return foo(x), x
    def foo_bwd(x, _):
      return jnp.float32(1.) * x.astype('float32'),
    foo.defvjp(foo_fwd, foo_bwd)

    @jax.jit
    @partial(jax.remat, policy=lambda *_, **__: True)
    def f(x):
      x = dot(x)
      return foo(x)

    x = (jnp.bfloat16(1) + eps) * jnp.eye(2, dtype='bfloat16')
    y, vjp = jax.vjp(f, x)
    y_, = vjp(jnp.ones_like(y))
    self.assertAllClose(y, y_, atol=0, rtol=0)

  def test_concreteness_error_includes_user_code(self):
    @jax.remat
    def f(x):
      if x > 0:
        return x
      else:
        return jnp.sin(x)

    try:
      f(3.)
    except TracerBoolConversionError:
      self.assertIn('x > 0', traceback.format_exc())
    else:
      assert False

  def test_concreteness_error_includes_user_code_with_static_argnums(self):
    @partial(jax.remat, static_argnums=(1,))
    def f(x, _):
      if x > 0:
        return x
      else:
        return jnp.sin(x)

    try:
      f(3., 1.)
    except TracerBoolConversionError:
      self.assertIn('x > 0', traceback.format_exc())
    else:
      assert False


@jtu.with_config(jax_pprint_use_color=False)
class JaxprTest(jtu.JaxTestCase):

  def test_scalar_literals(self):
    jaxpr = api.make_jaxpr(lambda x: x + 2)(42)
    self.assertLen(jaxpr.jaxpr.constvars, 0)

  def test_abstract_inputs(self):
    jaxpr = api.make_jaxpr(lambda x: x + 2.)(
        types.SimpleNamespace(shape=(), dtype=np.dtype(np.float32)))
    self.assertEqual(jaxpr.in_avals[0].shape, ())
    self.assertEqual(jaxpr.in_avals[0].dtype, np.float32)

  def test_const(self):
    def fun(x):
      return (x, 1., np.zeros(1, dtype=jnp.float32))

    dtype = "f64" if config.enable_x64.value else "f32"
    expected = f"{{ lambda a:f32[1]; b:f32[]. let  in (b, 1.0:{dtype}[], a) }}"
    jaxpr = api.make_jaxpr(fun)(jnp.float32(0.))
    self.assertMultiLineStrippedEqual(expected, str(jaxpr))

  def test_cond(self):
    def f(x):
      return lax.cond(x >= 0.,
                      x + 1.,
                      lambda xt: xt + x,
                      x + 2.,
                      lambda xf: xf - x)
    expected = """{ lambda ; a:f32[]. let
    b:bool[] = ge a 0.0:f32[]
    c:f32[] = add a 1.0:f32[]
    d:f32[] = add a 2.0:f32[]
    e:i32[] = convert_element_type[new_dtype=int32 weak_type=False] b
    f:f32[] = cond[
      branches=(
        { lambda ; g:f32[] h:f32[] i:f32[]. let j:f32[] = sub i g in (j,) }
        { lambda ; k:f32[] l:f32[] m:f32[]. let n:f32[] = add l k in (n,) }
      )
    ] e a c d
  in (f,) }"""
    jaxpr = api.make_jaxpr(f)(jnp.float32(3.))
    self.assertMultiLineStrippedEqual(expected, str(jaxpr))

  def test_make_jaxpr_static_argnums(self):
    def f(x, y):
      return x + y

    jaxpr = api.make_jaxpr(f, static_argnums=(1,))(2, 3)
    self.assertIn('3', str(jaxpr))

  def test_make_jaxpr_return_shape(self):
    _, shape_tree = api.make_jaxpr(lambda x: (x + 1, jnp.zeros(2, jnp.float32)),
                                   return_shape=True)(jnp.int32(1))
    expected = (api.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
                api.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32))
    self.assertEqual(shape_tree, expected)

  def test_make_jaxpr_axis_env(self):
    def f(x):
      return x - lax.psum(x, 'i')
    jaxpr = api.make_jaxpr(f, axis_env=[('i', 4)])(2)
    self.assertIn('psum', str(jaxpr))

  def test_weak_type_jit_invariance(self):
    y = jnp.broadcast_to(3., (3,))
    self.assertTrue(y.aval.weak_type)

    def f():
      return lax.convert_element_type(y, 'float32')

    self.assertEqual(f().aval.weak_type, api.jit(f)().aval.weak_type)

  def test_elide_trivial_convert_element_types(self):
    # since we apply convert_element_type to a numpy.ndarray, the primitive is
    # still bound and thus would appear in the jaxpr if we didn't clean it up
    if config.enable_x64.value:
      x = np.arange(3, dtype='float64')
    else:
      x = np.arange(3, dtype='float32')

    cet = partial(lax.convert_element_type, new_dtype=x.dtype)
    jaxpr = api.make_jaxpr(lambda: cet(cet(cet(x))))()
    self.assertLen(jaxpr.eqns, 0)

  def test_elide_trivial_broadcasts(self):
    # since we apply broadcast to a numpy.ndarray, the primitive is still bound
    # and thus would appear in the jaxpr if we didn't clean it up
    jaxpr = api.make_jaxpr(lambda: lax.broadcast(np.float32(3), ()))()
    self.assertLen(jaxpr.jaxpr.eqns, 0)

  def test_convert_element_type_literal_constant_folding(self):
    # this convert_element_type is nontrivial, but because it's on a scalar we
    # constant-fold it
    cet = partial(lax.convert_element_type, new_dtype='float16')
    jaxpr = api.make_jaxpr(lambda: cet(3.))()
    self.assertLen(jaxpr.eqns, 0)

  def test_eqn_repr_with_no_lhs(self):
    def f(x):
      jax.debug.print("{}", x)
      return x
    jaxpr = jax.make_jaxpr(f)(np.int32(0))
    self.assertEqual(jaxpr.eqns[0].primitive, debugging.debug_callback_p)
    self.assertStartsWith(str(jaxpr.eqns[0]), "debug_callback[", )


class DCETest(jtu.JaxTestCase):

  def assert_dce_result(self, jaxpr: core.Jaxpr, used_outputs: list[bool],
                        expected_used_inputs: list[bool],
                        expected_num_eqns: int | None = None,
                        check_diff: bool = True):
    jaxpr_dce, used_inputs = pe.dce_jaxpr(jaxpr, used_outputs)
    core.check_jaxpr(jaxpr_dce)
    self.assertEqual(used_inputs, expected_used_inputs)
    if expected_num_eqns is not None:
      all_jaxprs = it.chain([jaxpr_dce], core.subjaxprs(jaxpr_dce))
      num_eqns = sum(len(subjaxpr.eqns) for subjaxpr in all_jaxprs)
      self.assertEqual(num_eqns, expected_num_eqns, msg=str(jaxpr_dce))

    rand_ = jtu.rand_small(np.random.RandomState(0))
    rand  = lambda v: rand_(v.aval.shape, v.aval.dtype)
    consts = [rand(v) for v in jaxpr.constvars]
    inputs = [rand(v) for v in jaxpr.invars   ]
    inputs_dce = [x for x, used in zip(inputs, used_inputs) if used]
    full_outs = core.eval_jaxpr(jaxpr    , consts, *inputs)
    expected_outs_dce = [y for y, used in zip(full_outs, used_outputs) if used]
    outs = core.eval_jaxpr(jaxpr_dce, consts, *inputs_dce)
    self.assertAllClose(outs, expected_outs_dce)

    if check_diff and expected_num_eqns != 0:
      f = lambda *args: core.eval_jaxpr(jaxpr_dce, consts, *args)
      jtu.check_grads(f, inputs_dce, order=2, modes=['rev'])

  def test_dce_jaxpr_scan_nontrivial_fixedpoint_carry(self):
    # The idea is that each element of the output carry tuple depends on the
    # corresponding carried input as well as the one to the left. The extensive
    # inputs and outputs aren't used here; just the carry depending on itself.
    def f(lst):
      def body(c, _):
        return [c[0]] + [c1 + c2 for c1, c2 in zip(c[:-1], c[1:])], None
      out, _ = jax.lax.scan(body, lst, None, length=len(lst))
      return out
    jaxpr = api.make_jaxpr(f)([1., 2., 3., 4.]).jaxpr
    self.assertLen(jaxpr.eqns, 1)
    self.assertLen(jaxpr.eqns[0].params['jaxpr'].jaxpr.eqns, 3)

    # If we use all but the last element, all but the first input is used, and
    # only one eqn is pruned.
    self.assert_dce_result(
        jaxpr,  used_outputs=[True, True, True, False],
        expected_used_inputs=[True, True, True, False],
        expected_num_eqns=1 + 2)  # one outer scan eqn, two adds in the body

    # Same as above if we just pull on the third element.
    self.assert_dce_result(
        jaxpr,  used_outputs=[False, False, True, False],
        expected_used_inputs=[True, True, True, False],
        expected_num_eqns=1 + 2)  # one outer scan eqn, two adds in the body

    # If we use all but the last two elements, the last two inputs are not used,
    # and two eqns can be pruned.
    self.assert_dce_result(
        jaxpr,  used_outputs=[True, True, False, False],
        expected_used_inputs=[True, True, False, False],
        expected_num_eqns=1 + 1)  # one outer scan eqn, one add in body

    # If we only use the last element, no eqns can be pruned.
    self.assert_dce_result(
        jaxpr,  used_outputs=[False, False, False, True],
        expected_used_inputs=[True, True, True, True],
        expected_num_eqns=1 + 3)  # one outer scan eqn, three adds in body

  def test_dce_jaxpr_scan_nontrivial_fixedpoint_carry_2(self):
    # This is much like the above test, except with a more interesting
    # dependence structure among the carry elements. Also add a const and
    # extensive input.
    hidden_sequence = [1, 2, 3, 5, 8]
    def f(lst):
      def body(c, _):
        _ = jnp.sin(np.array([3., 1., 4.]))
        sub_c = [c[i] for i in hidden_sequence]
        sub_c = [sub_c[0]] + [c1 * c2 for c1, c2 in zip(sub_c[:-1], sub_c[1:])]
        new_c = list(c)
        for i, elt in zip(hidden_sequence, sub_c):
          new_c[i] = elt
        return new_c, None
      out, _ = jax.lax.scan(body, lst, np.arange(len(lst), dtype='float32'))
      return out
    jaxpr = api.make_jaxpr(f)([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]).jaxpr
    self.assertLen(jaxpr.eqns, 1)
    self.assertLen(jaxpr.eqns[0].params['jaxpr'].jaxpr.eqns, 5)

    # If we use the value at index 8 only, all the hidden sequence must be kept
    # and no eqns can be pruned.
    used_outputs = [False] * 10
    used_outputs[8] = True
    expected_used_inputs = [False] * 10
    for i in hidden_sequence:
      expected_used_inputs[i] = True
    self.assert_dce_result(
        jaxpr,   used_outputs=used_outputs,
        expected_used_inputs=expected_used_inputs,
        expected_num_eqns=1 + 4)

    # If we use the value at any indices not in the hidden sequence, none of the
    # hidden sequence must be kept and we can prune all body eqns.
    used_outputs = [False] * 10
    expected_used_inputs = [False] * 10
    used_outputs[9] = expected_used_inputs[9] = True
    self.assert_dce_result(
        jaxpr,   used_outputs=used_outputs,
        expected_used_inputs=expected_used_inputs,
        expected_num_eqns=0)
    used_outputs[7] = expected_used_inputs[7] = True
    used_outputs[6] = expected_used_inputs[6] = True
    self.assert_dce_result(
        jaxpr,   used_outputs=used_outputs,
        expected_used_inputs=expected_used_inputs,
        expected_num_eqns=0)

    # If we use the value at index 3 only, some of the hidden sequence must be
    # kept but the rest pruned.
    used_outputs = [False] * 10
    used_outputs[3] = True
    expected_used_inputs = [False] * 10
    expected_used_inputs[1] = expected_used_inputs[2] = \
        expected_used_inputs[3] = True
    self.assert_dce_result(
        jaxpr,   used_outputs=used_outputs,
        expected_used_inputs=expected_used_inputs,
        expected_num_eqns=1 + 2)

  def test_dce_jaxpr_scan_nontrivial_fixedpoint_extensive_output(self):
    # Here we test how using the extensive output affects the carry.
    def f(lst):
      def body(c, _):
        return [c[-1], *c[:-1]], c[-1]
      _, ys = jax.lax.scan(body, lst, None, length=len(lst))
      return ys
    jaxpr = api.make_jaxpr(f)([1., 2., 3., 4.]).jaxpr
    self.assertLen(jaxpr.eqns, 1)

    # If we only use the extensive output, all carry elements are needed, and we
    # need to keep the scan itself.
    self.assert_dce_result(
        jaxpr,   used_outputs=[True],
        expected_used_inputs=[True, True, True, True],
        expected_num_eqns=1)

    # If we don't use the extensive output, no carry elements are needed, and we
    # don't need to keep the scan.
    self.assert_dce_result(
        jaxpr,   used_outputs=[False],
        expected_used_inputs=[False, False, False, False],
        expected_num_eqns=0)

  def test_dce_jaxpr_scan_extensive_input(self):
    # Here we test an extensive input affecting the carry.
    def cumprod(xs):
      def body(c, x):
        return c * x, c
      c, ys = jax.lax.scan(body, jnp.float32(1.), xs)
      return c, ys
    jaxpr = api.make_jaxpr(cumprod)(jnp.arange(1., 5., dtype='float32')).jaxpr

    # If we only use the carry output or extensive output, we need the input.
    self.assert_dce_result(
        jaxpr,   used_outputs=[True, False],
        expected_used_inputs=[True],
        expected_num_eqns=2)
    self.assert_dce_result(
        jaxpr,   used_outputs=[False, True],
        expected_used_inputs=[True],
        expected_num_eqns=2)

    # If we don't use either output, the scan is eliminated.
    self.assert_dce_result(
        jaxpr,   used_outputs=[False, False],
        expected_used_inputs=[False],
        expected_num_eqns=0)

  def test_dce_jaxpr_scan_overpruning(self):
    # This is a regression test for a specific issue.
    @jax.remat
    def scanned_f(c, x):
      out = jnp.tanh(c * x)
      return out, out

    def f(xs):
      return lax.scan(scanned_f, jnp.array(1., 'float32'), xs)

    xs = jnp.arange(10., dtype='float32')
    jaxpr = api.make_jaxpr(lambda xs: api.linearize(f, xs)[1])(xs).jaxpr

    jaxpr, used_inputs = pe.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars))
    self.assertLen(jaxpr.eqns, 1)
    self.assertLen(jaxpr.eqns[-1].params['jaxpr'].jaxpr.eqns, 2)

  def test_dce_jaxpr_scan_const_in_jvp(self):
    # The main point of this test is to check for a crash.
    @jax.custom_jvp
    def f(x):
      return x * np.arange(3.)
    @f.defjvp
    def f_jvp(primals, tangents):
      (x,), (xdot,) = primals, tangents
      return f(x), xdot * np.arange(3.)

    def g(x):
      def body(c, _):
        return f(c), None
      y, _ = jax.lax.scan(body, x, None, length=1)
      return y

    jaxpr = api.make_jaxpr(lambda x, xdot: api.jvp(g, (x,), (xdot,))
                           )(np.arange(3.), np.arange(3.)).jaxpr

    self.assert_dce_result(
        jaxpr,   used_outputs=[True, True],
        expected_used_inputs=[True, True])

    self.assert_dce_result(
        jaxpr,   used_outputs=[True, False],
        expected_used_inputs=[True, False])

  def test_dce_jaxpr_scan_results(self):
    # This doesn't test whether DCE is doing nontrivial work; instead it tests
    # whether the result after applying DCE computes different values. If
    # dce_jaxpr were an identity function, it'd pass this test!
    def f(cs, xs):
      def body(c, x):
        return (c[0], c[0] + c[1], jnp.arange(3.)), x
      cs, xs = jax.lax.scan(body, cs, xs)
      return cs[::2], xs[::2]

    cs = 1., 2., jnp.arange(3.)
    xs = jnp.arange(3.), jnp.arange(3.) + 5
    jaxpr_ = jax.make_jaxpr(f)(cs, xs)
    jaxpr, consts = jaxpr_.jaxpr, jaxpr_.consts
    jaxpr_pruned, used_inputs = pe.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars))

    args = (*cs, *xs)
    result1 = core.eval_jaxpr(jaxpr       , consts, *cs, *xs)
    pruned_args = [x for x, used in zip(args, used_inputs) if used]
    result2 = core.eval_jaxpr(jaxpr_pruned, consts, *pruned_args)
    self.assertAllClose(result1, result2)

  def test_dce_jaxpr_cond_trivial(self):
    x = jnp.array(1., dtype='float32')

    # start with 7 eqns, use both outputs so nothing can be pruned
    def f(x1, x2):
      return lax.cond(x1 > 0,
                      lambda x1, x2: (jnp.sin(x1), jnp.sin(x2)),
                      lambda x1, x2: (jnp.sin(x1), jnp.sin(x2)),
                      x1, x2)
    jaxpr = jax.make_jaxpr(f)(x, x).jaxpr
    self.assert_dce_result(jaxpr, [True, True], [True, True], 7)

    # use neither output so everything can be pruned
    self.assert_dce_result(jaxpr, [False, False], [False, False], 0)

  def test_dce_jaxpr_cond_nontrivial(self):
    x = jnp.array(1., dtype='float32')

    # start with 7 eqns, don't use an output so an eqn can be trimmed on each
    # side and x2 _can_ be pruned
    def f(x1, x2):
      return lax.cond(x1 > 0,
                      lambda x1, x2: (jnp.sin(x1), jnp.sin(x2)),
                      lambda x1, x2: (jnp.sin(x1), jnp.sin(x1)),
                      x1, x2)
    jaxpr = jax.make_jaxpr(f)(x, x).jaxpr
    self.assert_dce_result(jaxpr, [True, False], [True, False], 5)

    # start with 7 eqns, don't use an output so an eqn can be trimmed on each
    # side, but x2 _can't_ be pruned b/c of a swap
    def f(x1, x2):
      return lax.cond(x1 > 0,
                      lambda x1, x2: (jnp.sin(x1), jnp.sin(x2)),
                      lambda x1, x2: (jnp.sin(x2), jnp.sin(x1)),
                      x1, x2)
    jaxpr = jax.make_jaxpr(f)(x, x).jaxpr
    self.assert_dce_result(jaxpr, [True, False], [True, True], 5)

    # start with 7 eqns, only use x1 on one side and x2 on the other, so we
    # can't prune any inputs or eqns
    def f(x1, x2):
      return lax.cond(x1 > 0,
                      lambda x1, x2: (jnp.sin(x1), jnp.sin(x1)),
                      lambda x1, x2: (jnp.sin(x2), jnp.sin(x2)),
                      x1, x2)
    jaxpr = jax.make_jaxpr(f)(x, x).jaxpr
    self.assert_dce_result(jaxpr, [True, True], [True, True], 7)
    # use only one output, so we can prune eqns but not inputs
    self.assert_dce_result(jaxpr, [True, False], [True, True], 5)


class BufferDonationTest(jtu.BufferDonationTestCase):

  @jtu.device_supports_buffer_donation()
  def test_pmap_donate_argnums_invalidates_input(self):
    move = api.pmap(lambda x: x + x - x, donate_argnums=0)
    n = jax.local_device_count()
    x = api.pmap(lambda x: x)(jnp.ones([n]))
    y = move(x)
    self.assertDeleted(x)
    np.testing.assert_allclose(y, [1.] * n)

  @jtu.device_supports_buffer_donation()
  def test_pmap_nested_donate_ignored(self):
    pmap_fun = jit(lambda x: api.pmap(lambda y: y ** 2, donate_argnums=0)(x))
    a = api.pmap(lambda x: x)(jnp.array([1]))

    # NOTE(mattjj): stopped raising error here and instead just ignored
    # with self.assertRaisesRegex(ValueError, "nested.*not supported"):
    #   pmap_fun(a)

    pmap_fun(a)  # doesn't crash


class NamedCallTest(jtu.JaxTestCase):

  def test_non_jaxtype_arg(self):
    # For the test to fail without the invalid JaxType filter we need to pass
    # in a valid JaxType that forces the invalid Jaxtype to be raised to an
    # abstract value.
    def f(not_a_jaxtype, a_jaxtype):
      # then Jax needs to try and evaluate the abstractified non-JaxType
      if not_a_jaxtype:
        return a_jaxtype
      return 0

    f = api.named_call(f, name="test")
    out = jax.jit(f, static_argnums=(0,))("not a Jaxtype", 1)
    self.assertEqual(out, 1)

  @parameterized.parameters(jax.jit, jax.grad, jax.vmap, jax.remat)
  def test_jax_transforms(self, transform):
    f = jnp.sum
    x = jnp.array([1.])

    unnamed_out = transform(f)(x)
    named_out = transform(api.named_call(f, name="test"))(x)

    self.assertEqual(unnamed_out, named_out)

  def test_static_argnums(self):
    f = api.named_call(lambda x, y: y if x else None, name="test")
    f = jax.jit(f, static_argnums=(0,))
    out = f(True, 5)
    self.assertEqual(out, 5)

  def test_partial_eval(self):
    f = api.named_call(lambda x, y: y if x else None, name="test")
    f = jax.jit(functools.partial(f, True))
    out = f(5)
    self.assertEqual(out, 5)

  @parameterized.parameters(
    [dict(func=func, jit=jit)
      for func in ['identity_trivial', 'identity', 'closure_trivial', 'closure',
                   'asarray', 'device_put']
      for jit in jtu.JIT_IMPLEMENTATION
      if not (jit._name == "noop" and func in ('identity', 'identity_trivial'))
    ],
  )
  def test_integer_overflow(self, jit, func):
    funcdict = {
      'identity_trivial': lambda x: x,  # may hit trivial dispatch path
      'identity': lambda x: x + 0,
      'closure_trivial': lambda x: jax.jit(lambda: x)(),
      'closure': lambda x: jax.jit(lambda: x + 0)(),
      'asarray': lambda x: jnp.asarray(x),  # add lambdas so no cross-test cache
      'device_put': lambda x: api.device_put(x),
    }

    f = jit(funcdict[func])

    int_dtype = dtypes.canonicalize_dtype(jnp.int64)
    int_max = np.iinfo(int_dtype).max
    int_min = np.iinfo(int_dtype).min

    # check before any jit cache entries
    self.assertRaises(OverflowError, f, int_max + 1)
    self.assertRaises(OverflowError, f, int_min - 1)

    self.assertEqual(f(int_max).dtype, int_dtype)
    self.assertEqual(f(int_min).dtype, int_dtype)
    self.assertAllClose(f(int_max), int_max)
    self.assertAllClose(f(int_min), int_min)

    # check after any cache entries
    self.assertRaises(OverflowError, f, int_max + 1)
    self.assertRaises(OverflowError, f, int_min - 1)
    if func in ('trivial', 'identity'):
      self.assertRaisesRegex(
          OverflowError, 'An overflow.*whose argument path is x.', f,
          int_max + 1)


class BackendsTest(jtu.JaxTestCase):

  @unittest.skipIf(not sys.executable, "test requires sys.executable")
  @jtu.run_on_devices("cpu")
  def test_no_backend_warning_on_cpu_if_platform_specified(self):
    warning_not_expected = (
      "import jax; "
      "jax.config.update('jax_platform_name', 'cpu'); "
      "jax.numpy.arange(10)")

    result = subprocess.run([sys.executable, '-c', warning_not_expected],
                            check=True, capture_output=True)
    assert "may be present" not in result.stderr.decode()


class CleanupTest(jtu.JaxTestCase):
  def test_call_wrapped_second_phase_cleanup(self):
    try:
      jax.vmap(lambda x: x, out_axes=None)(jnp.arange(3))
    except:
      assert core.trace_state_clean()  # this is the hard one
    assert core.trace_state_clean()


class EnvironmentInfoTest(jtu.JaxTestCase):
  @parameterized.parameters([True, False])
  @jtu.thread_unsafe_test()
  def test_print_environment_info(self, return_string):
    # Flush stdout buffer before checking.
    sys.stdout.flush()
    with jtu.capture_stdout() as stdout:
      result = jax.print_environment_info(return_string=return_string)
    if return_string:
      self.assertEmpty(stdout())
    else:
      self.assertIsNone(result)
      result = stdout()
    assert f"jax:    {jax.__version__}" in result
    assert f"jaxlib: {lib.version_str}" in result
    assert f"numpy:  {np.__version__}" in result

class AutodidaxTest(jtu.JaxTestCase):
  def test_autodidax_smoketest(self):
    autodidax_file = os.path.join(
      os.path.dirname(os.path.dirname(__file__)),
      'docs',
      'autodidax.py')
    if not os.path.exists(autodidax_file):
      self.skipTest("Cannot locate autodidax.py")
    spec = importlib.util.spec_from_file_location('autodidax', autodidax_file)
    autodidax_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(autodidax_module)

class GarbageCollectionTest(jtu.JaxTestCase):

  @jtu.thread_unsafe_test()  # GC isn't predictable
  def test_xla_gc_callback(self):
    # https://github.com/jax-ml/jax/issues/14882
    x_np = np.arange(10, dtype='int32')
    x_jax = jax.device_put(x_np)
    x_np_weakref = weakref.ref(x_np)

    del x_np
    del x_jax
    gc.collect()

    assert x_np_weakref() is None


class OverrideLoweringTest(jtu.JaxTestCase):

  def test_sharding_constraint_as_noop(self):
    def f(x):
      return jax.lax.with_sharding_constraint(
          x, jax.sharding.SingleDeviceSharding(jax.devices()[0]))

    def wsc_as_noop(ctx, operand, *args, **kwargs):
      del ctx, args, kwargs
      return [operand]

    rules = ((jax.lax.sharding_constraint_p, wsc_as_noop),)
    lowered_ir = (
        jax.jit(f)
        .trace(jax.ShapeDtypeStruct((2, 4), dtype=jnp.bfloat16))
        .lower(_private_parameters=mlir.LoweringParameters(
            override_lowering_rules=rules))
        .as_text()
    )
    self.assertNotIn("stablehlo.custom_call @Sharding", lowered_ir)


class InputSavedVJPTest(jtu.JaxTestCase):

  def test_basic(self):
    def f(x, y):
      return x * y

    primals = 2., 3.
    y, f_vjp = api.si_vjp(f, [True, True], *primals)
    arg_cts = f_vjp(1., *primals)
    self.assertAllClose(y, 6.)
    self.assertAllClose(arg_cts, (3., 2.))

  def test_basic_pass_through_jit(self):
    def f(x, y):
      return x * y

    @jax.jit
    def g():
      primals = 2., 3.
      y, f_vjp = api.si_vjp(f, [True, True], *primals)
      return y, f_vjp

    @jax.jit
    def h(f_vjp):
      return f_vjp(1., 2., 3.)

    y, f_vjp = g()
    arg_cts = h(f_vjp)
    self.assertAllClose(y, 6.)
    self.assertAllClose(arg_cts, (3., 2.))

  def test_basic_unused(self):
    f = jnp.sin
    primals = 3.,
    y, f_vjp = api.si_vjp(f, [True], *primals)
    x_ct, = f_vjp(1., *primals)
    self.assertAllClose(y, jnp.sin(3.))
    self.assertAllClose(x_ct, jnp.cos(3.))

    with self.assertRaisesRegex(Exception, "not used by the backward pass: x"):
      _ = api.si_vjp(f, [True], *primals, allow_unused=False)

  def test_basic_opaque(self):
    f = jnp.sin
    primals = 3.,
    with self.assertRaisesRegex(Exception, "the backward pass requires opaque"):
      _ = api.si_vjp(f, [True], *primals, allow_opaque=False)

  def test_basic_pytree_error(self):
    def f(x):
      return [x['hi'] * x['bye']]

    y, f_vjp = api.si_vjp(f, [True], {'hi': 2., 'bye': 3.})
    arg_ct, = f_vjp([1.], {'hi': 2., 'bye': 3.})
    self.assertAllClose(y, [6.])
    self.assertAllClose(arg_ct, {'hi': 3., 'bye': 2.})

    with self.assertRaisesRegex(ValueError, "but the structures differ"):
      f_vjp(1., {'hi': 2.})

  def test_fsdp(self):
    # see https://github.com/jax-ml/jax/pull/27017 for why this is called "fsdp"
    def f2(x, w):
      x = 1. * x
      x = x @ w
      x = 2. * x
      return x

    x = jnp.ones((3, 4))
    w = jnp.ones((4, 4))
    y, f2_sivjp = api.si_vjp(f2, [False, True], x, w)
    y_grad = jnp.ones_like(y)
    x_grad, w_grad = f2_sivjp(y_grad, w)
    self.assertAllClose(x_grad, 2. * y_grad @ w.T)
    self.assertAllClose(w_grad, 2. * x.T @ y_grad)

  def test_doesnt_leak_symbolic_zeros(self):
    _, vjp = api.si_vjp(lambda x: 1., [False], 3.14)
    ans, = vjp(1.0)
    self.assertIsInstance(ans, jax.Array)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
