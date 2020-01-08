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

import collections
from contextlib import contextmanager
import copy
from functools import partial
import unittest
import warnings
import weakref

from absl import logging
from absl.testing import absltest, parameterized
import numpy as onp
import six

if six.PY3:
  import concurrent.futures

import jax
import jax.numpy as np
from jax import jit, grad, device_put, jacfwd, jacrev, hessian
from jax import api, core, lax, lax_reference
from jax.core import Primitive
from jax.interpreters import ad
from jax.interpreters import xla
from jax.abstract_arrays import concretization_err_msg
from jax.lib import xla_bridge as xb
from jax import test_util as jtu
from jax import tree_util

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

class APITest(jtu.JaxTestCase):

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

  def test_jit_static_args(self):
    side = []

    def f(x, y, z, flag=False, flag2=False):
      assert flag
      side.append(None)
      return 100*x + 10*y + z

    f1 = jit(f, static_argnums=(3, 4))
    assert f1(1, 2, 3, True, False) == 123
    assert len(side) == 1
    assert f1(2, 1, 3, True, False) == 213
    assert len(side) == 1
    assert f1(2, 1, 3, True, True) == 213
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

  def test_jit_kwargs(self):
    side = []

    def f(x, y, z):
      side.append(None)
      return 100*x + 10*y + z

    f = jit(f)
    assert f(1, 2, 3) == 123
    assert len(side) == 1
    assert f(1, 2, 3) == 123
    assert len(side) == 1

    assert f(1, 2, z=3) == 123
    assert len(side) == 2  # actually recompiles from kwarg
    assert f(1, 2, z=3) == 123
    assert len(side) == 2  # but should still cache

    f(1, 2, z=onp.zeros(3))  # doesn't crash

  def test_jit_many_args_tuples(self):
    @jit
    def f(args_list):
      return sum(args_list)

    make_tuple = xla.make_tuple

    counts = [0]
    def make_tuple_and_count(*args, **kwargs):
      counts[0] += 1
      return make_tuple(*args, **kwargs)

    try:
      xla.make_tuple = make_tuple_and_count
      ans = f(list(range(500)))
    finally:
      xla.make_tuple = make_tuple

    expected = sum(range(500))
    self.assertEqual(counts[0], 1)  # formed a tuple on dispatch
    self.assertEqual(ans, expected)  # computed the correct result

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

  def test_bad_input(self):
    def f(x):
      return x

    self.assertRaisesRegex(
      TypeError, ".* 'foo' of type <.*'str'> is not a valid JAX type",
      lambda: grad(f)("foo"))

    self.assertRaisesRegex(
      TypeError, ".* 'foo' of type <.*'str'> is not a valid JAX type",
      lambda: jit(f)("foo"))

  def test_grad_tuple_output(self):
    jtu.check_raises(lambda: grad(lambda x: (x,x))(1.0), TypeError,
                     "Gradient only defined for scalar-output functions. ")

  def test_grad_unit_output(self):
    jtu.check_raises(lambda: grad(lambda x: ())(onp.zeros(3)), TypeError,
                     "Gradient only defined for scalar-output functions. ")

  def test_grad_nonscalar_output(self):
    jtu.check_raises(lambda: grad(lambda x: x)(onp.zeros(3)), TypeError,
                     "Gradient only defined for scalar-output functions. ")

  def test_unwrapped_numpy(self):
    def f(x):
      return onp.exp(x)

    jtu.check_raises(lambda: grad(f)(onp.zeros(3)), Exception,
                     "Tracer can't be used with raw numpy functions. "
                     "You might have\n  import numpy as np\ninstead of\n"
                     "  import jax.numpy as np")

  def test_binop_mismatch(self):
    def f(x, y):
      return x + y

    jtu.check_raises(
        lambda: f(np.zeros(3), np.zeros(4)),
        TypeError,
        "add got incompatible shapes for broadcasting: (3,), (4,).")

    jtu.check_raises(
        lambda: grad(f)(onp.zeros(3), onp.zeros(4)),
        TypeError,
        "add got incompatible shapes for broadcasting: (3,), (4,).")

  def test_dot_mismatch(self):
    def f(x, y):
      return np.dot(x, y)

    self.assertRaisesRegex(
      TypeError, "Incompatible shapes for dot: got \\(3L?,\\) and \\(4L?,\\).",
      lambda: grad(f)(onp.zeros(3), onp.zeros(4)))

  def test_switch_value_jit(self):
    def f(x):
      y = x > 0
      if y:
        return x
      else:
        return -x

    assert grad(f)(1.0) == 1.0
    assert grad(f)(-1.0) == -1.0
    jtu.check_raises(lambda: jit(f)(1), TypeError, concretization_err_msg(bool))

  def test_range_err(self):
    def f(x, n):
      for i in range(n):
        x = x + i
      return x

    assert jit(f, static_argnums=(1,))(0, 5) == 10
    self.assertRaisesRegex(
        TypeError,
        "('JaxprTracer' object cannot be interpreted as an integer"
        "|Abstract value passed to .*)",
        lambda: jit(f)(0, 5))

  def test_casts(self):
    for castfun in [float, complex, hex, oct] + list(six.integer_types):
      f = lambda x: castfun(x)
      self.assertRaisesRegex(
          TypeError,
          "('JaxprTracer' object cannot be interpreted as an integer"
          "|Abstract value passed to .*)", lambda: jit(f)(0))

  def test_unimplemented_interpreter_rules(self):
    foo_p = Primitive('foo')
    def foo(x):
      return foo_p.bind(x)

    jtu.check_raises(lambda: foo(1.0), NotImplementedError,
                     "Evaluation rule for 'foo' not implemented")

    jtu.check_raises(lambda: jit(foo)(1.0), NotImplementedError,
                     "Abstract evaluation for 'foo' not implemented")

    jtu.check_raises(lambda: grad(foo)(1.0), NotImplementedError,
                     "Forward-mode differentiation rule for 'foo' not implemented")

    foo_p.def_abstract_eval(lambda x: x)

    jtu.check_raises(lambda: jit(foo)(1.0), NotImplementedError,
                     "XLA translation rule for primitive 'foo' not found")

    foo_p.def_impl(lambda x: x)
    ad.defjvp(foo_p, lambda g, x: foo(g))

    jtu.check_raises(lambda: grad(foo)(1.0), NotImplementedError,
                     "Reverse-mode differentiation rule for 'foo' not implemented")

  def test_device_put_and_get(self):
    x = onp.arange(12.).reshape((3, 4)).astype("float32")
    dx = api.device_put(x)
    self.assertIsInstance(dx, xla.DeviceArray)
    x2 = api.device_get(dx)
    self.assertIsInstance(x2, onp.ndarray)
    assert onp.all(x == x2)

    y = [x, (2 * x, 3 * x)]
    dy = api.device_put(y)
    y2 = api.device_get(dy)
    self.assertIsInstance(y2, list)
    self.assertIsInstance(y2[0], onp.ndarray)
    assert onp.all(y2[0] == x)
    self.assertIsInstance(y2[1], tuple)
    self.assertIsInstance(y2[1][0], onp.ndarray)
    assert onp.all(y2[1][0] == 2 * x)
    self.assertIsInstance(y2[1][1], onp.ndarray)
    assert onp.all(y2[1][1] == 3 * x)

  def test_device_put_across_devices(self):
    if xb.device_count() == 1:
      raise unittest.SkipTest("this test requires multiple devices")
    d1, d2 = xb.local_devices()[:2]
    x = api.device_put(onp.array([1,2,3]), device=d1)
    self.assertEqual(x.device_buffer.device(), d1)
    y = api.device_put(x, device=d2)
    self.assertEqual(y.device_buffer.device(), d2)
    # Make sure these don't crash
    api.device_put(x)
    api.device_put(y)

  @jtu.skip_on_devices("cpu")
  def test_device_put_across_platforms(self):
    default_device = jax.devices()[0]
    cpu_device = jax.devices("cpu")[0]

    onp_arr = onp.array([1,2,3])
    scalar = 1
    device_arr = np.array([1,2,3])
    assert device_arr.device_buffer.device() is default_device

    for val in [onp_arr, device_arr, scalar]:
      x = api.device_put(val, device=cpu_device)
      self.assertEqual(x.device_buffer.device(), cpu_device)

    y = api.device_put(x)
    self.assertEqual(y.device_buffer.device(), default_device)

  @jtu.skip_on_devices("tpu")
  def test_jacobian(self):
    R = onp.random.RandomState(0).randn
    A = R(4, 3)
    x = R(3)

    f = lambda x: np.dot(A, x)
    assert onp.allclose(jacfwd(f)(x), A)
    assert onp.allclose(jacrev(f)(x), A)

    f = lambda x: np.tanh(np.dot(A, x))
    assert onp.allclose(jacfwd(f)(x), jacrev(f)(x))

  @jtu.skip_on_devices("tpu")
  def test_hessian(self):
    R = onp.random.RandomState(0).randn
    A = R(4, 4)
    x = R(4)

    f = lambda x: np.dot(x, np.dot(A, x))
    assert onp.allclose(hessian(f)(x), A + A.T)

  def test_std_basis(self):
    basis = api._std_basis(np.zeros(3))
    assert getattr(basis, "shape", None) == (3, 3)
    assert onp.allclose(basis, onp.eye(3))

    basis = api._std_basis(np.zeros((3, 3)))
    assert getattr(basis, "shape", None) == (9, 3, 3)
    assert onp.allclose(basis, onp.eye(9).reshape(9, 3, 3))

    basis = api._std_basis([0., (np.zeros(3), np.zeros((3, 4)))])
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

      R = onp.random.RandomState(0).randn
      x = R(2)
      y = R(3)
      ans = jacfun(lambda x, y: {'x': x, 'xy': np.outer(x, y)})(x, y)
      expected = {'x': onp.eye(2),
                  'xy': onp.kron(onp.eye(2), y[:, None]).reshape(2, 3, 2)}
      self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("tpu")
  def test_hessian_on_pytrees(self):
    ans = hessian(lambda x: np.array(x)**2)((1., 2.))
    expected = ((onp.array([2., 0.]), onp.array([0., 0.])),
                (onp.array([0., 0.]), onp.array([0., 2.])))
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("tpu")
  def test_issue1372(self):
    def quad(x):
      return np.dot(x, x)

    def f(x, u):
      return quad(x) + quad(u)

    x, u = np.ones(5), np.ones(2)

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

  def test_disable_jit(self):
    effects = []

    @api.jit
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

  def test_large_device_constant(self):
    ans = jit(lambda x: 2 * x)(np.ones(int(2e6)))  # doesn't crash
    self.assertAllClose(ans, onp.ones(int(2e6)) * 2., check_dtypes=False)

  def test_grad_and_aux_basic(self):
    g, aux = grad(lambda x: (x**3, [x**2]), has_aux=True)(3.)
    self.assertAllClose(g, grad(lambda x: x**3)(3.), check_dtypes=True)
    self.assertAllClose(aux, [9.], check_dtypes=False)

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
      return aux[0] * np.sin(x)

    f2 = lambda x: x**3 * np.sin(x)

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
    # see https://github.com/google/jax/issues/1950
    def f(x):
      aux = dict(identity=x, p1=x+1)
      return x ** 2, aux

    _, aux = jax.grad(f, has_aux=True)(3.)
    self.assertIsInstance(aux, dict)
    for val in aux.values():
      self.assertNotIsInstance(val, core.Tracer)

  def test_jvp_mismatched_arguments(self):
    self.assertRaisesRegex(
      TypeError,
      ("primal and tangent arguments to jax.jvp must have the same tree "
       "structure"),
      lambda: api.jvp(lambda x, y: x * y, (onp.float32(2),), ()))
    # If primals and tangents must both be tuples or both lists
    self.assertRaisesRegex(
      TypeError,
      ("primal and tangent arguments to jax.jvp must have the same tree "
       "structure"),
      lambda: api.jvp(lambda x, y: x * y, (onp.float32(2),), [onp.float32(2)]))
    self.assertRaisesRegex(
      TypeError,
      "primal and tangent arguments to jax.jvp must have equal types",
      lambda: api.jvp(lambda x: -x, (onp.float16(2),), (onp.float32(4),)))

  def test_jvp_non_tuple_arguments(self):
    def f(x, y): return x + y
    self.assertRaisesRegex(
        TypeError,
        "primal and tangent arguments to jax.jvp must be tuples or lists; found float and tuple.",
        lambda: partial(api.jvp(f, 0., (1.,))))
    self.assertRaisesRegex(
        TypeError,
        "primal and tangent arguments to jax.jvp must be tuples or lists; found tuple and ndarray.",
        lambda: partial(api.jvp(f, (0.,), onp.array([1., 2.]))))

  def test_vjp_mismatched_arguments(self):
    _, pullback = api.vjp(lambda x, y: x * y, onp.float32(3), onp.float32(4))
    self.assertRaisesRegex(
      TypeError,
      "Tree structure of cotangent input.*does not match",
      lambda: pullback((onp.float32(7), onp.float32(100))))
    self.assertRaisesRegex(
      TypeError,
      "Type of cotangent input to vjp pullback.*does not match type",
      lambda: pullback((onp.float16(42))))

  def test_jarrett_jvps(self):
    def f1(x):
      return np.sin(np.sin(np.sin(x)))
    f2 = api.jarrett(f1)

    for x in [3., onp.array([2., 3., 4.])]:
      self.assertAllClose(f1(x), f2(x), check_dtypes=True)

      _, f1_vjp = api.vjp(f1, x)
      _, f2_vjp = api.vjp(f2, x)
      self.assertAllClose(f1_vjp(x), f2_vjp(x), check_dtypes=True)

      # TODO(mattjj): test that constants/literals are set up properly
      # jaxpr2 = api.make_jaxpr(f2_vjp)(x)
      # assert len(jaxpr2.constvars) == 1

  def test_jarrett_jvps2(self):
    def f1(x, y):
      return np.sin(x) * np.cos(y) * np.sin(x) * np.cos(y)
    f2 = api.jarrett(f1)

    # TODO(mattjj): doesn't work for (3., onp.array([4., 5.]))
    for x, y in [(3., 4.), (onp.array([5., 6.]), onp.array([7., 8.]))]:
      self.assertAllClose(f1(x, y), f2(x, y), check_dtypes=True)

      _, f1_vjp = api.vjp(f1, x, y)
      _, f2_vjp = api.vjp(f2, x, y)
      self.assertAllClose(f1_vjp(y), f2_vjp(y), check_dtypes=True)

      # TODO(mattjj): test that constants/literals are set up properly
      # jaxpr2 = api.make_jaxpr(f2_vjp)(y)
      # assert len(jaxpr2.constvars) == 2

  def test_jvp_jit_cached(self):
    """Bug in caching in presence of JVP and JIT."""

    def func(x):
      def inner(y):
        return y * x

      # Must have two calls to the inner jit (the second one hits the cache)
      res1 = api.jit(inner)(4.)
      res2 = api.jit(inner)(5.)
      return res1 + res2

    self.assertAllClose((45., 9.), api.jvp(func, (5.,), (1.,)), check_dtypes=True)


  def test_complex_grad_raises_error(self):
    self.assertRaises(TypeError, lambda: grad(lambda x: np.sin(x))(1 + 2j))

  def test_holomorphic_grad(self):
    out = grad(lambda x: np.sin(x), holomorphic=True)(1 + 2j)
    expected = 2.0327230070196656 - 3.0518977991518j
    self.assertAllClose(out, expected, check_dtypes=False)

  def test_nonholomorphic_grad(self):
    zs = 0.5j * onp.arange(5) + onp.arange(5)

    def f(z):
      return np.sum(np.cos(np.abs(z)))

    ans = grad(f)(zs)
    expected = onp.array([ 0.        +0.j,
                          -0.80430663+0.40215331j,
                          -0.70368982+0.35184491j,
                           0.1886467 -0.09432335j,
                           0.86873727-0.43436864j])
    self.assertAllClose(ans, expected, check_dtypes=False,
                        atol=jtu.default_gradient_tolerance,
                        rtol=jtu.default_gradient_tolerance)

  def test_complex_output_jacrev_raises_error(self):
    self.assertRaises(TypeError, lambda: jacrev(lambda x: np.sin(x))(1 + 2j))

  def test_nonholomorphic_jacrev(self):
    # code based on https://github.com/google/jax/issues/603
    zs = 0.5j * onp.arange(5) + onp.arange(5)

    def f(z):
      return np.cos(np.linalg.norm(2 * z))

    ans = jacrev(f)(zs)
    expected = grad(f)(zs)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def test_complex_input_jacfwd_raises_error(self):
    self.assertRaises(TypeError, lambda: jacfwd(lambda x: np.sin(x))(1 + 2j))

  def test_defvjp_all(self):
    foo_p = Primitive('foo')
    def foo(x): return 2. * foo_p.bind(x)

    ad.defvjp_all(foo_p, lambda x: (x**2, lambda g: (4 * g * np.sin(x),)))
    val_ans, grad_ans = api.value_and_grad(foo)(3.)
    self.assertAllClose(val_ans, 2 * 3.**2, check_dtypes=False)
    self.assertAllClose(grad_ans, 4 * 2 * onp.sin(3.), check_dtypes=False)

  def test_defvjp_all_const(self):
    foo_p = Primitive('foo')
    def foo(x): return foo_p.bind(x)

    ad.defvjp_all(foo_p, lambda x: (x**2, lambda g: (12.,)))
    val_ans, grad_ans = api.value_and_grad(foo)(3.)
    self.assertAllClose(val_ans, 9., check_dtypes=False)
    self.assertAllClose(grad_ans, 12., check_dtypes=True)

  def test_defvjp_all_higher_order_revmode(self):
    foo_p = Primitive('foo')
    def foo(x): return 2. * foo_p.bind(x)

    ad.defvjp_all(foo_p, lambda x: (x**2, lambda g: (g * x ** 2,)))
    ans = api.grad(api.grad(foo))(3.)
    self.assertAllClose(ans, 2 * 2 * 3., check_dtypes=False)

  def test_defvjp_all_multiple_arguments(self):
    # also tests passing in symbolic zero tangents b/c we differentiate wrt only
    # the first argument in one case

    foo_p = Primitive('foo')
    def foo(x, y): return foo_p.bind(x, y)

    def vjpfun(x, y):
      out = x**2 + y**3
      vjp = lambda g: (g + x + y, g * x * 9.)
      return out, vjp

    ad.defvjp_all(foo_p, vjpfun)
    val_ans, grad_ans = api.value_and_grad(foo)(3., 4.)
    self.assertAllClose(val_ans, 3.**2 + 4.**3, check_dtypes=False)
    self.assertAllClose(grad_ans, 1. + 3. + 4., check_dtypes=False)

    ans = api.grad(foo, (0, 1))(3., 4.)
    self.assertAllClose(ans, (1. + 3. + 4., 1. * 3. * 9.), check_dtypes=False)

  def test_defvjp_all(self):
    @api.custom_transforms
    def foo(x):
      return np.sin(x)

    api.defvjp_all(foo, lambda x: (np.sin(x), lambda g: (g * x,)))
    val_ans, grad_ans = api.value_and_grad(foo)(3.)
    self.assertAllClose(val_ans, onp.sin(3.), check_dtypes=False)
    self.assertAllClose(grad_ans, 3., check_dtypes=False)

  # TODO(mattjj): add defvjp_all test with pytree arguments

  def test_defvjp(self):
    @api.custom_transforms
    def foo(x, y):
      return np.sin(x * y)

    api.defvjp(foo, None, lambda g, _, x, y: g * x * y)
    val_ans, grad_ans = api.value_and_grad(foo)(3., 4.)
    self.assertAllClose(val_ans, onp.sin(3. * 4.), check_dtypes=False)
    self.assertAllClose(grad_ans, 0., check_dtypes=False)

    ans_0, ans_1 = api.grad(foo, (0, 1))(3., 4.)
    self.assertAllClose(ans_0, 0., check_dtypes=False)
    self.assertAllClose(ans_1, 3. * 4., check_dtypes=False)

  def test_defvjp_higher_order(self):
    @api.custom_transforms
    def foo(x):
      return np.sin(2. * x)

    api.defvjp(foo, lambda g, _, x: g * np.cos(x))
    ans = api.grad(api.grad(foo))(2.)
    expected = api.grad(api.grad(np.sin))(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_defvjp_use_ans(self):
    @api.custom_transforms
    def foo(x, y):
      return np.sin(x * y)

    api.defvjp(foo, None, lambda g, ans, x, y: g * x * y + np.cos(ans))
    val_ans, grad_ans = api.value_and_grad(foo, 1)(3., 4.)
    self.assertAllClose(val_ans, onp.sin(3. * 4.), check_dtypes=False)
    self.assertAllClose(grad_ans, 3. * 4. + onp.cos(onp.sin(3. * 4)),
                        check_dtypes=False)

  # TODO
  # def test_defjvp_closure_error(self):
  #   def foo(x):
  #     @api.custom_transforms
  #     def bar(y):
  #       return x * y

  #     api.defjvp(bar, lambda y_dot, ans, y: x * y)
  #     return bar(x)
  #   jtu.check_raises(
  #       lambda: api.jvp(foo, (1.,), (1.,)), ValueError,
  #       "Detected differentiation with respect to closed-over values with "
  #       "custom JVP rule, which isn't supported.")

  # TODO
  # def test_defvjp_closure_error(self):
  #   def foo(x):
  #     @api.custom_transforms
  #     def bar(y):
  #       return x * y

  #     api.defvjp(bar, lambda g, ans, y: x * y)
  #     return bar(x)
  #   jtu.check_raises(
  #       lambda: grad(foo)(1.,), ValueError,
  #       "Detected differentiation w.r.t. variables from outside "
  #       "the scope of <jax.custom_transforms function bar>, but defvjp and "
  #       "defvjp_all only support differentiation w.r.t. positional arguments.")

  def test_custom_transforms_eval_with_pytrees(self):
    @api.custom_transforms
    def f(x):
      a, b = x[0], x[1]
      return {'hi': 2 * a, 'bye': 2 * b}

    ans = f((1, 2))
    self.assertEqual(ans, {'hi': 2 * 1, 'bye': 2 * 2})

  def test_custom_transforms_jit_with_pytrees(self):
    @api.custom_transforms
    def f(x):
      a, b = x[0], x[1]
      return {'hi': 2 * a, 'bye': 2 * b}

    ans = jit(f)((1, 2))
    self.assertEqual(ans, {'hi': 2 * 1, 'bye': 2 * 2})

  def test_custom_transforms_jit_with_pytrees_consts(self):
    # The purpose of this test is to exercise the custom_transforms default
    # translation rule in how it deals with constants that are too large to be
    # treated as literals (at the time of writing).
    z = onp.arange(10.)

    @api.custom_transforms
    def f(x):
      a, b = x[0], x[1]
      return {'hi': 2 * a, 'bye': z * b}

    ans = jit(f)((1, 2))
    self.assertAllClose(ans, {'hi': 2 * 1, 'bye': z * 2}, check_dtypes=False)

  def test_custom_transforms_jvp_with_pytrees(self):
    @api.custom_transforms
    def f(x):
      a, b = x[0], x[1]
      return {'hi': 2 * a, 'bye': 2 * b}

    ans, out_tangent = api.jvp(f, ((1, 2),), ((3, 4),))
    self.assertEqual(ans, {'hi': 2 * 1, 'bye': 2 * 2})
    self.assertEqual(out_tangent, {'hi': 2 * 3, 'bye': 2 * 4})

  def test_custom_transforms_vmap_with_pytrees(self):
    @api.custom_transforms
    def f(x):
      a, b = x[0], x[1]
      return {'hi': 2 * a, 'bye': 2 * b}

    ans = api.vmap(f)((onp.arange(3), onp.ones((3, 2))))
    expected = {'hi': 2 * onp.arange(3), 'bye': 2 * onp.ones((3, 2))}
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_custom_transforms_jvp_with_closure(self):
    def f(x):
      @api.custom_transforms
      def g(y):
        return x * y
      return g(x)

    ans = api.grad(f)(1.)
    expected = 2.
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_custom_gradient(self):
    @api.custom_gradient
    def f(x):
      return x ** 2, lambda g: (g * x,)

    self.assertAllClose(f(3.), 9., check_dtypes=False)
    self.assertAllClose(api.grad(f)(3.), 3., check_dtypes=False)

  def test_legacy_devicearray_repr(self):
    dx = device_put(3.)
    str(dx.item())  # doesn't crash

  def test_devicearray_repr(self):
    x = device_put(np.zeros(3))
    self.assertIsInstance(x, xla.DeviceArray)
    repr(x)  # doesn't crash

    x = device_put(np.ones(3) + 1j * np.ones(3))
    self.assertIsInstance(x, xla.DeviceArray)
    repr(x)  # doesn't crash

  def test_devicearray_delete(self):
    x = device_put(1.)
    x.delete()
    self.assertRaisesRegex(ValueError, "DeviceValue has been deleted.",
                            lambda: repr(x))

  def test_devicearray_block_until_ready(self):
    x = device_put(1.)
    y = x.block_until_ready()
    # Tests mostly that block_until_ready() does not produce an error.
    self.assertTrue(y is x)

  def test_namedtuple_transparency(self):
    # See https://github.com/google/jax/issues/446
    Point = collections.namedtuple("Point", ["x", "y"])

    def f(pt):
      return np.sqrt(pt.x ** 2 + pt.y ** 2)

    pt = Point(1., 2.)

    f(pt)  # doesn't crash
    g = api.grad(f)(pt)
    self.assertIsInstance(g, Point)

    f_jit = api.jit(f)
    self.assertAllClose(f(pt), f_jit(pt), check_dtypes=False)

  def test_namedtuple_subclass_transparency(self):
    # See https://github.com/google/jax/issues/806
    Point = collections.namedtuple("Point", ["x", "y"])

    class ZeroPoint(Point):
      def is_zero(self):
        return (self.x == 0) and (self.y == 0)

    pt = ZeroPoint(0., 0.)

    def f(pt):
      return 0. if pt.is_zero() else np.sqrt(pt.x ** 2 + pt.y ** 2)

    f(pt)  # doesn't crash
    g = api.grad(f)(pt)
    self.assertIsInstance(pt, ZeroPoint)

  def test_eval_shape(self):
    def fun(x, y):
      return np.tanh(np.dot(x, y) + 3.)

    x = np.ones((2, 3))
    y = np.ones((3, 4))
    out_shape = api.eval_shape(fun, x, y)

    self.assertEqual(out_shape.shape, (2, 4))

  def test_eval_shape_constants(self):
    def fun():
      x = np.ones((2, 3))
      y = np.ones((3, 4))
      return np.tanh(np.dot(x, y) + 3.)

    out_shape = api.eval_shape(fun)

    self.assertEqual(out_shape.shape, (2, 4))

  def test_eval_shape_tuple_unpacking(self):
    def fun(x, y):
      a, b = x
      return a + b + y

    x = (np.ones(2), np.ones(2))
    y = 3.
    out_shape = api.eval_shape(fun, x, y)

    self.assertEqual(out_shape.shape, (2,))

  def test_eval_shape_tuple_itemgetting(self):
    def fun(x, y):
      return x[0] + x[1] + y

    x = (np.ones(2), np.ones(2))
    y = 3.
    out_shape = api.eval_shape(fun, x, y)

    self.assertEqual(out_shape.shape, (2,))

  def test_eval_shape_output_dict(self):
    def fun(x, y):
      return {'hi': x[0] + x[1] + y}

    x = (np.ones(2), np.ones(2))
    y = 3.
    out_shape = api.eval_shape(fun, x, y)
    out_shape = tree_util.tree_map(onp.shape, out_shape)

    self.assertEqual(out_shape, {'hi': (2,)})

  def test_eval_shape_shape_error(self):
    def fun(x, y):
      return np.tanh(np.dot(x, y) + 3.)

    x = np.ones((3, 3))
    y = np.ones((4, 4))

    self.assertRaises(TypeError, lambda: api.eval_shape(fun, x, y))

  def test_eval_shape_duck_typing(self):
    def fun(A, b, x):
      return np.dot(A, x) + b

    class MyArgArray(object):
      def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    A = MyArgArray((3, 4), np.float32)
    b = MyArgArray((5,), np.float32)
    x = MyArgArray((4, 5), np.float32)
    out_shape = api.eval_shape(fun, A, b, x)

    self.assertEqual(out_shape.shape, (3, 5))

  def test_issue_871(self):
    T = np.array([[1., 2.], [3., 4.], [5., 6.]])
    x = np.array([1, 2, 3])

    y, f_jvp = api.linearize(np.sum, x)
    jtu.check_raises(lambda: f_jvp(T), ValueError,
                     ("linearized function called on tangent values "
                      "inconsistent with the original primal values."))

    y, f_jvp = api.linearize(api.jit(np.sum), x)
    jtu.check_raises(lambda: f_jvp(T), ValueError,
                     ("linearized function called on tangent values "
                      "inconsistent with the original primal values."))

  def test_partial_eval_lower(self):
    # this is a simplified model of a bug that arose when we first used @jit in
    # a jvp rule. it's in this file because we want to use make_jaxpr.

    # NOTE(mattjj): I no longer understand what this was meant to test. My guess
    # is it was related to staging out the broadcast into a jaxpr to be
    # transposed, but after #1749 that's no longer a problem. After changing
    # make_jaxpr (and jit) to stage out sub-calls fully, this test started to
    # fail; I left it in as skipped because deleting tests feels wrong.
    raise unittest.SkipTest("obsolete test")

    @api.jit
    def f(a, b, c):
      a = lax.broadcast(a, (2,))
      return lax.select(a, b, c)

    a = onp.ones((3, 3), dtype=onp.bool_)
    b = onp.ones((2, 3, 3))
    c = onp.ones((2, 3, 3))

    jaxpr = api.make_jaxpr(lambda b, c: f(a, b, c))(b, c)
    subjaxpr = next(eqn.bound_subjaxprs[0][0] for eqn in jaxpr.jaxpr.eqns
                    if eqn.bound_subjaxprs)
    self.assertEqual(len(subjaxpr.eqns), 1)

  def test_grad_of_int_errors(self):
    dfn = grad(lambda x: x ** 2)
    self.assertRaisesRegex(
      TypeError,
      "Primal inputs to reverse-mode differentiation must be of float or "
      "complex type, got type int..", lambda: dfn(3))

  def test_xla_computation(self):
    # these tests basically check the examples in the xla_computation docstring

    def h(x):
      return np.sin(np.cos(x))
    c = api.xla_computation(h)(2.)
    self.assertIn('cosine', c.GetHloText())
    self.assertIn('sine', c.GetHloText())

    def f(x):
      return x - lax.psum(x, 'i')
    axis_env = [('i', 4)]
    c = api.xla_computation(f, axis_env=axis_env)(2)
    self.assertIn('all-reduce', c.GetHloText())
    self.assertIn('replica_groups={{0,1,2,3}}', c.GetHloText())

    def g(x):
      rowsum = lax.psum(x, 'i')
      colsum = lax.psum(x, 'j')
      allsum = lax.psum(x, ('i', 'j'))
      return rowsum, colsum, allsum
    axis_env = [('i', 4), ('j', 2)]
    c = api.xla_computation(g, axis_env=axis_env)(5.)
    self.assertIn('all-reduce', c.GetHloText())
    self.assertIn('replica_groups={{0,2,4,6},{1,3,5,7}}', c.GetHloText())
    self.assertIn('replica_groups={{0,1},{2,3},{4,5},{6,7}}', c.GetHloText())
    self.assertIn('replica_groups={{0,1,2,3,4,5,6,7}}', c.GetHloText())

  def test_xla_computation_args(self):
    def foo(x, y, z):
      return x + y + z

    c = api.xla_computation(foo)(1., 2., 3.)
    self.assertEqual(len(c.GetProgramShape().parameter_shapes()), 3)

    c = api.xla_computation(foo, tuple_args=True)(1., 2., 3.)
    param_shapes = c.GetProgramShape().parameter_shapes()
    self.assertEqual(len(param_shapes), 1)
    self.assertEqual(param_shapes[0].xla_element_type(),
                     xb.xla_client.PrimitiveType.TUPLE)

  def test_staging_out_multi_replica(self):
    def f(x):
      return api.pmap(np.mean)(x)
    xla_comp = api.xla_computation(f)
    xla_comp(np.arange(8)).GetHloText()  # doesn't crash

  def test_xla_computation_instantiate_constant_outputs(self):
    def f():
      return np.zeros((3, 4))

    xla_comp = api.xla_computation(f, instantiate_const_outputs=True)()
    out_shape, = xla_comp.GetReturnValueShape().tuple_shapes()
    self.assertEqual(out_shape.dimensions(), (3, 4))

  def test_jit_device(self):
    device = xb.devices()[-1]
    x = api.jit(lambda x: x, device=device)(3.)
    self.assertIsInstance(x, xla.DeviceArray)
    self.assertEqual(x.device_buffer.device(), device)

  def test_jit_of_noncallable(self):
    self.assertRaisesRegex(TypeError, "Expected a callable value.*",
                            lambda: api.jit(3))

  def test_issue_1062(self):
    # code from https://github.com/google/jax/issues/1062 @shoyer
    # this tests, among other things, whether ShardedDeviceTuple constants work
    device_count = xb.device_count()

    @jit
    def multi_step(state, count):
      return lax.fori_loop(0, count, lambda i, s: s, state)

    @jit
    def multi_step_pmap(state, count=2):
      @partial(api.pmap, axis_name='x')
      def pmapped_multi_step(state):
        return multi_step(state, count)

      return pmapped_multi_step(state)

    u = np.ones((device_count, 100))
    u_final = multi_step_pmap(u)  # doesn't crash

  @unittest.skipIf(six.PY2, "Test requires Python 3")
  def test_concurrent_device_get_and_put(self):
    def f(x):
      for _ in range(100):
        y = jax.device_put(x)
        x = jax.device_get(y)
      return x

    xs = [onp.random.randn(i) for i in range(10)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(partial(f, x)) for x in xs]
      ys = [f.result() for f in futures]
    for x, y in zip(xs, ys):
      self.assertAllClose(x, y, check_dtypes=True)

  @unittest.skipIf(six.PY2, "Test requires Python 3")
  def test_concurrent_jit(self):
    @jit
    def f(x):
      return x + x - 3.

    xs = [onp.random.randn(i) for i in range(10)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(partial(f, x)) for x in xs]
      ys = [f.result() for f in futures]
    for x, y in zip(xs, ys):
      self.assertAllClose(x * 2 - 3., y, check_dtypes=True)

  def test_dtype_warning(self):
    # cf. issue #1230
    if FLAGS.jax_enable_x64:
      return  # test only applies when x64 is disabled

    def check_warning(warn, nowarn):
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        nowarn()  # get rid of extra startup warning

        prev_len = len(w)
        nowarn()
        assert len(w) == prev_len

        warn()
        assert len(w) > 0
        msg = str(w[-1].message)
        expected_prefix = "Explicitly requested dtype "
        self.assertEqual(expected_prefix, msg[:len(expected_prefix)])

        prev_len = len(w)
        nowarn()
        assert len(w) == prev_len

    check_warning(lambda: np.array([1, 2, 3], dtype="float64"),
                  lambda: np.array([1, 2, 3], dtype="float32"),)
    check_warning(lambda: np.ones(3, dtype=onp.float64),
                  lambda: np.ones(3))
    check_warning(lambda: np.ones_like(3, dtype=onp.int64),
                  lambda: np.ones_like(3, dtype=onp.int32))
    check_warning(lambda: np.zeros(3, dtype="int64"),
                  lambda: np.zeros(3, dtype="int32"))
    check_warning(lambda: np.zeros_like(3, dtype="float64"),
                  lambda: np.zeros_like(3, dtype="float32"))
    check_warning(lambda: np.full((2, 3), 1, dtype="int64"),
                  lambda: np.full((2, 3), 1))
    check_warning(lambda: np.ones(3).astype("float64"),
                  lambda: np.ones(3).astype("float32"))
    check_warning(lambda: np.eye(3, dtype=onp.float64),
                  lambda: np.eye(3))
    check_warning(lambda: np.arange(3, dtype=onp.float64),
                  lambda: np.arange(3, dtype=onp.float32))
    check_warning(lambda: np.linspace(0, 3, dtype=onp.float64),
                  lambda: np.linspace(0, 3, dtype=onp.float32))
    check_warning(lambda: np.tri(2, dtype="float64"),
                  lambda: np.tri(2, dtype="float32"))

  def test_custom_vjp_zeros(self):
    @api.custom_transforms
    def f(x, y):
      return 2 * x, 3 * y

    def f_vjp(x, y):
      return (2 * x, 3 * y), lambda ts: (4 * ts[0], 5 * ts[1])

    api.defvjp_all(f, f_vjp, )
    api.grad(lambda x, y: f(x, y)[0])(1., 2.)  # doesn't crash

  def test_custom_transforms_vjp_nones(self):
    # issue rasied by jsnoek@ and jumper@
    @jax.custom_transforms
    def solve(a, b):
      return np.dot(np.linalg.inv(a), b)
    # print(solve(a, b))

    def solve_vjp(a, b):
      x = solve(a, b)
      def vjp(x_tangent):
        dx = np.dot(solve(a, x_tangent), x.T)
        out = (dx, b * 0.)
        return out
      return x, vjp
    jax.defvjp_all(solve, solve_vjp)
    gf = grad(lambda a,b: np.sum(solve(a, b)))

    n = 3
    a_in = np.linspace(0, 1, n)[:, None]
    a = np.dot(a_in, a_in.T) + np.eye(n) * 0.1
    real_x = onp.random.RandomState(0).randn(n)
    b = np.dot(a + np.eye(a.shape[0]), real_x)
    print(gf(a, b))  # doesn't crash

  def test_vmap_in_axes_tree_prefix_error(self):
    # https://github.com/google/jax/issues/795
    self.assertRaisesRegex(
        ValueError,
        "axes specification must be a tree prefix of the corresponding "
        r"value, got specification \(0, 0\) for value "
        r"PyTreeDef\(tuple, \[\*\]\).",
        lambda: api.vmap(lambda x: x, in_axes=(0, 0))(np.ones(3))
    )

  def test_vmap_unbatched_object_passthrough_issue_183(self):
    # https://github.com/google/jax/issues/183
    fun = lambda f, x: f(x)
    vfun = api.vmap(fun, (None, 0))
    ans = vfun(lambda x: x + 1, np.arange(3))
    self.assertAllClose(ans, onp.arange(1, 4), check_dtypes=False)

  def test_vmap_mismatched_axis_sizes_error_message_issue_705(self):
    # https://github.com/google/jax/issues/705
    def h(a, b):
      return np.sum(a) + np.sum(b)

    X = onp.random.randn(10, 4)
    U = onp.random.randn(10, 2)

    self.assertRaisesRegex(
        ValueError,
        "vmap got inconsistent sizes for array axes to be mapped:\n"
        r"arg 0 has shape \(10, 4\) and axis 0 is to be mapped" "\n"
        r"arg 1 has shape \(10, 2\) and axis 1 is to be mapped" "\n"
        "so\n"
        "arg 0 has an axis to be mapped of size 10\n"
        "arg 1 has an axis to be mapped of size 2",
        lambda: api.vmap(h, in_axes=(0, 1))(X, U))

    self.assertRaisesRegex(
        ValueError,
        "vmap got inconsistent sizes for array axes to be mapped:\n"
        r"arg 0 has shape \(10, 4\) and axis 0 is to be mapped" "\n"
        r"arg 1 has shape \(10, 2\) and axis 1 is to be mapped" "\n"
        r"arg 2 has shape \(10, 4\) and axis 0 is to be mapped" "\n"
        "so\n"
        "args 0, 2 have axes to be mapped of size 10\n"
        "arg 1 has an axis to be mapped of size 2",
        lambda: api.vmap(lambda x, y, z: None, in_axes=(0, 1, 0))(X, U, X))

    self.assertRaisesRegex(
        ValueError,
        "vmap got inconsistent sizes for array axes to be mapped:\n"
        "the tree of axis sizes is:\n"
        r"\(10, \[2, 2\]\)",
        lambda: api.vmap(h, in_axes=(0, 1))(X, [U, U]))

  def test_vmap_structured_in_axes(self):

    A, B, C, D = 2, 3, 4, 5
    K = 6  # batch size
    x = onp.ones((K, A, B))  # batch axis in different locations
    y = onp.ones((B, K, C))
    z = onp.ones((C, D, K))

    def foo(tree_arg):
      x, (y, z) = tree_arg
      return np.dot(x, np.dot(y, z))

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
      return np.dot(x, np.dot(y, z))

    tree = (x, {'a':y, 'b':z})
    vfoo = api.vmap(foo, in_axes=((0, {'a':1, 'b':2}),))
    self.assertEqual(vfoo(tree).shape, (6, 2, 5))

    tree = (x, collections.OrderedDict([('a', y), ('b', z)]))
    vfoo = api.vmap(
        foo, in_axes=((0, collections.OrderedDict([('a', 1), ('b', 2)])),))
    self.assertEqual(vfoo(tree).shape, (6, 2, 5))

  def test_jit_reference_dropping(self):
    x = onp.ones(10)
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

  def test_jit_global_cache(self):
    def f(x):
      assert python_should_be_executing
      return x

    python_should_be_executing = True
    api.jit(f)(2)
    python_should_be_executing = False
    api.jit(f)(3)

  def test_jit_shallow_copy(self):
    def f(x):
      return copy.copy(x)
    api.jit(f)(1)

  def test_jit_deep_copy(self):
    def f(x):
      return copy.deepcopy(x)
    api.jit(f)(1)

  def test_pmap_global_cache(self):
    def f(x):
      assert python_should_be_executing
      return x

    x = onp.ones(1)

    python_should_be_executing = True
    api.pmap(f)(x)
    python_should_be_executing = False
    api.pmap(f)(x)

    python_should_be_executing = True
    api.pmap(f, 'i')(x)
    python_should_be_executing = False
    api.pmap(f, 'i')(x)

  def test_device_array_repr(self):
    rep = repr(np.ones(()) + 1.)
    self.assertStartsWith(rep, 'DeviceArray')

  def test_grad_without_enough_args_error_message(self):
    # https://github.com/google/jax/issues/1696
    def f(x, y): return x + y
    df = api.grad(f, argnums=0)
    self.assertRaisesRegex(
        TypeError,
        "differentiating with respect to argnums=0 requires at least 1 "
        "positional arguments to be passed by the caller, but got only 0 "
        "positional arguments.",
        lambda: partial(df, x=0.)(y=1.))

  def test_grad_of_jit_compilation_caching(self):
    if not hasattr(self, "assertLogs"):
      raise unittest.SkipTest("test requires assertLogs (python 3)")

    lax.add(1, 2)  # make sure some initial warnings are already printed

    sin = api.jit(np.sin)

    prev_level = logging.get_verbosity()
    try:
      logging.set_verbosity('DEBUG')
      with self.assertLogs(level=logging.DEBUG) as l:
        ans1 = api.grad(sin)(2.)
        ans2 = api.grad(sin)(3.)
    finally:
      logging.set_verbosity(prev_level)
    self.assertLen(l.output, 2)

    self.assertAllClose(ans1, onp.cos(2.), check_dtypes=False)
    self.assertAllClose(ans2, onp.cos(3.), check_dtypes=False)

  def test_remat_basic(self):
    @api.remat
    def g(x):
      return lax.sin(lax.sin(x)), 3.

    def f(x):
      x, _ = g(x)
      return x

    ans = f(2.)
    expected = onp.sin(onp.sin(2.))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans, f_lin = api.linearize(f, 2.)
    expected = onp.sin(onp.sin(2.))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = f_lin(3.)
    expected = onp.cos(onp.sin(2.)) * onp.cos(2.) * 3.
    self.assertAllClose(ans, expected, check_dtypes=False)

    sin_calls = []
    cos_calls = []
    sin_impl = lax.sin_p.impl
    cos_impl = lax.cos_p.impl
    try:
      lax.sin_p.def_impl(lambda x: sin_calls.append(1) or sin_impl(x))
      lax.cos_p.def_impl(lambda x: cos_calls.append(1) or cos_impl(x))
      f_lin(3.)
    finally:
      lax.sin_p.def_impl(sin_impl)
      lax.cos_p.def_impl(cos_impl)
    self.assertEqual(len(sin_calls), 1)
    self.assertEqual(len(cos_calls), 2)

  def test_remat_freevars(self):
    def f1(x):
      y = 2 * np.sin(x)
      z = np.cos(x) * np.sin(y)
      return z

    def f2(x):
      y = 2 * np.sin(x)
      z = api.remat(lambda x: np.cos(x) * np.sin(y))(x)
      return z

    ans, f_lin = api.linearize(f2, 2.)
    expected, f_lin_expected = api.linearize(f1, 2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = f_lin(3.)
    expected = f_lin_expected(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_remat_grad_python_control_flow(self):
    @partial(api.remat, concrete=True)
    def g(x):
      if x > 0:
        return lax.sin(x), 3.
      else:
        return lax.cos(x), 4.

    def f(x):
      x, _ = g(x)
      return x

    ans = f(2.)
    expected = onp.sin(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(f)(2.)
    expected = onp.cos(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_remat_jit(self):
    @api.remat
    def g(x):
      return lax.sin(lax.sin(x))

    def f_(x):
      return g(x)
    f = api.jit(f_)

    ans = f(2.)
    expected = onp.sin(onp.sin(2.))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(f)(2.)
    expected = onp.cos(onp.sin(2.)) * onp.cos(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jit(api.grad(f_))(2.)
    expected = onp.cos(onp.sin(2.)) * onp.cos(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_remat_vmap(self):
    @api.remat
    def g(x):
      return lax.sin(lax.sin(x))

    x = onp.arange(3.)

    ans = api.vmap(g)(x)
    expected = onp.sin(onp.sin(x))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jacfwd(g)(x)
    expected = onp.diag(onp.cos(onp.sin(x)) * onp.cos(x))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jacrev(g)(x)
    expected = onp.diag(onp.cos(onp.sin(x)) * onp.cos(x))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_remat_higher_order_autodiff(self):
    def f(x):
      return lax.cos(lax.sin(x))
    g = api.remat(f)

    ans = api.grad(api.grad(g))(3.)
    expected = api.grad(api.grad(f))(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_remat_scan(self):
    to_scan = lambda c, x: (np.sin(c), None)

    def f_noremat(x):
      y, _ = lax.scan(to_scan, x, onp.arange(3.))
      return y

    def f_yesremat(x):
      y, _ = lax.scan(api.remat(to_scan), x, onp.arange(3.))
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

  def test_remat_no_redundant_flops(self):
    # see https://github.com/google/jax/pull/1749#issuecomment-558267584

    @api.jit
    def g(x):
      return f(2., x)

    @api.remat
    def f(x, y):
      return np.sin(x) * y

    # We swap out sin_p's impl rule to count how many times it's invoked
    called = []
    sin_impl = lax.sin_p.impl
    try:
      lax.sin_p.def_impl(lambda x: called.append(1) or sin_impl(x))
      api.grad(g)(3.)
    finally:
      lax.sin_p.def_impl(sin_impl)
    num_calls = len(called)
    self.assertEqual(num_calls, 1)

  def test_remat_binomial_checkpointing(self):
    def binom_checkpoint(funs):
      if len(funs) == 1:
        return funs[0]
      else:
        f1 = binom_checkpoint(funs[:len(funs)//2])
        f2 = binom_checkpoint(funs[len(funs)//2:])
        return api.remat(lambda x: f1(f2(x)))

    f1 = binom_checkpoint([np.sin, np.sin, np.sin, np.sin])
    f2 = lambda x: np.sin(np.sin(np.sin(np.sin(x))))
    x = 4.
    self.assertAllClose(f1(x), f2(x), check_dtypes=False)
    self.assertAllClose(api.grad(f1)(x), api.grad(f2)(x), check_dtypes=False)

  def test_remat_symbolic_zeros(self):
    # code from https://github.com/google/jax/issues/1907
    test_remat = True
    test_scan = True

    key = jax.random.PRNGKey(0)
    key, split = jax.random.split(key)
    n = 5

    def func(D0):
      def shift(R, dR, **unused_kwargs):
        return R + dR

      def apply_fn(R):
        return D0 * R

      Rinit = jax.random.uniform(split, (n,3), minval=0.0, maxval=5.0,
                                 dtype=np.float32)

      def move(R,i):
        F = apply_fn(R)
        return shift(R, 0.001 * F), np.array([0.])

      move = api.remat(move)
      R, temp = lax.scan(move, Rinit, np.arange(2))
      return R[0, 0]

    api.grad(func)(5.0)  # doesn't crash

  def test_trivial_computations(self):
    x = np.array([1, 2, 3])
    y = api.jit(lambda x: x)(x)
    self.assertIs(x, y)

    z1, z2 = api.jit(lambda x: (x, x))(x)
    self.assertIs(z1, z2)

    x1, x2 = np.array([1, 2]), np.array([2, 3])
    z1, z2, z3 = api.jit(lambda x, y: (y, 1, x))(x1, x2)
    self.assertIs(z1, x2)
    self.assertIs(z3, x1)
    self.assertEqual(z2, 1)

  def test_nested_jit_hoisting(self):
    @api.jit
    def f(x, y):
      z = 2 * x
      return y + z, 3

    @api.jit
    def g(x):
      return f(2, x)

    jaxpr_subcomp = xla.jaxpr_subcomp

    jaxprs = []
    def jaxpr_subcomp_and_collect(c, jaxpr, *args, **kwargs):
      jaxprs.append(jaxpr)
      return jaxpr_subcomp(c, jaxpr, *args, **kwargs)

    try:
      xla.jaxpr_subcomp = jaxpr_subcomp_and_collect
      ans = g(3)
    finally:
      xla.jaxpr_subcomp = jaxpr_subcomp

    self.assertEqual(ans, (7, 3))
    self.assertLen(jaxprs, 2)
    outer_jaxpr, inner_jaxpr = jaxprs

    self.assertLen(outer_jaxpr.eqns, 1)
    self.assertEqual(outer_jaxpr.eqns[0].primitive.name, 'xla_call')
    (subjaxpr_1, _,  _), = outer_jaxpr.eqns[0].bound_subjaxprs
    self.assertEqual(str(subjaxpr_1), str(inner_jaxpr))
    self.assertLen(inner_jaxpr.eqns, 2)
    self.assertEqual(inner_jaxpr.eqns[0].primitive.name, 'mul')
    self.assertEqual(inner_jaxpr.eqns[1].primitive.name, 'add')

  def test_primitive_compilation_cache(self):
    with jtu.count_primitive_compiles() as count:
      lax.add(1, 2)
      lax.add(2, 3)
    self.assertEqual(count[0], 1)


class JaxprTest(jtu.JaxTestCase):

  def test_scalar_literals(self):
    jaxpr = api.make_jaxpr(lambda x: x + 2)(42)
    self.assertLen(jaxpr.jaxpr.constvars, 0)

  def test_const(self):
    def fun(x):
      return (x, 1., np.zeros(1))

    jaxpr = api.make_jaxpr(fun)(0.)
    self.assertMultiLineStrippedEqual(str(jaxpr), """
    { lambda b ;  ; a.
        let
        in [a, 1.0, b] }
    """)

  def test_cond(self):
    def f(x):
      return lax.cond(x >= 0.,
                      x + 1.,
                      lambda xt: xt + x,
                      x + 2.,
                      lambda xf: xf - x)
    jaxpr = api.make_jaxpr(f)(3.)
    self.assertMultiLineStrippedEqual(str(jaxpr), """
    { lambda  ;  ; a.
      let b = ge a 0.0
          c = add a 1.0
          d = add a 2.0
          e = cond[ false_jaxpr={ lambda  ;  ; b a.
                                  let c = sub a b
                                  in [c] }
                    false_nconsts=1
                    true_jaxpr={ lambda  ;  ; b a.
                                 let c = add a b
                                 in [c] }
                    true_nconsts=1 ] b a c a d
      in [e] }
        """)


class LazyTest(jtu.JaxTestCase):

  @contextmanager
  def count_compiles(self):

    make_computation_builder = xb.make_computation_builder
    count = [0]

    def make_computation_builder_and_count(*args, **kwargs):
      count[0] += 1
      return make_computation_builder(*args, **kwargs)

    xb.make_computation_builder = make_computation_builder_and_count
    try:
      yield count
    finally:
      xb.make_computation_builder = make_computation_builder

  @jtu.skip_on_devices("tpu")
  def test_lazy_jit_closed_over_values(self):
    if not core.skip_checks:
      raise SkipTest("oom test skipped when core.skip_checks is False")

    y = np.arange(int(1e12))  # will likely oom if materialized
    ans = jit(lambda x: (x + y)[1])(1)
    self.assertEqual(ans, 2)

  def test_jit_forces_arguments(self):

    @api.jit
    def f(x):
      assert python_should_be_executing
      return np.sum(x)

    x = np.arange(10, dtype=np.int32)
    assert xla.is_device_constant(x)  # lazy iota

    python_should_be_executing = True
    _ = f(x)

    python_should_be_executing = False  # should not recompile
    x = onp.arange(10, dtype=onp.int32)
    _ = f(x)

  @parameterized.parameters(jtu.cases_from_list(range(10000)))
  def test_random_lazy_program(self, seed):

    def random_array(rng):
      kind = rng.choice(['arr', 'iota', 'eye', 'tri'])
      if kind == 'arr':
        dtype = [onp.float32, onp.int32][rng.choice(2)]
        dim = rng.randint(4)
        shape = rng.randint(4, size=dim)
        onp_x = onp.asarray(rng.randn(*shape), dtype=dtype)
        jax_x = np.array(onp_x, dtype=dtype)
      elif kind == 'iota':
        dtype = [onp.float32, onp.int32][rng.choice(2)]
        size = rng.randint(5)
        onp_x = onp.arange(size, dtype=dtype)
        jax_x = lax.iota(dtype, size)
      elif kind == 'eye':
        dtype = [onp.float32, onp.int32][rng.choice(2)]
        N = rng.randint(2, 5)
        M = None if rng.rand() < 0.5 else rng.randint(2, 5)
        k = rng.choice([-1, 0, 1])
        onp_x = onp.eye(N, M, k, dtype=dtype)
        jax_x = np.eye(N, M, k, dtype=dtype)
      elif kind == 'tri':
        dtype = [onp.float32, onp.int32][rng.choice(2)]
        N = rng.randint(2, 5)
        M = None if rng.rand() < 0.5 else rng.randint(2, 5)
        k = rng.choice([-1, 0, 1])
        onp_x = onp.tri(N, M, k, dtype=dtype)
        jax_x = np.tri(N, M, k, dtype=dtype)
      else:
        assert False
      assert type(onp_x) is onp.ndarray and type(jax_x) is xla.DeviceArray
      return onp_x, jax_x

    def random_op(rng, shape):
      kind = rng.choice(['transpose', 'broadcast', 'reshape'])
      if kind == 'transpose':
        perm = tuple(rng.permutation(len(shape)))
        return Op(partial(onp.transpose, axes=perm),
                  partial(lax.transpose, permutation=perm))
      elif kind == 'broadcast':
        n = rng.randint(1, 3)
        new_sizes = rng.randint(1, 4, size=n)
        new_ndim  = n + len(shape)
        bcast_dims = tuple(sorted(rng.permutation(new_ndim)[:len(shape)]))
        shape_iter = iter(shape)
        new_sizes = iter(rng.randint(1, 4, size=n))
        new_shape = [next(shape_iter) if i in  bcast_dims else next(new_sizes)
                    for i in range(new_ndim)]
        return Op(partial(lax_reference.broadcast_in_dim, shape=new_shape,
                          broadcast_dimensions=bcast_dims),
                  partial(lax.broadcast_in_dim, shape=new_shape,
                          broadcast_dimensions=bcast_dims))
      elif kind == 'reshape':
        new_shape = list(shape)
        for _ in range(rng.randint(1, 3)):
          loc = len(new_shape) and rng.randint(len(new_shape))
          new_shape.insert(loc, 1)
        new_shape = tuple(new_shape)
        return Op(partial(onp.reshape, newshape=new_shape),
                  partial(lax.reshape, new_sizes=new_shape))
      else:
        assert False
    Op = collections.namedtuple('Op', ['onp_fn', 'jax_fn'])

    rng = onp.random.RandomState(seed)
    onp_x, jax_x = _, orig_x = random_array(rng)
    ops = []
    with jtu.count_primitive_compiles() as count:
      for _ in range(rng.randint(5)):
        op = random_op(rng, onp.shape(onp_x))
        onp_x = op.onp_fn(onp_x)
        jax_x = op.jax_fn(jax_x)
        ops.append(op)
    self.assertEqual(count[0], 0)

    kind = rng.choice(['closure', 'npy_value', 'add'])
    if kind == 'closure':
      result = api.jit(lambda x: x + jax_x)(0)
      self.assertAllClose(onp_x, result, check_dtypes=False)
    elif kind == 'npy_value':
      self.assertAllClose(onp_x, jax_x, check_dtypes=False)
    elif kind == 'add':
      result = jax_x + onp.zeros(jax_x.shape, dtype=jax_x.dtype)
      self.assertAllClose(onp_x, result, check_dtypes=False)
    else:
      assert False

    @jit
    def apply_ops(x):
      for op in ops:
        x = op.jax_fn(x)
      return x

    jit_result = apply_ops(orig_x)
    self.assertAllClose(jit_result, onp_x, check_dtypes=False)

    @jit
    def apply_ops_closure():
      x = orig_x
      for op in ops:
        x = op.jax_fn(x)
      return x

    jit_result = apply_ops_closure()
    self.assertAllClose(jit_result, onp_x, check_dtypes=False)

  def test_constant_forcing_computations_cached(self):
    # from https://github.com/google/jax/issues/1909
    xla.xla_primitive_callable.cache_clear()  # clear compile cache
    big_lazy_x = np.ones((api.device_count(), 100))
    f = api.pmap(lambda x: 2 * x)
    _ = f(big_lazy_x)

    with self.count_compiles() as count:
      _ = f(big_lazy_x)
    self.assertEqual(count[0], 0)

  def test_zeros_ones_compilation(self):
    xla.xla_primitive_callable.cache_clear()  # clear compile cache

    with self.count_compiles() as count:
      x = np.ones(3) + np.zeros(3)
      y = np.ones(3) + np.ones(3)

    self.assertEqual(count[0], 1)
    self.assertAllClose(x, onp.ones(3), check_dtypes=False)
    self.assertAllClose(y, onp.ones(3) + onp.ones(3), check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
