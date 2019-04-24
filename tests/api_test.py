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

import six

import numpy as onp
from absl.testing import absltest
from jax import test_util as jtu

import jax.numpy as np
from jax import jit, grad, device_get, device_put, jacfwd, jacrev, hessian
from jax import api
from jax.core import Primitive
from jax.interpreters.ad import defjvp, defvjp, defvjp2, defvjp_all
from jax.interpreters.xla import DeviceArray
from jax.abstract_arrays import concretization_err_msg

from jax.config import config
config.parse_flags_with_absl()

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

    jtu.check_raises_regexp(lambda: grad(f)("foo"), TypeError,
                     "Argument 'foo' of type <.*'str'> is not a valid JAX type")

    jtu.check_raises_regexp(lambda: jit(f)("foo"), TypeError,
                     "Argument 'foo' of type <.*'str'> is not a valid JAX type")

  # TODO(dougalm): enable when we remove 'None' from pytree nodes
  # def test_bad_output(self):
  #   def f(x):
  #     pass

  #   grad(f)(onp.zeros(3))
  #   jit(f)(onp.zeros(3))
  #   assert False

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

    jtu.check_raises(lambda: grad(f)(onp.zeros(3), onp.zeros(4)),
                     ValueError,
                     "Incompatible shapes for broadcasting: ((3,), (4,))")

  def test_dot_mismatch(self):
    def f(x, y):
      return np.dot(x, y)

    jtu.check_raises(lambda: grad(f)(onp.zeros(3), onp.zeros(4)),
                     TypeError,
                     "Incompatible shapes for dot: got (3,) and (4,).")

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
    jtu.check_raises_regexp(
        lambda: jit(f)(0, 5), TypeError,
        "('JaxprTracer' object cannot be interpreted as an integer"
        "|Abstract value passed to .*)")

  def test_casts(self):
    for castfun in [float, complex, hex, oct] + list(six.integer_types):
      f = lambda x: castfun(x)
      jtu.check_raises_regexp(
          lambda: jit(f)(0), TypeError,
          "('JaxprTracer' object cannot be interpreted as an integer"
          "|Abstract value passed to .*)")

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
                     "XLA translation rule for 'foo' not implemented")

    foo_p.def_impl(lambda x: x)
    defjvp(foo_p, lambda g, x: foo(g))

    jtu.check_raises(lambda: grad(foo)(1.0), NotImplementedError,
                     "Reverse-mode differentiation rule for 'foo' not implemented")

  def test_device_put_and_get(self):
    x = onp.arange(12.).reshape((3, 4)).astype("float32")
    dx = device_put(x)
    assert isinstance(dx, DeviceArray)
    x2 = device_get(dx)
    assert isinstance(x2, onp.ndarray)
    assert onp.all(x == x2)

    y = [x, (2 * x, 3 * x)]
    dy = device_put(y)
    y2 = device_get(dy)
    assert isinstance(y2, list)
    assert isinstance(y2[0], onp.ndarray)
    assert onp.all(y2[0] == x)
    assert isinstance(y2[1], tuple)
    assert isinstance(y2[1][0], onp.ndarray)
    assert onp.all(y2[1][0] == 2 * x)
    assert isinstance(y2[1][1], onp.ndarray)
    assert onp.all(y2[1][1] == 3 * x)

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
    self.assertAllClose(aux, [9.], check_dtypes=True)

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

  def test_jarrett_jvps(self):
    def f1(x):
      return np.sin(np.sin(np.sin(x)))
    f2 = api.jarrett(f1)

    for x in [3., onp.array([2., 3., 4.])]:
      self.assertAllClose(f1(x), f2(x), check_dtypes=True)

      _, f1_vjp = api.vjp(f1, x)
      _, f2_vjp = api.vjp(f2, x)
      self.assertAllClose(f1_vjp(x), f2_vjp(x), check_dtypes=True)

      jaxpr2 = api.make_jaxpr(f2_vjp)(x)
      assert len(jaxpr2.constvars) == 1

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

      jaxpr2 = api.make_jaxpr(f2_vjp)(y)
      assert len(jaxpr2.constvars) == 2

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
    self.assertAllClose(ans, expected, check_dtypes=False)

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

    defvjp_all(foo_p, lambda x: (x**2, lambda g: (4 * g * np.sin(x),)))
    val_ans, grad_ans = api.value_and_grad(foo)(3.)
    self.assertAllClose(val_ans, 2 * 3.**2, check_dtypes=False)
    self.assertAllClose(grad_ans, 4 * 2 * onp.sin(3.), check_dtypes=False)

  def test_defvjp_all_const(self):
    foo_p = Primitive('foo')
    def foo(x): return foo_p.bind(x)

    defvjp_all(foo_p, lambda x: (x**2, lambda g: (12.,)))
    val_ans, grad_ans = api.value_and_grad(foo)(3.)
    self.assertAllClose(val_ans, 9., check_dtypes=False)
    self.assertAllClose(grad_ans, 12., check_dtypes=True)

  def test_defvjp_all_higher_order_revmode(self):
    foo_p = Primitive('foo')
    def foo(x): return 2. * foo_p.bind(x)

    defvjp_all(foo_p, lambda x: (x**2, lambda g: (g * x ** 2,)))
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

    defvjp_all(foo_p, vjpfun)
    val_ans, grad_ans = api.value_and_grad(foo)(3., 4.)
    self.assertAllClose(val_ans, 3.**2 + 4.**3, check_dtypes=False)
    self.assertAllClose(grad_ans, 1. + 3. + 4., check_dtypes=False)

    ans = api.grad(foo, (0, 1))(3., 4.)
    self.assertAllClose(ans, (1. + 3. + 4., 1. * 3. * 9.), check_dtypes=False)

  def test_defvjp(self):
    @api.custom_transforms
    def foo(x, y):
      return np.sin(x * y)

    defvjp(foo.primitive, None, lambda g, x, y: g * x * y)
    val_ans, grad_ans = api.value_and_grad(foo)(3., 4.)
    self.assertAllClose(val_ans, onp.sin(3. * 4.), check_dtypes=False)
    self.assertAllClose(grad_ans, 0., check_dtypes=False)

    ans_0, ans_1 = api.grad(foo, (0, 1))(3., 4.)
    self.assertAllClose(ans_0, 0., check_dtypes=False)
    self.assertAllClose(ans_1, 3. * 4., check_dtypes=False)

  def test_defvjp2(self):
    @api.custom_transforms
    def foo(x, y):
      return np.sin(x * y)

    defvjp2(foo.primitive, None, lambda g, ans, x, y: g * x * y + np.cos(ans))
    val_ans, grad_ans = api.value_and_grad(foo, 1)(3., 4.)
    self.assertAllClose(val_ans, onp.sin(3. * 4.), check_dtypes=False)
    self.assertAllClose(grad_ans, 3. * 4. + onp.cos(onp.sin(3. * 4)),
                        check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
