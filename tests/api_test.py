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
from jax import jit, grad, device_get, device_put, jacfwd, jacrev
from jax.core import Primitive
from jax.interpreters.partial_eval import def_abstract_eval
from jax.interpreters.ad import defjvp
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


  def test_jit_static_args(self):
    side = []

    def f(x, y, z, flag=False, flag2=False):
      assert flag
      side.append(None)
      return 100*x + 10*y + z

    f1 = jit(f)
    assert f1(1, 2, 3, flag=True) == 123
    assert len(side) == 1
    assert f1(2, 1, 3, flag=True) == 213
    assert len(side) == 1
    assert f1(2, 1, 3, flag=True, flag2=True) == 213
    assert len(side) == 2

    side[:] = []
    f2 = jit(f, static_argnums=[0,2])
    assert f2(1, 2, 3, flag=True) == 123
    assert len(side) == 1
    assert f2(1, 3, 3, flag=True) == 133
    assert len(side) == 1
    assert f2(2, 2, 3, flag=True) == 223
    assert len(side) == 2
    assert f2(2, 4, 3, flag=True) == 243
    assert len(side) == 2
    assert f2(2, 4, 3, flag=True, flag2=True) == 243
    assert len(side) == 3
    assert f2(2, 5, 3, flag=True, flag2=True) == 253
    assert len(side) == 3

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
        lambda: jit(f)(0, 5), TypeError, "Abstract value passed to .*")

  def test_casts(self):
    for castfun in [float, complex, hex, oct] + list(six.integer_types):
      f = lambda x: castfun(x)
      jtu.check_raises_regexp(
          lambda: jit(f)(0), TypeError, "Abstract value passed to .*")

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

    def_abstract_eval(foo_p, lambda x: x)

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


if __name__ == '__main__':
  absltest.main()
