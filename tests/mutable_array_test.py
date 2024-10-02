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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import jax
from jax._src import core
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp

from jax._src.state.types import (RefEffect)

config.parse_flags_with_absl()

class MutableArrayTest(jtu.JaxTestCase):

  @parameterized.parameters([True, False])
  def test_basic(self, jit):
    def f(x_mut):
      x_mut[...] += 1.
      x_mut[0] += 1
      x_mut[1] += 5

    if jit:
      f = jax.jit(f)

    x_mut = core.mutable_array(jnp.zeros(3))
    f(x_mut)

    self.assertAllClose(x_mut[...], jnp.array([2., 6., 1.]),
                        check_dtypes=False)

    jaxpr = jax.make_jaxpr(f)(x_mut)
    self.assertTrue(any(isinstance(e, RefEffect) for e in jaxpr.effects))

  @parameterized.parameters([True, False])
  def test_multiple_inputs_and_outputs(self, jit):
    def f(x_mut, y, z_mut, w):
      x_mut[...] += 1
      z_mut[...] += 1
      return x_mut[...] + y + z_mut[...] + w, y + w

    if jit:
      f = jax.jit(f)

    x_mut = core.mutable_array(jnp.zeros((1, 3)))
    y = jnp.ones((2, 3))
    z_mut = core.mutable_array(jnp.zeros((2, 3)))
    w = jnp.ones((2, 1))

    out1, out2 = f(x_mut, y, z_mut, w)

    self.assertAllClose(x_mut[...], jnp.ones((1, 3)), check_dtypes=False)
    self.assertAllClose(z_mut[...], jnp.ones((2, 3)), check_dtypes=False)
    self.assertAllClose(out1, 4 * jnp.ones((2, 3)), check_dtypes=False)
    self.assertAllClose(out2, y + w, check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_closed_over_basic(self, jit):
    x_mut = core.mutable_array(jnp.zeros(3))
    def f():
      x_mut[...] += 1.
      x_mut[0] += 1
      x_mut[1] += 5

    if jit:
      f = jax.jit(f)

    f()

    self.assertAllClose(x_mut[...], jnp.array([2., 6., 1.]),
                        check_dtypes=False)

    jaxpr = jax.make_jaxpr(f)()
    self.assertTrue(any(isinstance(e, RefEffect) for e in jaxpr.effects))

  @parameterized.parameters([True, False])
  def test_closed_over_nested(self, jit):
    x_mut = core.mutable_array(jnp.zeros(3))

    @jax.jit
    def f(y_mut, z):
      x_mut[...] += 1.
      x_mut[0] += 1
      x_mut[1] += 5

      y_mut[2] += 7
      return z + 9

    if jit:
      f = jax.jit(f)

    y_mut = core.mutable_array(np.zeros(3))

    w = f(y_mut, 1)

    self.assertAllClose(x_mut[...], jnp.array([2., 6., 1.]),
                        check_dtypes=False)
    self.assertAllClose(y_mut[...], jnp.array([0., 0., 7.]),
                        check_dtypes=False)
    self.assertAllClose(w, 10, check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_internal_mutarray_basic(self, jit):
    def f():
      x_mut = core.mutable_array(jnp.zeros(3))
      x_mut[0] += 1
      x_mut[0] += 1
      x_mut[2] += 1
      return x_mut[...]

    if jit:
      f = jax.jit(f)

    out = f()
    self.assertAllClose(out, jnp.array([2., 0., 1.]), check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_refs_in_vjps(self, jit):
    def gradient_history_calculator_fwd(x, ref):
      return x, ref

    def gradient_history_calculator_bwd(amax_history, grad_output):
      amax_update = jnp.max(jnp.abs(grad_output))
      shifted = jnp.roll(amax_history[:], 1)
      shifted = shifted.at[0].set(amax_update)
      amax_history[:] = shifted
      amax_from_history = jnp.max(amax_history[:])
      grad_output = grad_output / amax_from_history
      return grad_output, None

    @jax.custom_vjp
    def gradient_history_calculator(x, ref):
      return x

    gradient_history_calculator.defvjp(
      gradient_history_calculator_fwd,
      gradient_history_calculator_bwd)

    class DotOp:
      def __init__(self):
        self.amax_history = core.mutable_array(jnp.zeros(5,))

      def forward(self, x, y):
        out = jnp.dot(x, y)
        out = gradient_history_calculator(out, self.amax_history)
        return out

    dot_op = DotOp()
    x_top = jnp.ones((5,))
    y_top = jnp.ones((5,))

    def loss(x, y):
      return dot_op.forward(x, y).sum()

    if jit:
      loss = jax.jit(loss)

    for i in range(3):
      jax.grad(loss, (0,1))(x_top, y_top)
      self.assertAllClose(dot_op.amax_history[:], jnp.zeros((5,)).at[:i+1].set(1.0), check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_scan_internal_mut_array(self, jit):
    def body_fun(_, x):
      x_mut = core.mutable_array(x)
      x_mut[...] += 2
      return ((), x_mut[...])
    doit = lambda: jax.lax.scan(body_fun, (), np.arange(5))
    if jit:
      doit = jax.jit(doit)
    _, xs = doit()
    self.assertAllClose(xs, (np.arange(5) + 2), check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_scan_closed_over_mut_array(self, jit):
    x_mut = core.mutable_array(0)
    def body_fun(_, x):
      x_mut[...] += 2
      return ((), x_mut[...])

    doit = lambda: jax.lax.scan(body_fun, (), np.arange(5))
    if jit:
      doit = jax.jit(doit)
    _, xs = doit()
    self.assertAllClose(x_mut[...], 10)
    self.assertAllClose(xs, np.arange(5) * 2 + 2, check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_scan_scanned_mut_array(self, jit):
    def body_fun(_, index_x):
      (index, x) = index_x
      x[...] += index
      # breakpoint()
      return ((), x[...])

    x_mut = core.mutable_array(np.arange(5))
    doit = lambda: jax.lax.scan(body_fun, (), (np.arange(5), x_mut))
    if jit:
      doit = jax.jit(doit)
    _, xs = doit()
    self.assertAllClose(xs, (np.arange(5) * 2), check_dtypes=False)

  def test_double_jit_mutable_array(self):
    @jax.jit
    @jax.jit
    def f():
      x_ref = core.mutable_array(jnp.zeros(8))
      return x_ref[...]
    x = f()
    self.assertArraysEqual(x, jnp.zeros(8))

  def test_grad_mutable_array(self):
    @jax.jit
    def f(x):
      x_ = core.mutable_array(x)
      x_[()] = x_[()] + x_[()]
      y = core.freeze(x_)
      return y

    ans = jax.grad(f)(1.)
    expected = 2.0
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_defensive_copy(self):
    x = jnp.arange(3.)
    _ = jax.jit(lambda x_ref: x_ref[...])(core.mutable_array(x))
    x + 1  # don't crash


@jtu.with_config(jax_mutable_array_checks=True)
class MutableArrayErrorsTest(jtu.JaxTestCase):
  def test_return_from_jit(self):
    with self.assertRaisesRegex(
        ValueError,
        r"traced for jit returned a mutable array reference.*\n\n"
        r".*was created on line"):
      jax.jit(core.mutable_array)(jnp.arange(3))

  def test_return_from_jit_arg(self):
    with self.assertRaisesRegex(
        ValueError,
        r"traced for jit returned a mutable array reference.*\n\n"
        r".*was passed in as the argument x_ref"):
      jax.jit(lambda x_ref: x_ref)(core.mutable_array(jnp.arange(3)))

  def test_return_from_jit_pytree(self):
    with self.assertRaisesRegex(
        ValueError,
        r"tree path \['hi'\]"):
      jax.jit(lambda x_ref: {'hi': x_ref})(core.mutable_array(jnp.arange(3)))

  def test_return_from_jit_closure(self):
    with self.assertRaisesRegex(
        ValueError,
        r"tree path \['hi'\]"):
      x_ref = core.mutable_array(jnp.arange(3))
      jax.jit(lambda: {'hi': x_ref})()

  def test_argument_aliases_jit(self):
    x_ref = core.mutable_array(0.)
    with self.assertRaisesRegex(
        ValueError, "appeared at both x_ref and y_ref"):
      jax.jit(lambda x_ref, y_ref: x_ref[...] + y_ref[...])(x_ref, x_ref)

  def test_closure_and_argument_aliases_jit(self):
    x_ref = core.mutable_array(0.)
    with self.assertRaisesRegex(
        ValueError, "closed over and passed as the argument y_ref"):
      jax.jit(lambda y_ref: x_ref[...] + y_ref[...])(x_ref)

  def test_return_from_scan(self):
    with self.assertRaisesRegex(
        ValueError, "traced for scan returned a mutable array reference of type"):
      jax.lax.scan(lambda c, x: (core.mutable_array(c), x), 0, jnp.arange(3))

  # TODO test_argument_aliases_scan
  # TODO test_closure_and_argument_aliases_scan

  def test_return_from_cond(self):
    with self.assertRaisesRegex(
        ValueError, "traced for cond returned a mutable array reference of type"):
      jax.lax.cond(True, lambda: core.mutable_array(1.0), lambda: core.mutable_array(2.0))

  # TODO test_argument_aliases_cond
  # TODO test_closure_and_argument_aliases_cond

  # TODO test_return_from_custom_jvp/vjp
  # TODO test_argument_aliases_custom_jvp/vjp
  # TODO test_closure_and_argument_aliases_custom_jvp/vjp

  # TODO(mattjj): enable when cond works with mutable arrays
  # @parameterized.parameters([False, True])
  # def test_cond_both_branches_close_over_same_mutable_array(self, jit):
  #   # see also test_cond_with_ref_reuse in state_test.py
  #   x_ref = core.mutable_array(0.)
  #   def f(pred):
  #     def true_fun():
  #       x_ref[()] = 1.
  #     def false_fun():
  #       x_ref[()] = 2.
  #     jax.lax.cond(pred, true_fun, false_fun)
  #   if jit:
  #     f = jax.jit(f)
  #   out_true = f(True)
  #   self.assertAllClose(x_ref[...], 1.)
  #   out_false = f(False)
  #   self.assertAllClose(x_ref[...], 2.)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
