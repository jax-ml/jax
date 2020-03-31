# Copyright 2019 Google LLC
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

"""Tests for the experimental/loops."""


from absl.testing import absltest
import numpy as onp
import re

from jax import api, lax, ops
from jax import numpy as np
from jax import test_util as jtu
from jax.experimental import loops

from jax.config import config
config.parse_flags_with_absl()

# Attempted fix for https://github.com/google/jax/issues/2507 based on resetting
# the global trace state. It could be that methods like _BodyTracer.end_subtrace
# are not cleaning up global trace state after exceptions because they don't use
# a try/finally pattern. This is just a guess though!
# TODO(mattjj,necula): check this attempted fix
from jax import core
def tearDownModule():
  core.trace_state = core.TraceState()

class LoopsTest(jtu.JaxTestCase):

  def test_scope_no_loops(self):
    def f_op(r):
      with loops.Scope() as s:
        s.x = r + 1
        return s.x
    self.assertAllClose(4.0, f_op(3.), check_dtypes=True)

  def test_loop_empty(self):
    def f_op(r):
      with loops.Scope() as s:
        for _ in s.range(5):
          pass
        return r

    self.assertAllClose(3.0, f_op(3.), check_dtypes=True)

  def test_loop_1(self):
    """One loop with one state var, with transforms."""
    def f_op(inc):
      with loops.Scope() as s:
        s.out = 10.
        for _ in s.range(5):
          s.out += inc
        return s.out
    def f_expected(inc):
      return 10 + 5 * inc
    self.assertAllClose(f_expected(2.), f_op(2.), check_dtypes=True)
    self.assertAllClose(f_expected(2.), api.jit(f_op)(2.), check_dtypes=True)
    self.assertAllClose(5., api.grad(f_op)(2.), check_dtypes=True)
    self.assertAllClose(5., api.grad(f_op)(2.), check_dtypes=True)
    inc_batch = onp.arange(5, dtype=np.float_)
    self.assertAllClose(np.array([f_expected(inc) for inc in inc_batch],
                                 dtype=np.float_),
                        api.vmap(f_op)(inc_batch), check_dtypes=True)


  def test_loop_2(self):
    """One loop, two state fields."""
    def f_op(inc):
      with loops.Scope() as s:
        s.out1 = 10.
        s.out2 = 20.
        for i in s.range(5):
          s.out1 += inc
          s.out2 += 1.
        return (s.out1, s.out2)

    self.assertAllClose((10. + 2. * 5, 20. + 1. * 5), f_op(2.), check_dtypes=True)


  def test_add_vectors(self):
    def add_vec(x, y):
      with loops.Scope() as s:
        n = x.shape[0]
        assert n == y.shape[0]
        s.out = np.zeros(shape=[n], dtype=np.float32)
        for i in s.range(n):
          s.out = ops.index_add(s.out, i, x[i] + y[i])
        return s.out

    x = np.array([1., 2., 3.], dtype=np.float32)
    y = np.array([4., 5., 6.], dtype=np.float32)
    self.assertAllClose(np.add(x, y), add_vec(x, y), check_dtypes=True)

  def test_matmul(self):
    def matmul(x, y):
      with loops.Scope() as s:
        n, m = x.shape
        m1, p = y.shape
        assert m == m1
        s.out = np.zeros(shape=[n, p], dtype=np.float32)
        for i in s.range(n):
          for j in s.range(p):
            for k in s.range(m):
              s.out = ops.index_add(s.out, (i, j), x[i, k] * y[k, j])
        return s.out

    x = np.array([[1., 2., 3.]], dtype=np.float32)  # 1x3
    y = np.array([[4.], [5.], [6.]], dtype=np.float32)  # 3x1
    self.assertAllClose(np.matmul(x, y), matmul(x, y), check_dtypes=True)

  def test_reuse_range(self):
    """Ranges can be reused, as long as not nested in each other."""
    def f_op():
      with loops.Scope() as s:
        r1 = s.range(5)
        s.out = 0
        for _ in r1:
          s.out += 1
        for _ in r1:
          s.out += 1
        return s.out

    self.assertEqual(10, f_op())


  def test_loop_nested(self):
    def f_op(inc):
      with loops.Scope() as s:
        s.out = 10.
        for i in s.range(5):
          s.out += inc
          for j in s.range(6):
            s.out += inc
        return s.out

    self.assertAllClose(10. + 5 * (2. + 6 * 2.), f_op(2.), check_dtypes=True)

  def test_example_doc(self):
    "The example from the module docstring."
    def f_expected():
      arr = onp.zeros(5, dtype=np.float_)
      for i in range(arr.shape[0]):
        arr[i] += 2.
        if i % 2 == 0:
          arr[i] += 1.
      return arr

    def f_op_jax():
      arr = np.zeros(5)
      def loop_body(i, acc_arr):
        arr1 = ops.index_update(acc_arr, i, acc_arr[i] + 2.)
        return lax.cond(i % 2 == 0,
                        arr1,
                        lambda arr1: ops.index_update(arr1, i, arr1[i] + 1.),
                        arr1,
                        lambda arr1: arr1)
      arr = lax.fori_loop(0, arr.shape[0], loop_body, arr)
      return arr

    def f_op_loops():
      with loops.Scope() as s:
        s.arr = np.zeros(5)  # Must create the mutable state of the loop as `scope` fields.
        for i in s.range(s.arr.shape[0]):
          s.arr = ops.index_update(s.arr, i, s.arr[i] + 2.)
          for _ in s.cond_range(i % 2 == 0):  # Conditionals are also sugared as loops with 0 or 1 iterations
            s.arr = ops.index_update(s.arr, i, s.arr[i] + 1.)
        return s.arr

    self.assertAllClose(f_expected(), f_op_jax(), check_dtypes=True)
    self.assertAllClose(f_expected(), f_op_loops(), check_dtypes=True)

  def test_loop_mutable_used_but_not_changed(self):
    def f_op(inc):
      with loops.Scope() as s:
        s.read_only = inc
        s.out = 10.
        for i in s.range(5):
          s.out += s.read_only
        # It is Ok to use regular Python variables outside loops.
        save_to_other_var = s.out

      return save_to_other_var

    self.assertAllClose(10. + 5 * 2., f_op(2.), check_dtypes=True)

  def test_range_locations(self):
    """Ranges have locations."""
    with loops.Scope() as s:
      r = s.range(5)
      cr = s.cond_range(True)
      wr = s.while_range(lambda: True)
      for range in [r, cr, wr]:
        self.assertIn("loops_test.py", range.location())
        self.assertIn(self._testMethodName, range.location())

  def test_error_reuse_range_nested(self):
    """Ranges cannot be reused nested in their own iteration."""
    def f_op():
      with loops.Scope() as s:
        r1 = s.range(5)
        s.out = 0
        for _ in r1:
          for _ in r1:
            s.out += 1
        return s.out

    with self.assertRaisesWithLiteralMatch(ValueError, "Range is reused nested inside itself."):
      f_op()

  def test_error_early_exit_range(self):
    """Ranges do not support early exit from loop body."""
    def bad_function(exit_how="break"):
      with loops.Scope() as s:
        for i in s.range(555):
          if exit_how == "break":
            break
          elif exit_how == "return":
            return 1.
          elif exit_how == "exception":
            raise ValueError("test exception")
        # Start another range, we get here after a "break" above
        for i in s.range(5):
          pass
        return 0.

    with self.assertRaisesRegex(ValueError,
                                re.compile(("Some ranges have exited prematurely. The innermost such range is at"
                                           ".*s.range.555."), re.DOTALL)):
      bad_function("break")
    with self.assertRaisesRegex(ValueError, "Some ranges have exited prematurely"):
      bad_function("return")
    # On exception exit, we let the exception propagate
    with self.assertRaisesRegex(ValueError, "test exception"):
      bad_function("exception")

  def test_error_early_exit_range_nested(self):
    """Exit early from a nested range."""
    def bad_function():
      with loops.Scope() as s:
        for i in s.range(5):  # When we end this range, we'll find the inner range still active
          for j in s.range(6):
            break
        return 0.

    with self.assertRaisesRegex(ValueError, "Some ranges have exited prematurely."):
      bad_function()

  def test_loop_index_var_live_expect_fail(self):
    """The index variable is live after the loop."""
    self.skipTest("Don't know how to check that index variable is not used after loop.")
    def f_op(r):
      with loops.Scope() as s:
        for i in s.range(r):
          pass
        return i

    self.assertAllClose(4, f_op(4), check_dtypes=True)

  def test_error_new_state_in_loop(self):
    """Error when creating new state in a loop."""
    def f_op(inc):
      with loops.Scope() as s:
        s.out = 10.
        for i in s.range(5):
          s.other_state = 1.
          s.out += inc
        return s.out

    with self.assertRaisesWithLiteralMatch(ValueError,
                                           "New mutable state 'other_state' cannot be created inside a loop."):
      f_op(2.)

  def test_error_range_ends_static(self):
    def f_op(start, end, inc):
      with loops.Scope() as s:
        s.out = 0.
        for i in s.range(start, end):
          s.out += inc
        return s.out

    self.assertAllClose(16., f_op(0, 4, 4.), check_dtypes=True)
    # Ok to jit, as long as the start and end are static
    self.assertAllClose(16., api.jit(f_op, static_argnums=(0, 1))(0, 4, 4.), check_dtypes=True)
    with self.assertRaisesRegex(TypeError, "Abstract value passed to `int`, which requires a concrete value"):
      self.assertAllClose(16., api.jit(f_op)(0, 4, 4.), check_dtypes=True)
    with self.assertRaisesRegex(TypeError, "Abstract value passed to `int`, which requires a concrete value"):
      self.assertAllClose(16., api.vmap(f_op)(np.zeros(10), np.ones(10), np.array([4.] * 10)), check_dtypes=True)

  def test_cond(self):
    def f_op(inc):
      with loops.Scope() as s:
        s.out = 10.
        for i in s.cond_range(inc > 0):
          s.out += inc
        return s.out

    self.assertAllClose(10. + 2., f_op(2.), check_dtypes=True)
    self.assertAllClose(10., f_op(-2.), check_dtypes=True)

  def test_cond_state(self):
    """Conditionals predicated on scope fields."""
    def f_op(init):
      with loops.Scope() as s:
        s.out = init
        for _ in s.cond_range(s.out > 0.):
          s.out *= 2.
        return s.out

    self.assertAllClose(2. * 2., f_op(2.), check_dtypes=True)
    self.assertAllClose(-2., f_op(-2.), check_dtypes=True)

  def test_cond_nested(self):
    """Nested conditionals."""
    def f_expected(init):
      """Multi-linear function.
      x in (..0)   x + 1.
      x in [0..10) x + 1 + 2 + 4
      x in [10..)  x + 1 + 2 + 4 + 8
      """
      out = init
      if out >= 0.:
        out += 2.
        if out - 2. >= 10.:
          out += 8.
        out += 4.
      out += 1.
      return out

    def f_op(init):
      with loops.Scope() as s:
        s.out = init
        for _ in s.cond_range(s.out >= 0.):
          s.out += 2.
          for _ in s.cond_range(s.out - 2. >= 10.):
            s.out += 8.
          s.out += 4.
        s.out += 1.
        return s.out

    for init in [-1., 0., 9., 10.]:
      self.assertAllClose(f_expected(init), f_op(init), check_dtypes=True)


  def test_error_cond_using_index_var(self):
    """Conditionals should not use the iteration index value."""
    def f_op(inc):
      with loops.Scope() as s:
        s.out = 10.
        for i in s.cond_range(inc > 0):
          s.out += i
        return s.out

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Body of cond_range or while_range should not use the index variable returned by iterator."):
      api.make_jaxpr(f_op)(2.)

  def test_while(self):
    def f_op(init):
      with loops.Scope() as s:
        s.out = init
        for _ in s.while_range(lambda: s.out < 5.):
          s.out += 2.
        s.out += 1.
        return s.out
    def f_expected(init):
      out = init
      while out < 5.:
        out += 2.
      out += 1.
      return out

    self.assertAllClose(f_expected(2.), f_op(2.), check_dtypes=True)
    self.assertAllClose(f_expected(2.), api.jit(f_op)(2.), check_dtypes=True)
    self.assertAllClose(f_expected(1.), f_op(1.), check_dtypes=True)
    init_batch = onp.array([1., 2., 3.], dtype=onp.float32)
    self.assertAllClose(onp.array([f_expected(init) for init in init_batch],
                                  dtype=onp.float32),
                        api.vmap(f_op)(init_batch), check_dtypes=True)

  def test_error_while_cond_mutation(self):
    """Disallow mutation in the while conditional."""
    def f_op(init):
      with loops.Scope() as s:
        s.out = init

        def cond_func():
          s.out += 1.  # Not allowed
          return s.out < 5.

        for _ in s.while_range(cond_func):
          s.out += 2.
        s.out += 1.
        return s.out

    with self.assertRaisesWithLiteralMatch(ValueError,
                                           "Conditional function modifies scope.out field."):
      f_op(0.)


if __name__ == '__main__':
  absltest.main()
