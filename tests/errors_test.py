# Copyright 2020 The JAX Authors.
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

import re
import sys
import traceback

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
from jax._src import config
from jax._src import core
from jax._src import source_info_util
from jax._src import test_util as jtu
from jax._src import traceback_util


config.parse_flags_with_absl()


def get_exception(etype, f):
  try:
    f()
  except etype as e:
    return e
  assert False

def check_filtered_stack_trace(test, etype, f, frame_patterns=(),
                               filter_mode="remove_frames"):
  with config.traceback_filtering(filter_mode):
    test.assertRaises(etype, f)
    e = get_exception(etype, f)
  c = e.__cause__
  if filter_mode == "quiet_remove_frames":
    if sys.version_info >= (3, 11):
      assert any("For simplicity" in x for x in e.__notes__)
    else:
      test.assertIsInstance(c, jax.errors.SimplifiedTraceback)
  elif filter_mode == "remove_frames":
    test.assertIsInstance(c, traceback_util.UnfilteredStackTrace)
  else:
    test.assertFalse(isinstance(c, traceback_util.UnfilteredStackTrace))

  if frame_patterns:
    frames = []
    for frame, lineno in traceback.walk_tb(e.__traceback__):
      if filter_mode == "tracebackhide":
        if "__tracebackhide__"  in frame.f_locals.keys():
          continue
      frames.append((frame, lineno))

    c_tb = traceback.format_list(traceback.StackSummary.extract(frames))
    for (fname_pat, line_pat), frame_fmt in zip(
        reversed(frame_patterns), reversed(c_tb)):
      file = re.escape(__file__)
      fname_pat = re.escape(fname_pat)
      line_pat = re.escape(line_pat)
      full_pat = (
          f'  File "{file}", line ' r'[0-9]+'
          f', in {fname_pat}' r'\n\s*' f'{line_pat}')
      test.assertRegex(frame_fmt, full_pat)


@jtu.with_config(jax_traceback_filtering='auto')  # JaxTestCase defaults to off.
@parameterized.named_parameters(
  {"testcase_name": f"_{f}", "filter_mode": f}
  for f in ("tracebackhide", "remove_frames", "quiet_remove_frames"))
class FilteredTracebackTest(jtu.JaxTestCase):

  def test_nested_jit(self, filter_mode):
    @jit
    def innermost(x):
      assert False
    @jit
    def inbetween(x):
      return 1 + innermost(x)
    @jit
    def outermost(x):
      return 2 + inbetween(x)

    f = lambda: outermost(jnp.array([1, 2]))

    check_filtered_stack_trace(self, AssertionError, f, [
        ('<lambda>', 'f = lambda: outermost'),
        ('outermost', 'return 2 + inbetween(x)'),
        ('inbetween', 'return 1 + innermost(x)'),
        ('innermost', 'assert False')],
        filter_mode=filter_mode)

  def test_nested_jit_and_vmap(self, filter_mode):
    @jit
    def innermost(x):
      assert False
    @jit
    def inbetween(x):
      return 1 + vmap(innermost)(x)
    @jit
    def outermost(x):
      return 2 + inbetween(x)

    f = lambda: outermost(jnp.array([1, 2]))

    check_filtered_stack_trace(self, AssertionError, f, [
        ('<lambda>', 'f = lambda: outermost'),
        ('outermost', 'return 2 + inbetween(x)'),
        ('inbetween', 'return 1 + vmap(innermost)(x)'),
        ('innermost', 'assert False')],
        filter_mode=filter_mode)

  def test_nested_jit_and_grad(self, filter_mode):
    @jit
    def innermost(x):
      assert False
    @jit
    def inbetween(x):
      return 1 + grad(innermost)(x)
    @jit
    def outermost(x):
      return 2 + inbetween(x)

    f = lambda: outermost(jnp.array([1, 2]))

    check_filtered_stack_trace(self, TypeError, f, [
        ('<lambda>', 'f = lambda: outermost'),
        ('outermost', 'return 2 + inbetween(x)'),
        ('inbetween', 'return 1 + grad(innermost)(x)'),
    ], filter_mode=filter_mode)

  def test_lax_cond(self, filter_mode):
    def err(_):
      assert False
      return ()

    def f():
      return lax.cond(True, err, lambda _: (), ())

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.cond(True, err, lambda _: (), ())'),
        ('err', 'assert False')],
        filter_mode=filter_mode)

  def test_lax_switch(self, filter_mode):
    def err(_):
      assert False
      return ()

    def f():
      branches = [lambda _: (), err, lambda _: ()]
      return lax.switch(1, branches, ())

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.switch(1, branches, ())'),
        ('err', 'assert False')], filter_mode=filter_mode)

  def test_lax_scan(self, filter_mode):
    def err(*_):
      assert False
      return ()

    def f():
      return lax.scan(err, (), (), 3)

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.scan(err, (), (), 3)'),
        ('err', 'assert False')], filter_mode=filter_mode)

  def test_lax_fori_loop(self, filter_mode):
    def err(*_):
      assert False
      return ()

    def f():
      return lax.fori_loop(0, 3, err, ())

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.fori_loop(0, 3, err, ())'),
        ('err', 'assert False')], filter_mode=filter_mode)

  def test_lax_while_loop(self, filter_mode):
    def err(*_):
      assert False
      return ()

    def f():
      pred = lambda _: False
      return lax.while_loop(pred, err, ())

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.while_loop(pred, err, ())'),
        ('err', 'assert False')], filter_mode=filter_mode)

  def test_lax_map(self, filter_mode):
    def err(_):
      assert False
      return ()

    def f():
      xs = jnp.ones(3)
      return lax.map(err, xs)

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.map(err, xs)'),
        ('err', 'assert False')], filter_mode=filter_mode)

  def test_lax_custom_root(self, filter_mode):
    def err(*_):
      assert False
      return ()

    def g(x): return (x - 1.) ** 2.
    def solve(*_): return 1.

    def f1():
      return lax.custom_root(g, 0., err, solve)
    def f2():
      return lax.custom_root(g, 0., solve, err)
    def f3():
      return lax.custom_root(err, 0., solve, solve)

    check_filtered_stack_trace(self, AssertionError, f1, [
        ('f1', 'return lax.custom_root(g, 0., err, solve)'),
        ('err', 'assert False')], filter_mode=filter_mode)
    check_filtered_stack_trace(self, AssertionError, f2, [
        ('f2', 'return lax.custom_root(g, 0., solve, err)'),
        ('err', 'assert False')], filter_mode=filter_mode)
    check_filtered_stack_trace(self, AssertionError, f3, [
        ('f3', 'return lax.custom_root(err, 0., solve, solve)'),
        ('err', 'assert False')], filter_mode=filter_mode)

  def test_lax_custom_linear_solve(self, filter_mode):
    def err(*_):
      assert False
      return ()

    matvec = lambda v: v
    solve = lambda mv, b: 1.
    b = 1.

    def f1():
      return lax.custom_linear_solve(err, b, solve)
    def f2():
      return lax.custom_linear_solve(matvec, b, err)

    check_filtered_stack_trace(self, AssertionError, f1, [
        ('f1', 'return lax.custom_linear_solve(err, b, solve)'),
        ('err', 'assert False')], filter_mode=filter_mode)
    check_filtered_stack_trace(self, AssertionError, f2, [
        ('f2', 'return lax.custom_linear_solve(matvec, b, err)'),
        ('err', 'assert False')], filter_mode=filter_mode)

  def test_lax_associative_scan(self, filter_mode):
    def err(*_):
      assert False
      return ()

    def f():
      xs = jnp.arange(4.)
      return lax.associative_scan(err, xs)

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.associative_scan(err, xs)'),
        ('err', 'assert False')], filter_mode=filter_mode)

  def test_custom_jvp(self, filter_mode):
    def err(*args):
      assert False
      return args

    @jax.custom_jvp
    def f(x):
      return err(x)

    @f.defjvp
    def f_jvp(x, tx):
      x = err(x)
      return x, tx

    check_filtered_stack_trace(self, AssertionError, lambda: f(1.), [
        ('f', 'return err(x)'),
        ('err', 'assert False')], filter_mode=filter_mode)
    check_filtered_stack_trace(self, AssertionError, lambda: jax.jvp(f, [1.], [1.]), [
        ('f_jvp', 'x = err(x)'),
        ('err', 'assert False')], filter_mode=filter_mode)

  def test_custom_vjp(self, filter_mode):
    def err(*args):
      assert False
      return args[0]

    @jax.custom_vjp
    def f(x):
      return err(x)

    def fwd(x):
      return x, ()

    def fwd_err(x):
      x = err(x)
      return x, ()

    def bwd(_, g):
      return (g,)

    def bwd_err(_, g):
      g = err(g)
      return (g,)

    f.defvjp(fwd_err, bwd)

    check_filtered_stack_trace(self, AssertionError, lambda: f(1.), [
        ('f', 'return err(x)'),
        ('err', 'assert False')], filter_mode=filter_mode)

    check_filtered_stack_trace(self, AssertionError, lambda: jax.grad(f)(1.), [
        ('fwd_err', 'x = err(x)'),
        ('err', 'assert False')], filter_mode=filter_mode)

    f.defvjp(fwd, bwd_err)

    check_filtered_stack_trace(self, AssertionError, lambda: jax.grad(f)(1.), [
        ('bwd_err', 'g = err(g)'),
        ('err', 'assert False')], filter_mode=filter_mode)

  def test_cause_chain(self, filter_mode):
    @jit
    def inner(x):
      raise ValueError('inner')
    @jit
    def outer(x):
      try:
        inner(x)
      except ValueError as e:
        raise TypeError('outer') from e

    f = lambda: outer(1.)

    check_filtered_stack_trace(self, TypeError, f, [
        ('<lambda>', 'f = lambda: outer'),
        ('outer', 'raise TypeError')], filter_mode=filter_mode)
    e = get_exception(TypeError, f)  # Uses the default JAX_TRACEBACK_FILTERING=auto
    if sys.version_info >= (3, 11):
      assert any("For simplicity" in x for x in e.__notes__)
      self.assertIsInstance(e.__cause__, ValueError)
    else:
      self.assertIsInstance(e.__cause__, jax.errors.SimplifiedTraceback)
      self.assertIsInstance(e.__cause__.__cause__, ValueError)

  def test_null_traceback(self, filter_mode):
    class TestA: pass
    def f(a): return a + 1

    def err():
      a = TestA()
      return jit(f)(a)

    check_filtered_stack_trace(self, TypeError, err, [
        ('err', 'return jit(f)(a)')], filter_mode=filter_mode)


@jtu.with_config(jax_traceback_filtering='auto')  # JaxTestCase defaults to off.
class UserContextTracebackTest(jtu.JaxTestCase):

  def test_grad_norm(self):
    e = None
    try:
      with jax.debug_nans(True):
        jax.grad(jnp.linalg.norm)(jnp.zeros((3, 3), jnp.float32))
    except FloatingPointError as exc:
      e = exc
    self.assertIsNot(e, None)
    self.assertIn("invalid value", str(e))
    if sys.version_info >= (3, 11):
      self.assertIsInstance(
          e.__cause__,
          source_info_util.JaxStackTraceBeforeTransformation)
    else:
      self.assertIsInstance(
          e.__cause__.__cause__,
          source_info_util.JaxStackTraceBeforeTransformation)


class CustomErrorsTest(jtu.JaxTestCase):
  @jtu.sample_product(
    errorclass=[
     errorclass for errorclass in dir(jax.errors)
     if errorclass.endswith('Error') and errorclass not in ['JaxIndexError', 'JAXTypeError']
    ],
  )
  def testErrorsURL(self, errorclass):
    class FakeTracer(core.Tracer):
      aval = None
    ErrorClass = getattr(jax.errors, errorclass)
    err = ErrorClass(FakeTracer(None))

    self.assertIn(f'https://jax.readthedocs.io/en/latest/errors.html#jax.errors.{errorclass}', str(err))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
