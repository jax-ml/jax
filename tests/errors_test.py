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

import re
import traceback
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import core, grad, jit, vmap, lax
import jax.numpy as jnp
from jax import test_util as jtu
from jax._src import source_info_util
from jax._src import traceback_util
from jax.lib import xla_extension


from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


def get_exception(etype, f):
  try:
    f()
  except etype as e:
    return e
  assert False

def check_filtered_stack_trace(test, etype, f, frame_patterns=[]):
  test.assertRaises(etype, f)
  e = get_exception(etype, f)
  c = e.__cause__
  test.assertIsInstance(c, traceback_util.UnfilteredStackTrace)
  c_tb = traceback.format_tb(e.__traceback__)
  # TODO(phawkins): remove this condition after jaxlib 0.1.66 is the minimum.
  if not hasattr(xla_extension, "replace_thread_exc_traceback"):
    c_tb = [t for t in c_tb if "reraise_with_filtered_traceback" not in t]
  if frame_patterns:
    for (fname_pat, line_pat), frame_fmt in zip(
        reversed(frame_patterns), reversed(c_tb)):
      file = re.escape(__file__)
      fname_pat = re.escape(fname_pat)
      line_pat = re.escape(line_pat)
      full_pat = (
          f'  File "{file}", line ' r'[0-9]+'
          f', in {fname_pat}' r'\n\s*' f'{line_pat}')
      test.assertRegex(frame_fmt, full_pat)


class FilteredTracebackTest(jtu.JaxTestCase):

  def test_nested_jit(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

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
        ('innermost', 'assert False')])

  def test_nested_jit_and_vmap(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

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
        ('innermost', 'assert False')])

  def test_nested_jit_and_grad(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

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
    ])

  def test_lax_cond(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

    def err(_):
      assert False
      return ()

    def f():
      return lax.cond(True, err, lambda _: (), ())

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.cond(True, err, lambda _: (), ())'),
        ('err', 'assert False')])

  def test_lax_switch(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

    def err(_):
      assert False
      return ()

    def f():
      branches = [lambda _: (), err, lambda _: ()]
      return lax.switch(1, branches, ())

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.switch(1, branches, ())'),
        ('err', 'assert False')])

  def test_lax_scan(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

    def err(*_):
      assert False
      return ()

    def f():
      return lax.scan(err, (), (), 3)

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.scan(err, (), (), 3)'),
        ('err', 'assert False')])

  def test_lax_fori_loop(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

    def err(*_):
      assert False
      return ()

    def f():
      return lax.fori_loop(0, 3, err, ())

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.fori_loop(0, 3, err, ())'),
        ('err', 'assert False')])

  def test_lax_while_loop(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

    def err(*_):
      assert False
      return ()

    def f():
      pred = lambda _: False
      return lax.while_loop(pred, err, ())

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.while_loop(pred, err, ())'),
        ('err', 'assert False')])

  def test_lax_map(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

    def err(_):
      assert False
      return ()

    def f():
      xs = jnp.ones(3)
      return lax.map(err, xs)

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.map(err, xs)'),
        ('err', 'assert False')])

  def test_lax_custom_root(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

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
        ('err', 'assert False')])
    check_filtered_stack_trace(self, AssertionError, f2, [
        ('f2', 'return lax.custom_root(g, 0., solve, err)'),
        ('err', 'assert False')])
    check_filtered_stack_trace(self, AssertionError, f3, [
        ('f3', 'return lax.custom_root(err, 0., solve, solve)'),
        ('err', 'assert False')])

  def test_lax_custom_linear_solve(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

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
        ('err', 'assert False')])
    check_filtered_stack_trace(self, AssertionError, f2, [
        ('f2', 'return lax.custom_linear_solve(matvec, b, err)'),
        ('err', 'assert False')])

  def test_lax_associative_scan(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

    def err(*_):
      assert False
      return ()

    def f():
      xs = jnp.arange(4.)
      return lax.associative_scan(err, xs)

    check_filtered_stack_trace(self, AssertionError, f, [
        ('f', 'return lax.associative_scan(err, xs)'),
        ('err', 'assert False')])

  def test_cause_chain(self):
    if not traceback_util.filtered_tracebacks_supported():
      raise unittest.SkipTest('Filtered tracebacks not supported')

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
        ('outer', 'raise TypeError')])
    e = get_exception(TypeError, f)
    self.assertIsInstance(e.__cause__, traceback_util.UnfilteredStackTrace)
    self.assertIsInstance(e.__cause__.__cause__, ValueError)


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
    # TODO(phawkins): make this test unconditional after jaxlib 0.1.66 is the
    # minimum.
    if jax.lib._xla_extension_version >= 19:
      self.assertIsInstance(
          e.__cause__.__cause__,
          source_info_util.JaxStackTraceBeforeTransformation)


class CustomErrorsTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}".format(errorclass), "errorclass": errorclass}
     for errorclass in dir(jax.errors)
     if errorclass.endswith('Error') and errorclass not in ['JaxIndexError', 'JAXTypeError']))
  def testErrorsURL(self, errorclass):
    class FakeTracer(core.Tracer):
      aval = None
    ErrorClass = getattr(jax.errors, errorclass)
    err = ErrorClass(FakeTracer(None))

    self.assertIn(f'https://jax.readthedocs.io/en/latest/errors.html#jax.errors.{errorclass}', str(err))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
