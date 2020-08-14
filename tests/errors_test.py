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

from jax import grad, jit, vmap
import jax.numpy as jnp
from jax import test_util as jtu
from jax import traceback_util


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
  c = traceback_util.last_cause(e)
  test.assertIsInstance(c, traceback_util.FilteredStackTrace)
  c_tb = traceback.format_tb(c.__traceback__)
  if frame_patterns:
    for (fname_pat, line_pat), frame_fmt in zip(
        reversed(frame_patterns), reversed(c_tb)):
      fname_pat = re.escape(fname_pat)
      line_pat = re.escape(line_pat)
      full_pat = (
          f'  File "{__file__}", line ' r'[0-9]+'
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
        ('inbetween', 'return 1 + grad(innermost)(x)')])

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
    self.assertIsInstance(e.__cause__, ValueError)
    self.assertIsInstance(e.__cause__.__cause__,
                          traceback_util.FilteredStackTrace)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
