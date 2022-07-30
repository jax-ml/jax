# Copyright 2022 Google LLC
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
import contextlib
import collections
import functools
import io
import textwrap
import unittest
from unittest import mock

from typing import Callable, Generator

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax.config import config
from jax.experimental import maps
from jax.experimental import pjit
from jax._src import debugging
from jax._src import lib as jaxlib
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()

debug_print = debugging.debug_print

@contextlib.contextmanager
def capture_stdout() -> Generator[Callable[[], str], None, None]:
  with mock.patch('sys.stdout', new_callable=io.StringIO) as fp:
    def _read() -> str:
      return fp.getvalue()
    yield _read

def _format_multiline(text):
  return textwrap.dedent(text).lstrip()

prev_xla_flags = None

def setUpModule():
  global prev_xla_flags
  # This will control the CPU devices. On TPU we always have 2 devices
  prev_xla_flags = jtu.set_host_platform_device_count(2)

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  prev_xla_flags()

# TODO(sharadmv): remove jaxlib guards for TPU tests when jaxlib minimum
#                 version is >= 0.3.15
disabled_backends = []
if jaxlib.version < (0, 3, 15):
  disabled_backends.append("tpu")

class DebugPrintTest(jtu.JaxTestCase):

  @jtu.skip_on_devices(*disabled_backends)
  def test_simple_debug_print_works_in_eager_mode(self):
    def f(x):
      debug_print('x: {}', x)
    with capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  @jtu.skip_on_devices(*disabled_backends)
  def test_debug_print_works_with_named_format_strings(self):
    def f(x):
      debug_print('x: {x}', x=x)
    with capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  @jtu.skip_on_devices(*disabled_backends)
  def test_multiple_debug_prints_should_print_multiple_values(self):
    def f(x):
      debug_print('x: {x}', x=x)
      debug_print('y: {y}', y=x + 1)
    with capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\ny: 3\n")

  @jtu.skip_on_devices(*disabled_backends)
  def test_can_stage_out_debug_print(self):
    @jax.jit
    def f(x):
      debug_print('x: {x}', x=x)
    with capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  @jtu.skip_on_devices(*disabled_backends)
  def test_can_stage_out_ordered_print(self):
    @jax.jit
    def f(x):
      debug_print('x: {x}', x=x, ordered=True)
    with capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  @jtu.skip_on_devices(*disabled_backends)
  def test_can_double_stage_out_ordered_print(self):
    @jax.jit
    @jax.jit
    def f(x):
      debug_print('x: {x}', x=x, ordered=True)
    with capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  @jtu.skip_on_devices(*disabled_backends)
  def test_can_stage_out_ordered_print_with_pytree(self):
    @jax.jit
    def f(x):
      struct = dict(foo=x)
      debug_print('x: {}', struct, ordered=True)
    with capture_stdout() as output:
      f(np.array(2, np.int32))
      jax.effects_barrier()
    self.assertEqual(output(), f"x: {str(dict(foo=np.array(2, np.int32)))}\n")

class DebugPrintTransformationTest(jtu.JaxTestCase):

  def test_debug_print_batching(self):
    @jax.vmap
    def f(x):
      debug_print('hello: {}', x)
    with capture_stdout() as output:
      f(jnp.arange(2))
      jax.effects_barrier()
    self.assertEqual(output(), "hello: 0\nhello: 1\n")

  def test_debug_print_batching_with_diff_axes(self):
    @functools.partial(jax.vmap, in_axes=(0, 1))
    def f(x, y):
      debug_print('hello: {} {}', x, y)
    with capture_stdout() as output:
      f(jnp.arange(2), jnp.arange(2)[None])
      jax.effects_barrier()
    self.assertEqual(output(), "hello: 0 [0]\nhello: 1 [1]\n")

  def tested_debug_print_with_nested_vmap(self):
    def f(x):
      debug_print('hello: {}', x)
    # Call with
    # [[0, 1],
    #  [2, 3],
    #  [4, 5]]
    with capture_stdout() as output:
      # Should print over 0-axis then 1-axis
      jax.vmap(jax.vmap(f))(jnp.arange(6).reshape((3, 2)))
      jax.effects_barrier()
    self.assertEqual(
        output(),
        "hello: 0\nhello: 2\nhello: 4\nhello: 1\nhello: 3\nhello: 5\n")
    with capture_stdout() as output:
      # Should print over 1-axis then 0-axis
      jax.vmap(jax.vmap(f, in_axes=0), in_axes=1)(jnp.arange(6).reshape((3, 2)))
      jax.effects_barrier()
    self.assertEqual(
        output(),
        "hello: 0\nhello: 1\nhello: 2\nhello: 3\nhello: 4\nhello: 5\n")

  def test_debug_print_jvp_rule(self):
    def f(x):
      debug_print('x: {}', x)
    with capture_stdout() as output:
      jax.jvp(f, (1.,), (1.,))
      jax.effects_barrier()
    self.assertEqual(output(), "x: 1.0\n")

  def test_debug_print_vjp_rule(self):
    def f(x):
      debug_print('x: {}', x)
    with capture_stdout() as output:
      jax.vjp(f, 1.)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 1.0\n")

  def test_debug_print_in_custom_jvp(self):

    @jax.custom_jvp
    def print_tangent(x):
      return x

    @print_tangent.defjvp
    def _(primals, tangents):
      (x,), (t,) = primals, tangents
      debug_print("x_tangent: {}", t)
      return x, t

    def f(x):
      x = jnp.sin(x)
      x = print_tangent(x)
      return x

    with capture_stdout() as output:
      x = jnp.array(1., jnp.float32)
      jax.jvp(f, (x,), (x,))
      jax.effects_barrier()
    expected = jnp.cos(jnp.array(1., jnp.float32))
    self.assertEqual(output(), f"x_tangent: {expected}\n")

  @unittest.skip("doesn't work yet!")  # TODO(mattjj,sharadmv)
  def test_debug_print_in_custom_jvp_linearize(self):

    @jax.custom_jvp
    def print_tangent(x):
      return x

    @print_tangent.defjvp
    def _(primals, tangents):
      (x,), (t,) = primals, tangents
      debug_print("x_tangent: {}", t)
      return x, t

    def f(x):
      x = jnp.sin(x)
      x = print_tangent(x)
      return x

    with capture_stdout() as output:
      x = jnp.array(1., jnp.float32)
      y, f_lin = jax.linearize(f, x)
      jax.effects_barrier()
    self.assertEqual(output(), "")

    with capture_stdout() as output:
      _ = f_lin(x)
      jax.effects_barrier()
    expected = jnp.cos(jnp.array(1., jnp.float32))
    self.assertEqual(output(), f"x_tangent: {expected}\n")

  def test_debug_print_grad_with_custom_vjp_rule(self):
    @jax.custom_vjp
    def print_grad(x):
      return x

    def print_grad_fwd(x):
      return x, None

    def print_grad_bwd(_, x_grad):
      debug_print("x_grad: {}", x_grad)
      return (x_grad,)

    print_grad.defvjp(print_grad_fwd, print_grad_bwd)
    def f(x):
      debug_print("x: {}", x)
      x = print_grad(x)
      return jnp.sin(x)

    with capture_stdout() as output:
      jax.grad(f)(jnp.array(1., jnp.float32))
      jax.effects_barrier()
    expected = jnp.cos(jnp.array(1., jnp.float32))
    self.assertEqual(output(), f"x: 1.0\nx_grad: {expected}\n")

  def test_debug_print_transpose_rule(self):
    def f(x):
      debug_print('should never be called: {}', x)
      return x
    with capture_stdout() as output:
      jax.linear_transpose(f, 1.)(1.)
      jax.effects_barrier()
    # `debug_print` should be dropped by `partial_eval` because of no
    # output data-dependence.
    self.assertEqual(output(), "")

class DebugPrintControlFlowTest(jtu.JaxTestCase):

  def _assertLinesEqual(self, text1, text2):

    def _count(lines):
      return collections.Counter(lines)

    self.assertDictEqual(_count(text1.split("\n")), _count(text2.split("\n")))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name="_ordered" if ordered else "", ordered=ordered)
         for ordered in [False, True]))
  @jtu.skip_on_devices(*disabled_backends)
  def test_can_print_inside_scan(self, ordered):
    def f(xs):
      def _body(carry, x):
        debug_print("carry: {carry}, x: {x}", carry=carry, x=x, ordered=ordered)
        return carry + 1, x + 1
      return lax.scan(_body, 2, xs)
    with capture_stdout() as output:
      f(jnp.arange(2))
      jax.effects_barrier()
    self.assertEqual(
        output(),
        _format_multiline("""
      carry: 2, x: 0
      carry: 3, x: 1
      """))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name="_ordered" if ordered else "", ordered=ordered)
         for ordered in [False, True]))
  @jtu.skip_on_devices(*disabled_backends)
  def test_can_print_inside_for_loop(self, ordered):
    def f(x):
      def _body(i, x):
        debug_print("i: {i}", i=i, ordered=ordered)
        debug_print("x: {x}", x=x, ordered=ordered)
        return x + 1
      return lax.fori_loop(0, 5, _body, x)
    with capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    expected = _format_multiline("""
      i: 0
      x: 2
      i: 1
      x: 3
      i: 2
      x: 4
      i: 3
      x: 5
      i: 4
      x: 6
      """)
    if ordered:
      self.assertEqual(output(), expected)
    else:
      self._assertLinesEqual(output(), expected)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name="_ordered" if ordered else "", ordered=ordered)
         for ordered in [False, True]))
  @jtu.skip_on_devices(*disabled_backends)
  def test_can_print_inside_while_loop_body(self, ordered):
    def f(x):
      def _cond(x):
        return x < 10
      def _body(x):
        debug_print("x: {x}", x=x, ordered=ordered)
        return x + 1
      return lax.while_loop(_cond, _body, x)
    with capture_stdout() as output:
      f(5)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      x: 5
      x: 6
      x: 7
      x: 8
      x: 9
      """))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name="_ordered" if ordered else "", ordered=ordered)
         for ordered in [False, True]))
  @jtu.skip_on_devices(*disabled_backends)
  def test_can_print_inside_while_loop_cond(self, ordered):
    def f(x):
      def _cond(x):
        debug_print("x: {x}", x=x, ordered=ordered)
        return x < 10
      def _body(x):
        return x + 1
      return lax.while_loop(_cond, _body, x)
    with capture_stdout() as output:
      f(5)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      x: 5
      x: 6
      x: 7
      x: 8
      x: 9
      x: 10
      """))

    with capture_stdout() as output:
      f(10)
      jax.effects_barrier()
    # Should run the cond once
    self.assertEqual(output(), _format_multiline("""
      x: 10
      """))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name="_ordered" if ordered else "", ordered=ordered)
         for ordered in [False, True]))
  @jtu.skip_on_devices(*disabled_backends)
  def test_can_print_inside_cond(self, ordered):
    def f(x):
      def true_fun(x):
        debug_print("true: {}", x, ordered=ordered)
        return x
      def false_fun(x):
        debug_print("false: {}", x, ordered=ordered)
        return x
      return lax.cond(x < 5, true_fun, false_fun, x)
    with capture_stdout() as output:
      f(5)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      false: 5
      """))
    with capture_stdout() as output:
      f(4)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      true: 4
      """))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name="_ordered" if ordered else "", ordered=ordered)
         for ordered in [False, True]))
  @jtu.skip_on_devices(*disabled_backends)
  def test_can_print_inside_switch(self, ordered):
    def f(x):
      def b1(x):
        debug_print("b1: {}", x, ordered=ordered)
        return x
      def b2(x):
        debug_print("b2: {}", x, ordered=ordered)
        return x
      def b3(x):
        debug_print("b3: {}", x, ordered=ordered)
        return x
      return lax.switch(x, (b1, b2, b3), x)
    with capture_stdout() as output:
      f(0)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      b1: 0
      """))
    with capture_stdout() as output:
      f(1)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      b2: 1
      """))
    with capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      b3: 2
      """))

class DebugPrintParallelTest(jtu.JaxTestCase):

  def _assertLinesEqual(self, text1, text2):

    def _count(lines):
      return collections.Counter(lines)

    self.assertDictEqual(_count(text1.split("\n")), _count(text2.split("\n")))

  @jtu.skip_on_devices(*disabled_backends)
  def test_ordered_print_not_supported_in_pmap(self):

    @jax.pmap
    def f(x):
      debug_print("{}", x, ordered=True)
    with self.assertRaisesRegex(
        ValueError, "Ordered effects not supported in `pmap`."):
      f(jnp.arange(jax.local_device_count()))

  @jtu.skip_on_devices(*disabled_backends)
  def test_unordered_print_works_in_pmap(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices.")

    @jax.pmap
    def f(x):
      debug_print("hello: {}", x, ordered=False)
    with capture_stdout() as output:
      f(jnp.arange(jax.local_device_count()))
      jax.effects_barrier()
    lines = [f"hello: {i}\n" for i in range(jax.local_device_count())]
    self._assertLinesEqual(output(), "".join(lines))

    @jax.pmap
    def f2(x):
      debug_print('hello: {}', x)
      debug_print('hello: {}', x + 2)
    with capture_stdout() as output:
      f2(jnp.arange(2))
      jax.effects_barrier()
    self._assertLinesEqual(output(), "hello: 0\nhello: 1\nhello: 2\nhello: 3\n")

  @jtu.skip_on_devices(*disabled_backends)
  def test_unordered_print_with_pjit(self):

    def f(x):
      debug_print("{}", x, ordered=False)
      return x
    f = pjit.pjit(f, in_axis_resources=pjit.PartitionSpec('dev'),
                  out_axis_resources=pjit.PartitionSpec('dev'))
    with maps.Mesh(np.array(jax.devices()), ['dev']):
      with capture_stdout() as output:
        f(jnp.arange(8, dtype=jnp.int32))
        jax.effects_barrier()
      self.assertEqual(output(), "[0 1 2 3 4 5 6 7]\n")

    def f2(x):
      y = x.dot(x)
      debug_print("{}", y, ordered=False)
      return y
    f2 = pjit.pjit(f2, in_axis_resources=pjit.PartitionSpec('dev'),
                   out_axis_resources=pjit.PartitionSpec())
    with maps.Mesh(np.array(jax.devices()), ['dev']):
      with capture_stdout() as output:
        f2(jnp.arange(8, dtype=jnp.int32))
        jax.effects_barrier()
      self.assertEqual(output(), "140\n")

  @jtu.skip_on_devices(*disabled_backends)
  def test_unordered_print_of_pjit_of_while(self):

    def f(x):
      def cond(carry):
        i, *_ = carry
        return i < 5
      def body(carry):
        i, x = carry
        debug_print("{}", x, ordered=False)
        x = x + 1
        return (i + 1, x)
      return lax.while_loop(cond, body, (0, x))[1]
    f = pjit.pjit(f, in_axis_resources=pjit.PartitionSpec('dev'),
                  out_axis_resources=pjit.PartitionSpec('dev'))
    with maps.Mesh(np.array(jax.devices()), ['dev']):
      with capture_stdout() as output:
        f(jnp.arange(8, dtype=jnp.int32))
        jax.effects_barrier()
      self.assertEqual(output(),
          "[0 1 2 3 4 5 6 7]\n"
          "[1 2 3 4 5 6 7 8]\n"
          "[2 3 4 5 6 7 8 9]\n"
          "[ 3  4  5  6  7  8  9 10]\n"
          "[ 4  5  6  7  8  9 10 11]\n")

  @jtu.skip_on_devices(*disabled_backends)
  def test_unordered_print_of_pjit_of_xmap(self):

    def f(x):
      def foo(x):
        idx = lax.axis_index('foo')
        debug_print("{idx}: {x}", idx=idx, x=x)
        return jnp.mean(x, axis=['foo'])
      out = maps.xmap(foo, in_axes=['foo'], out_axes=[...])(x)
      debug_print("Out: {}", out)
      return out
    f = pjit.pjit(f, in_axis_resources=pjit.PartitionSpec('dev'),
                  out_axis_resources=pjit.PartitionSpec())
    with maps.Mesh(np.array(jax.devices()), ['dev']):
      with capture_stdout() as output:
        f(jnp.arange(8, dtype=jnp.int32) * 2)
        lines = ["0: 0", "1: 2", "2: 4", "3: 6", "4: 8", "5: 10", "6: 12",
                 "7: 14", "Out: 7.0", ""]
        jax.effects_barrier()
        self._assertLinesEqual(output(), "\n".join(lines))

  @jtu.skip_on_devices(*disabled_backends)
  def test_unordered_print_with_xmap(self):

    def f(x):
      debug_print("{}", x, ordered=False)
    f = maps.xmap(f, in_axes=['a'], out_axes=None, backend='cpu',
                  axis_resources={'a': 'dev'})
    with maps.Mesh(np.array(jax.devices()), ['dev']):
      with capture_stdout() as output:
        f(jnp.arange(40))
        jax.effects_barrier()
      lines = [f"{i}\n" for i in range(40)]
      self._assertLinesEqual(output(), "".join(lines))

  @jtu.skip_on_devices(*disabled_backends)
  def test_unordered_print_works_in_pmap_of_while(self):

    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices.")

    @jax.pmap
    def f(x):
      def cond(x):
        return x < 3
      def body(x):
        debug_print("hello: {}", x, ordered=False)
        return x + 1
      return lax.while_loop(cond, body, x)

    with capture_stdout() as output:
      f(jnp.arange(2))
      jax.effects_barrier()

    self._assertLinesEqual(
        output(), "hello: 0\nhello: 1\nhello: 2\n"
        "hello: 1\nhello: 2\n")

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
