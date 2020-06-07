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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import logging
import numpy as np
import os
import re
from typing import Callable, Sequence
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

from jax import api
from jax import lax
from jax import numpy as jnp
from jax import test_util as jtu
from jax.config import config
from jax.experimental import host_callback as hcb
from jax.lib import xla_bridge


config.parse_flags_with_absl()
FLAGS = config.FLAGS

def skip_if_jit_not_enabled():
  if os.getenv("JAX_ENABLE_JIT_PRINT", "false") == "false":
    raise SkipTest("print jit not enabled yet; use JAX_ENABLE_JIT_PRINT env.")

def supported_dtypes():
  return sorted(jtu.supported_dtypes(), key=lambda x: np.dtype(x).name)

class _TestingOutputStream(object):
  """Use as `output_stream` for tests."""

  def __init__(self):
    self._output = []
    self.testMethodName = None

  def write(self, what: str) -> None:
    print(f"output_stream[{self.testMethodName}]: {what}", end="")
    self._output.append(what)

  @property
  def output(self):
    return "".join(self._output)

  def __str__(self):
    return "TestingOutputStream"

  def reset(self):
    self._output = []


testing_stream = _TestingOutputStream()


def fun1(a):
  y = hcb.id_print(a * 2., what="a * 2", output_stream=testing_stream)
  y = hcb.id_print(y * 3., what="y * 3", output_stream=testing_stream, result=y)
  return y**2  # Some computation to make the gradient interesting


def fun1_equiv(a):  # Numerical equivalent of fun`
  return (a * 2.)**2

def assertMultiLineStrippedEqual(tst: jtu.JaxTestCase, expected: str, what: str):
  """A variant that preprocesses the string to eliminate non-determinism in
  floating point values, and several uninteresting id_tap primitive params."""
  # Sometimes we get floating points in the output; we round them
  def repl_floats(match_group):
    matched = match_group.group(0)
    if matched == ".": return matched
    # TODO: why can't we use here np.around?
    x = np.around(float(matched), decimals=2)
    return f"{x:.2f}"
  what = re.sub(r"\-?\d*\.[\-\def]*", repl_floats, what)
  what = re.sub(r"output_stream=[^\]\n]*", "", what)
  what = re.sub(r"threshold=[^\]\n]*", "", what)
  # Empty lines
  what = re.sub(r"^\s*\n", "", what, flags=re.MULTILINE)
  def repl_func(match_group):
    matched = match_group.group(0)
    if "function _print_consumer" in matched:
      return "func=_print"
    else:
      return "..."
  what = re.sub(r"func=(.*)", repl_func, what)
  tst.assertMultiLineStrippedEqual(expected, what)

class HostCallbackTest(jtu.JaxTestCase):

  def setUp(self):
    testing_stream.reset()
    testing_stream.testMethodName = self._testMethodName
    self.old_flags = os.getenv("XLA_FLAGS", "")

  def tearDown(self) -> None:
    if os.getenv("XLA_FLAGS") != self.old_flags:
      os.environ["XLA_FLAGS"] = self.old_flags
      xla_bridge.get_backend.cache_clear()

  def helper_set_devices(self, nr_devices):
    flags_str = os.getenv("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (
        flags_str +
        " --xla_force_host_platform_device_count={}".format(nr_devices))
    # Clear any cached backends so new CPU backend will pick up the env var.
    xla_bridge.get_backend.cache_clear()
    return api.devices()

  def helper_set_hlo_dump(self):
    flags_str = os.getenv("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = f"{flags_str} --xla_dump_to=/tmp/xla_dump"
    # Clear any cached backends so new CPU backend will pick up the env var.
    xla_bridge.get_backend.cache_clear()

  def test_eval(self):
    # TODO: renable jaxpr golden tests when changing host_callback
    #assertMultiLineStrippedEqual(self, "", str(api.make_jaxpr(fun1)(5.)))

    with hcb.outfeed_receiver():
      self.assertAllClose((5. * 2.) ** 2, fun1(5.))
    assertMultiLineStrippedEqual(self, """
what: a * 2
10.00
what: y * 3
30.00""", testing_stream.output)
    testing_stream.reset()

  def test_with_tuple_results(self):
    def func2(x):
      x1, y1 = hcb.id_print((x * 2., x * 3.), output_stream=testing_stream)
      return x1 + y1

    #assertMultiLineStrippedEqual(self, "", str(api.make_jaxpr(func2)(3.)))
    with hcb.outfeed_receiver():
      self.assertEqual(3. * (2. + 3.), func2(3.))
    assertMultiLineStrippedEqual(self, """
[ 6.00
  9.00 ]""", testing_stream.output)
    testing_stream.reset()

  def test_with_dict_results(self):
    def func2(x):
      res = hcb.id_print(dict(a=x * 2., b=x * 3.), output_stream=testing_stream)
      return res["a"] + res["b"]

    with hcb.outfeed_receiver():
      self.assertEqual(3. * (2. + 3.), func2(3.))
    assertMultiLineStrippedEqual(self, """
{ a=6.00
  b=9.00 }""", testing_stream.output)
    testing_stream.reset()

  def test_with_result(self):
    def func2(x):
      x1 = hcb.id_print((x * 2., x * 3.), result=x * 4.,
                        output_stream=testing_stream)
      return x1

    with hcb.outfeed_receiver():
      self.assertEqual(3. * 4., func2(3.))
    assertMultiLineStrippedEqual(self, """
[ 6.00
  9.00 ]""", testing_stream.output)
    testing_stream.reset()

  def test_eval_tap_exception(self):
    # Simulate a tap error
    def tap_err(*args, **kwargs):
      raise NotImplementedError

    def func(x):
      x1 = hcb.id_print(x + 1, what="x1", output_stream=testing_stream)
      x2 = hcb.id_tap(tap_err, x1 + 1, what="err")
      x3 = hcb.id_print(x2 + 1, what="x3", output_stream=testing_stream)
      return x3

    with self.assertRaises(hcb.TapFunctionException):
      with hcb.outfeed_receiver():
        _ = func(0)

    # We should have received everything before the error
    assertMultiLineStrippedEqual(self, """
what: x1
1
what: x3
3""", testing_stream.output)
    testing_stream.reset()

  def test_jit_simple(self):
    jit_fun1 = api.jit(lambda x: 3. * hcb.id_print(
        2. * x, what="here", output_stream=testing_stream))

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      res = jit_fun1(5.)

    self.assertAllClose(6. * 5., res)
    assertMultiLineStrippedEqual(self, """
what: here
10.00""", testing_stream.output)
    testing_stream.reset()

  def test_jit_constant(self):
    def func(x):
      return hcb.id_print(42, result=x, output_stream=testing_stream)

    #assertMultiLineStrippedEqual(self, "", str(api.make_jaxpr(api.jit(func))(5)))

    with hcb.outfeed_receiver():
      self.assertAllClose(5, api.jit(func)(5))
    assertMultiLineStrippedEqual(self, """
42""", testing_stream.output)
    testing_stream.reset()

  def test_jit_sequence1(self):
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      return hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

    logging.info("%s: %s", self._testMethodName,
          api.make_jaxpr(func)(1))
    logging.info("%s: %s", self._testMethodName,
          api.xla_computation(func)(1).as_hlo_text())

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(2, api.jit(func)(1))
    assertMultiLineStrippedEqual(self, """
where: 1
1
where: 2
2""", testing_stream.output)
    testing_stream.reset()

  def test_jit2(self):
    """A sequence of JIT."""
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)
      return x2

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(2, api.jit(func)(1))
      self.assertEqual(11, api.jit(func)(10))

    assertMultiLineStrippedEqual(self, """
where: 1
1
where: 2
2
where: 1
10
where: 2
11""", testing_stream.output)
    testing_stream.reset()

  def test_jit_nested(self):
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      def func_nested(x):
        x2 = hcb.id_print(x + 1, where="nested", output_stream=testing_stream)
        return x2
      x3 = api.jit(func_nested)(x1)
      return hcb.id_print(x3 + 1, where="3", output_stream=testing_stream)

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(3, api.jit(func)(1))
    assertMultiLineStrippedEqual(self, """
where: 1
1
where: nested
2
where: 3
3""", testing_stream.output)
    testing_stream.reset()

  def test_jit_devices(self):
    """Running on multiple devices."""
    devices = api.local_devices()
    logging.info(f"{self._testMethodName}: has devices {devices}")
    def func(x, device_id):
      x1 = hcb.id_print(x, dev=str(device_id), output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, dev=str(device_id), output_stream=testing_stream)
      return x2

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      for d in devices:
        self.assertEqual(112, api.jit(func, device=d, static_argnums=1)(111, d.id))
    logging.info(f"{self._testMethodName}: found output {testing_stream.output}")
    self.assertEqual(len(devices), len(re.findall(r"111", testing_stream.output)))
    self.assertEqual(len(devices), len(re.findall(r"112", testing_stream.output)))
    testing_stream.reset()

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_with_jit_{with_jit}",
              with_jit=with_jit)
          for with_jit in [True, False]))
  def test_pytree(self, with_jit=False):
    def func(x, what=""):
      """Returns some pytrees depending on x"""
      if what == "pair_1_x":
        return (1, x)
      elif what == "pair_x_2x":
        return (x, 2 * x)
      elif what == "dict":
        return dict(a=2 * x, b=3 * x)
      else:
        assert False
    tap_count = 0
    def tap_func(a, what=""):
      nonlocal tap_count
      tap_count += 1
      self.assertEqual(func(5, what), a)

    transform = api.jit if with_jit else lambda f: f
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      for what in ("pair_1_x", "pair_x_2x", "dict"):
        self.assertEqual(func(10, what),
                         transform(lambda x: hcb.id_tap(tap_func, func(x, what),
                                                        result=func(x * 2, what),
                                                        what=what))(5))
    # Wait for receivers to be done
    self.assertEqual(3, tap_count)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_with_jit_{with_jit}",
              with_jit=with_jit)
          for with_jit in [True, False]))
  def test_cond(self, with_jit=False):
    """A conditional"""
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

      x4 = lax.cond(x % 2 == 0,
                    lambda x: hcb.id_print(x, where="cond_t", output_stream=testing_stream),
                    lambda x: hcb.id_print(-1, where="cond_f", result=x, output_stream=testing_stream),
                    x2 + 1)
      x5 = hcb.id_print(x4 + 1, where="end", output_stream=testing_stream)
      return x5

    transform = api.jit if with_jit else lambda f: f
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(4, transform(func)(1))
    assertMultiLineStrippedEqual(self, """
where: 1
1
where: 2
2
where: cond_f
-1
where: end
4""", testing_stream.output)
    testing_stream.reset()

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_with_jit_{with_jit}",
              with_jit=with_jit)
          for with_jit in [True, False]))
  def test_while_cond(self, with_jit=False):
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)
      def body(x):
        x3 = hcb.id_print(x, where="w_b_1", output_stream=testing_stream)
        x4 = lax.cond(x % 2 == 0,
                      lambda x: hcb.id_print(x, where="w_b_t",
                                             output_stream=testing_stream),
                      lambda x: hcb.id_print(-1, where="w_b_f",
                                             result=x, output_stream=testing_stream),
                      x3 + 1)
        return hcb.id_print(x4, where="w_b_2", output_stream=testing_stream)
      x10 = lax.while_loop(lambda x: x <= 3, body, x2)
      res = hcb.id_print(x10, where="end", output_stream=testing_stream)
      return res

    transform = api.jit if with_jit else lambda f: f
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(4, transform(func)(1))
    assertMultiLineStrippedEqual(self, """
where: 1
1
where: 2
2
where: w_b_1
2
where: w_b_t
3
where: w_b_2
3
where: w_b_1
3
where: w_b_f
-1
where: w_b_2
4
where: end
4""", testing_stream.output)
    testing_stream.reset()

  def test_jit_while_pred_tap(self):
    """While with printing in the conditional."""
    def func(x):
      x1 = hcb.id_print(x, where="1")
      x10 = lax.while_loop(lambda x: hcb.id_print(x < 3,
                                                  where="w_p",
                                                  output_stream=testing_stream),
                           lambda x: hcb.id_print(x + 1, where="w_b",
                                                  output_stream=testing_stream),
                           x1)
      res = hcb.id_print(x10, where="3", output_stream=testing_stream)
      return res

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(3, api.jit(func)(1))
    assertMultiLineStrippedEqual(self,
      """
where: w_p
True
where: w_b
2
where: w_p
True
where: w_b
3
where: w_p
False
where: 3
3""", testing_stream.output)
    testing_stream.reset()

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_with_jit_{with_jit}",
              with_jit=with_jit)
          for with_jit in [True, False]))
  def test_scan_cond(self, with_jit=False):
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

      def body(c, x):
        x3 = hcb.id_print(x, where="s_1", output_stream=testing_stream)
        x4 = lax.cond(x % 2 == 0,
                      lambda x: hcb.id_print(x, where="s_t", output_stream=testing_stream),
                      lambda x: hcb.id_print(-1, where="s_f", result=x, output_stream=testing_stream),
                      x3 + 1)
        return (c, hcb.id_print(x4, where="s_2", output_stream=testing_stream))

      _, x10 = lax.scan(body, x2, jnp.arange(3))
      res = hcb.id_print(x10, where="10", output_stream=testing_stream)
      return res

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      if with_jit:
        func = api.jit(func)
      res = func(1)
      self.assertAllClose(jnp.array([1, 2, 3]), res)
    assertMultiLineStrippedEqual(self, """
where: 1
1
where: 2
2
where: s_1
0
where: s_t
1
where: s_2
1
where: s_1
1
where: s_f
-1
where: s_2
2
where: s_1
2
where: s_t
3
where: s_2
3
where: 10
[1 2 3]""", testing_stream.output)
    testing_stream.reset()

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_shape_{shape}_dtype_{dtype}_nr_args={nr_args}",
              shape=shape,
              dtype=dtype,
              nr_args=nr_args) for nr_args in [1, 2]
          for shape in [(), (2,), (2, 3), (2, 3, 4)]
          for dtype in supported_dtypes()))
  def test_jit_types(self, nr_args=2, dtype=jnp.int16, shape=(2,)):
    if dtype in (jnp.complex64, jnp.complex128, jnp.bool_):
      raise SkipTest(f"id_print jit not implemented for {dtype}.")
    if jtu.device_under_test() == "tpu":
      if dtype in (jnp.int16,):
        raise SkipTest(f"transfering {dtype} not supported on TPU")
    args = [jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)]
    if nr_args > 1:
      args = args * nr_args
    jit_fun1 = api.jit(lambda xs: hcb.id_print(
        xs,
        a_new_test="************",
        testcase_name=f"shape_{shape}_dtype_{dtype}_nr_args={nr_args}"))
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      _ = jit_fun1(args)
    # self.assertAllClose(args, res)

  def test_jit_large(self):
    arg = jnp.arange(10000, dtype=jnp.int32).reshape((10, 10, 5, -1))
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      api.jit(hcb.id_print)(arg)

  def test_jit_several_together(self):
    arg = jnp.arange(50, dtype=jnp.int32).reshape((10, 5))
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      api.jit(lambda x, y: hcb.id_print((x, y, x * 2.)))(arg, jnp.ones(100, dtype=jnp.int32))

  def test_jit_interleaving(self):
    # Several jit's without data dependencies; they may interfere
    count = 0  # Count tap invocations
    nr_arrays = 5
    def tap_func(arg, **kwargs):
      nonlocal count
      assert len(arg) == nr_arrays
      count += 1
    # This is the function that we'll run multiple times
    def func(x, count):
      for i in range(count):
        x = hcb.id_tap(tap_func, [x + i for i in range(nr_arrays)], i=i)[-1]
      return x
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      x = jnp.array(1, dtype=np.int32)
      res = 0
      for i in range(10):
        # No dependencies between the jit invocations
        res += api.jit(lambda x: func(x, 10))(x)
    self.assertEqual(100, count)

  def test_jit_tap_exception(self):
    # Simulate a tap error
    def tap_err(*args, **kwargs):
      raise NotImplementedError
    def func(x):
      x1 = hcb.id_print(x + 1, what="x1", output_stream=testing_stream)
      x2 = hcb.id_tap(tap_err, x1 + 1, what="err")
      x3 = hcb.id_print(x2 + 1, what="x3", output_stream=testing_stream)
      return x3

    with self.assertRaises(hcb.TapFunctionException):
      with hcb.outfeed_receiver(receiver_name=self._testMethodName):
        res = api.jit(func)(0)
    # Even though the receiver thread raised, the main thread should still
    # return 3.
    self.assertEqual(3, res)
    # We should have received all others
    assertMultiLineStrippedEqual(self, """
what: x1
1
what: x3
3""", testing_stream.output)
    testing_stream.reset()

  def test_jit_unknown_tap(self):
    # Simulate an unknown tap function
    def func(x):
      x1 = hcb.id_print(x + 1, what="x1", output_stream=testing_stream)
      x2 = hcb.id_tap(hcb._unknown_testing_consumer, x1 + 1, what="err")
      x3 = hcb.id_print(x2 + 1, what="x3", output_stream=testing_stream)
      return x3

    with self.assertRaises(hcb.TapFunctionException):
      with hcb.outfeed_receiver(receiver_name=self._testMethodName):
        res = api.jit(func)(0)
    # Even though the receiver thread raised, the main thread should still
    # return 3.
    self.assertEqual(3, res)
    # We should have received all others
    assertMultiLineStrippedEqual(self, """
what: x1
1
what: x3
3""", testing_stream.output)
    testing_stream.reset()

  # On CPU and GPU the device code blocks
  # On GPU it seems that there is a 5 min timeout?
  # On TPU the client does not block, but messes up the rest somehow
  @jtu.skip_on_devices("cpu", "gpu", "tpu")
  def test_jit_receiver_ends_prematurely(self):
    # Simulate an unknown tap function
    def func(x):
      x1 = hcb.id_print(x + 1, what="x1", output_stream=testing_stream)
      x2 = hcb.id_tap(hcb._end_consumer, result=x1 + 1)  # Will end the consumer loop
      x3 = hcb.id_print(x2 + 1, what="x3", output_stream=testing_stream)
      return x3

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      _ = api.jit(func)(0)

    assert False  # It seems that the previous jit blocks above

  def test_jit_error_no_consumer(self):
    # Check for errors if starting jit without a consumer active
    with self.assertRaisesRegex(ValueError, "outfeed_receiver is not started"):
      api.jit(lambda x: hcb.id_print(x))(0)

  def test_while(self):
    """Executing while, even without JIT uses compiled code"""
    y = jnp.ones(5)  # captured const

    def func(x):
      return lax.while_loop(
        lambda c: c[1] < 5,
        lambda c: (y, hcb.id_print(c[1], output_stream=testing_stream) + 1),
        (x, 1))
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      func(y)
    assertMultiLineStrippedEqual(self, """
1
2
3
4""", testing_stream.output)
    testing_stream.reset()

  def test_while_error_no_receiver(self):
    """Executing while needs the receiver"""
    y = jnp.ones(5)  # captured const
    def func(x):
      return lax.while_loop(
        lambda c: c[1] < 5,
        lambda c: (y, hcb.id_print(c[1], output_stream=testing_stream) + 1),
        (x, 1))

    with self.assertRaisesRegex(ValueError, ".*outfeed_receiver.*not started"):
      func(y).block_until_ready()


  def test_jvp(self):
    jvp_fun1 = lambda x, xt: api.jvp(fun1, (x,), (xt,))
    #assertMultiLineStrippedEqual(self, "",
    #                             str(api.make_jaxpr(jvp_fun1)(jnp.float32(5.), jnp.float32(0.1))))
    with hcb.outfeed_receiver():
      res_primals, res_tangents = jvp_fun1(jnp.float32(5.), jnp.float32(0.1))
    self.assertAllClose(100., res_primals, check_dtypes=False)
    self.assertAllClose(4., res_tangents, check_dtypes=False)
    assertMultiLineStrippedEqual(self, """
what: a * 2
10.00
transforms: ({'name': 'jvp'},) what: a * 2
0.20
what: y * 3
30.00
transforms: ({'name': 'jvp'},) what: y * 3
0.60""", testing_stream.output)
    testing_stream.reset()

  def test_grad_primal_unused(self):
    # The output of id_print is not needed for backwards pass
    def func(x):
      return 2. * hcb.id_print(x * 3., what="x * 3", output_stream=testing_stream)

    grad_func = api.grad(func)
    with hcb.outfeed_receiver():
      assertMultiLineStrippedEqual(self, """
{ lambda  ; a.
  let
  in (6.00,) }""", str(api.make_jaxpr(grad_func)(5.)))

    # Just making the Jaxpr invokes the id_print once
    assertMultiLineStrippedEqual(self, """
transforms: ({'name': 'jvp'}, {'name': 'transpose'}) what: x * 3
2.00""", testing_stream.output)
    testing_stream.reset()

    with hcb.outfeed_receiver():
      res_grad = grad_func(jnp.float32(5.))

    self.assertAllClose(6., res_grad, check_dtypes=False)
    assertMultiLineStrippedEqual(self, """
what: x * 3
15.00
transforms: ({'name': 'jvp'}, {'name': 'transpose'}) what: x * 3
2.00""", testing_stream.output)
    testing_stream.reset()

  def test_grad_simple(self):
    def func(x):
      y = hcb.id_print(x * 2., what="x * 2", output_stream=testing_stream)
      return x * hcb.id_print(y * 3., what="y * 3", output_stream=testing_stream)
    grad_func = api.grad(func)
    #assertMultiLineStrippedEqual(self, "", str(api.make_jaxpr(grad_func)(5.)))

    with hcb.outfeed_receiver():
      res_grad = grad_func(jnp.float32(5.))
    self.assertAllClose(2. * 5. * 6., res_grad, check_dtypes=False)
    assertMultiLineStrippedEqual(self, """
what: x * 2
10.00
what: y * 3
30.00
transforms: ({'name': 'jvp'}, {'name': 'transpose'}) what: y * 3
5.00
transforms: ({'name': 'jvp'}, {'name': 'transpose'}) what: x * 2
15.00""", testing_stream.output)
    testing_stream.reset()

  def test_grad_double(self):
    def func(x):
      y = hcb.id_print(x * 2., what="x * 2", output_stream=testing_stream)
      return x * (y * 3.)

    grad_func = api.grad(api.grad(func))
    with hcb.outfeed_receiver():
      _ = api.make_jaxpr(grad_func)(5.)
      # Just making the Jaxpr invokes the id_print twiceonce
      assertMultiLineStrippedEqual(self, """
transforms: ({'name': 'jvp'}, {'name': 'transpose'}) what: x * 2
3.00
transforms: ({'name': 'jvp'}, {'name': 'transpose'}, {'name': 'jvp'}, {'name': 'transpose'}) what: x * 2
2.00""", testing_stream.output)
      testing_stream.reset()
      res_grad = grad_func(jnp.float32(5.))

    self.assertAllClose(12., res_grad, check_dtypes=False)
    assertMultiLineStrippedEqual(self, """
what: x * 2
10.00
transforms: ({'name': 'jvp'}, {'name': 'transpose'}) what: x * 2
15.00
transforms: ({'name': 'jvp'}, {'name': 'transpose'}, {'name': 'jvp'}, {'name': 'transpose'}) what: x * 2
2.00
transforms: ({'name': 'jvp'}, {'name': 'transpose'}) what: x * 2
3.00""", testing_stream.output)
    testing_stream.reset()

  def test_vmap(self):
    vmap_fun1 = api.vmap(fun1)
    vargs = jnp.array([jnp.float32(4.), jnp.float32(5.)])
    #assertMultiLineStrippedEqual(self, "", str(api.make_jaxpr(vmap_fun1)(vargs)))
    with hcb.outfeed_receiver():
      _ = vmap_fun1(vargs)
    assertMultiLineStrippedEqual(self, """
transforms: ({'name': 'batch', 'batch_dims': (0,)},) what: a * 2
[ 8.00 10.00]
transforms: ({'name': 'batch', 'batch_dims': (0, 0)},) what: y * 3
[24.00 30.00]""", testing_stream.output)
    testing_stream.reset()

  def test_vmap_not_batched(self):
    x = 3.
    def func(y):
      # x is not mapped, y is mapped
      _, y = hcb.id_print((x, y), output_stream=testing_stream)
      return x + y

    vmap_func = api.vmap(func)
    vargs = jnp.array([jnp.float32(4.), jnp.float32(5.)])
    #assertMultiLineStrippedEqual(self, "", str(api.make_jaxpr(vmap_func)(vargs)))
    with hcb.outfeed_receiver():
      _ = vmap_func(vargs)
    assertMultiLineStrippedEqual(self, """
transforms: ({'name': 'batch', 'batch_dims': (None, 0)},)
[ 3.00
  [4.00 5.00] ]""", testing_stream.output)
    testing_stream.reset()

  def test_double_vmap(self):
    # A 2D tensor with x[i, j] = i + j using 2 vmap
    def sum(x, y):
      return hcb.id_print(x + y, output_stream=testing_stream)
    def sum_rows(xv, y):
      return api.vmap(sum, in_axes=(0, None))(xv, y)
    def sum_all(xv, yv):
      return api.vmap(sum_rows, in_axes=(None, 0))(xv, yv)

    xv = jnp.arange(5, dtype=np.int32)
    yv = jnp.arange(3, dtype=np.int32)
    #assertMultiLineStrippedEqual(self, "", str(api.make_jaxpr(sum_all)(xv, yv)))
    with hcb.outfeed_receiver():
      _ = sum_all(xv, yv)
    assertMultiLineStrippedEqual(self, """
transforms: ({'name': 'batch', 'batch_dims': (0,)}, {'name': 'batch', 'batch_dims': (0,)})
[[0 1 2 3 4]
 [1 2 3 4 5]
 [2 3 4 5 6]]""", testing_stream.output)
    testing_stream.reset()

  def test_vmap_while(self):
    """Vmap of while."""

    def func(x):
      # like max(x, 2)
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = lax.while_loop(lambda x: x < 2,
                           lambda x: hcb.id_print(x + 1, where="w_b",
                                                  output_stream=testing_stream),
                           x1)
      res = hcb.id_print(x2, where="3", output_stream=testing_stream)
      return res

    inputs = np.arange(5, dtype=np.int32)
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertAllClose(np.array([2, 2, 2, 3, 4]), api.jit(api.vmap(func))(inputs),
                          check_dtypes=False)
    assertMultiLineStrippedEqual(self, """
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: 1
[0 1 2 3 4]
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: w_b
[1 2 3 4 5]
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: w_b
[2 3 3 4 5]
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: 3
[2 2 2 3 4]""", testing_stream.output)
    testing_stream.reset()

  def test_vmap_while_tap_cond(self):
    """Vmap of while, with a tap in the conditional."""

    def func(x):
      # like max(x, 2)
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = lax.while_loop(lambda x: hcb.id_print(x < 2, where="w_c",
                                                 output_stream=testing_stream),
                          lambda x: hcb.id_print(x + 1, where="w_b",
                                                 output_stream=testing_stream),
                          x1)
      res = hcb.id_print(x2, where="3", output_stream=testing_stream)
      return res

    inputs = np.arange(5, dtype=np.int32)
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertAllClose(np.array([2, 2, 2, 3, 4]), api.jit(api.vmap(func))(inputs),
                          check_dtypes=False)
    assertMultiLineStrippedEqual(self, """
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: 1
[0 1 2 3 4]
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: w_c
[ True  True False False False]
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: w_b
[1 2 3 4 5]
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: w_c
[ True False False False False]
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: w_b
[2 3 3 4 5]
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: w_c
[False False False False False]
transforms: ({'name': 'batch', 'batch_dims': (0,)},) where: 3
[2 2 2 3 4]""", testing_stream.output)
    testing_stream.reset()

  def test_pmap(self):
    vargs = 2. + jnp.arange(api.local_device_count(), dtype=jnp.float32)

    pmap_fun1 = api.pmap(fun1, axis_name="i")
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      res = pmap_fun1(vargs)
    expected_res = jnp.stack([fun1_equiv(2. + a) for a in range(api.local_device_count())])
    self.assertAllClose(expected_res, res, check_dtypes=False)

  def test_pmap_error_no_receiver(self):
    # Check for errors if starting jit without a consumer active
    vargs = 2. + jnp.arange(api.local_device_count(), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "outfeed_receiver is not started"):
      api.pmap(lambda x: hcb.id_print(x))(vargs)

  def test_mask(self):
    # TODO(necula)
    raise SkipTest("masking has regressed")
    @partial(api.mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return jnp.sum(hcb.id_print(x, what="x", output_stream=testing_stream))
    args = [jnp.arange(4)], dict(n=np.int64(2))
    assertMultiLineStrippedEqual(self, """
{ lambda c f ; a b.
  let d = lt c b
      e = id_tap[ func=_print
                  logical_shapes=[(Traced<ShapedArray(int32[]):JaxprTrace(level=0/0)>,)]
                  transforms=('mask',)
                  what=x ] a
      g = select d e f
      h = reduce_sum[ axes=(0,) ] g
  in (h,) }""", str(api.make_jaxpr(padded_sum)(*args)))

    _ = padded_sum(*args)
    self.assertMultiLineStrippedEqual("""
logical_shapes: [(2,)] transforms: ('mask',) what: x
[0 1 2 3]
   """, testing_stream.output)
    testing_stream.reset()

class OutfeedRewriterTest(jtu.JaxTestCase):
  def assertRewrite(self, expected: str, func: Callable, args: Sequence,
                    has_input_token=True, has_output_token=True):
    """Check that the rewrite of func(*args) matches expected."""
    _ = api.make_jaxpr(func)(*args)
    # TODO: re-enable when we change the host_callback rewriter
    #assertMultiLineStrippedEqual(self, expected,
    #  str(hcb._rewrite_typed_jaxpr(jaxpr, has_input_token, has_output_token)[0]))

  def test_no_outfeed(self):
    self.assertRewrite("""
{ lambda  ; a.
  let b = mul a a
      c = add a b
  in (c,) }""", lambda x: x + x * x, [0], has_input_token=False, has_output_token=False)
    self.assertRewrite("""
{ lambda  ; a d.
  let b = mul a a
      c = add a b
  in (c,) }""", lambda x: x + x * x, [0], has_output_token=False)
    self.assertRewrite("""
{ lambda  ; a d.
  let b = mul a a
      c = add a b
  in (c, d) }""", lambda x: x + x * x, [0])

  def test_simple_outfeed(self):
    self.assertRewrite("""
{ lambda  ; a d.
  let b = add a a
      c e = id_tap[ arg_treedef=*
                    func=_print
                     ] b d
  in (c, e) }""", lambda x: hcb.id_print(x + x), [0])

  def test_cond(self):
    y = jnp.ones(5)  # captured const
    def func(x, z):
      return lax.cond(z > 0, (1, 2), lambda a: (a[0], jnp.zeros(5)),
                      z, lambda a: (hcb.id_print(a), y))
    self.assertRewrite("""
{ lambda e f ; a b i.
  let c = gt b 0
      d = convert_element_type[ new_dtype=int32
                                old_dtype=bool ] c
      g h j = cond[ branches=( { lambda  ; f_ e a b c g.
                                 let d h = id_tap[ arg_treedef=*
                                                   func=_print
                                                   ] c g
                                 in (d, e, h) }
                               { lambda  ; d g_ a b c h.
                                 let
                                 in (a, d, h) } )
                    linear=(False, False, False, False, False, False) ] d e f 1 2 b i
  in (g, h, j) }""", func, [y, 5])

  def test_while(self):
    ct_body = jnp.ones(5, np.float32)     # captured const for the body
    ct_cond = jnp.ones(5, np.float32) # captured const for the conditional

    def func(x):
      return lax.while_loop(lambda c: c[1] < jnp.sum(c[0] + ct_cond),
                            lambda c: (ct_body, hcb.id_print(c[1]) + 1.),
                            (x, np.float32(1.)))
    # TODO: we should not need to start a receiver here!!! I believe this is
    # because of the partial evaluation of while, which calls impl, which
    # uses JIT.
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertRewrite("""
{ lambda b c ; a f.
  let d e g = while[ body_jaxpr={ lambda  ; c a b f.
                                  let d g = id_tap[ arg_treedef=*
                                                    func=_print
                                                    ] b f
                                      e = add d 1.00
                                  in (c, e, g) }
                     body_nconsts=1
                     cond_jaxpr={ lambda  ; c a b g.
                                  let d = add a c
                                      e = reduce_sum[ axes=(0,) ] d
                                      f = lt b e
                                  in (f,) }
                     cond_nconsts=1 ] b c a 1.00 f
  in (d, e, g) }""", func, [ct_body])

  def test_while_pred_outfeed(self):
    """A while with outfeed in the pred."""
    ct_body = jnp.ones(5)  # captured const for the body
    ct_cond = jnp.ones(2) # captured const for the conditional

    def func(x):
      return lax.while_loop(lambda c: hcb.id_print(ct_cond, result=c[1]) < 5,
                            lambda c: (ct_body, hcb.id_print(c[1]) + 1),
                            (x, 1))

    # TODO: we should not need to start a receiver here!!! I believe this is
    # because of the partial evaluation of while, which calls impl, which
    # uses JIT.
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertRewrite("""
{ lambda b c ; a f.
  let h i = xla_call[ call_jaxpr={ lambda  ; c a b g.
                                   let d e h = id_tap[ arg_treedef=*
                                                       func=_print
                                                       nr_untapped=1
                                                       ] c b g
                                       f = lt e 5
                                   in (f, h) }
                      name=cond_before ] b a 1 f
      y d e g = while[ body_jaxpr={ lambda  ; n o p q r s.
                                    let t u v = xla_call[ call_jaxpr={ lambda  ; c a b f.
                                                                       let d g = id_tap[ arg_treedef=*
                                                                                         func=_print
                                                                                         ] b f
                                                                           e = add d 1
                                                                       in (c, e, g) }
                                                          name=body ] o q r s
                                        w x = xla_call[ call_jaxpr={ lambda  ; c a b g.
                                                                     let d e h = id_tap[ arg_treedef=*
                                                                                         func=_print
                                                                                         nr_untapped=1
                                                                                         ] c b g
                                                                         f = lt e 5
                                                                     in (f, h) }
                                                        name=cond_body ] n t u v
                                    in (w, t, u, x) }
                       body_nconsts=2
                       cond_jaxpr={ lambda  ; j k l m.
                                    let
                                    in (j,) }
                       cond_nconsts=0 ] b c h a 1 i
  in (d, 5, g) }""", func, [ct_body])

  def test_scan(self):
    y = jnp.ones(5)  # captured const
    def func(x):
      return lax.scan(lambda c, a: (hcb.id_print(c), y), (1, 2), x)
    self.assertRewrite("""
{ lambda b ; a f.
  let c d g e = scan[ jaxpr={ lambda  ; f a b g c.
                              let d e h = id_tap[ arg_treedef=PyTreeDef(tuple, [*,*])
                                                  func=_print
                                                   ] a b g
                              in (d, e, h, f) }
                      length=5
                      linear=(False, False, False, False, False)
                      num_carry=3
                      num_consts=1
                      reverse=False ] b 1 2 f a
  in (c, d, e, g) }""", func, [y])


if __name__ == "__main__":
  absltest.main()
