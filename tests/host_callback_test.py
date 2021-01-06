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

from functools import partial
import itertools
import logging
import os
import re
import threading
import time
from typing import Callable, Optional, Sequence
from unittest import SkipTest, skipIf

from absl.testing import absltest
from absl.testing import parameterized

from jax import api
from jax.config import config
from jax import dtypes
from jax.experimental import host_callback as hcb
from jax import lax
from jax import numpy as jnp
from jax import test_util as jtu
from jax.lib import xla_bridge

import numpy as np

config.parse_flags_with_absl()
FLAGS = config.FLAGS

class _TestingOutputStream(object):
  """Use as `output_stream` for tests."""

  def __init__(self):
    self._output = []
    self.test_method_name = None

  def write(self, what: str) -> None:
    print(f"output_stream[{self.test_method_name}]: {what}", end="")
    self._output.append(what)

  @property
  def output(self):
    return "".join(self._output)

  @property
  def output_sorted_by_device(self):
    # Assume that the output is a sequence of strings including metadata
    # and data, with metadata containing `device: xxx`
    by_device = []  # each element is a pair (device, str_list)
    for s in self._output:
      m = re.match(r'.*device: (\S+)', s)
      if m:
        by_device.append((m.group(1), []))
      by_device[-1][1].append(s)

    sorted_by_device = sorted(by_device, key=lambda x: x[0])
    return "\n".join(itertools.chain(*[s[1] for s in sorted_by_device]))

  def __str__(self):
    return "TestingOutputStream"

  def reset(self):
    self._output = []


testing_stream = _TestingOutputStream()


def fun1(a):
  y = hcb.id_print(a * 2., what="a * 2", output_stream=testing_stream)
  y = hcb.id_print(y * 3., what="y * 3", output_stream=testing_stream, result=y)
  return y ** 2  # Some computation to make the gradient interesting


def fun1_equiv(a):  # Numerical equivalent of fun`
  return (a * 2.) ** 2


def maybe_print(do_print: bool, arg, what: str, tap_with_device: Optional[bool] = False):
  """Conditionally print on testing_string"""
  if do_print:
    return hcb.id_print(arg, what=what,
                        output_stream=testing_stream, tap_with_device=tap_with_device)
  else:
    return arg


ignore_jit_of_pmap_warning = partial(
    jtu.ignore_warning, message=".*jit-of-pmap.*")


def assertMultiLineStrippedEqual(tst: jtu.JaxTestCase,
                                 expected: str, what: str):
  """A variant that preprocesses the string to eliminate non-determinism in
  floating point values, and several uninteresting id_tap primitive params.
  """

  # Sometimes we get floating points in the output; we round them
  def repl_floats(match_group):
    matched = match_group.group(0)
    if matched == ".": return matched
    x = np.around(float(matched), decimals=2)
    return f"{x:.2f}"

  what = re.sub(r"\-?\d*\.[\-\def]*", repl_floats, what)
  what = re.sub(r"output_stream=[^\]\n,]*,?", "", what)
  what = re.sub(r"threshold=[^\]\n,]*,?", "", what)
  what = re.sub(r"bwd=[^\]\n]*", "", what)
  what = re.sub(r"out_trees=[^\]\n]*", "", what)
  what = re.sub(r"fwd_jaxpr_thunk=[^\]\n]*", "", what)
  what = re.sub(r"jvp_jaxpr_thunk=[^\]\n]*", "", what)
  # Empty lines
  what = re.sub(r"^\s*\n", "", what, flags=re.MULTILINE)

  def repl_func(match_group):
    matched = match_group.group(0)
    if "function _print_consumer" in matched:
      return "tap_func_=_print"
    else:
      return "..."

  what = re.sub(r"tap_func_=([^\]\n,]*),?", repl_func, what)
  tst.assertMultiLineStrippedEqual(expected, what)


def helper_set_hlo_dump():
  flags_str = os.getenv("XLA_FLAGS", "")
  import shutil
  dump_dir = "/tmp/xla_dump"
  os.environ["XLA_FLAGS"] = f"{flags_str} --xla_dump_to={dump_dir}"
  if os.path.isdir(dump_dir):
    logging.warning(f"Deleting old XLA dump directory {dump_dir}")
    shutil.rmtree(dump_dir)
  logging.warning(f"Setting XLA dump directory {dump_dir}")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()


def helper_print_optimized_hlo(fun, *args):
  backend = api.lib.xla_bridge.get_backend()
  c = api.xla_computation(fun)(*args)
  print(re.sub(r", metadata.*", "",
               backend.compile(c).hlo_modules()[0].to_string()))


prev_xla_flags = None


def setUpModule():
  global prev_xla_flags
  # This will control the CPU devices. On TPU we always have 2 devices
  prev_xla_flags = jtu.set_host_platform_device_count(2)


# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  prev_xla_flags()


def assertMultiDeviceOutputEqual(tst: jtu.JaxTestCase,
                                 expected_2CPUs: str):
  """Check that the multi-device output is equal to the expected.

  The tests run with 2 CPU devices on CPU (due to the flag), also
  on TPU (due to how the TPU tests are set up), but only 1 device on
  GPU. We adjust the expected output here for 1 device.

  Args:
    expected_2CPUs: the expected output for 2 CPUs. If there is only
      one device, this is trimmed to the first device. If the current
      device_under_test is not a CPU, then we change the names
  """
  assert api.device_count() in (1, 2)
  expected = expected_2CPUs
  if api.device_count() == 1:
    start_device_1 = expected.find('device: cpu:1')
    if start_device_1 >= 0:
      expected = expected[0:start_device_1]

  def replace_device_name(m) -> str:
    return str(api.devices()[int(m.group(1))])

  expected = re.sub(r'cpu:(\d+)', replace_device_name, expected)
  what = testing_stream.output_sorted_by_device
  return assertMultiLineStrippedEqual(tst, expected, what)


class HostCallbackIdTapTest(jtu.JaxTestCase):

  def setUp(self):
    testing_stream.reset()
    testing_stream.test_method_name = self._testMethodName
    self.old_flags = os.getenv("XLA_FLAGS", "")
    super().setUp()

  def tearDown(self) -> None:
    if os.getenv("XLA_FLAGS") != self.old_flags:
      os.environ["XLA_FLAGS"] = self.old_flags
      xla_bridge.get_backend.cache_clear()
    hcb.barrier_wait("HostCallbackTest.tearDown")

  def test_eval(self):
    self.assertAllClose((5. * 2.) ** 2, fun1(5.))
    hcb.barrier_wait()
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

    self.assertEqual(3. * (2. + 3.), func2(3.))
    hcb.barrier_wait()

    assertMultiLineStrippedEqual(self, """
        ( 6.00
          9.00 )""", testing_stream.output)
    testing_stream.reset()

  def test_with_dict_results(self):
    def func2(x):
      res = hcb.id_print(dict(a=x * 2., b=x * 3.), output_stream=testing_stream)
      return res["a"] + res["b"]

    self.assertEqual(3. * (2. + 3.), func2(3.))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        { a=6.00
          b=9.00 }""", testing_stream.output)
    testing_stream.reset()

  def test_with_result(self):
    def func2(x):
      x1 = hcb.id_print((x * 2., x * 3.), result=x * 4.,
                        output_stream=testing_stream)
      return x1

    self.assertEqual(3. * 4., func2(3.))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        ( 6.00
          9.00 )""", testing_stream.output)
    testing_stream.reset()

  def test_print_with_device(self):
    def func2(x):
      x1 = hcb.id_print((x * 2., x * 3.), result=x * 4.,
                        output_stream=testing_stream,
                        tap_with_device=True)
      return x1

    self.assertEqual(3. * 4., func2(3.))
    hcb.barrier_wait()
    assertMultiDeviceOutputEqual(self, """
      device: cpu:0
      ( 6.00
        9.00 )""")
    testing_stream.reset()

  def test_eval_tap_exception(self):
    # Simulate a tap error
    def tap_err(*args, **kwargs):
      raise ValueError("Some user message")

    def func(x):
      x1 = hcb.id_print(x + 1, what="x1", output_stream=testing_stream)
      x2 = hcb.id_tap(tap_err, x1 + 1)
      x3 = hcb.id_print(x2 + 1, what="x3", output_stream=testing_stream)
      return x3

    with self.assertRaisesRegex(
       hcb.CallbackException,
       re.compile("There were exceptions during callback processing. Last one was:.*"
                  "ValueError: Some user message", re.DOTALL)):
      func(0)
      hcb.barrier_wait()

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
    self.assertAllClose(6. * 5., jit_fun1(5.))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        what: here
        10.00""", testing_stream.output)
    testing_stream.reset()

  def test_jit_no_invars(self):
    def func():  # jitted function does not take arguments
      return hcb.id_print(42, output_stream=testing_stream)

    self.assertAllClose(42, api.jit(func)())
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
    42""", testing_stream.output)
    testing_stream.reset()

  def test_jit_multiple_invars(self):
    def func(x1, x2):
      return hcb.id_print(x1 + x2, output_stream=testing_stream)

    self.assertAllClose(42, api.jit(func)(40, 2))
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
    42""", testing_stream.output)
    testing_stream.reset()

  def test_jit_constant(self):
    def func(x):
      return hcb.id_print(42, result=x, output_stream=testing_stream)

    self.assertAllClose(5, api.jit(func)(5))
    hcb.barrier_wait()
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
    self.assertEqual(2, api.jit(func)(1))
    hcb.barrier_wait()

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

    self.assertEqual(2, api.jit(func)(1))
    self.assertEqual(11, api.jit(func)(10))
    hcb.barrier_wait()
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

  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_jit_result_unused(self):
    """We can id_print even if we don't use the result."""

    def func(x):
      hcb.id_print(x, where="1", output_stream=testing_stream)
      hcb.id_print(x + 1, where="2", output_stream=testing_stream)
      return x + 1

    self.assertEqual(2, api.jit(func)(1))
    self.assertEqual(11, api.jit(func)(10))
    hcb.barrier_wait()
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

    self.assertEqual(3, api.jit(func)(1))
    hcb.barrier_wait()
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

    for d in devices:
      self.assertEqual(112, api.jit(func, device=d, static_argnums=1)(111, d.id))
    hcb.barrier_wait()
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

    def tap_func(a, _, *, what=""):
      nonlocal tap_count
      tap_count += 1
      self.assertEqual(func(5, what), a)

    transform = api.jit if with_jit else lambda f: f
    for what in ("pair_1_x", "pair_x_2x", "dict"):
      transformed = transform(
          lambda x: hcb.id_tap(
              partial(tap_func, what=what),
              func(x, what),
              result=func(x * 2, what))
      )(5)
      self.assertEqual(func(10, what), transformed)
    hcb.barrier_wait()  # Wait for receivers to be done
    self.assertEqual(3, tap_count)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_concurrent_{concurrent}",
              concurrent=concurrent)
          for concurrent in [True, False]))
  def test_multiple_tap(self, concurrent=False):
    """Call id_tap multiple times, concurrently or in sequence. """
    if concurrent and jtu.device_under_test() == "gpu":
      # TODO(necula): it seems that on GPU if multiple host threads run
      # a jit computation, the multiple computations are interleaved on the
      # GPU. This can result in the outfeed trains being interleaved, which
      # will trigger an error. The solution is to fix on GPU the receiving
      # logic so that we can outfeed the train as one tuple, and receive it
      # one piece as a time. Then the trains should be atomic.
      # See also b/160692602.
      raise SkipTest("concurrent id_tap not supported on GPU")
    received = set()
    count = 5

    def pause_tap(idx, _):
      received.add(int(idx))
      logging.info(f"Starting do_tap {idx}. Sleeping 1sec ...")
      time.sleep(0.3)
      logging.info(f"Finish do_tap {idx}")

    def do_tap(idx):
      api.jit(lambda idx: hcb.id_tap(pause_tap, idx))(idx)

    if concurrent:
      threads = [
          threading.Thread(
              name=f"enqueue_tap_{idx}", target=do_tap, args=(idx,))
          for idx in range(count)
      ]
      [t.start() for t in threads]
      [t.join() for t in threads]
    else:
      for idx in range(count):
        do_tap(idx)

    hcb.barrier_wait()
    self.assertEqual(received, set(range(count)))

  # TODO(necula): see comment for test_multiple_tap.
  @jtu.skip_on_devices("gpu")
  def test_multiple_barriers(self):
    """Call barrier_wait concurrently."""

    def pause_tap(*args, **kwargs):
      logging.info("pause_tap waiting")
      time.sleep(0.3)
      logging.info("pause_tap done")

    def long_run(x):
      return hcb.id_tap(pause_tap, x)

    api.jit(long_run)(5.)

    def try_barrier(idx):
      logging.info(f"Starting test barrier {idx}")
      hcb.barrier_wait()
      logging.info(f"Finished test barrier {idx}")

    threads = [
        threading.Thread(
            name=f"barrier_{idx}", target=try_barrier, args=(idx,))
        for idx in range(3)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_with_jit_{with_jit}",
              with_jit=with_jit)
          for with_jit in [True, False]))
  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_cond(self, with_jit=False):
    """A conditional"""

    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

      x4 = lax.cond(x % 2 == 0,
                    lambda x: hcb.id_print(x, where="cond_t",
                                           output_stream=testing_stream),
                    lambda x: hcb.id_print(-1, where="cond_f", result=x,
                                           output_stream=testing_stream),
                    x2 + 1)
      x5 = hcb.id_print(x4 + 1, where="end", output_stream=testing_stream)
      return x5

    transform = api.jit if with_jit else lambda f: f
    self.assertEqual(4, transform(func)(1))
    hcb.barrier_wait()
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
          dict(testcase_name=f"_with_jit_{with_jit}",
               with_jit=with_jit)
          for with_jit in [True, False]))
  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
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
    self.assertEqual(4, transform(func)(1))
    hcb.barrier_wait()
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

    self.assertEqual(3, api.jit(func)(1))
    hcb.barrier_wait()
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
  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_scan_cond(self, with_jit=True):
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

    if with_jit:
      func = api.jit(func)
    res = func(1)
    self.assertAllClose(jnp.array([1, 2, 3]), res)
    hcb.barrier_wait()
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
          for dtype in jtu.dtypes.all))
  def test_jit_types(self, nr_args=2, dtype=jnp.int16, shape=(2,)):
    if dtype in (jnp.complex64, jnp.complex128, jnp.bool_):
      raise SkipTest(f"id_print jit not implemented for {dtype}.")
    args = [jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)]
    if nr_args > 1:
      args = args * nr_args
    jit_fun1 = api.jit(lambda xs: hcb.id_print(
        xs,
        a_new_test="************",
        testcase_name=f"shape_{shape}_dtype_{dtype}_nr_args={nr_args}"))

    res = jit_fun1(args)
    self.assertAllClose(args, res)

  def test_jit_large(self):
    arg = jnp.arange(10000, dtype=jnp.int32).reshape((10, 10, 5, -1))
    api.jit(hcb.id_print)(arg)

  def test_jit_several_together(self):
    arg = jnp.arange(50, dtype=jnp.int32).reshape((10, 5))
    api.jit(lambda x, y: hcb.id_print((x, y, x * 2.)))(arg, jnp.ones(100, dtype=jnp.int32))

  def test_jit_interleaving(self):
    # Several jit's without data dependencies; they may interfere
    count = 0  # Count tap invocations
    nr_arrays = 5

    def tap_func(arg, _):
      nonlocal count
      assert len(arg) == nr_arrays
      count += 1

    # This is the function that we'll run multiple times
    def func(x, count):
      for i in range(count):
        x = hcb.id_tap(tap_func, [x + i for i in range(nr_arrays)])[-1]
      return x

    x = jnp.array(1, dtype=np.int32)
    res = 0
    for _ in range(10):
      # No dependencies between the jit invocations
      res += api.jit(lambda x: func(x, 10))(x)
    hcb.barrier_wait()
    self.assertEqual(100, count)

  def test_jit_tap_exception(self):
    # Simulate a tap error
    def tap_err(*args, **kwargs):
      raise NotImplementedError

    def func(x):
      x1 = hcb.id_print(x + 1, what="x1", output_stream=testing_stream)
      x2 = hcb.id_tap(tap_err, x1 + 1)
      x3 = hcb.id_print(x2 + 1, what="x3", output_stream=testing_stream)
      return x3

    res = api.jit(func)(0)  # No error yet
    with self.assertRaises(hcb.CallbackException):
      hcb.barrier_wait()

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

  def test_while(self):
    """Executing while, even without JIT uses compiled code"""
    y = jnp.ones(5)  # captured const

    def func(x):
      return lax.while_loop(
          lambda c: c[1] < 5,
          lambda c: (y, hcb.id_print(c[1], output_stream=testing_stream) + 1),
          (x, 1))

    func(y)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        1
        2
        3
        4""", testing_stream.output)
    testing_stream.reset()

  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_jvp(self):
    jvp_fun1 = lambda x, xt: api.jvp(fun1, (x,), (xt,))
    res_primals, res_tangents = jvp_fun1(jnp.float32(5.), jnp.float32(0.1))
    self.assertAllClose(100., res_primals, check_dtypes=False)
    self.assertAllClose(4., res_tangents, check_dtypes=False)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        transforms: ['jvp'] what: a * 2
        ( 10.00
          0.20 )
        transforms: ['jvp'] what: y * 3
        ( 30.00
          0.60 )""", testing_stream.output)
    testing_stream.reset()

  def test_grad_primal_unused(self):
    if not config.omnistaging_enabled:
      raise SkipTest("Test requires omnistaging")

    # The output of id_print is not needed for backwards pass
    def func(x):
      return 2. * hcb.id_print(x * 3., what="x * 3",
                               output_stream=testing_stream)

    grad_func = api.grad(func)
    jaxpr = str(api.make_jaxpr(grad_func)(5.))
    # making the Jaxpr does not print anything
    hcb.barrier_wait()

    assertMultiLineStrippedEqual(self, """
        { lambda  ; a.
          let b = mul a 3.00
              c = id_tap[ arg_treedef_=*
                          tap_func_=_print   what='x * 3')
                          transforms=(  ) ] b
              _ = mul c 2.00
              d = mul 1.00 2.00
              e = id_tap[ arg_treedef_=*
                          tap_func_=_print   what='x * 3')
                          transforms=(('jvp',), ('transpose',)) ] d
              f = mul e 3.00
          in (f,) }""", jaxpr)
    assertMultiLineStrippedEqual(self, "", testing_stream.output)
    testing_stream.reset()

    res_grad = grad_func(jnp.float32(5.))
    hcb.barrier_wait()

    self.assertAllClose(6., res_grad, check_dtypes=False)
    assertMultiLineStrippedEqual(self, """
        what: x * 3
        15.00
        transforms: ['jvp', 'transpose'] what: x * 3
        2.00""", testing_stream.output)
    testing_stream.reset()

  def test_grad_simple(self):
    def func(x):
      y = hcb.id_print(x * 2., what="x * 2", output_stream=testing_stream)
      return x * hcb.id_print(y * 3., what="y * 3",
                              output_stream=testing_stream)

    grad_func = api.grad(func)

    res_grad = grad_func(jnp.float32(5.))
    self.assertAllClose(2. * 5. * 6., res_grad, check_dtypes=False)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        what: x * 2
        10.00
        what: y * 3
        30.00
        transforms: ['jvp', 'transpose'] what: y * 3
        5.00
        transforms: ['jvp', 'transpose'] what: x * 2
        15.00""", testing_stream.output)
    testing_stream.reset()

  def test_grad_grad(self):
    if not config.omnistaging_enabled:
      raise SkipTest("Test requires omnistaging")

    def func(x):
      y = hcb.id_print(x * 2., what="x * 2", output_stream=testing_stream)
      return x * (y * 3.)

    grad_func = api.grad(api.grad(func))
    # making the Jaxpr does not print anything
    _ = api.make_jaxpr(grad_func)(5.)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, "", testing_stream.output)

    res_grad = grad_func(jnp.float32(5.))

    self.assertAllClose(12., res_grad, check_dtypes=False)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        what: x * 2
        10.00
        transforms: ['jvp', 'transpose'] what: x * 2
        15.00
        transforms: ['jvp', 'transpose'] what: x * 2
        3.00
        transforms: ['jvp', 'transpose', 'jvp', 'transpose'] what: x * 2
        2.00""", testing_stream.output)
    testing_stream.reset()

  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_grad_pytree(self):
    def func(x):
      x4, x5 = hcb.id_print((x * 2., x * 3.), what="pair",
                            result=(x * 4., x * 5.),
                            output_stream=testing_stream)
      return x4 + 2. * x5

    x = jnp.float32(5.)
    grad_func = api.grad(func)
    print(api.make_jaxpr(grad_func)(x))
    res_grad = grad_func(x)
    self.assertAllClose(14., res_grad, check_dtypes=False)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        what: pair
        ( 10.00
          15.00 )
        transforms: ['jvp', 'transpose'] what: pair
        ( 0.00
          0.00 )""", testing_stream.output)
    testing_stream.reset()

  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_jvp_float0(self):
    def f(x, yint):
      x, yint = hcb.id_tap(lambda arg, _: arg, (x, yint))
      return x * yint

    res = api.jvp(f, (2., 3), (0.2, np.zeros((), dtypes.float0)))
    self.assertAllClose((6., 0.6), res)

  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_grad_float0(self):
    def func(x, yint):
      x, yint = hcb.id_print((x, yint), what="pair", output_stream=testing_stream)
      return x * yint

    grad_func = api.grad(func)

    res_grad = grad_func(jnp.float32(5.), jnp.int32(2))
    self.assertAllClose(2., res_grad, check_dtypes=False)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        what: pair
        ( 5.00
          2 )
        transforms: ['jvp', 'transpose'] what: pair
        ( 2.00
          False )""", testing_stream.output)
    testing_stream.reset()

  def test_vmap(self):
    vmap_fun1 = api.vmap(fun1)
    vargs = jnp.array([jnp.float32(4.), jnp.float32(5.)])
    vmap_fun1(vargs)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        transforms: [('batch', {'batch_dims': (0,)})] what: a * 2
        [ 8.00 10.00]
        transforms: [('batch', {'batch_dims': (0,)})] what: y * 3
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
    _ = vmap_func(vargs)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
      transforms: [('batch', {'batch_dims': (None, 0)})]
      ( 3.00
        [4.00 5.00] )""", testing_stream.output)
    testing_stream.reset()

  def test_vmap_vmap(self):
    # A 2D tensor with x[i, j] = i + j using 2 vmap
    def sum(x, y):
      return hcb.id_print(x + y, output_stream=testing_stream)

    def sum_rows(xv, y):
      return api.vmap(sum, in_axes=(0, None))(xv, y)

    def sum_all(xv, yv):
      return api.vmap(sum_rows, in_axes=(None, 0))(xv, yv)

    xv = jnp.arange(5, dtype=np.int32)
    yv = jnp.arange(3, dtype=np.int32)
    # assertMultiLineStrippedEqual(self, "", str(api.make_jaxpr(sum_all)(xv, yv)))
    _ = sum_all(xv, yv)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        transforms: [('batch', {'batch_dims': (0,)}), ('batch', {'batch_dims': (0,)})]
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
    self.assertAllClose(np.array([2, 2, 2, 3, 4]), api.jit(api.vmap(func))(inputs),
                        check_dtypes=False)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        transforms: [('batch', {'batch_dims': (0,)})] where: 1
        [0 1 2 3 4]
        transforms: [('batch', {'batch_dims': (0,)})] where: w_b
        [1 2 3 4 5]
        transforms: [('batch', {'batch_dims': (0,)})] where: w_b
        [2 3 3 4 5]
        transforms: [('batch', {'batch_dims': (0,)})] where: 3
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
    res = api.jit(api.vmap(func))(inputs)
    hcb.barrier_wait()
    self.assertAllClose(np.array([2, 2, 2, 3, 4]), res, check_dtypes=False)
    assertMultiLineStrippedEqual(self, """
        transforms: [('batch', {'batch_dims': (0,)})] where: 1
        [0 1 2 3 4]
        transforms: [('batch', {'batch_dims': (0,)})] where: w_c
        [ True  True False False False]
        transforms: [('batch', {'batch_dims': (0,)})] where: w_b
        [1 2 3 4 5]
        transforms: [('batch', {'batch_dims': (0,)})] where: w_c
        [ True False False False False]
        transforms: [('batch', {'batch_dims': (0,)})] where: w_b
        [2 3 3 4 5]
        transforms: [('batch', {'batch_dims': (0,)})] where: w_c
        [False False False False False]
        transforms: [('batch', {'batch_dims': (0,)})] where: 3
        [2 2 2 3 4]""", testing_stream.output)
    testing_stream.reset()

  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_composed(self):
    def power(x, n):
      x, n = hcb.id_print((x, n), output_stream=testing_stream)
      return x * x * n * x

    def f(x, n):
      return x * power(x + 1., n)

    x = 3.
    print("impl = ", f(x, 2.))
    hcb.barrier_wait()
    expected = """
        ( 4.
          2. )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print("jvp = ", api.jvp(lambda x: f(x, 2.), (x,), (1.,)))
    hcb.barrier_wait()
    expected = """
        transforms: ['jvp']
        ( ( 4.
            2. )
          ( 1.
            0. ) )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    print("grad = ", api.grad(f)(x, 2.))
    hcb.barrier_wait()
    expected = """
        ( 4.
          2. )
        transforms: ['jvp', 'transpose']
        ( 288.
          192. )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()

    xv = np.array([3., 4.])
    print("vmap o grad = ", api.vmap(api.grad(f))(xv, np.array([2., 3.])))
    hcb.barrier_wait()
    expected = """
        transforms: [('batch', {'batch_dims': (0, 0)})]
        ( [4. 5.]
          [2. 3.] )
        transforms: ['jvp', 'transpose', ('batch', {'batch_dims': (0, 0)})]
        ( [288. 900.]
          [192. 500.] )"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()


  def test_pmap(self):
    xv = jnp.arange(api.device_count(), dtype=jnp.int32)

    def fun1(x, do_print=False):  # x: i32
      return maybe_print(do_print, x * 2, "x * 2", tap_with_device=True)

    pmap_fun1 = api.pmap(partial(fun1, do_print=True))
    res = pmap_fun1(xv)
    hcb.barrier_wait()
    expected_res = api.pmap(partial(fun1, do_print=False))(xv)
    self.assertAllClose(expected_res, res, check_dtypes=False)
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(self, """
        device: cpu:0 what: x * 2
        0
        device: cpu:1 what: x * 2
        2""")
    testing_stream.reset()

  def test_pmap_vmap(self):
    # A matrix M[ij] = i * 10 + j
    nr_devices = api.device_count()
    shape = (nr_devices, 3)
    matrix = np.fromfunction(lambda i, j: 10. * i + j, shape,
                             dtype=np.int32)

    def fun1(x, do_print=False):  # x: i32
      return maybe_print(do_print, x * 2, "x * 2", tap_with_device=True)

    pmap_vmap_fun1 = api.pmap(api.vmap(partial(fun1, do_print=True)))

    res = pmap_vmap_fun1(matrix)
    hcb.barrier_wait()
    expected_res = api.pmap(api.vmap(partial(fun1, do_print=False)))(matrix)
    self.assertAllClose(expected_res, res, check_dtypes=False)
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(self, """
        device: cpu:0 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
        [0.00 2.00 4.00]
        device: cpu:1 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
        [20.00 22.00 24.00]""")
    testing_stream.reset()

  def test_pmap_pmap_vmap(self):
    # A matrix M[ijk] = i * 100 + j * 10 + k
    nr_devices = api.local_device_count()
    if nr_devices % 2 != 0:
      raise SkipTest("test works only on even number of devices")

    shape = (2, nr_devices // 2, 3)
    matrix = np.fromfunction(lambda i, j, k: 100. * i + 10. * j + k, shape,
                             dtype=np.float32)

    def fun1(x, do_print=False):  # x: f32
      y = maybe_print(do_print, x * 2., "x * 2", tap_with_device=True)
      return y ** 2

    pmap_fun1 = api.pmap(api.pmap(api.vmap(partial(fun1, do_print=True))))
    res = pmap_fun1(matrix)
    hcb.barrier_wait()
    expected_res = api.pmap(api.pmap(api.vmap(partial(fun1, do_print=False))))(matrix)
    self.assertAllClose(expected_res, res, check_dtypes=False)
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(self, """
        device: cpu:0 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
        [0.00 2.00 4.00]
        device: cpu:1 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
        [200.00 202.00 204.00]""")
    testing_stream.reset()

  @ignore_jit_of_pmap_warning()
  def test_pmap_pmap_extra(self):
    """pmap of a pmap surrounded by extra code."""
    # A matrix M[ij] = i * 10 + j
    nr_devices = api.local_device_count()
    if nr_devices != 2:
      raise SkipTest("test works only on 2 devices")
    shape = (2, 1, 3)
    matrix = np.fromfunction(lambda i, j, k: 100. * i + 10. * j + k, shape,
                             dtype=np.float32)

    def fun(xv, do_print=False):
      # This will be printed on all devices, with shape [1, 3]
      xv = maybe_print(do_print, xv + 1., "before", tap_with_device=True)
      res = api.pmap(lambda x: maybe_print(do_print, x * 2., "inside", tap_with_device=True))(xv)
      # This will be printed on all devices, with shape [1, 3]
      return maybe_print(do_print, res + 1., "after", tap_with_device=True)

    res = api.pmap(partial(fun, do_print=True))(matrix)
    self.assertAllClose(fun(matrix, do_print=False), res, check_dtypes=False)
    hcb.barrier_wait()
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(self, """
      device: cpu:0 what: before
      [[1.00 2.00 3.00]]
      device: cpu:0 what: inside
      [2.00 4.00 6.00]
      device: cpu:0 what: after
      [[3.00 5.00 7.00]]
      device: cpu:1 what: before
      [[101.00 102.00 103.00]]
      device: cpu:1 what: inside
      [202.00 204.00 206.00]
      device: cpu:1 what: after
      [[203.00 205.00 207.00]]""")

    testing_stream.reset()

  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_jvp_pmap_vmap(self):
    # A matrix M[ijk] = i * 100 + j * 10 * k
    nr_devices = api.local_device_count()
    shape = (nr_devices, 2, 3)
    matrix = np.fromfunction(lambda i, j, k: 100. * i + 10. * j + k, shape,
                             dtype=np.float32)

    def fun(xv, do_print=False):
      # x: f32[3]
      return api.jvp(api.pmap(api.vmap(lambda x: maybe_print(do_print, x * 2., "x * 2", tap_with_device=True))),
                     (xv,), (.1 * jnp.ones_like(xv),))

    res = fun(matrix, do_print=True)
    hcb.barrier_wait()
    expected_res = fun(matrix, do_print=False)
    self.assertAllClose(expected_res, res, check_dtypes=False)
    # Assertion text is for 2 devices (also works for 1 device)
    # Device 0 will get to execute api.jvp(api.vmap(...)) for matrix[0, :, :]
    assertMultiDeviceOutputEqual(self, """
      device: cpu:0 transforms: [('batch', {'batch_dims': (0,)}), 'jvp'] what: x * 2
      ( [[ 0.00  2.00  4.00]
         [20.00 22.00 24.00]]
        [[0.20 0.20 0.20]
         [0.20 0.20 0.20]] )
      device: cpu:1 transforms: [('batch', {'batch_dims': (0,)}), 'jvp'] what: x * 2
      ( [[200.00 202.00 204.00]
         [220.00 222.00 224.00]]
        [[0.20 0.20 0.20]
         [0.20 0.20 0.20]] )""")
    testing_stream.reset()

  def test_vmap_pmap(self):
    # A matrix M[ijk] = i * 100 + j * 10 * k
    nr_devices = api.local_device_count()
    shape = (2, nr_devices, 3)
    matrix = np.fromfunction(lambda i, j, k: 100. * i + 10. * j + k, shape,
                             dtype=np.float32)

    def fun(xv, do_print=False):
      # x: f32[3]
      return api.vmap(api.pmap(lambda x: maybe_print(do_print, x * 2., "x * 2", tap_with_device=True)))(xv)

    res = fun(matrix, do_print=True)
    hcb.barrier_wait()
    expected_res = fun(matrix, do_print=False)
    self.assertAllClose(expected_res, res, check_dtypes=False)
    # Assertion text is for 2 devices (also works for 1 device)
    # Device 0 will get to execute api.jvp(api.vmap(...)) for matrix[:, 0, :]
    assertMultiDeviceOutputEqual(self, """
      device: cpu:0 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
      [[  0.00   2.00   4.00]
       [200.00 202.00 204.00]]
      device: cpu:1 transforms: [('batch', {'batch_dims': (0,)})] what: x * 2
      [[ 20.00  22.00  24.00]
       [220.00 222.00 224.00]]""")
    testing_stream.reset()

  @ignore_jit_of_pmap_warning()
  def test_jit_pmap_extra(self):
    """jit of a pmap surrounded by extra code."""
    # A matrix M[ij] = i * 10 + j
    nr_devices = api.local_device_count()
    assert nr_devices in (1, 2)
    shape = (nr_devices, 3)
    matrix = np.fromfunction(lambda i, j: 10. * i + j, shape,
                             dtype=np.float32)

    def fun(xv, do_print=False):
      # This will be printed on all devices with shape (nr_devices, 3)
      xv = maybe_print(do_print, xv + 1., "before", tap_with_device=True)
      res = api.pmap(lambda x: maybe_print(do_print, x * 2., "inside", tap_with_device=True))(xv)
      # This will be printed on all devices with shape (nr_devices, 3)
      return maybe_print(do_print, res + 1., "after", tap_with_device=True)

    res = api.jit(partial(fun, do_print=True))(matrix)
    self.assertAllClose(fun(matrix, do_print=False), res, check_dtypes=False)
    hcb.barrier_wait()
    if api.device_count() == 2:
      assertMultiDeviceOutputEqual(self, """
        device: cpu:0 what: before
        [[ 1.00  2.00  3.00]
         [11.00 12.00 13.00]]
        device: cpu:0 what: inside
        [2.00 4.00 6.00]
        device: cpu:0 what: after
        [[ 3.00  5.00  7.00]
         [23.00 25.00 27.00]]
        device: cpu:1 what: before
        [[ 1.00  2.00  3.00]
         [11.00 12.00 13.00]]
        device: cpu:1 what: inside
        [22.00 24.00 26.00]
        device: cpu:1 what: after
        [[ 3.00  5.00  7.00]
         [23.00 25.00 27.00]]""")
    else:
      assert api.device_count() == 1
      assertMultiDeviceOutputEqual(self, """
        device: cpu:0 what: before
        [[1.00 2.00 3.00]]
        device: cpu:0 what: inside
        [2.00 4.00 6.00]
        device: cpu:0 what: after
        [[3.00 5.00 7.00]]""")

    testing_stream.reset()

  def test_cond_pmap(self):
    raise SkipTest("cond of pmap does not work in JAX. Issue #5178.")
    # A matrix M[ij] = i * 10 + j
    nr_devices = api.local_device_count()
    shape = (nr_devices, 3)
    matrix = np.fromfunction(lambda i, j: 10. * i + j, shape,
                             dtype=np.float32)

    def fun1(x, do_print=False):
      return maybe_print(do_print, x * 2., "x * 2")

    def fun2(cond, xv, do_print=False):
      return lax.cond(cond, api.pmap(partial(fun1, do_print=do_print)),
                      lambda xv: xv, xv)

    res = fun2(True, matrix)
    self.assertAllClose(fun2(True, matrix, do_print=False), res, check_dtypes=False)
    hcb.barrier_wait()
    assertMultiLineStrippedEqual(self, """
        TBD""", testing_stream.output)
    testing_stream.reset()

  def test_scan_custom_jvp(self):
    """custom JVP, inside scan.
    This exercises the custom_jvp_call_jaxpr primitives."""

    @api.custom_jvp
    def f(x):
      return x * hcb.id_print(x, output_stream=testing_stream, what="x")

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * hcb.id_print(x_dot, output_stream=testing_stream, what="x_dot")
      return primal_out, tangent_out

    def g(x):
      # Sum f(x_i)
      return lax.scan(lambda carry, inp: (carry + f(inp), 0.),
                      np.full(x.shape[1:], 0.),  # Like x w/o leading dim
                      x)[0]

    arg = np.full((2,), 0.7)
    self.assertAllClose(0.7 * 0.7 * 2, g(arg))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        what: x
        0.7
        what: x
        0.7""", testing_stream.output)
    testing_stream.reset()

    self.assertAllClose(np.array([2.1, 2.1]), api.grad(g)(arg), check_dtypes=False)
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        what: x
        0.7
        what: x
        0.7
        transforms: ['transpose'] what: x_dot
        2.1
        transforms: ['transpose'] what: x_dot
        2.1""", testing_stream.output)

  def test_scan_custom_vjp(self):
    """custom VJP, inside scan.
    This exercises the custom_vjp_call_jaxpr primitives."""

    @api.custom_vjp
    def f(x):
      return x * hcb.id_print(x, output_stream=testing_stream, what="x")

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), 3. * x

    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * hcb.id_print(ct_b, output_stream=testing_stream, what="ct_b"),

    f.defvjp(f_fwd, f_bwd)

    def g(x):
      # Sum f(x_i)
      return lax.scan(lambda carry, inp: (carry + f(inp), 0.),
                      np.full(x.shape[1:], 0.),  # Like x w/o leading dim
                      x)[0]

    arg = np.full((2,), 0.7)

    self.assertAllClose(0.7 * 0.7 * 2, g(arg))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        what: x
        0.7
        what: x
        0.7""", testing_stream.output)
    testing_stream.reset()

    self.assertAllClose(np.array([2.1, 2.1]), api.grad(g)(arg), check_dtypes=False)
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        what: x
        0.7
        what: x
        0.7
        what: ct_b
        1.
        what: ct_b
        1.""", testing_stream.output)

  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_mask(self):

    @partial(api.mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      three_x = hcb.id_print((x, 2 * x), result=3 * x, what="x",
                             output_stream=testing_stream)
      return jnp.sum(three_x)

    x = np.arange(5.)

    self.assertAllClose(9., padded_sum([x], dict(n=3)))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        transforms: [('mask', {'logical_shapes': 5})] what: x
        ( ( [0. 1. 2. 3. 4.]
            [0. 2. 4. 6. 8.] )
          ( ( 3 )
            ( 3 ) ) )""", testing_stream.output)
    testing_stream.reset()

    # With VMAP
    xv = np.arange(10.).reshape((2, 5))  # logical_shape = 5
    self.assertAllClose(
        np.array([9., 78.]),
        # batch_size = 2, n=3 and 4 for the two elements
        api.vmap(padded_sum)([xv],
                             dict(n=np.array([3., 4.]))))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        transforms: [('mask', {'logical_shapes': 5}), ('batch', {'batch_dims': (0, 0, 0, 0)})] what: x
        ( ( [[0. 1. 2. 3. 4.]
             [5. 6. 7. 8. 9.]]
            [[ 0.  2.  4.  6.  8.]
             [10. 12. 14. 16. 18.]] )
          ( ( [3. 4.] )
            ( [3. 4.] ) ) )""", testing_stream.output)
    testing_stream.reset()

    # With JVP
    self.assertAllClose((9., 0.9),
                        api.jvp(lambda arg: padded_sum([arg], dict(n=3)),
                                (x,), (x * 0.1,)))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
        transforms: [('mask', {'logical_shapes': 5}), 'jvp'] what: x
        ( ( ( [0. 1. 2. 3. 4.]
              [0. 2. 4. 6. 8.] )
            ( ( 3 )
              ( 3 ) ) )
          ( ( [0.  0.1 0.2 0.3 0.4]
              [0.  0.2 0.4 0.6 0.8] )
            ( ( False )
              ( False ) ) ) )""", testing_stream.output)
    testing_stream.reset()

    # Now with JIT
    self.assertAllClose(9., api.jit(padded_sum)([x], dict(n=3)))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual("""
      transforms: [('mask', {'logical_shapes': 5})] what: x
      ( ( [0. 1. 2. 3. 4.]
          [0. 2. 4. 6. 8.] )
        ( ( 3 )
          ( 3 ) ) )""", testing_stream.output)
    testing_stream.reset()

  def test_callback_delay(self):
    hcb.callback_extra = lambda dev: time.sleep(1)

    def func(x):
      for i in range(5):
        x = hcb.id_print(x * i, what="x times i")
      return x

    api.jit(func)(np.arange(6, dtype=np.float32).reshape((2, 3)))

  def test_callback_delay_barrier(self):
    hcb.callback_extra = lambda dev: time.sleep(2)

    def func(x):
      for i in range(1, 4):
        x = hcb.id_print(x * i, what="x times i", output_stream=testing_stream)
      return x

    api.jit(func)(np.arange(6, dtype=np.float32).reshape((2, 3)))
    # Wait for the results
    hcb.barrier_wait()
    expected = """
        what: x times i
        [[0. 1. 2.]
        [3. 4. 5.]]
        what: x times i
        [[ 0.  2.  4.]
        [ 6.  8. 10.]]
        what: x times i
        [[ 0.  6. 12.]
        [18. 24. 30.]]"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)
    testing_stream.reset()
    # Call again
    api.jit(func)(np.arange(6, dtype=np.float32).reshape((2, 3)))
    hcb.barrier_wait()
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)

  def test_error_bad_consumer_id(self):
    """Try to use reserved consumer ID 0.

    Check that we get the proper error from the runtime."""
    comp = xla_bridge.make_computation_builder(self._testMethodName)
    token = hcb.xops.CreateToken(comp)
    hcb._initialize_outfeed_receiver()  # Needed if this is the sole test
    with self.assertRaisesRegex(RuntimeError,
                                "Consumer ID cannot be a reserved value: 0"):
      hcb._outfeed_receiver.receiver.add_outfeed(
          comp, token, 0,
          [xla_bridge.constant(comp, np.zeros((2, 3), dtype=np.float32))])

  def test_error_different_shapes(self):
    """Try to register different shapes for the same consumer ID."""
    comp = xla_bridge.make_computation_builder(self._testMethodName)
    token = hcb.xops.CreateToken(comp)
    hcb._initialize_outfeed_receiver()  # Needed if this is the sole test
    hcb._outfeed_receiver.receiver.add_outfeed(
        comp, token, 123,
        [xla_bridge.constant(comp, np.zeros((2, 3), dtype=np.float32))])
    with self.assertRaisesRegex(
        RuntimeError, ".*does not match previous shape element_type.*"):
      hcb._outfeed_receiver.receiver.add_outfeed(
          comp, token, 123,
          [xla_bridge.constant(comp, np.zeros((2, 3), dtype=np.int32))])
    with self.assertRaisesRegex(
        RuntimeError, ".*does not match previous shape element_type.*"):
      hcb._outfeed_receiver.receiver.add_outfeed(
          comp, token, 123,
          [xla_bridge.constant(comp, np.zeros((2,), dtype=np.float32))])

  def test_id_tap_removed_kwargs(self):
    def func(x, transforms, y):
      pass

    with self.assertRaisesRegex(TypeError, r"Support for \*\*kwargs in ``id_tap``"):
      hcb.id_tap(func, 1, y=2)

  def test_odeint(self):
    # TODO: find a smaller repro for bug #4015
    # Seems to be xla_call(scan(xla_call)), all under grad.
    from jax.experimental.ode import odeint

    def f(x, t, k):
      x = hcb.id_print(x)
      return -k * x

    def loss(k=1.0):
      t = jnp.linspace(0, 0.001, num=2)
      xs = odeint(f, 1.0, t, k)
      return xs[-1]

    api.grad(loss)(1.0)  # should not fail

  def test_remat(self):
    def f(i, k):
      x = hcb.id_print(k + i, output_stream=testing_stream)
      return k * x

    def loss(k):
      return lax.fori_loop(0, 2, api.remat(f), k)

    print(loss(3))
    hcb.barrier_wait()
    expected = """
      3
      10"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)

  def test_named_call(self):
    if not config.omnistaging_enabled:
      raise SkipTest("Test requires omnistaging")

    def tap_scalar(init, do_print=False):
      @partial(api.named_call, name="step")
      def step(acc, step_nr):
        acc = acc + step_nr
        maybe_print(do_print, step_nr, what="step_nr")
        return acc, None

      return lax.scan(step, init, np.arange(2))

    self.assertAllClose(tap_scalar(3., do_print=False), tap_scalar(3., do_print=True))
    hcb.barrier_wait()
    expected = """
      what: step_nr
      0
      what: step_nr
      1"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)




class HostCallbackCallTest(jtu.JaxTestCase):
  """Tests for hcb.call"""

  def setUp(self):
    testing_stream.reset()
    testing_stream.test_method_name = self._testMethodName
    super().setUp()

  def tearDown(self) -> None:
    hcb.barrier_wait("HostCallbackCallTest.tearDown")
    super().tearDown()

  def call_log_testing_stream(self, func, arg, *, result_shape, name=""):
    """Call `func` and log inputs and outputs to the testing stream"""

    def call_log(arg):
      def val2str(v):
        return np.array2string(np.array(arg))
      testing_stream.write(f"Call {name}({val2str(arg)})\n")
      res = func(arg)
      testing_stream.write(f"  = {val2str(res)}\n")
      return res
    return hcb.call(call_log, arg, result_shape=result_shape)

  def test_call_simple(self):
    def f_outside(args):
      x, y = args
      return x * y

    def fun(x, use_outside=True):
      return 2 * (hcb.call(f_outside, (x, x + 1),
                           result_shape=x)
                  if use_outside else f_outside((x, x + 1)))

    res_inside = fun(2, use_outside=False)
    self.assertAllClose(res_inside, fun(2, use_outside=True))

  @skipIf(not config.omnistaging_enabled,
          "test works only with omnistaging enabled")
  def test_call_no_result(self):
    def f_outside(arg):
      self.call_log_testing_stream(lambda x: None, arg,
                                   result_shape=None,
                                   name="outside")
      return arg

    self.assertAllClose((3., 4.), f_outside((3., 4.)))
    hcb.barrier_wait()
    expected = """
        Call outside([3. 4.])
          = [3. 4.]"""
    self.assertMultiLineStrippedEqual(expected, testing_stream.output)

  def test_call_cond(self):
    def f_outside(args):
      x, y = args
      return x * y

    def loop(x, use_outside=True):
      def body(i, acc):
        return lax.cond(i % 2 == 1,
                        lambda _: (hcb.call(f_outside, (acc, i),
                                            result_shape=acc)
                                   if use_outside else f_outside((acc, i))),
                        lambda _: acc,
                        None)

      return lax.fori_loop(0, 18, body, x)

    res_inside = loop(1.2, use_outside=False)
    self.assertAllClose(res_inside, loop(1.2, use_outside=True))

  def test_jit_scan_call(self):
    def f_outside(x):
      return x

    def loop(x, use_outside=True):
      def body(carry, i):
        if use_outside:
          return carry + hcb.call(f_outside, i,
                                  result_shape=i), None
        else:
          return carry + i, None

      return lax.scan(body, 0, x)

    x = np.arange(5, dtype=np.int32)

    res_outside = api.jit(partial(loop, use_outside=True))(x)
    self.assertAllClose(res_outside, loop(x, use_outside=False))

  def test_doc_example1(self):
    """Examples from the documentation: simplest, call a function"""

    def host_eig(x):
      return np.linalg.eigvals(x)

    shape = (2, 5, 4, 4)

    m = np.ones(shape, dtype=np.float32)

    def fun(m):
      eig_m = hcb.call(host_eig, m,
                       result_shape=api.ShapeDtypeStruct(m.shape[:-1], m.dtype))
      return eig_m

    expected_res = np.linalg.eigvals(m)
    self.assertAllClose(expected_res, fun(m))

  def test_doc_example_hlo(self):
    """Examples from the documentation: simplest, call a function"""

    def fun(m):
      return jnp.sin(hcb.call(lambda x: np.cos,
                              jnp.cos(m),
                              result_shape=m))

    m = np.ones((2,), np.float32)
    helper_print_optimized_hlo(fun, m)

    def fun(m):
      x = hcb.call(lambda x: None, 2, result_shape=())
      return x

    m = np.ones((2,), np.float32)
    helper_print_optimized_hlo(fun, m)

  def test_call_with_device(self):
    def callback_func(x, device=None):
      testing_stream.write(f"device: {device}\n Called with {x}")
      return x

    def func(x):
      return hcb.call(callback_func, x,
                      result_shape=x,
                      call_with_device=True)

    self.assertEqual(3., func(3.))
    assertMultiDeviceOutputEqual(self, """
        device: cpu:0
         Called with 3.00""")
    testing_stream.reset()

  def test_call_pmap(self):
    # Works for 1 or 2 devices
    def callback_func(x, device=None):
      testing_stream.write(f"device: {device}\n Called with {x}")
      return x * np.array(3, np.int32)

    def fun(x):  # x: i32
      return hcb.call(callback_func, x * 2,
                      result_shape=x,
                      call_with_device=True)

    xv = jnp.arange(api.device_count(), dtype=jnp.int32)
    res = api.pmap(fun)(xv)
    self.assertAllClose(api.pmap(lambda x: x * 6)(xv), res)
    # Assertion text is for 2 devices (also works for 1 device)
    assertMultiDeviceOutputEqual(self, """
        device: cpu:0
         Called with 0
        device: cpu:1
         Called with 2""")
    testing_stream.reset()

  def test_call_vmap(self):
    def f_outside(x): return x

    def fun(x):
      return hcb.call(f_outside, x, result_shape=x)

    with self.assertRaisesRegex(NotImplementedError, "Batching rule for 'outside_call' not implemented"):
      api.vmap(fun)(np.ones((2, 3)))

  def test_error_bad_result_shape(self):
    with self.assertRaisesRegex(
        ValueError,
        "The values must be either numeric scalars, or must have 'shape' and 'dtype' attributes"):
      hcb.call(lambda x: x, 3., result_shape="string")

    with self.assertRaisesRegex(
        ValueError,
        "The values must be either numeric scalars, or must have 'shape' and 'dtype' attributes"):
      hcb.call(lambda x: x, 3., result_shape=lambda x: x)
      hcb.barrier_wait("wait for error")

  def helper_check_callback_errors(self, thunk: Callable,
                                   expected_exc_txt: str):
    """Calls thunk() and checks for expected exceptions.
    """
    if jtu.device_under_test() == "cpu":
      # On CPU the runtime crashes, and the tests are all aborted
      raise SkipTest("TODO: CPU runtime crashes on unexpected infeed")
    elif jtu.device_under_test() == "gpu":
      # On GPU we get a nice error back to Python
      with self.assertRaisesRegex(
          RuntimeError,
          "RET_CHECK failure .* Mismatch between infeed source buffer shape s8.12345."):
        thunk()
    elif jtu.device_under_test() == "tpu":
      # On TPU we get no error!!!
      raise SkipTest("TODO: TPU runtime does not check infeed, and just computes with garbage")

    # Both on GPU and TPU we also get an error during the barrier_wait at the
    # end of the test. Run a barrier_wait now, to consume that error.
    with self.assertRaisesRegex(
        hcb.CallbackException,
        re.compile(
            "There were exceptions during callback processing.*Last one was:.*" +
            expected_exc_txt,
            re.DOTALL)):
      hcb.barrier_wait("Waiting for error")

  def test_error_callback_throws_exception(self):
    def f_outside(x):
      raise ValueError("user exception")
    def fun(x):
      return hcb.call(f_outside, x, result_shape=x)

    self.helper_check_callback_errors(lambda: fun(3.),
                                      "ValueError: user exception")

  def test_error_callback_returns_unexpected_shape(self):
    def fun(x):
      return hcb.call(lambda x: (x, x), x, result_shape=x)

    self.helper_check_callback_errors(lambda: fun(3.),
                                      "Callback func .* should have returned a result with pytree")

  def test_error_then_compute(self):
    # Continue computation on device after error
    def f_outside(x):
      raise ValueError("user exception")
    def fun(x):
      x1 = hcb.call(f_outside, x, result_shape=x)
      return x1
    arg = np.arange(3, dtype=np.int32)
    self.helper_check_callback_errors(lambda: self.assertAllClose(arg, fun(arg)),
                                      "ValueError: user exception")


def call_jax_other_device(jax_outside_fun, arg, *, device):
  """Calls a JAX function on a specific device with simple support for reverse AD.

  Functions whose name starts with "jax_outside" are called on another device,
  by way of hcb.call.
  """

  def run_jax_outside_fun(arg):
    return api.jit(jax_outside_fun)(api.device_put(arg, device))

  @api.custom_vjp
  def make_call(arg):
    return hcb.call(run_jax_outside_fun, arg,
                    result_shape=api.eval_shape(jax_outside_fun, arg))

  # Define the fwd and bwd custom_vjp functions
  def make_call_vjp_fwd(arg):
    # Return the primal argument as the residual. Use `make_call` for the
    # primal computation to enable higher-order AD.
    return make_call(arg), arg  # Return the primal argument as the residual

  def make_call_vjp_bwd(res, ct_res):
    arg = res  # residual is the primal argument

    def jax_outside_vjp_fun(arg_and_ct):
      arg, ct = arg_and_ct
      _, f_vjp = api.vjp(jax_outside_fun, arg)
      ct_in, = f_vjp(ct)
      return ct_in

    return (call_jax_other_device(jax_outside_vjp_fun, (arg, ct_res), device=device),)

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return make_call(arg)


class CallJaxTest(jtu.JaxTestCase):
  """Tests using `call_jax_other_device`."""

  def setUp(self):
    if jtu.device_under_test() != "cpu":
      assert api.devices("cpu")
      self.outside_device = api.devices("cpu")[0]
    else:
      if len(api.devices("cpu")) == 1:
        raise SkipTest("Test needs at least two devices. On CPU use XLA_FLAGS=--xla_force_host_platform_device_count=2")
      self.outside_device = api.devices("cpu")[1]
    super().setUp()

  def test_impl(self):
    def f_jax(x):
      return jnp.sin(x)

    def f_outside(x):
      return call_jax_other_device(f_jax, x, device=self.outside_device)

    self.assertAllClose(f_jax(3.), f_outside(3.))
    self.assertAllClose(f_jax(3.), api.jit(f_outside)(3.))

  def test_impl_pytree(self):
    def f_jax(x):
      # x : dict(a=..., b=...) and output is a list of two elements
      return [jnp.sin(x["a"]), jnp.sin(x["b"])]

    def f_outside(x):
      return call_jax_other_device(f_jax, x, device=self.outside_device)

    x = dict(a=3., b=4.)
    res_jax = f_jax(x)
    # print(f"outside_jaxpr = {api.make_jaxpr(f_outside)(x)}")
    res_outside = f_outside(x)
    self.assertAllClose(res_jax, res_outside)

  def test_grad(self):
    def f_jax(x):
      return 2. * jnp.sin(x)

    def f_outside(x):
      return 2. * call_jax_other_device(jnp.sin, x, device=self.outside_device)

    res_jax = api.grad(f_jax)(3.)
    self.assertAllClose(res_jax, api.grad(f_outside)(3.))

  def test_grad_pytree(self):
    def f_jax(x):
      # x : dict(a=..., b=...) and output is a float
      return 3. * jnp.sin(x["a"]) + jnp.sin(x["b"])

    def f_outside(x):
      return call_jax_other_device(f_jax, x, device=self.outside_device)

    x = dict(a=3., b=4.)
    res_jax = api.grad(f_jax)(x)
    self.assertAllClose(res_jax, api.grad(f_outside)(x))

  def test_grad_of_grad(self):
    def f_jax(x):
      return 2. * x * x * x

    def f_outside(x):
      return 2. * call_jax_other_device(lambda x: x * x * x, x, device=self.outside_device)

    res_jax = api.grad(api.grad(f_jax))(5.)
    res_outside = api.grad(api.grad(f_outside))(5.)
    self.assertAllClose(res_jax, res_outside)


class OutfeedRewriterTest(jtu.JaxTestCase):

  def assertRewrite(self, expected: str, func: Callable, args: Sequence,
                    has_input_token=True, has_output_token=True):
    """Check that the rewrite of func(*args) matches expected."""
    jaxpr = api.make_jaxpr(func)(*args)
    rewritten = hcb._rewrite_closed_jaxpr(jaxpr,  # noqa: F841
                                          has_input_token, has_output_token)
    # Since it is somewhat annoying to update the Jaxpr assertions when we change
    # the Jaxpr printing, we do not check these by default. It is recommended that
    # before making changes to the code generation and Jaxpr rewriting, turn on
    # the checking, update the expected Jaxpr, and then make the changes.
    # assertMultiLineStrippedEqual(self, expected, str(rewritten))
    del rewritten

  def test_no_outfeed(self):
    self.assertRewrite("""
        { lambda  ; a.
          let b = mul a a
              c = add a b
          in (c,) }""", lambda x: x + x * x, [0], has_input_token=False,
                       has_output_token=False)
    self.assertRewrite("""
        { lambda  ; a d e.
          let b = mul a a
              c = add a b
          in (c,) }""", lambda x: x + x * x, [0], has_output_token=False)
    self.assertRewrite("""
        { lambda  ; a d e.
          let b = mul a a
              c = add a b
          in (c, d, e) }""", lambda x: x + x * x, [0])

  def test_simple_outfeed(self):
    self.assertRewrite("""
        { lambda  ; a d e.
          let b = add a a
              c f = id_tap[ arg_treedef_=*
                            has_token_=True
                            tap_func_=_print  ] b d
              g = id e
          in (c, f, g) }""", lambda x: hcb.id_print(x + x), [0])

  def test_simple_outfeed_without_input_token(self):
    self.assertRewrite("""
        { lambda  ; a b.
          let e = create_token a b
              f = create_token a b
              c = add a b
              d g = id_tap[ arg_treedef_=*
                            has_token_=True
                            tap_func_=_print  ] c e
              h = id f
          in (d,) }""", lambda x1, x2: hcb.id_print(x1 + x2), [1, 2],
                       has_input_token=False, has_output_token=False)

  def test_simple_outfeed_without_input_token_nor_invars(self):
    self.assertRewrite("""
        { lambda  ; .
          let b = create_token
              c = create_token
              a d = id_tap[ arg_treedef_=*
                            has_token_=True
                            tap_func_=_print  ] 42 b
              e = id c
          in (a,) }""", lambda: hcb.id_print(42), [],
                       has_input_token=False, has_output_token=False)

  def test_multiple_tap_without_dependencies(self):
    def f(x):
      hcb.id_print(x, what="x")
      hcb.id_print(x + 1, what="x + 1")
      return 2

    self.assertRewrite("""
        { lambda  ; a c d.
          let _ e = id_tap[ arg_treedef_=*
                            has_token_=True
                            tap_func_=_print   what='x') ] a c
              f = id d
              b = add a 1
              _ g = id_tap[ arg_treedef_=*
                            has_token_=True
                            tap_func_=_print   what='x + 1') ] b e
              h = id f
          in (2, g, h) }""", f, [1])

  def test_cond(self):
    y = jnp.ones(5)  # captured const

    def func(x, z):
      return lax.cond(z > 0, (1, 2), lambda a: (a[0], jnp.zeros(5)),
                      z, lambda a: (hcb.id_print(a), y))

    self.assertRewrite("""
        { lambda a ; b c h i.
          let d = gt c 0
              e = convert_element_type[ new_dtype=int32 ] d
              f g j k =
                cond[ branches=( { lambda  ; a b c d f g.
                                   let e h = id_tap[ arg_treedef_=*
                                                     has_token_=True
                                                     tap_func_=_print  ] d f
                                       i = id g
                                   in (e, a, h, i) }
                                 { lambda  ; f_ a b c g h.
                                   let d = broadcast_in_dim[ broadcast_dimensions=(  )
                                                             shape=(5,) ] 0.00
                                   in (a, d, g, h) } )
                      linear=(False, False, False, False, False, False) ] e a 1 2 c h i
          in (f, g, j, k) }""", func, [y, 5])

  def test_while(self):
    ct_body = jnp.ones(5, np.float32)  # captured const for the body
    ct_cond = jnp.ones(5, np.float32)  # captured const for the conditional

    def func(x):
      # x: f32[5]
      # c: (f32[5], f32)
      return lax.while_loop(lambda c: c[1] < jnp.sum(c[0] + ct_cond),
                            lambda c: (ct_body, hcb.id_print(c[1]) + 1.),
                            (x, np.float32(1.)))

    self.assertRewrite("""
        { lambda a b ; c f g.
          let d e h i =
                while[ body_jaxpr={ lambda  ; a b c f g.
                                    let d h = id_tap[ arg_treedef_=*
                                                      has_token_=True
                                                      tap_func_=_print  ] c f
                                        i = id g
                                        e = add d 1.00
                                    in (a, e, h, i) }
                       body_nconsts=1
                       cond_jaxpr={ lambda  ; a b c g h.
                                    let d = add b a
                                        e = reduce_sum[ axes=(0,) ] d
                                        f = lt c e
                                    in (f,) }
                       cond_nconsts=1 ] a b c 1.00 f g
          in (d, e, h, i) }""", func, [ct_body])

  def test_while_pred_outfeed(self):
    """A while with outfeed in the pred."""
    ct_body = jnp.ones(5)  # captured const for the body
    ct_cond = jnp.ones(2)  # captured const for the conditional

    def func(x):
      return lax.while_loop(lambda c: hcb.id_print(ct_cond, result=c[1]) < 5,
                            lambda c: (ct_body, hcb.id_print(c[1]) + 1),
                            (x, 1))

    self.assertRewrite("""
        { lambda a b ; c f g.
          let j k l = xla_call[ call_jaxpr={ lambda  ; a b c g h.
                                             let d i = id_tap[ arg_treedef_=*
                                                               has_token_=True
                                                               tap_func_=_print  ] a g
                                                 j = id h
                                                 e = id_tap_dep c d
                                                 f = lt e 5
                                             in (f, i, j) }
                                donated_invars=(False, False, False, False, False)
                                name=cond_before ] a c 1 f g
              bf d e h i =
                while[ body_jaxpr={ lambda  ; r s t u v w x.
                                    let y z ba bb =
                                          xla_call[ call_jaxpr={ lambda  ; a b c f g.
                                                                 let d h = id_tap[ arg_treedef_=*
                                                                                   has_token_=True
                                                                                   tap_func_=_print  ] c f
                                                                     i = id g
                                                                     e = add d 1
                                                                 in (a, e, h, i) }
                                                    donated_invars=(False, False, False, False, False)
                                                    name=body ] s u v w x
                                        bc bd be =
                                          xla_call[ call_jaxpr={ lambda  ; a b c g h.
                                                                 let d i = id_tap[ arg_treedef_=*
                                                                                   has_token_=True
                                                                                   tap_func_=_print  ] a g
                                                                     j = id h
                                                                     e = id_tap_dep c d
                                                                     f = lt e 5
                                                                 in (f, i, j) }
                                                    donated_invars=(False, False, False, False, False)
                                                    name=cond_body ] r y z ba bb
                                    in (bc, y, z, bd, be) }
                       body_nconsts=2
                       cond_jaxpr={ lambda  ; m n o p q.
                                    let
                                    in (m,) }
                       cond_nconsts=0 ] a b j c 1 k l
          in (d, e, h, i) }""", func, [ct_body])

  def test_scan(self):
    y = jnp.ones(5)  # captured const

    def func(x):
      return lax.scan(lambda c, a: (hcb.id_print(c), y), (1, 2), x)

    self.assertRewrite("""
        { lambda a ; b f g.
          let c d h i e =
                scan[ jaxpr={ lambda  ; a b c g h d.
                              let e f i = id_tap[ arg_treedef_=PyTreeDef(tuple, [*,*])
                                                  has_token_=True
                                                  tap_func_=_print  ] b c g
                                  j = id h
                              in (e, f, i, j, a) }
                      length=5
                      linear=(False, False, False, False, False, False)
                      num_carry=4
                      num_consts=1
                      reverse=False
                      unroll=1 ] a 1 2 f g b
          in (c, d, e, h, i) }""", func, [y])

  def test_scan_custom_jvp(self):
    """custom JVP, inside scan.
    This exercises the custom_jvp_call_jaxpr primitives."""

    @api.custom_jvp
    def f(x):
      return x * hcb.id_print(x)

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * hcb.id_print(x_dot)
      return primal_out, tangent_out

    def g(x):
      # Sum f(x_i)
      return lax.scan(lambda carry, inp: (carry + f(inp), 0.),
                      np.full(x.shape[1:], 0.),  # Like x w/o leading dim
                      x)[0]

    arg = np.full((5,), 0.7)
    self.assertRewrite("""
        { lambda  ; a c d.
          let b e f _ =
                scan[ jaxpr={ lambda  ; a e f b.
                              let c g h = custom_jvp_call_jaxpr[ fun_jaxpr={ lambda  ; a d e.
                                                                             let b f = id_tap[ arg_treedef_=*
                                                                                               has_token_=True
                                                                                               tap_func_=_print  ] a d
                                                                                 g = id e
                                                                                 c = mul a b
                                                                             in (c, f, g) }
                                                                 num_consts=0 ] b e f
                                  d = add a c
                              in (d, g, h, 0.00) }
                      length=5
                      linear=(False, False, False, False)
                      num_carry=3
                      num_consts=0
                      reverse=False
                      unroll=1 ] 0.00 c d a
          in (b, e, f) }""", g, [arg])
    self.assertRewrite("""
        { lambda  ; a d e.
          let _ _ f g _ b =
                scan[ jaxpr={ lambda  ; a b h i c d.
                              let e j k = custom_jvp_call_jaxpr[ fun_jaxpr={ lambda  ; a d e.
                                                                             let b f = id_tap[ arg_treedef_=*
                                                                                               has_token_=True
                                                                                               tap_func_=_print  ] a d
                                                                                 g = id e
                                                                                 c = mul a b
                                                                             in (c, f, g) }
                                                                 num_consts=0 ] c h i
                                  f = add a e
                                  g = mul c 3.00
                              in (f, *, j, k, 0.00, g) }
                      length=5
                      linear=(False, True, False, False, False, True)
                      num_carry=4
                      num_consts=0
                      reverse=False
                      unroll=1 ] 0.00 * d e a *
              _ _ h i _ c =
                scan[ jaxpr={ lambda  ; a b g h c d.
                              let e = mul b d
                                  f i = id_tap[ arg_treedef_=*
                                                has_token_=True
                                                tap_func_=_print
                                                transforms=(('transpose',),) ] e g
                                  j = id h
                              in (*, b, i, j, *, f) }
                      length=5
                      linear=(True, True, False, False, True, False)
                      num_carry=4
                      num_consts=0
                      reverse=True
                      unroll=1 ] * 1.00 f g * b
          in (c, h, i) }""", api.grad(g), [arg])

  def test_scan_custom_vjp(self):
    """custom VJP, inside scan.
    This exercises the custom_vjp_call_jaxpr primitives."""

    @api.custom_vjp
    def f(x):
      return x * hcb.id_print(x)

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), 3. * x

    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * hcb.id_print(ct_b),

    f.defvjp(f_fwd, f_bwd)

    def g(x):
      # Sum f(x_i)
      return lax.scan(lambda carry, inp: (carry + f(inp), 0.),
                      np.full(x.shape[1:], 0.),  # Like x w/o leading dim
                      x)[0]

    arg = np.full((2,), 0.7)
    self.assertRewrite("""
        { lambda  ; a c d.
          let b e f _ =
                scan[ jaxpr={ lambda  ; a e f b.
                              let c g h = custom_vjp_call_jaxpr[
                                                                 fun_jaxpr={ lambda  ; a d e.
                                                                             let b f = id_tap[ arg_treedef_=*
                                                                                               has_token_=True
                                                                                               tap_func_=_print  ] a d
                                                                                 g = id e
                                                                                 c = mul a b
                                                                             in (c, f, g) }
                                                                 num_consts=0
                                                                 ] b e f
                                  d = add a c
                              in (d, g, h, 0.00) }
                      length=2
                      linear=(False, False, False, False)
                      num_carry=3
                      num_consts=0
                      reverse=False
                      unroll=1 ] 0.00 c d a
          in (b, e, f) }""", g, [arg])
    self.assertRewrite("""
        { lambda  ; a d e.
          let _ _ f g _ b =
                scan[ jaxpr={ lambda  ; a b h i c d.
                              let e j k = custom_vjp_call_jaxpr[
                                                                 fun_jaxpr={ lambda  ; a d e.
                                                                             let b f = id_tap[ arg_treedef_=*
                                                                                               has_token_=True
                                                                                               tap_func_=_print  ] a d
                                                                                 g = id e
                                                                                 c = mul a b
                                                                             in (c, f, g) }
                                                                 num_consts=0
                                                                 ] c h i
                                  f = add a e
                                  g = mul c 3.00
                              in (f, *, j, k, 0.00, g) }
                      length=2
                      linear=(False, True, False, False, False, True)
                      num_carry=4
                      num_consts=0
                      reverse=False
                      unroll=1 ] 0.00 * d e a *
              _ _ h i _ c =
                scan[ jaxpr={ lambda  ; a b g h c d.
                              let e i = id_tap[ arg_treedef_=*
                                                has_token_=True
                                                tap_func_=_print  ] b g
                                  j = id h
                                  f = mul d e
                              in (*, b, i, j, *, f) }
                      length=2
                      linear=(True, True, False, False, True, False)
                      num_carry=4
                      num_consts=0
                      reverse=True
                      unroll=1 ] * 1.00 f g * b
          in (c, h, i) }""", api.grad(g), [arg])

  def test_remat_loop(self):
    def f(k, x):
      x = hcb.id_print(k + x)
      return -k * x

    def loss(k):
      return lax.fori_loop(0, 1, api.remat(f), k)

    self.assertRewrite("""
        { lambda  ; a c d.
          let _ _ b e f =
                while[ body_jaxpr={ lambda  ; a b c f g.
                                    let d = add a 1
                                        e h i = remat_call[ call_jaxpr={ lambda  ; a b g h.
                                                                         let c = add a b
                                                                             d i = id_tap[ arg_treedef_=*
                                                                                           has_token_=True
                                                                                           tap_func_=_print  ] c g
                                                                             j = id h
                                                                             e = neg a
                                                                             f = mul e d
                                                                         in (f, i, j) }
                                                            concrete=False
                                                            name=f ] a c f g
                                    in (d, b, e, h, i) }
                       body_nconsts=0
                       cond_jaxpr={ lambda  ; a b c e f.
                                    let d = lt a b
                                    in (d,) }
                       cond_nconsts=0 ] 0 1 a c d
          in (b, e, f) }""", loss, [2])

  def test_named_call(self):
    def tap_scalar(init, do_print=False):
      @partial(api.named_call, name="step")
      def step(acc, step_nr):
        acc = acc + step_nr
        maybe_print(do_print, step_nr, what="step_nr")
        return acc, None

      return lax.scan(step, init, np.arange(2, dtype=np.int32))

    self.assertRewrite("""
        { lambda a ; b d e.
          let c = scan[ jaxpr={ lambda  ; a b.
                                let c = named_call[ call_jaxpr={ lambda  ; a b.
                                                                 let c = add a b
                                                                 in (c,) }
                                                    name=step ] a b
                                in (c,) }
                        length=2
                        linear=(False, False)
                        num_carry=1
                        num_consts=0
                        reverse=False
                        unroll=1 ] b a
          in (c, d, e) }""", tap_scalar, [np.int32(3)])

  def test_pmap(self):
    def f(xv):
      api.pmap(lambda x: jnp.sin(hcb.id_print(x, tap_with_device=True)),
               axis_name="i")(xv)

    self.assertRewrite("""
        { lambda  ; a b c.
          let _ d e = xla_pmap[ axis_name=i
                                axis_size=1
                                backend=None
                                call_jaxpr={ lambda  ; a e f.
                                             let b g = id_tap[ arg_treedef_=*
                                                               has_token_=True
                                                               tap_func_=_print
                                                               tap_with_device_=True ] a e
                                                 h = id f
                                                 c = convert_element_type[ new_dtype=float32 ] b
                                                 d = sin c
                                             in (d, g, h) }
                                devices=None
                                donated_invars=(False, False, False)
                                global_arg_shapes=(None,)
                                global_axis_size=None
                                in_axes=(0, 0, 0)
                                name=<lambda>
                                out_axes=(0, 0, 0) ] a b c
          in (d, e) }""", f, [np.array([2])])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
