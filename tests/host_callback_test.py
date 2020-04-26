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

import logging
import numpy as onp
import os
import re
import threading
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

from jax import api
from jax import lax
from jax import numpy as np
from jax import test_util as jtu
from jax.config import config
from jax.experimental import host_callback as hcb
from jax.interpreters import xla
from jax.interpreters import partial_eval as pe
from jax.lib import xla_bridge


config.parse_flags_with_absl()
FLAGS = config.FLAGS

def skip_if_jit_not_enabled():
  if os.getenv("JAX_ENABLE_JIT_PRINT", "false") == "false":
    raise SkipTest("print jit not enabled yet; use JAX_ENABLE_JIT_PRINT env.")

class _TestingOutputStream(object):
  """Use as `output_stream` for tests."""

  def __init__(self):
    self._output = []

  def write(self, what: str) -> None:
    # Sometimes we get floating points in the output; we round them
    def repl(match_group):
      matched = match_group.group(0)
      if matched == ".": return matched
      # TODO: why can't we use here np.around?
      x = onp.around(float(matched), decimals=2)
      return f"{x:.2f}"

    what = re.sub(r"\-?\d*\.[\-\def]*", repl, what)
    print(f"output_stream: {what}")
    self._output.append(what)

  @property
  def output(self):
    return "\n".join(self._output)

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


class HostCallbackTest(jtu.JaxTestCase):

  def setUp(self):
    testing_stream.reset()

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


  def test_with_tuple_result(self):

    def func2(x):
      x1, y1 = hcb.id_print(x * 2., x * 3., output_stream=testing_stream)
      return x1 + y1

    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = mul a 2.0
      c = mul a 3.0
      d e = id_print[ output_stream=TestingOutputStream ] b c
      f = add d e
  in (f,) }""", str(api.make_jaxpr(func2)(3.)))
    self.assertEqual(3. * (2. + 3.), func2(3.))
    self.assertMultiLineStrippedEqual("""
(6.00, 9.00)  {}""", testing_stream.output)
    testing_stream.reset()

  def test_eval(self):
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = mul a 2.0
      c = id_print[ output_stream=TestingOutputStream
                    what=a * 2 ] b
      d = mul c 3.0
      e f = id_print[ nr_results=1
                      output_stream=TestingOutputStream
                      what=y * 3 ] d c
      g = pow f 2.0
  in (g,) }""", str(api.make_jaxpr(fun1)(5.)))
    self.assertEqual("", testing_stream.output)

    self.assertEqual((5. * 2.)**2, fun1(5.))
    self.assertMultiLineStrippedEqual(
        """
(10.00,)  {'what': 'a * 2'}
(30.00, 10.00)  {'what': 'y * 3', 'nr_results': 1}""", testing_stream.output)
    testing_stream.reset()

  def test_jit_simple(self):
    jit_fun1 = api.jit(lambda x: 3. * hcb.id_print(
        2. * x, what="here"))
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = xla_call[ backend=None
                    call_jaxpr={ lambda  ; a.
                                 let b = mul a 2.0
                                     c = id_print[ what=here ] b
                                     d = mul c 3.0
                                 in (d,) }
                    device=None
                    name=<lambda> ] a
  in (b,) }""", str(api.make_jaxpr(jit_fun1)(5.)))
    logging.warning("%s: %s",
                 self._testMethodName, api.xla_computation(jit_fun1)(5.).GetHloText())
    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      res = jit_fun1(5.)

    self.assertAllClose(6. * 5., res, check_dtypes=True)
    self.assertMultiLineStrippedEqual(
      """
10.00""", testing_stream.output)
    testing_stream.reset()

  def test_jit_sequence1(self):
    def func(x):
      x1 = hcb.id_print(x, where="1")
      return hcb.id_print(x1 + 1, where="2")

    logging.info("%s: %s", self._testMethodName,
          api.make_jaxpr(func)(1))
    logging.info("%s: %s", self._testMethodName,
          api.xla_computation(func)(1).GetHloText())

    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      self.assertEqual(2, api.jit(func)(1))
    self.assertMultiLineStrippedEqual(
            """
1
2""", testing_stream.output)
    testing_stream.reset()

  def test_jit2(self):
    """A sequence of JIT."""
    def func(x):
      x1 = hcb.id_print(x, where="1")
      x2 = hcb.id_print(x1 + 1, where="2")
      return x2

    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      self.assertEqual(2, api.jit(func)(1))
      self.assertEqual(11, api.jit(func)(10))

    self.assertMultiLineStrippedEqual(
      """
1
2
10
11""", testing_stream.output)
    testing_stream.reset()

  def test_jit_nested(self):
    def func(x):
      x1 = hcb.id_print(x, where="1")
      def func_nested(x):
        x2 = hcb.id_print(x + 1, where="nested")
        return x2
      x3 = api.jit(func_nested)(x1)
      return hcb.id_print(x3 + 1, where="2")

    logging.warning("%s: %s", self._testMethodName,
                 api.make_jaxpr(func)(1))
    logging.warning("%s: %s", self._testMethodName,
                 api.xla_computation(func)(1).GetHloText())
    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      self.assertEqual(3, api.jit(func)(1))
    self.assertMultiLineStrippedEqual(
      """
1
2
3""", testing_stream.output)
    testing_stream.reset()

  def test_jit_devices(self):
    """Running on multiple devices."""
    devices = api.local_devices()
    def func(x, device_id):
      x1 = hcb.id_print(x, dev=str(device_id))
      x2 = hcb.id_print(x1 + 1, dev=str(device_id))
      return x2

    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      for d in devices:
        self.assertEqual(2, api.jit(func, device=d, static_argnums=1)(1, d.id))

    self.assertEqual(len(devices), len(re.findall(r"1", testing_stream.output)))
    self.assertEqual(len(devices), len(re.findall(r"2", testing_stream.output)))
    testing_stream.reset()

  def test_jit_cond1(self):
    """A conditional"""
    def func(x):
      x1 = hcb.id_print(x, where="1")
      x2 = hcb.id_print(x1 + 1, where="2")

      x4 = lax.cond(x % 2 == 0,
                    x2 + 1, lambda x: hcb.id_print(x, where="cond_t"),
                    x2 + 1, lambda x: hcb.id_print(-1, where="cond_f", result=x))
      x5 = hcb.id_print(x4 + 1, where="w.2")
      return x5

    logging.warning("%s: %s", self._testMethodName, api.make_jaxpr(func)(1))
    logging.warning("%s: %s", self._testMethodName,
                    api.xla_computation(func)(1).GetHloText())

    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      self.assertEqual(4, api.jit(func)(1))
    self.assertMultiLineStrippedEqual("""
1
2
-1
4""", testing_stream.output)
    testing_stream.reset()


  def test_jit_while_cond(self):
    def func(x):
      x1 = hcb.id_print(x, where="1")
      x2 = hcb.id_print(x1 + 1, where="2")
      def body(x):
        x3 = hcb.id_print(x, where="w.1")
        x4 = lax.cond(x % 2 == 0,
                      x3 + 1, lambda x: hcb.id_print(x, where="w.t"),
                      x3 + 1, lambda x: hcb.id_print(-1, where="w.f", result=x))
        return hcb.id_print(x4 + 1, where="w.2")
      x10 = lax.while_loop(lambda x: x < 10, body, x2)
      res = hcb.id_print(x10, where="10")
      return res
    logging.warning("%s: %s", self._testMethodName, api.make_jaxpr(func)(1))
    logging.warning("%s: %s", self._testMethodName,
          api.xla_computation(func)(1).GetHloText())

    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      self.assertEqual(10, api.jit(func)(1))
    self.assertMultiLineStrippedEqual(
        """
1
2
2
3
4
4
5
6
6
7
8
8
9
10
10""", testing_stream.output)
    testing_stream.reset()

  def test_jit_while_pred_printing(self):
    raise SkipTest("Not yet implemented")
    """While with printing in the conditional."""
    def func(x):
      x1 = hcb.id_print(x, where="1")

      def body(x):
        x3 = hcb.id_print(x, where="w.1")
        return hcb.id_print(x3 + 1, where="w.2")

      x10 = lax.while_loop(lambda x: hcb.id_print(x < 10, where="w.p"),
                           body, x1)
      res = hcb.id_print(x10, where="10")
      return res

    logging.warning("%s: %s", self._testMethodName, api.make_jaxpr(func)(1))
    logging.warning("%s: %s", self._testMethodName,
                    api.xla_computation(func)(1).GetHloText())

    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      self.assertEqual(10, api.jit(func)(1))
    self.assertMultiLineStrippedEqual(
      """
""", testing_stream.output)
    testing_stream.reset()


  def test_jit_scan_cond(self):
    def func(x):
      x1 = hcb.id_print(x, where="1")
      x2 = hcb.id_print(x1 + 1, where="2")

      def body(c, x):
        x3 = hcb.id_print(x, where="s.1")
        x4 = lax.cond(x % 2 == 0,
                      x3 + 1, lambda x: hcb.id_print(x, where="s.t"),
                      x3 + 1, lambda x: hcb.id_print(-1, where="s.f", result=x))
        return (c, hcb.id_print(x4 + 1, where="w.2"))

      _, x10 = lax.scan(body, x2, np.arange(3))
      res = hcb.id_print(x10, where="10")
      return res

    logging.warning("%s: %s", self._testMethodName, api.make_jaxpr(func)(1))
    logging.warning("%s: %s", self._testMethodName,
                    api.xla_computation(func)(1).GetHloText())

    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      res = api.jit(func)(1)
      self.assertAllClose(np.array([2, 3, 4]), res, check_dtypes=True)
    self.assertMultiLineStrippedEqual(
      """
1
2
0
1
2
1
-1
3
2
3
4
[2 3 4]""", testing_stream.output)
    testing_stream.reset()

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_shape_{shape}_dtype_{dtype}_nr_args={nr_args}",
              shape=shape,
              dtype=dtype,
              nr_args=nr_args) for nr_args in [1, 2]
          for shape in [(), (2,), (2, 3), (2, 3, 4)]
          for dtype in jtu.supported_dtypes()))
  def test_jit_types(self, nr_args=2, dtype=np.int16, shape=(2,)):
    if dtype in (np.complex64, np.complex128, np.bool_):
      raise SkipTest(f"id_print jit not implemented for {dtype}.")
    if jtu.device_under_test() == "tpu":
      if dtype in (np.int16,):
        raise SkipTest(f"transfering {dtype} not supported on TPU")
    self.helper_set_hlo_dump()
    args = [np.arange(np.prod(shape), dtype=dtype).reshape(shape)]
    if nr_args > 1:
      args = args * nr_args
    jit_fun1 = api.jit(lambda xs: hcb.id_print(
        *xs,
        a_new_test="************",
        testcase_name=f"shape_{shape}_dtype_{dtype}_nr_args={nr_args}"))
    with hcb.print_receiver(receiver_name=self._testMethodName):
      res = jit_fun1(args)
    # self.assertAllClose(args, res, check_dtypes=True)

  def test_jit_large(self):
    arg = np.arange(10000, dtype=np.int32).reshape((10, 10, 5, -1))
    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      api.jit(hcb.id_print)(arg)

  def test_jvp(self):
    jvp_fun1 = lambda x, xt: api.jvp(fun1, (x,), (xt,))
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a b.
  let c = mul a 2.0
      d = id_print[ output_stream=TestingOutputStream
                    what=a * 2 ] c
      e = mul d 3.0
      f g = id_print[ nr_results=1
                      output_stream=TestingOutputStream
                      what=y * 3 ] e d
      h = pow g 2.0
      i = mul b 2.0
      j = id_print[ output_stream=TestingOutputStream
                    transforms=('jvp',)
                    what=a * 2 ] i
      k = mul j 3.0
      l m = id_print[ nr_results=1
                      output_stream=TestingOutputStream
                      transforms=('jvp',)
                      what=y * 3 ] k j
      n = pow g 1.0
      o = mul 2.0 n
      p = mul m o
  in (h, p) }""",
        str(api.make_jaxpr(jvp_fun1)(np.float32(5.), np.float32(0.1))))

    res_primals, res_tangents = jvp_fun1(np.float32(5.), np.float32(0.1))
    self.assertMultiLineStrippedEqual(
        """
(DeviceArray(10.00, dtype=float32),)  {'what': 'a * 2'}
(DeviceArray(0.20, dtype=float32),)  {'what': 'a * 2', 'transforms': ('jvp',)}
(DeviceArray(30.00, dtype=float32), DeviceArray(10.00, dtype=float32))  {'what': 'y * 3', 'nr_results': 1}
(DeviceArray(0.60, dtype=float32), DeviceArray(0.20, dtype=float32))  {'what': 'y * 3', 'nr_results': 1, 'transforms': ('jvp',)}
  """, testing_stream.output)
    testing_stream.reset()



  def test_jit_nested_cond_no_print(self):
    """A nested conditional, without any prints"""
    # raise SkipTest("skip this")
    @api.jit
    def cfun(x):
      return lax.cond(
          lax.lt(x, 2),
          x, lambda x: x,
          x, lambda x: lax.cond(x < 5,
                                3, lambda x: x,
                                4, lambda y: y))
    print(self._testMethodName, api.xla_computation(cfun)(1).GetHloText())
    cfun(1)

  def test_grad(self):
    grad_fun1 = api.grad(fun1)
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = mul a 2.0
      c = id_print[ output_stream=TestingOutputStream
                    what=a * 2 ] b
      d = mul c 3.0
      e f = id_print[ nr_results=1
                      output_stream=TestingOutputStream
                      what=y * 3 ] d c
      g = pow f 1.0
      h = mul 2.0 g
      i = mul 1.0 h
      j k = id_print[ nr_results=1
                      output_stream=TestingOutputStream
                      transforms=('jvp', 'transpose')
                      what=y * 3 ] 0.0 i
      l = mul j 3.0
      m = add_any k l
      n = id_print[ output_stream=TestingOutputStream
                    transforms=('jvp', 'transpose')
                    what=a * 2 ] m
      o = mul n 2.0
  in (o,) }""", str(api.make_jaxpr(grad_fun1)(5.)))


    res_grad = grad_fun1(np.float32(5.))
    self.assertMultiLineStrippedEqual(
        """
(DeviceArray(10.00, dtype=float32),)  {'what': 'a * 2'}
(DeviceArray(30.00, dtype=float32), DeviceArray(10.00, dtype=float32))  {'what': 'y * 3', 'nr_results': 1}
(array(0.00, dtype=float32), DeviceArray(20.00, dtype=float32))  {'what': 'y * 3', 'nr_results': 1, 'transforms': ('jvp', 'transpose')}
(DeviceArray(20.00, dtype=float32),)  {'what': 'a * 2', 'transforms': ('jvp', 'transpose')}
   """, testing_stream.output)
    testing_stream.reset()

  def test_vmap(self):
    vmap_fun1 = api.vmap(fun1)
    vargs = np.array([np.float32(4.), np.float32(5.)])
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = mul a 2.0
      c = id_print[ output_stream=TestingOutputStream
                    transforms=('batch',)
                    what=a * 2 ] b
      d = mul c 3.0
      e f = id_print[ nr_results=1
                      output_stream=TestingOutputStream
                      transforms=('batch',)
                      what=y * 3 ] d c
      g = pow f 2.0
  in (g,) }""", str(api.make_jaxpr(vmap_fun1)(vargs)))

    res_vmap = vmap_fun1(vargs)
    self.assertMultiLineStrippedEqual(
        """
(DeviceArray([ 8.00, 10.00], dtype=float32),)  {'what': 'a * 2', 'transforms': ('batch',)}
(DeviceArray([24.00, 30.00], dtype=float32), DeviceArray([ 8.00, 10.00], dtype=float32))  {'what': 'y * 3', 'nr_results': 1, 'transforms': ('batch',)}
     """, testing_stream.output)
    testing_stream.reset()

  def test_pmap(self):
    vargs = 2. + np.arange(api.local_device_count(), dtype=np.float32)

    pmap_fun1 = api.pmap(fun1, axis_name="i")
    with hcb.print_receiver(output_stream=testing_stream,
                            receiver_name=self._testMethodName):
      res = pmap_fun1(vargs)
    expected_res = np.stack([fun1_equiv(2. + a) for a in range(api.local_device_count())])
    self.assertAllClose(expected_res, res, check_dtypes=False)


if __name__ == "__main__":
  absltest.main()
