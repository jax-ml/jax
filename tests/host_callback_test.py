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
import numpy as onp
import os
import re
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
    print(f"output_stream: {what}", end="")
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

  def assertMultiLineStrippedEqual(self, expected, what):
    """A variant that preprocesses the string to eliminate non-determinism."""
    # Sometimes we get floating points in the output; we round them
    def repl_floats(match_group):
      matched = match_group.group(0)
      if matched == ".": return matched
      # TODO: why can't we use here np.around?
      x = onp.around(float(matched), decimals=2)
      return f"{x:.2f}"
    what = re.sub(r"\-?\d*\.[\-\def]*", repl_floats, what)
    # We rewrite consumer_id because it changes
    what = re.sub(r"consumer_id=(\d+)", "consumer_id=...", what)
    what = re.sub(r"output_stream=[^\]\n]*", "output_stream=...", what)
    def repl_func(match_group):
      matched = match_group.group(0)
      if "function _print_consumer" in matched:
        return "func=_print"
      else:
        return "..."
    what = re.sub(r"func=(.*)", repl_func, what)
    super().assertMultiLineStrippedEqual(expected, what)

  def test_eval(self):
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = mul a 2.00
      c = id_tap[ func=_print
                  output_stream=...
                  what=a * 2 ] b
      d = mul c 3.00
      e f = id_tap[ func=_print
                    nr_results=1
                    output_stream=...
                    what=y * 3 ] d c
      g = pow f 2.00
  in (g,) }""", str(api.make_jaxpr(fun1)(5.)))
    self.assertEqual("", testing_stream.output)

    self.assertEqual((5. * 2.) ** 2, fun1(5.))
    self.assertMultiLineStrippedEqual(
        """
what: a * 2
10.00
what: y * 3
30.00""", testing_stream.output)
    testing_stream.reset()

  def test_with_tuple_results(self):
    def func2(x):
      x1, y1 = hcb.id_print(x * 2., x * 3., output_stream=testing_stream)
      return x1 + y1

    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = mul a 2.00
      c = mul a 3.00
      d e = id_tap[ func=_print
                    output_stream=...] b c
      f = add d e
  in (f,) }""", str(api.make_jaxpr(func2)(3.)))
    self.assertEqual(3. * (2. + 3.), func2(3.))
    self.assertMultiLineStrippedEqual("""
6.00
9.00""", testing_stream.output)
    testing_stream.reset()

  def test_with_result(self):
    def func2(x):
      x1 = hcb.id_print(x * 2., x * 3., result=x * 4.,
                        output_stream=testing_stream)
      return x1

    self.assertMultiLineStrippedEqual(
      """
{ lambda  ; a.
  let b = mul a 2.00
      c = mul a 3.00
      d = mul a 4.00
      e f g = id_tap[ func=_print
                      nr_results=1
                      output_stream=...] b c d
  in (g,) }""", str(api.make_jaxpr(func2)(3.)))
    self.assertEqual(3. * 4., func2(3.))
    self.assertMultiLineStrippedEqual("""
6.00
9.00""", testing_stream.output)
    testing_stream.reset()


  def test_jit_simple(self):
    jit_fun1 = api.jit(lambda x: 3. * hcb.id_print(
        2. * x, what="here", output_stream=testing_stream))
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = xla_call[ backend=None
                    call_jaxpr={ lambda  ; a.
                                 let b = mul a 2.00
                                     c = id_tap[ func=_print
                                                 output_stream=...
                                                 what=here ] b
                                     d = mul c 3.00
                                 in (d,) }
                    device=None
                    name=<lambda> ] a
  in (b,) }""", str(api.make_jaxpr(jit_fun1)(5.)))
    logging.warning("%s: %s",
                 self._testMethodName, api.xla_computation(jit_fun1)(5.).GetHloText())
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      res = jit_fun1(5.)

    self.assertAllClose(6. * 5., res, check_dtypes=True)
    self.assertMultiLineStrippedEqual(
      """
what: here
10.00""", testing_stream.output)
    testing_stream.reset()

  def test_jit_sequence1(self):
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      return hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

    logging.info("%s: %s", self._testMethodName,
          api.make_jaxpr(func)(1))
    logging.info("%s: %s", self._testMethodName,
          api.xla_computation(func)(1).GetHloText())

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(2, api.jit(func)(1))
    self.assertMultiLineStrippedEqual(
            """
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

    self.assertMultiLineStrippedEqual(
      """
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
      return hcb.id_print(x3 + 1, where="2", output_stream=testing_stream)

    logging.warning("%s: %s", self._testMethodName,
                 api.make_jaxpr(func)(1))
    logging.warning("%s: %s", self._testMethodName,
                 api.xla_computation(func)(1).GetHloText())
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(3, api.jit(func)(1))
    self.assertMultiLineStrippedEqual(
      """
where: 1
1
where: nested
2
where: 2
3""", testing_stream.output)
    testing_stream.reset()

  def test_jit_devices(self):
    """Running on multiple devices."""
    devices = api.local_devices()
    def func(x, device_id):
      x1 = hcb.id_print(x, dev=str(device_id), output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, dev=str(device_id), output_stream=testing_stream)
      return x2

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      for d in devices:
        self.assertEqual(2, api.jit(func, device=d, static_argnums=1)(1, d.id))

    self.assertEqual(len(devices), len(re.findall(r"1", testing_stream.output)))
    self.assertEqual(len(devices), len(re.findall(r"2", testing_stream.output)))
    testing_stream.reset()

  def test_jit_cond1(self):
    """A conditional"""
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

      x4 = lax.cond(x % 2 == 0,
                    x2 + 1, lambda x: hcb.id_print(x, where="cond_t", output_stream=testing_stream),
                    x2 + 1, lambda x: hcb.id_print(-1, where="cond_f", result=x, output_stream=testing_stream))
      x5 = hcb.id_print(x4 + 1, where="w_2", output_stream=testing_stream)
      return x5

    logging.warning("%s: %s", self._testMethodName, api.make_jaxpr(func)(1))
    logging.warning("%s: %s", self._testMethodName,
                    api.xla_computation(func)(1).GetHloText())

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(4, api.jit(func)(1))
    self.assertMultiLineStrippedEqual("""
where: 1
1
where: 2
2
where: cond_f
-1
where: w_2
4""", testing_stream.output)
    testing_stream.reset()


  def test_jit_while_cond(self):
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)
      def body(x):
        x3 = hcb.id_print(x, where="w_1", output_stream=testing_stream)
        x4 = lax.cond(x % 2 == 0,
                      x3 + 1, lambda x: hcb.id_print(x, where="w_t", output_stream=testing_stream),
                      x3 + 1, lambda x: hcb.id_print(-1, where="w_f", result=x, output_stream=testing_stream))
        return hcb.id_print(x4 + 1, where="w_2", output_stream=testing_stream)
      x10 = lax.while_loop(lambda x: x < 10, body, x2)
      res = hcb.id_print(x10, where="10", output_stream=testing_stream)
      return res
    logging.warning("%s: %s", self._testMethodName, api.make_jaxpr(func)(1))
    logging.warning("%s: %s", self._testMethodName,
          api.xla_computation(func)(1).GetHloText())

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(10, api.jit(func)(1))
    self.assertMultiLineStrippedEqual(
        """
where: 1
1
where: 2
2
where: w_1
2
where: w_t
3
where: w_2
4
where: w_1
4
where: w_t
5
where: w_2
6
where: w_1
6
where: w_t
7
where: w_2
8
where: w_1
8
where: w_t
9
where: w_2
10
where: 10
10""", testing_stream.output)
    testing_stream.reset()

  def test_jit_while_pred_printing(self):
    raise SkipTest("Not yet implemented")
    """While with printing in the conditional."""
    def func(x):
      x1 = hcb.id_print(x, where="1")

      def body(x):
        x3 = hcb.id_print(x, where="w_1", output_stream=testing_stream)
        return hcb.id_print(x3 + 1, where="w_2", output_stream=testing_stream)

      x10 = lax.while_loop(lambda x: hcb.id_print(x < 10, where="w_p", output_stream=testing_stream),
                           body, x1)
      res = hcb.id_print(x10, where="10", output_stream=testing_stream)
      return res

    logging.warning("%s: %s", self._testMethodName, api.make_jaxpr(func)(1))
    logging.warning("%s: %s", self._testMethodName,
                    api.xla_computation(func)(1).GetHloText())

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      self.assertEqual(10, api.jit(func)(1))
    self.assertMultiLineStrippedEqual(
      """
""", testing_stream.output)
    testing_stream.reset()


  def test_jit_scan_cond(self):
    def func(x):
      x1 = hcb.id_print(x, where="1", output_stream=testing_stream)
      x2 = hcb.id_print(x1 + 1, where="2", output_stream=testing_stream)

      def body(c, x):
        x3 = hcb.id_print(x, where="s_1", output_stream=testing_stream)
        x4 = lax.cond(x % 2 == 0,
                      x3 + 1, lambda x: hcb.id_print(x, where="s_t", output_stream=testing_stream),
                      x3 + 1, lambda x: hcb.id_print(-1, where="s_f", result=x, output_stream=testing_stream))
        return (c, hcb.id_print(x4 + 1, where="w_2", output_stream=testing_stream))

      _, x10 = lax.scan(body, x2, np.arange(3))
      res = hcb.id_print(x10, where="10", output_stream=testing_stream)
      return res

    logging.warning("%s: %s", self._testMethodName, api.make_jaxpr(func)(1))
    logging.warning("%s: %s", self._testMethodName,
                    api.xla_computation(func)(1).GetHloText())

    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      res = api.jit(func)(1)
      self.assertAllClose(np.array([2, 3, 4]), res, check_dtypes=True)
    self.assertMultiLineStrippedEqual(
      """
where: 1
1
where: 2
2
where: s_1
0
where: s_t
1
where: w_2
2
where: s_1
1
where: s_f
-1
where: w_2
3
where: s_1
2
where: s_t
3
where: w_2
4
where: 10
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
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      res = jit_fun1(args)
    # self.assertAllClose(args, res, check_dtypes=True)

  def test_jit_large(self):
    arg = np.arange(10000, dtype=np.int32).reshape((10, 10, 5, -1))
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      api.jit(hcb.id_print)(arg)

  def test_jit_several(self):
    arg = np.arange(50, dtype=np.int32).reshape((10, 5))
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      api.jit(lambda x, y: hcb.id_print(x, y, x * 2.))(arg, np.ones(100, dtype=np.int32))

  def test_jvp(self):
    jvp_fun1 = lambda x, xt: api.jvp(fun1, (x,), (xt,))
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a b.
  let c = mul a 2.00
      d = mul b 2.00
      e f = id_tap[ func=_print
                    nr_results=0
                    output_stream=...
                    transforms=('jvp',)
                    what=a * 2 ] c d
      g = mul e 3.00
      h = mul f 3.00
      i j k l = id_tap[ func=_print
                        nr_results=2
                        output_stream=...
                        transforms=('jvp',)
                        what=y * 3 ] g h e f
      m = pow j 2.00
      n = pow j 1.00
      o = mul 2.00 n
      p = mul l o
  in (m, p) }""",
        str(api.make_jaxpr(jvp_fun1)(np.float32(5.), np.float32(0.1))))

    res_primals, res_tangents = jvp_fun1(np.float32(5.), np.float32(0.1))
    self.assertMultiLineStrippedEqual(
        """
transforms: ('jvp',) what: a * 2
10.00
0.20
transforms: ('jvp',) what: y * 3
30.00
0.60""", testing_stream.output)
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
  let b = mul a 2.00
      c = id_tap[ consumer_id=xxx
                  output_stream=yyy
                  what=a * 2 ] b
      d = mul c 3.00
      e f = id_tap[ consumer_id=xxx
                    nr_results=1
                    output_stream=yyy
                    what=y * 3 ] d c
      g = pow f 1.00
      h = mul 2.00 g
      i = mul 1.00 h
      j k = id_tap[ consumer_id=xxx
                    nr_results=1
                    output_stream=yyy
                    transforms=('jvp', 'transpose')
                    what=y * 3 ] 0.00 i
      l = mul j 3.00
      m = add_any k l
      n = id_tap[ consumer_id=xxx
                  output_stream=yyy
                  transforms=('jvp', 'transpose')
                  what=a * 2 ] m
      o = mul n 2.00
  in (o,) }""", str(api.make_jaxpr(grad_fun1)(5.)))


    res_grad = grad_fun1(np.float32(5.))
    self.assertMultiLineStrippedEqual(
        """
what: a * 2
10.00
what: y * 3
30.00
10.00
transforms: ('jvp',) what: y * 3
0.00
20.00
transforms: ('jvp',) what: a * 2
20.00
   """, testing_stream.output)
    testing_stream.reset()

  def test_vmap(self):
    vmap_fun1 = api.vmap(fun1)
    vargs = np.array([np.float32(4.), np.float32(5.)])
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = mul a 2.00
      c = id_tap[ batch_dims=(0,)
                  func=_print
                  output_stream=...
                  transforms=('batch',)
                  what=a * 2 ] b
      d = mul c 3.00
      e f = id_tap[ batch_dims=(0, 0)
                    func=_print
                    nr_results=1
                    output_stream=...
                    transforms=('batch',)
                    what=y * 3 ] d c
      g = pow f 2.00
  in (g,) }""", str(api.make_jaxpr(vmap_fun1)(vargs)))

    res_vmap = vmap_fun1(vargs)
    self.assertMultiLineStrippedEqual(
        """
batch_dims: (0,) transforms: ('batch',) what: a * 2
[ 8.00 10.00]
batch_dims: (0, 0) transforms: ('batch',) what: y * 3
[24.00 30.00]""", testing_stream.output)
    testing_stream.reset()

  def test_vmap_not_batched(self):
    x = 3.
    def func(y):
      # x is not mapped, y is mapped
      _, y = hcb.id_print(x, y, output_stream=testing_stream)
      return x + y

    vmap_func = api.vmap(func)
    vargs = np.array([np.float32(4.), np.float32(5.)])
    self.assertMultiLineStrippedEqual(
      """
{ lambda  ; a.
  let b c = id_tap[ batch_dims=(None, 0)
                    func=_print
                    output_stream=...
                    transforms=('batch',) ] 3.00 a
      d = add c 3.00
  in (d,) }""", str(api.make_jaxpr(vmap_func)(vargs)))

    res_vmap = vmap_func(vargs)
    self.assertMultiLineStrippedEqual(
      """
batch_dims: (None, 0) transforms: ('batch',)
3.00
[4.00 5.00]
   """, testing_stream.output)
    testing_stream.reset()


  def test_pmap(self):
    vargs = 2. + np.arange(api.local_device_count(), dtype=np.float32)

    pmap_fun1 = api.pmap(fun1, axis_name="i")
    with hcb.outfeed_receiver(receiver_name=self._testMethodName):
      res = pmap_fun1(vargs)
    expected_res = np.stack([fun1_equiv(2. + a) for a in range(api.local_device_count())])
    self.assertAllClose(expected_res, res, check_dtypes=False)

  def test_mask(self):

    @partial(api.mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return np.sum(hcb.id_print(x, what="x", output_stream=testing_stream))
    args = [np.arange(4)], dict(n=onp.int64(2))
    self.assertMultiLineStrippedEqual(
      """
{ lambda c f ; a b.
  let d = lt c b
      e = id_tap[ func=_print
                  logical_shapes=[(Traced<ShapedArray(int32[]):JaxprTrace(level=0/0)>,)]
                  output_stream=...
                  transforms=('mask',)
                  what=x ] a
      g = select d e f
      h = reduce_sum[ axes=(0,) ] g
  in (h,) }""", str(api.make_jaxpr(padded_sum)(*args)))

    res = padded_sum(*args)
    self.assertMultiLineStrippedEqual(
      """
logical_shapes: [(2,)] transforms: ('mask',) what: x
[0 1 2 3]
   """, testing_stream.output)
    testing_stream.reset()

if __name__ == "__main__":
  absltest.main()
