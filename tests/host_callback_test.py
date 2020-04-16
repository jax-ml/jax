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

import collections
import functools
from functools import partial, reduce
import os
import re
import time
from typing import Callable
import unittest
import warnings
import weakref
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp
import six

if six.PY3:
  import concurrent.futures

from absl.testing import parameterized

import jax
import jax.numpy as np
jnp = np
from jax import api, lax, lax_reference
from jax import core

from jax import test_util as jtu
from jax.experimental import host_callback as hcb
from jax.lib import xla_bridge

from benchmarks import benchmark

from jax.config import config

from jax.scipy.special import logsumexp

config.parse_flags_with_absl()
FLAGS = config.FLAGS


class _TestingOutputStream(object):
  """Use as `output_stream` for tests."""

  def __init__(self):
    self.output = ""

  def write(self, other: str) -> None:
    print(f"output_stream: {other}")
    self.output = other if not self.output else f"{self.output}\n{other}"

  def __str__(self):
    return "TestingOutputStream"

  def reset(self):
    self.output = ""


testing_stream = _TestingOutputStream()


def fun1(a):
  y = hcb.id_print(a * 2., what="a * 2", output_stream=testing_stream)
  y = hcb.id_print(y * 3., what="y * 3", output_stream=testing_stream, tie_in=y)
  return y**4  # Some computation to make the gradient interesting


def fun1_equiv(a):  # Numerical equivalent of fun`
  return (a * 2.)**4


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
    from jax import anecula_utils
    anecula_utils.helper_set_hlo_dump()

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
(6.0, 9.0)  {}""", testing_stream.output)
    testing_stream.reset()

  def test_eval(self):
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = mul a 2.0
      c = id_print[ output_stream=TestingOutputStream
                    what=a * 2 ] b
      d = mul c 3.0
      e = id_print[ output_stream=TestingOutputStream
                    what=y * 3 ] d
      f = tie_in e c
      g = pow f 4.0
  in (g,) }""", str(api.make_jaxpr(fun1)(5.)))
    self.assertEqual("", testing_stream.output)

    self.assertEqual((5. * 2.)**4, fun1(5.))
    self.assertMultiLineStrippedEqual(
        """
(10.0,)  {'what': 'a * 2'}
(30.0,)  {'what': 'y * 3'}""", testing_stream.output)
    testing_stream.reset()

  def test_jit_simple(self):
    self.helper_set_hlo_dump()
    jit_fun1 = api.jit(lambda x: 3. * hcb.id_print(
        2. * x, what="here", output_stream=testing_stream))
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = xla_call[ backend=None
                    call_jaxpr={ lambda  ; a.
                                 let b = mul a 2.0
                                     c = id_print[ output_stream=TestingOutputStream
                                                   what=here ] b
                                     d = mul c 3.0
                                 in (d,) }
                    device=None
                    name=<lambda> ] a
  in (b,) }""", str(api.make_jaxpr(jit_fun1)(5.)))
    print(api.xla_computation(jit_fun1)(5.).GetHloText())
    res = jit_fun1(5.)
    self.assertAllClose(6. * 5., res, check_dtypes=True)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(
              testcase_name=f"_shape_{shape}_dtype_{dtype}_nr_args={nr_args}",
              shape=shape,
              dtype=dtype,
              nr_args=nr_args) for nr_args in [1, 2]
          for shape in [(), (2,), (2, 3), (2, 3, 4)]
          for dtype in jtu.supported_dtypes()))
  def test_jit_types(self, nr_args=1, dtype=np.bfloat16, shape=()):
    self.helper_set_hlo_dump()
    args = [np.arange(np.prod(shape), dtype=dtype).reshape(shape)]
    if nr_args > 1:
      args = args * nr_args
    jit_fun1 = api.jit(lambda xs: hcb.id_print(
        *xs,
        a_new_test="************",
        testcase_name=f"shape_{shape}_dtype_{dtype}_nr_args={nr_args}"))
    res = jit_fun1(args)

  def test_jit_large(self):
    arg = np.arange(10000).reshape((10, 10, 5, -1))
    api.jit(lambda x: hcb.id_print(x))(arg)

  def test_jvp(self):
    jvp_fun1 = lambda x, xt: api.jvp(fun1, (x,), (xt,))
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a b.
  let c = mul a 2.0
      d = id_print[ output_stream=TestingOutputStream
                    what=a * 2 ] c
      e = mul d 3.0
      f = id_print[ output_stream=TestingOutputStream
                    what=y * 3 ] e
      g = tie_in f d
      h = pow g 4.0
      i = mul b 2.0
      j = id_print[ output_stream=TestingOutputStream
                    transforms=('jvp',)
                    what=a * 2 ] i
      k = mul j 3.0
      l = id_print[ output_stream=TestingOutputStream
                    transforms=('jvp',)
                    what=y * 3 ] k
      m = tie_in l j
      n = pow g 3.0
      o = mul 4.0 n
      p = mul m o
  in (h, p) }""", str(api.make_jaxpr(jvp_fun1)(5., 0.1)))

    res_primals, res_tangents = jvp_fun1(5., .1)
    self.assertMultiLineStrippedEqual(
        """
(DeviceArray(10., dtype=float32),)  {'what': 'a * 2'}
(DeviceArray(0.2, dtype=float32),)  {'what': 'a * 2', 'transforms': ('jvp',)}
(DeviceArray(30., dtype=float32),)  {'what': 'y * 3'}
(DeviceArray(0.6, dtype=float32),)  {'what': 'y * 3', 'transforms': ('jvp',)}
  """, testing_stream.output)
    testing_stream.reset()

  def test_grad(self):
    grad_fun1 = api.grad(fun1)
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = mul a 2.0
      c = id_print[ output_stream=TestingOutputStream
                    what=a * 2 ] b
      d = mul c 3.0
      e = id_print[ output_stream=TestingOutputStream
                    what=y * 3 ] d
      f = tie_in e c
      g = pow f 3.0
      h = mul 4.0 g
      i = mul 1.0 h
      j = id_print[ output_stream=TestingOutputStream
                    transforms=('jvp', 'transpose')
                    what=a * 2 ] i
      k = mul j 2.0
  in (k,) }""", str(api.make_jaxpr(grad_fun1)(5.)))

    # This comes from the actual partial evaluation
    self.assertMultiLineStrippedEqual(
        """
(Zero,)  {'what': 'y * 3', 'transforms': ('jvp', 'transpose')}
  """, testing_stream.output)
    testing_stream.reset()

    res_grad = grad_fun1(5.)
    self.assertMultiLineStrippedEqual(
        """
(DeviceArray(10., dtype=float32),)  {'what': 'a * 2'}
(DeviceArray(30., dtype=float32),)  {'what': 'y * 3'}
(Zero,)  {'what': 'y * 3', 'transforms': ('jvp', 'transpose')}
(DeviceArray(4000., dtype=float32),)  {'what': 'a * 2', 'transforms': ('jvp', 'transpose')}
   """, testing_stream.output)
    testing_stream.reset()

  def test_vmap(self):
    vmap_fun1 = api.vmap(fun1)
    vargs = onp.array([4., 5.])
    self.assertMultiLineStrippedEqual(
        """
{ lambda  ; a.
  let b = mul a 2.0
      c = id_print[ output_stream=TestingOutputStream
                    transforms=('batch',)
                    what=a * 2 ] b
      d = mul c 3.0
      e = id_print[ output_stream=TestingOutputStream
                    transforms=('batch',)
                    what=y * 3 ] d
      f = tie_in e c
      g = pow f 4.0
  in (g,) }""", str(api.make_jaxpr(vmap_fun1)(vargs)))

    res_vmap = vmap_fun1(vargs)
    self.assertMultiLineStrippedEqual(
        """
(DeviceArray([ 8., 10.], dtype=float32),)  {'what': 'a * 2', 'transforms': ('batch',)}
(DeviceArray([24., 30.], dtype=float32),)  {'what': 'y * 3', 'transforms': ('batch',)}
     """, testing_stream.output)
    testing_stream.reset()

  def test_pmap(self):
    self.helper_set_devices(4)
    vargs = np.arange(api.local_device_count(), dtype=np.float32)

    pmap_fun1 = api.pmap(fun1, axis_name="i")
    res = pmap_fun1(vargs)


if __name__ == "__main__":
  absltest.main()
