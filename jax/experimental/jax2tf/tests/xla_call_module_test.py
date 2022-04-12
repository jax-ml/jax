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
"""Tests for call_xla."""

import logging
from typing import Callable, Dict, Optional
import unittest

from absl.testing import absltest

import jax
from jax import lax
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax.config import config
import jax.interpreters.mlir as mlir
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util

import numpy as np

import tensorflow as tf  # type: ignore[import]

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla  # type: ignore[import]
# pylint: enable=g-direct-tensorflow-import

config.parse_flags_with_absl()


def get_serialized_computation(
    f_jax: Callable,
    *args,
    abstracted_axes: Optional[Dict[int, str]] = None) -> str:
  lowered = jax.jit(f_jax, abstracted_axes=abstracted_axes).lower(*args)
  mhlo_module = lowered.compiler_ir(dialect='mhlo')
  mhlo_module_text = mlir.module_to_string(mhlo_module)
  logging.info(f'Serialized ir.Module = {mhlo_module_text}')
  return mhlo_module_text


class XlaCallModuleTest(tf_test_util.JaxToTfTestCase):

  def test_simple(self):

    def f_jax(x):
      return jnp.sin(x)

    x = np.ones((2, 3), dtype=np.float32)

    jax_res = f_jax(x)
    res = tfxla.call_module([x],
                            module=get_serialized_computation(f_jax, x),
                            Tout=[jax_res.dtype],
                            Sout=[jax_res.shape])
    logging.info(f'Result = {res}')

  def test_while(self):
    # With nested computation
    def f_jax(count, x):
      return lax.while_loop(lambda carry: carry[0] < count, lambda carry:
                            (carry[0] + 1, carry[1] + 1.), (0, x))[1]

    count = np.int32(5)
    x = np.ones((2, 3), dtype=np.float32)

    jax_res = f_jax(count, x)
    res = tfxla.call_module([count, x],
                            module=get_serialized_computation(f_jax, count, x),
                            Tout=[jax_res.dtype],
                            Sout=[jax_res.shape])
    logging.info(f'Result = {res}')

  def test_multiple_args_results(self):

    def f_jax(x1, x2):
      return (jnp.sin(x1), jnp.cos(x2))

    x1 = np.ones((2, 3), dtype=np.float32)
    x2 = np.ones((3, 4), dtype=np.float32)

    jax_res = f_jax(x1, x2)

    def f_tf(x1_tf, x2_tf):
      return tfxla.call_module([x1_tf, x2_tf],
                               module=get_serialized_computation(f_jax, x1, x2),
                               Tout=[jax_res[0].dtype, jax_res[1].dtype],
                               Sout=[jax_res[0].shape, jax_res[1].shape])

    res = tf.function(f_tf, jit_compile=True, autograph=False)(x1, x2)
    logging.info(f'Result = {res}')

  @unittest.skip('TODO(necula): not yet working')
  def test_shape_poly_arange(self):

    def f_jax(x):  # x: f32[b]
      return jnp.arange(x.shape[0]) + x

    x1 = np.ones((5,), dtype=np.float32)

    jax_res = f_jax(x1)

    def f_tf(x1_tf):
      return tfxla.call_module([x1_tf],
                               module=get_serialized_computation(
                                   f_jax, x1,
                                   abstracted_axes=({
                                       0: 'b'
                                   },)),
                               Tout=[jax_res.dtype],
                               Sout=[jax_res.shape],
                               dim_args_spec=('0.0',))

    res = tf.function(f_tf, jit_compile=True, autograph=False)(x1)
    logging.info(f'Result = {res}')


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
