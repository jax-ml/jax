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

import contextlib
import logging
import numpy as np
from typing import Any, Callable, Optional, Tuple
import tensorflow as tf  # type: ignore[import]

import jax
from jax.config import config
from jax import dtypes
from jax.experimental import jax2tf
from jax import test_util as jtu
from jax import numpy as jnp

class JaxToTfTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    # Ensure that all TF ops are created on the proper device (TPU or GPU or CPU)
    # TODO(necula): why doesn't TF do this automatically?
    tf_preferred_devices = (
        tf.config.list_logical_devices("TPU") +
        tf.config.list_logical_devices("GPU") +
        tf.config.list_logical_devices())
    self.tf_default_device = tf_preferred_devices[0]
    logging.info(f"Running jax2tf converted code on {self.tf_default_device}.")
    if jtu.device_under_test() != "gpu":
      # TODO(necula): Change the build flags to ensure the GPU is seen by TF
      # It seems that we need --config=cuda build flag for this to work?
      self.assertEqual(jtu.device_under_test().upper(),
                       self.tf_default_device.device_type)

    with contextlib.ExitStack() as stack:
      stack.enter_context(tf.device(self.tf_default_device))
      self.addCleanup(stack.pop_all().close)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
    """Compares dtypes across JAX and TF dtypes. Overrides super method."""
    def to_numpy_dtype(dt):
      return dt if isinstance(dt, np.dtype) else dt.as_numpy_dtype

    if not config.FLAGS.jax_enable_x64 and canonicalize_dtypes:
      self.assertEqual(dtypes.canonicalize_dtype(to_numpy_dtype(jtu._dtype(x))),
                       dtypes.canonicalize_dtype(to_numpy_dtype(jtu._dtype(y))))
    else:
      self.assertEqual(to_numpy_dtype(jtu._dtype(x)),
                       to_numpy_dtype(jtu._dtype(y)))

  def ConvertAndCompare(self, func_jax: Callable, *args,
                        with_function: bool = False,
                        atol=None,
                        rtol=None) -> Tuple[Any, Any]:
    """Compares jax_func(*args) with convert(jax_func)(*args)."""
    func_tf = jax2tf.convert(func_jax)
    if with_function:
      func_tf = tf.function(func_tf, autograph=False)
    res_jax = func_jax(*args)
    #logging.info(f"res_jax is {res_jax} on {res_jax.device_buffer.device()}")
    res_tf = func_tf(*args)
    #logging.info(f"res_tf is {res_tf} on {res_tf.backing_device}")
    self.assertAllClose(res_jax, res_tf, atol=atol, rtol=rtol)
    return (res_jax, res_tf)

  def TransformConvertAndCompare(self, func: Callable,
                                 arg,
                                 transform: Optional[str],
                                 with_function: bool = False):
    """Like ConvertAndCompare but first applies a transformation.

    `func` must be a function from one argument to one result. `arg` is
    the argument before the transformation.

    `transform` can be None, "jvp", "grad", "vmap", "jvp_vmap", "grad_vmap"
    """
    if transform is None:
      return self.ConvertAndCompare(func, arg, with_function=with_function)
    if transform == "jvp":
      t_func = lambda x, xt: jax.jvp(func, (x,), (xt,))
      return self.ConvertAndCompare(t_func, arg, np.full_like(arg, 0.1),
                                    with_function=with_function)
    if transform == "grad":
      return self.ConvertAndCompare(jax.grad(func), arg,
                                    with_function=with_function)
    if transform == "vmap":
      t_arg = np.stack([arg] * 4)
      return self.ConvertAndCompare(jax.vmap(func),
                                    t_arg, with_function=with_function)
    if transform == "jvp_vmap":
      jvp_func = lambda x, xt: jax.jvp(jax.vmap(func), (x,), (xt,))
      t_arg = np.stack([arg] * 4)
      return self.ConvertAndCompare(jvp_func, t_arg,
                                    np.full_like(t_arg, 0.1),
                                    with_function=with_function)
    if transform == "grad_vmap":
      grad_func = jax.grad(lambda x: jnp.sum(jax.vmap(func)(x)))
      t_arg = np.stack([arg] * 4)
      return self.ConvertAndCompare(grad_func, t_arg,
                                    with_function=with_function)
    assert False, transform
