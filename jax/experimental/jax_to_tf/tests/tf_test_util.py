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

import numpy as np
from typing import Any, Callable, Tuple
import tensorflow as tf  # type: ignore[import]

from jax.config import config
from jax import dtypes
from jax.experimental import jax_to_tf
from jax import test_util as jtu

class JaxToTfTestCase(jtu.JaxTestCase):

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
    func_tf = jax_to_tf.convert(func_jax)
    if with_function:
      func_tf = tf.function(func_tf)
    res_jax = func_jax(*args)
    res_tf = func_tf(*args)
    self.assertAllClose(res_jax, res_tf, atol=atol, rtol=rtol)
    return (res_jax, res_tf)
