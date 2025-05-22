# Copyright 2025 The JAX Authors.
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


"""Test the Triton dialect lowering for a variety of atomic operations."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.pallas.pallas_call import _trace_kernel_to_jaxpr
from jax.experimental import pallas as pl
import jax.numpy as jnp

config.parse_flags_with_absl()


@jtu.with_config(jax_traceback_filtering="off")
class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if jtu.test_device_matches(["cpu"]):
      if not self.INTERPRET:
        self.skipTest("On CPU the test works only in interpret mode")
    elif jtu.test_device_matches(["gpu"]):
      if not jtu.is_cuda_compute_capability_at_least("9.0"):
        self.skipTest("Only works on GPU with capability >= sm90")
    else:
      self.skipTest("Test only works on CPU and GPU")

    super().setUp()
    _trace_kernel_to_jaxpr.cache_clear()

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)


DTYPE_LIST = [jnp.float32, jnp.float16, jnp.bfloat16,
              jnp.float8_e4m3fn, jnp.float8_e5m2]


class TritonPallasTest(PallasBaseTest):
  INTERPRET = False

  @parameterized.product(src_dtype=DTYPE_LIST, dst_dtype=DTYPE_LIST)
  def test_fp_dtype_cast(self, src_dtype, dst_dtype):
    if src_dtype == dst_dtype:
      self.skipTest("No need to test the same dtype")
    if dtypes.bit_width(src_dtype) == 8 and dtypes.bit_width(dst_dtype) == 8:
      self.skipTest("Not casting between 8-bit types")

    def body(x_ref, y_ref):
      y_ref[...] = x_ref[...].astype(dst_dtype)

    x = 10 * jax.random.normal(jax.random.key(0), (64, 64), dtype=src_dtype)
    y = self.pallas_call(body,
        in_specs=[pl.BlockSpec((64, 64), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((64, 64), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((64, 64), dst_dtype),
        grid=(1,),
    )(x)
    self.assertEqual(y.dtype, dst_dtype)
    self.assertArraysEqual(y, x.astype(dst_dtype))

if __name__ == "__main__":
  absltest.main()
