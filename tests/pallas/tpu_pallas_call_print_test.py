# Copyright 2023 The JAX Authors.
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

"""Test TPU-specific extensions to pallas print call."""

import functools
import re
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax._src.pallas import pallas_test_util as ptu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec

partial = functools.partial


@jtu.skip_under_pytest("Requires pytest -s (no capture) to pass, which is not enabled in CI")
@jtu.thread_unsafe_test_class()  # debug print test is not thread safe
class PallasCallPrintTest(ptu.PallasTPUTest):

  def test_debug_print(self):
    @jax.jit(compiler_options={'xla_tpu_enable_log_recorder': 'true'})
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      pl.debug_print('It works!')

    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape((8, 128))
    with jtu.capture_stderr() as get_output:
      jax.block_until_ready(kernel(x))
    self.assertIn('It works!', get_output())

  @parameterized.product(arg_type=[int, float])
  def test_debug_print_with_values(self, arg_type):
    @jax.jit(compiler_options={'xla_tpu_enable_log_recorder': 'true'})
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    def kernel(o_ref):
      del o_ref  # Unused.
      pl.debug_print('DONE ', arg_type(123))

    with jtu.capture_stderr() as get_output:
      jax.block_until_ready(kernel())

    if arg_type is int:
      self.assertIn('DONE s32[] 123', get_output())
    else:
      self.assertIn('DONE f32[] 123', get_output())

  def test_debug_print_in_index_map(self):
    def index_map(i):
      pl.debug_print('It works!')
      return (i, 0)

    @jax.jit(compiler_options={'xla_tpu_enable_log_recorder': 'true'})
    @functools.partial(
        self.pallas_call,
        grid=(1,),
        in_specs=(pl.BlockSpec(index_map=index_map),),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape((8, 128))
    with jtu.capture_stderr() as get_output:
      jax.block_until_ready(kernel(x))
    self.assertIn('It works!', get_output())

  @parameterized.parameters(
      (jnp.int32, 42),
      (jnp.uint32, 42),
      (jnp.int32, -42),
      (jnp.float32, 42.0),
  )
  def test_debug_print_with_formatting(self, dtype, value):
    @jax.jit(compiler_options={'xla_tpu_enable_log_recorder': 'true'})
    @functools.partial(
        self.pallas_call,
        in_specs=(pl.BlockSpec(memory_space=pltpu.SMEM),),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      del o_ref  # Only used for awaiting the result.
      pl.debug_print('x[0] == {}', x_ref[0])

    x = jnp.array([value], dtype=dtype)
    with jtu.capture_stderr() as get_output:
      jax.block_until_ready(kernel(x))
    output = get_output()
    self.assertIn(f'x[0] == {value}', output)

  @parameterized.parameters(
      'x[0] == {} and x[1] == {}',
      'x[0] == {} and x[1] == {} and trailing text',
  )
  def test_debug_print_multiple_with_formatting(self, fmt):
    @jax.jit(compiler_options={'xla_tpu_enable_log_recorder': 'true'})
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
    )
    def kernel(x_ref, o_ref):
      del o_ref  # Only used for awaiting the result.
      pl.debug_print(fmt, x_ref[0], x_ref[1])

    x = jnp.array([1, 2], dtype=jnp.int32)
    with jtu.capture_stderr() as get_output:
      jax.block_until_ready(kernel(x))
    output = get_output()
    self.assertIn(fmt.format(x[0], x[1]), output)

  @parameterized.named_parameters(
      (f"{'_'.join(map(str, shape))}_{dtype.__name__}", shape, dtype)
      for shape in (
          (2, 8, 128),
          # test unaligned shapes
          (3,),
          (3, 4),
          (2, 3, 4),
          (2, 9, 129),
      )
      for dtype in (jnp.int32, jnp.uint32, jnp.float32)
  )
  def test_debug_print_vector(self, shape, dtype):
    @jax.jit(compiler_options={"xla_tpu_enable_log_recorder": "true"})
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
    )
    def kernel(x_ref, o_ref):
      pl.debug_print("{}", x_ref[...])
      o_ref[...] = x_ref[...]

    n = np.prod(shape)
    x = jnp.arange(n, dtype=dtype).reshape(shape)
    with jtu.capture_stderr() as get_output:
      jax.block_until_ready(kernel(x))
    output = get_output()
    numbers = [
        int(num)
        for line in output.splitlines()
        if (match := re.search(r"\{(.*)", line))  # extract contents after `{`
        for num in re.findall(r"\d+", match.group(1))
    ]
    # Check if the numbers in the output match the values generated by `arange`.
    self.assertLen(numbers, n)
    self.assertTrue(all(num == i for i, num in enumerate(numbers)))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
