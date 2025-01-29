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

from functools import partial
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.ad_checkpoint
from jax import lax
from jax.sharding import PartitionSpec as P
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp

from jax.experimental.shard_map import shard_map


config.parse_flags_with_absl()

class RaggedCollectiveTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_single_axis_name', axis_name='x', mesh_axes=dict(x=2)
      ),
  )
  def test_ragged_all_to_all(self, axis_name, mesh_axes):
    device_type = jax.devices()[0].platform
    if device_type == 'tpu' and jtu.get_tpu_version() == 3:
      raise unittest.SkipTest(
          'UNSUPPORTED: HLO opcode `ragged-all-to-all` is not supported by'
          ' TPU v3'
      )
    mesh = jtu.create_mesh(tuple(mesh_axes.values()), tuple(mesh_axes.keys()))
    operand = jax.device_put(
        jnp.array([[1, 2, 2], [3, 4, 0]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    output = jax.device_put(
        jnp.zeros((2, 4), dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    input_offsets = jax.device_put(
        jnp.array([[0, 1], [0, 1]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    send_sizes = jax.device_put(
        jnp.array([[1, 2], [1, 1]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    output_offsets = jax.device_put(
        jnp.array([[0, 0], [1, 2]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    recv_sizes = jax.device_put(
        jnp.array([[1, 1], [2, 1]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )

    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P(axis_name, None),
            P(axis_name, None),
            P(axis_name, None),
            P(axis_name, None),
            P(axis_name, None),
            P(axis_name, None),
        ),
        out_specs=P(axis_name),
        check_rep=False,
    )
    def fwd(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    ):
      operand = operand.reshape(operand.shape[1:])
      output = output.reshape(output.shape[1:])
      input_offsets = input_offsets.reshape(input_offsets.shape[1:])
      send_sizes = send_sizes.reshape(send_sizes.shape[1:])
      output_offsets = output_offsets.reshape(output_offsets.shape[1:])
      recv_sizes = recv_sizes.reshape(recv_sizes.shape[1:])
      return lax.ragged_all_to_all(
          operand,
          output,
          input_offsets,
          send_sizes,
          output_offsets,
          recv_sizes,
          axis_name=axis_name,
      )

    mlir_module = fwd.lower(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    ).as_text()
    self.assertIn('stablehlo.custom_call @ragged_all_to_all', mlir_module)
    self.assertIn(
        'replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>', mlir_module
    )

    c = fwd(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    ).reshape((2, 4))
    self.assertAllClose(
        c, jnp.array([[1, 3, 0, 0], [2, 2, 4, 0]], dtype=jnp.int32)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_single_axis_name', axis_name='x', mesh_axes=dict(x=4)
      ),
  )
  def test_ragged_all_to_all_axis_index_groups(self, axis_name, mesh_axes):
    device_type = jax.devices()[0].platform
    if device_type == 'tpu' and jtu.get_tpu_version() == 3:
      raise unittest.SkipTest(
          'UNSUPPORTED: HLO opcode `ragged-all-to-all` is not supported by'
          ' TPU v3'
      )
    mesh = jtu.create_mesh(tuple(mesh_axes.values()), tuple(mesh_axes.keys()))
    operand = jax.device_put(
        jnp.array([[1, 2, 2], [3, 4, 0],
                   [10, 20, 20], [30, 40, 0]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    output = jax.device_put(
        jnp.zeros((4, 4), dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    input_offsets = jax.device_put(
        jnp.array([[0, 1], [0, 1],
                   [0, 1], [0, 1]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    send_sizes = jax.device_put(
        jnp.array([[1, 2], [1, 1],
                   [1, 2], [1, 1]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    output_offsets = jax.device_put(
        jnp.array([[0, 0], [1, 2],
                   [0, 0], [1, 2]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    recv_sizes = jax.device_put(
        jnp.array([[1, 1], [2, 1],
                   [1, 1], [2, 1]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    axis_index_groups = ((0, 1), (2, 3))

    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P(axis_name, None),
            P(axis_name, None),
            P(axis_name, None),
            P(axis_name, None),
            P(axis_name, None),
            P(axis_name, None),
        ),
        out_specs=P(axis_name),
        check_rep=False,
    )
    def fwd(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    ):
      operand = operand.reshape(operand.shape[1:])
      output = output.reshape(output.shape[1:])
      input_offsets = input_offsets.reshape(input_offsets.shape[1:])
      send_sizes = send_sizes.reshape(send_sizes.shape[1:])
      output_offsets = output_offsets.reshape(output_offsets.shape[1:])
      recv_sizes = recv_sizes.reshape(recv_sizes.shape[1:])
      return lax.ragged_all_to_all(
          operand,
          output,
          input_offsets,
          send_sizes,
          output_offsets,
          recv_sizes,
          axis_name=axis_name,
          axis_index_groups=axis_index_groups,
      )

    mlir_module = fwd.lower(
        operand, output, input_offsets, send_sizes, output_offsets,
        recv_sizes).as_text()
    self.assertIn("stablehlo.custom_call @ragged_all_to_all", mlir_module)
    self.assertIn("replica_groups = dense<[[0, 1], [2, 3]]> :"
                  " tensor<2x2xi64>", mlir_module)

    c = fwd(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    ).reshape((4, 4))
    self.assertAllClose(
        c, jnp.array([[1, 3, 0, 0], [2, 2, 4, 0],
                      [10, 30, 0, 0], [20, 20, 40, 0]], dtype=jnp.int32)
    )


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
