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
from jax import lax
from jax import vmap
from jax.sharding import PartitionSpec as P
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp

from jax._src.shard_map import shard_map


config.parse_flags_with_absl()

class RaggedCollectiveTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(['cpu']):
      self.skipTest('ragged-all-to-all is not supported on CPU')

  @parameterized.named_parameters(
      dict(
          testcase_name='_single_axis_name', axis_name='x', mesh_axes=dict(x=2)
      ),
  )
  def test_ragged_all_to_all(self, axis_name, mesh_axes):
    device_type = jax.devices()[0].platform
    if device_type == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest(
          'UNSUPPORTED: HLO opcode `ragged-all-to-all` is not supported by TPU'
          f' v{jtu.get_tpu_version()}'
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
          testcase_name='_single_axis_name', axis_name='x', mesh_axes=dict(x=2)
      ),
  )
  def test_ragged_all_to_all_grad(self, axis_name, mesh_axes):
    device_type = jax.devices()[0].platform
    if device_type == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest(
          'UNSUPPORTED: HLO opcode `ragged-all-to-all` is not supported by TPU'
          f' v{jtu.get_tpu_version()}'
      )
    mesh = jtu.create_mesh(tuple(mesh_axes.values()), tuple(mesh_axes.keys()))
    operand = jax.device_put(
        jnp.array([[1, 2, 2], [3, 4, 0]], dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    output = jax.device_put(
        jnp.zeros((2, 4), dtype=jnp.float32),
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

    args = input_offsets, send_sizes, output_offsets, recv_sizes
    jtu.check_grads(lambda op, out: fwd(op, out, *args), (operand, output), order=1)

  @parameterized.named_parameters(
      dict(
          testcase_name='_single_axis_name', axis_name='x', mesh_axes=dict(x=4)
      ),
  )
  def test_ragged_all_to_all_axis_index_groups(self, axis_name, mesh_axes):
    device_type = jax.devices()[0].platform
    if device_type == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest(
          'UNSUPPORTED: HLO opcode `ragged-all-to-all` is not supported by TPU'
          f' v{jtu.get_tpu_version()}'
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
    self.assertIn('stablehlo.custom_call @ragged_all_to_all', mlir_module)
    self.assertIn('replica_groups = dense<[[0, 1], [2, 3]]> :'
                  ' tensor<2x2xi64>', mlir_module)

    c = fwd(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    ).reshape((4, 4))
    self.assertAllClose(
        c, jnp.array([[1, 3, 0, 0], [2, 2, 4, 0],
                      [10, 30, 0, 0], [20, 20, 40, 0]], dtype=jnp.int32)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_single_axis_name', axis_name='x', mesh_axes=dict(x=2)
      ),
  )
  def test_ragged_all_to_all_degenerate_groups(self, axis_name, mesh_axes):
    device_type = jax.devices()[0].platform
    if device_type == 'tpu':
      raise unittest.SkipTest(
          'UNSUPPORTED: HLO opcode `ragged-all-to-all` with singleton group is'
          ' not supported by TPU'
      )
    mesh = jtu.create_mesh(tuple(mesh_axes.values()), tuple(mesh_axes.keys()))
    operand = jax.device_put(
        jnp.array([[1, 0, 0, 0], [2, 3, 4, 0]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    output = jax.device_put(
        jnp.zeros((2, 4), dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    input_offsets = jax.device_put(
        jnp.array([[0], [0]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    send_sizes = jax.device_put(
        jnp.array([[1], [3]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    output_offsets = jax.device_put(
        jnp.array([[2], [1]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    recv_sizes = jax.device_put(
        jnp.array([[1], [3]], dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)),
    )
    axis_index_groups = ((0,), (1,))

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
    self.assertIn('stablehlo.custom_call @ragged_all_to_all', mlir_module)
    self.assertIn('replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>',
                  mlir_module)

    c = fwd(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    ).reshape((2, 4))
    self.assertAllClose(
        c, jnp.array([[0, 0, 1, 0], [0, 2, 3, 4]], dtype=jnp.int32)
    )

  def test_ragged_all_to_all_vmap_multi_dim_operand(self):
    device_type = jax.devices()[0].platform
    if device_type == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest(
          'UNSUPPORTED: HLO opcode `ragged-all-to-all` is not supported by TPU'
          f' v{jtu.get_tpu_version()}'
      )

    axis_name = 'x'
    mesh = jtu.create_mesh((2,), ('x',))
    data_sharding = P(axis_name, None, None)
    operand_data = jnp.zeros((2, 2, 3), dtype=jnp.int32)
    output_data = jnp.zeros((2, 2, 4), dtype=jnp.int32)
    input_offsets_data = jnp.zeros((2, 2, 2), dtype=jnp.int32)
    send_sizes_data = jnp.zeros((2, 2, 2), dtype=jnp.int32)
    output_offsets_data = jnp.zeros((2, 2, 2), dtype=jnp.int32)
    recv_sizes_data = jnp.zeros((2, 2, 2), dtype=jnp.int32)

    operand = jax.device_put(operand_data, jax.sharding.NamedSharding(mesh, data_sharding))
    output = jax.device_put(output_data, jax.sharding.NamedSharding(mesh, data_sharding))
    input_offsets = jax.device_put(input_offsets_data, jax.sharding.NamedSharding(mesh, data_sharding))
    send_sizes = jax.device_put(send_sizes_data, jax.sharding.NamedSharding(mesh, data_sharding))
    output_offsets = jax.device_put(output_offsets_data, jax.sharding.NamedSharding(mesh, data_sharding))
    recv_sizes = jax.device_put(recv_sizes_data, jax.sharding.NamedSharding(mesh, data_sharding))

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
    )
    def fwd(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    ):
      return lax.ragged_all_to_all(
          operand=operand.reshape(operand.shape[1:]),
          output=output.reshape(output.shape[1:]),
          input_offsets=input_offsets.reshape(input_offsets.shape[1:]),
          send_sizes=send_sizes.reshape(send_sizes.shape[1:]),
          output_offsets=output_offsets.reshape(output_offsets.shape[1:]),
          recv_sizes=recv_sizes.reshape(recv_sizes.shape[1:]),
          axis_name=axis_name,
      )

    res = vmap(fwd, in_axes=0, out_axes=0
               )(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes)
    ref = jnp.stack(list(map(fwd, operand, output, input_offsets, send_sizes, output_offsets, recv_sizes)))
    self.assertEqual(res.shape, ref.shape)
    self.assertAllClose(res, ref, check_dtypes=False)

  @parameterized.named_parameters(
    dict(
        testcase_name='_batch_0_data_shard_axis_0_input_0',
        axis_name='x',
        vmap_axis_name='y',
        mesh_axes=dict(x=2, y=2),
        vmap_batch_axis=0,
        data_shard_axis=0,
        input_config=0,
    ),
    dict(
        testcase_name='_batch_0_data_shard_axis_1_input_0',
        axis_name='x',
        vmap_axis_name='y',
        mesh_axes=dict(x=2, y=2),
        vmap_batch_axis=0,
        data_shard_axis=1,
        input_config=0,
    ),
    dict(
        testcase_name='_batch_1_data_shard_axis_0_input_1',
        axis_name='x',
        vmap_axis_name='y',
        mesh_axes=dict(x=2, y=2),
        vmap_batch_axis=1,
        data_shard_axis=0,
        input_config=1,
    ),
    dict(
        testcase_name='_batch_1_data_shard_axis_1_input_1',
        axis_name='x',
        vmap_axis_name='y',
        mesh_axes=dict(x=2, y=2),
        vmap_batch_axis=1,
        data_shard_axis=1,
        input_config=1,
    ),
  )
  def test_ragged_all_to_all_vmap(
      self,
      axis_name,
      vmap_axis_name,
      mesh_axes,
      vmap_batch_axis,
      data_shard_axis,
      input_config,
  ):
    device_type = jax.devices()[0].platform
    if device_type == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest(
          'UNSUPPORTED: HLO opcode `ragged-all-to-all` is not supported by TPU'
          f' v{jtu.get_tpu_version()}'
      )
    mesh = jtu.create_mesh(tuple(mesh_axes.values()), tuple(mesh_axes.keys()))

    def get_data_sharding(axis):
      if axis == 0:
        return P(axis_name, None, None)
      elif axis == 1:
        return P(None, axis_name, None)
      else:
        raise ValueError("Invalid data_shard_axis")

    data_sharding = get_data_sharding(data_shard_axis)

    if input_config == 0:
      operand_data = jnp.array([[[1, 2, 3], [4, 5, 6]],
                                [[1, 2, 3], [4, 5, 6]]], dtype=jnp.int32)
      send_sizes_data = jnp.array([[[1, 2], [1, 1]],
                                   [[1, 2], [1, 1]]], dtype=jnp.int32)
      output_offsets_data = jnp.array([[[0, 0], [1, 2]],
                                       [[0, 0], [1, 2]]], dtype=jnp.int32)
      recv_sizes_data = jnp.array([[[1, 1], [2, 1]],
                                   [[1, 1], [2, 1]]], dtype=jnp.int32)
    elif input_config == 1:
      operand_data = jnp.array([[[1, 2, 3], [1, 2, 3]],
                                [[4, 5, 6], [4, 5, 6]]], dtype=jnp.int32)
      send_sizes_data = jnp.array([[[1, 2], [1, 2]],
                                   [[1, 1], [1, 1]]], dtype=jnp.int32)
      output_offsets_data = jnp.array([[[0, 0], [0, 0]],
                                       [[1, 2], [1, 2]]], dtype=jnp.int32)
      recv_sizes_data = jnp.array([[[1, 1], [1, 1]],
                                   [[2, 1], [2, 1]]], dtype=jnp.int32)
    else:
      raise ValueError("Invalid input config")

    output_data = jnp.zeros((2, 2, 4), dtype=jnp.int32)
    input_offsets_data = jnp.array([[[0, 1], [0, 1]],
                                    [[0, 1], [0, 1]]], dtype=jnp.int32)

    operand = jax.device_put(operand_data, jax.sharding.NamedSharding(mesh, data_sharding))
    output = jax.device_put(output_data, jax.sharding.NamedSharding(mesh, data_sharding))
    input_offsets = jax.device_put(input_offsets_data, jax.sharding.NamedSharding(mesh, data_sharding))
    send_sizes = jax.device_put(send_sizes_data, jax.sharding.NamedSharding(mesh, data_sharding))
    output_offsets = jax.device_put(output_offsets_data, jax.sharding.NamedSharding(mesh, data_sharding))
    recv_sizes = jax.device_put(recv_sizes_data, jax.sharding.NamedSharding(mesh, data_sharding))

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
    )
    def fwd(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    ):
      return lax.ragged_all_to_all(
          operand=operand.reshape(operand.shape[1:]),
          output=output.reshape(output.shape[1:]),
          input_offsets=input_offsets.reshape(input_offsets.shape[1:]),
          send_sizes=send_sizes.reshape(send_sizes.shape[1:]),
          output_offsets=output_offsets.reshape(output_offsets.shape[1:]),
          recv_sizes=recv_sizes.reshape(recv_sizes.shape[1:]),
          axis_name=axis_name,
      )

    res = vmap(
        fwd, in_axes=vmap_batch_axis, out_axes=0,
    )(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    )

    expected_res = []
    vmap_size = output_data.shape[vmap_batch_axis]
    args = operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    for i in range(vmap_size):
      args_ = [jax.lax.index_in_dim(x, i, vmap_batch_axis, False) for x in args]
      expected_res.append(fwd(*args_))
    expected_res = jnp.stack(expected_res)
    self.assertAllClose(res, expected_res)

  def test_ragged_all_to_all_vmap_unsupported_axis_index_groups(self):
    device_type = jax.devices()[0].platform
    if device_type == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest(
          'UNSUPPORTED: HLO opcode `ragged-all-to-all` is not supported by TPU'
          f' v{jtu.get_tpu_version()}'
      )

    axis_name = 'x'
    mesh_axes = dict(x=2)
    mesh = jtu.create_mesh(tuple(mesh_axes.values()), tuple(mesh_axes.keys()))
    data_sharding = P(axis_name, None, None)
    operand_data = jnp.zeros((2, 2, 3), dtype=jnp.int32)
    output_data = jnp.zeros((2, 2, 4), dtype=jnp.int32)
    input_offsets_data = jnp.zeros((2, 2, 2), dtype=jnp.int32)
    send_sizes_data = jnp.zeros((2, 2, 2), dtype=jnp.int32)
    output_offsets_data = jnp.zeros((2, 2, 2), dtype=jnp.int32)
    recv_sizes_data = jnp.zeros((2, 2, 2), dtype=jnp.int32)

    operand = jax.device_put(operand_data, jax.sharding.NamedSharding(mesh, data_sharding))
    output = jax.device_put(output_data, jax.sharding.NamedSharding(mesh, data_sharding))
    input_offsets = jax.device_put(input_offsets_data, jax.sharding.NamedSharding(mesh, data_sharding))
    send_sizes = jax.device_put(send_sizes_data, jax.sharding.NamedSharding(mesh, data_sharding))
    output_offsets = jax.device_put(output_offsets_data, jax.sharding.NamedSharding(mesh, data_sharding))
    recv_sizes = jax.device_put(recv_sizes_data, jax.sharding.NamedSharding(mesh, data_sharding))

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
    )
    def fwd(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
    ):
      return lax.ragged_all_to_all(
          operand=operand.reshape(operand.shape[1:]),
          output=output.reshape(output.shape[1:]),
          input_offsets=input_offsets.reshape(input_offsets.shape[1:]),
          send_sizes=send_sizes.reshape(send_sizes.shape[1:]),
          output_offsets=output_offsets.reshape(output_offsets.shape[1:]),
          recv_sizes=recv_sizes.reshape(recv_sizes.shape[1:]),
          axis_name=axis_name,
          axis_index_groups=[[0, 1]],
      )

    with self.assertRaisesWithLiteralMatch(
        NotImplementedError, 'Please open a feature request!'):
      vmap(fwd, in_axes=0, out_axes=0, axis_name='b')(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes)

  def test_ragged_all_to_all_errors(self):
    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float32)
    output = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    input_offsets = jnp.array([0, 1, 3], dtype=jnp.int32)
    send_sizes = jnp.array([1, 2, 3], dtype=jnp.int32)
    output_offsets = jnp.array([0, 1, 3], dtype=jnp.int32)
    recv_sizes = jnp.array([1, 2, 3], dtype=jnp.int32)
    axis_name = 'x'

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'ragged_all_to_all input_offsets must be integer type.'
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, jnp.array([0.0, 1.0, 3.0], dtype=jnp.float32),
          send_sizes, output_offsets, recv_sizes, axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'ragged_all_to_all send_sizes must be integer type.'
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, input_offsets,
          jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), output_offsets,
          recv_sizes, axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'ragged_all_to_all output_offsets must be integer type.'
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, input_offsets, send_sizes,
          jnp.array([0.0, 1.0, 3.0], dtype=jnp.float32), recv_sizes,
          axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'ragged_all_to_all recv_sizes must be integer type.'
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, input_offsets, send_sizes, output_offsets,
          jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'ragged_all_to_all input_offsets must be rank 1 with positive dimension'
        ' size, but got shape (1, 3)',
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, jnp.array([[0, 1, 3]], dtype=jnp.int32), send_sizes,
          output_offsets, recv_sizes, axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'ragged_all_to_all input_offsets must be rank 1 with positive dimension'
        ' size, but got shape (0,)',
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, jnp.array([], dtype=jnp.int32), send_sizes,
          output_offsets, recv_sizes, axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'ragged_all_to_all send_sizes must be rank 1 with positive dimension'
        ' size, but got shape (1, 3)',
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, input_offsets,
          jnp.array([[1, 2, 3]], dtype=jnp.int32), output_offsets, recv_sizes,
          axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'ragged_all_to_all send_sizes must be rank 1 with positive dimension'
        ' size, but got shape (0,)',
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, input_offsets, jnp.array([], dtype=jnp.int32),
          output_offsets, recv_sizes, axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'ragged_all_to_all output_offsets must be rank 1 with positive'
        ' dimension size, but got shape (1, 3)',
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, input_offsets, send_sizes,
          jnp.array([[0, 1, 3]], dtype=jnp.int32), recv_sizes,
          axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'ragged_all_to_all output_offsets must be rank 1 with positive'
        ' dimension size, but got shape (0,)',
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, input_offsets, send_sizes,
          jnp.array([], dtype=jnp.int32), recv_sizes, axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'ragged_all_to_all recv_sizes must be rank 1 with positive dimension'
        ' size, but got shape (1, 3)',
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, input_offsets, send_sizes, output_offsets,
          jnp.array([[1, 2, 3]], dtype=jnp.int32), axis_name=axis_name)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'ragged_all_to_all recv_sizes must be rank 1 with positive dimension'
        ' size, but got shape (0,)',
    ):
      jax.jit(lax.ragged_all_to_all, static_argnames='axis_name').lower(
          operand, output, input_offsets, send_sizes, output_offsets,
          jnp.array([], dtype=jnp.int32), axis_name=axis_name)

  def test_vmap_basic(self):
    device_type = jax.devices()[0].platform
    if device_type == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest(
          'UNSUPPORTED: HLO opcode `ragged-all-to-all` is not supported by TPU'
          f' v{jtu.get_tpu_version()}'
      )
    num_devices = len(jax.devices()) # we expect either 4 or 8 total devices
    if num_devices not in (4, 8):
      raise unittest.SkipTest("test requires 4 or 8 devices")

    expert_parallelism = 2
    pipeline_parallelism = num_devices // expert_parallelism # We expect this is either 2 or 4
    batch = 2 * expert_parallelism**2
    model = 3
    axis_name = "expert"

    # Define a mesh with PP + EP
    mesh = jtu.create_mesh((expert_parallelism, pipeline_parallelism), ('expert', 'pipeline'))
    x_partition_spec = jax.sharding.PartitionSpec("expert", None)
    x_sharding = jax.sharding.NamedSharding(mesh, x_partition_spec)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            x_partition_spec,
            x_partition_spec,
            x_partition_spec,
            x_partition_spec,
            x_partition_spec,
            x_partition_spec,
            ),
        out_specs=(x_partition_spec),
    )
    def ra2a_wrapper(x, output_shape, input_offsets, send_sizes, output_offsets, recv_sizes):
        input_offsets = input_offsets.reshape(input_offsets.shape[1:])
        send_sizes = send_sizes.reshape(send_sizes.shape[1:])
        output_offsets = output_offsets.reshape(output_offsets.shape[1:])
        recv_sizes = recv_sizes.reshape(recv_sizes.shape[1:])
        return jax.lax.ragged_all_to_all(
            x, output_shape, input_offsets, send_sizes, output_offsets,
            recv_sizes, axis_name=axis_name,)

    # create an array x which is [batch, model] and has elements like
    # [[0,0,0],
    #  [1,1,1],
    #  ...
    x = jnp.arange(0.0, batch)
    x = jnp.expand_dims(x, axis=1)
    x = jnp.tile(x, (1, model))
    x = jax.device_put(x, x_sharding)

    output_shape = x.copy()

    input_offsets = jnp.array([[0, 2],[0,2]], dtype=jnp.int32)
    input_offsets = jax.device_put(input_offsets, x_sharding)

    send_sizes = jnp.array([[2, 2],[2,2]], dtype=jnp.int32)
    send_sizes = jax.device_put(send_sizes, x_sharding)

    output_offsets = jnp.array([[0, 0],[2,2]], dtype=jnp.int32)
    output_offsets = jax.device_put(output_offsets, x_sharding)

    recv_sizes = jnp.array([[2, 2],[2,2]], dtype=jnp.int32)
    recv_sizes = jax.device_put(recv_sizes, x_sharding)

    expected_array = jnp.array([[0,0,0],[1,1,1],[4,4,4],[5,5,5],[2,2,2],[3,3,3],[6,6,6],[7,7,7]], dtype=jnp.int32)

    ##### Non-vmap #####
    jit_wrapper = jax.jit(ra2a_wrapper)
    x_a2a = jit_wrapper(x, output_shape, input_offsets, send_sizes, output_offsets, recv_sizes)
    self.assertEqual(x_a2a.shape, (batch, model))
    self.assertAllClose(x_a2a, expected_array, check_dtypes=False)


    #### Vmap #####
    vmap_func = jax.vmap(
        ra2a_wrapper,
    )
    jit_vmap_func = jax.jit(vmap_func)

    vmap_sharding = jax.sharding.NamedSharding(mesh, jax.P("pipeline", "expert", None))

    def expand_array_for_vmap(arr):
      arr = jnp.expand_dims(arr, axis=0)
      arr = jnp.tile(arr, (pipeline_parallelism, 1, 1))
      arr = jax.device_put(arr, vmap_sharding)
      return arr

    x_vmap = expand_array_for_vmap(x)
    output_shape_vmap = expand_array_for_vmap(output_shape)
    input_offsets_vmap = expand_array_for_vmap(input_offsets)
    send_sizes_vmap = expand_array_for_vmap(send_sizes)
    output_offsets_vmap = expand_array_for_vmap(output_offsets)
    recv_sizes_vmap = expand_array_for_vmap(recv_sizes)

    vmap_output = jit_vmap_func(
        x_vmap, output_shape_vmap, input_offsets_vmap, send_sizes_vmap,
        output_offsets_vmap, recv_sizes_vmap)
    self.assertEqual(vmap_output.shape, (pipeline_parallelism, batch, model))
    for i in range(pipeline_parallelism):
      self.assertAllClose(vmap_output[i,:,:], expected_array, check_dtypes=False)

  def test_ragged_all_to_all_vma_invariant_output_buffer(self):
    if jax.devices()[0].platform == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest('ragged-all-to-all requires TPU v4+')
    mesh = jtu.create_mesh((2,), ('x',))
    io = oo = jnp.zeros((2, 2), jnp.int32)
    ss = rs = jnp.array([[0, 1], [1, 0]], jnp.int32)
    x = jnp.array([[1.0], [2.0]])
    w = jnp.array([[10.0], [20.0]])

    def body(x, io, ss, oo, rs):
      out = jnp.zeros((1,), dtype=x.dtype)
      y = lax.ragged_all_to_all(x[0], out, io[0], ss[0], oo[0], rs[0], axis_name='x')
      self.assertEqual(jax.typeof(y).mat.varying, frozenset({'x'}))
      return y[None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P('x'), out_specs=P('x')))
    ref = lambda x: x[::-1]
    self.assertAllClose(f(x, io, ss, oo, rs), ref(x))
    got = jax.grad(lambda x: jnp.sum(f(x, io, ss, oo, rs) * w))(x)
    want = jax.grad(lambda x: jnp.sum(ref(x) * w))(x)
    self.assertAllClose(got, want)

    with self.assertRaises(ValueError):
      jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P('x'), out_specs=P())).lower(
          x, io, ss, oo, rs)

  def test_ragged_all_to_all_vma_offsets_vary_over_second_axis(self):
    if jax.devices()[0].platform == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest('ragged-all-to-all requires TPU v4+')
    if len(jax.devices()) < 4:
      raise unittest.SkipTest('test requires at least 4 devices')
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    io = oo = jnp.zeros((2, 2, 2), jnp.int32)
    ss = rs = jnp.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], jnp.int32)
    x = jnp.array([[1.0], [2.0]])
    out = jnp.zeros((2, 1))
    w = jnp.array([[[1.0], [2.0]], [[4.0], [8.0]]])

    def smap(out_specs, lead):
      def body(x, out, io, ss, oo, rs):
        y = lax.ragged_all_to_all(x[0], out[0], io[0, 0], ss[0, 0], oo[0, 0], rs[0, 0], axis_name='x')
        self.assertEqual(jax.typeof(y).mat.varying, frozenset({'x', 'y'}))
        return y.reshape((1,) * lead + y.shape)
      return jax.jit(jax.shard_map(
          body, mesh=mesh, in_specs=(P('x'), P('x')) + (P('x', 'y'),) * 4,
          out_specs=out_specs))

    f = smap(P('x', 'y'), 2)
    ref = lambda x: jnp.stack([x[::-1], x], axis=1)
    self.assertAllClose(f(x, out, io, ss, oo, rs), ref(x))
    got = jax.grad(lambda x: jnp.sum(f(x, out, io, ss, oo, rs) * w))(x)
    want = jax.grad(lambda x: jnp.sum(ref(x) * w))(x)
    self.assertAllClose(got, want)

    with self.assertRaises(ValueError):
      smap(P('x'), 1).lower(x, out, io, ss, oo, rs)

  def test_ragged_all_to_all_vma_replicated_operand(self):
    if jax.devices()[0].platform == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest('ragged-all-to-all requires TPU v4+')
    mesh = jtu.create_mesh((2,), ('x',))
    io = jnp.array([[0, 0], [1, 0]], jnp.int32)
    oo = jnp.zeros((2, 2), jnp.int32)
    ss = rs = jnp.array([[0, 1], [1, 0]], jnp.int32)
    x = jnp.array([[1.0], [2.0]])
    out = jnp.zeros((2, 1))
    w = jnp.array([[10.0], [20.0]])

    f = jax.jit(jax.shard_map(
        lambda x, out, io, ss, oo, rs: lax.ragged_all_to_all(
            x, out, io[0], ss[0], oo[0], rs[0], axis_name='x'),
        mesh=mesh, in_specs=(P(), P('x'), P('x'), P('x'), P('x'), P('x')),
        out_specs=P('x')))
    ref = lambda x: x[::-1]
    self.assertAllClose(f(x, out, io, ss, oo, rs), ref(x))
    got = jax.grad(lambda x: jnp.sum(f(x, out, io, ss, oo, rs) * w))(x)
    want = jax.grad(lambda x: jnp.sum(ref(x) * w))(x)
    self.assertAllClose(got, want)

  def test_ragged_all_to_all_transpose_zero_size_update(self):
    if jax.devices()[0].platform == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest('ragged-all-to-all requires TPU v4+')
    mesh = jtu.create_mesh((2,), ('x',))
    io = jnp.array([[0, 0], [0, 0]], jnp.int32)
    ss = jnp.array([[0, 0], [2, 0]], jnp.int32)
    oo = jnp.array([[0, 0], [0, 0]], jnp.int32)
    rs = jnp.array([[0, 2], [0, 0]], jnp.int32)
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = jnp.zeros((2, 2))
    w = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    @jax.shard_map(mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def ra2a(x, out, io, ss, oo, rs):
      return lax.ragged_all_to_all(x[0], out[0], io[0], ss[0], oo[0], rs[0], axis_name='x')[None]

    ref = lambda x, out: out.at[0, 0:2].set(x[1, 0:2])
    self.assertAllClose(ra2a(x, out, io, ss, oo, rs), ref(x, out))
    got = jax.grad(lambda out: jnp.sum(ra2a(x, out, io, ss, oo, rs) * w))(out)
    want = jax.grad(lambda out: jnp.sum(ref(x, out) * w))(out)
    self.assertAllClose(got, want)

  def test_ragged_all_to_all_transpose_axis_index_groups(self):
    if jax.devices()[0].platform == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest('ragged-all-to-all requires TPU v4+')
    if len(jax.devices()) < 4:
      raise unittest.SkipTest('test requires at least 4 devices')
    mesh = jtu.create_mesh((4,), ('x',))
    groups = ((0, 1), (2, 3))
    x = jnp.arange(8.0).reshape(4, 2)
    out = jnp.zeros((4, 2))
    w = jnp.arange(1.0, 9.0).reshape(4, 2)
    ref = lambda x, out: out.at[0, 1].set(x[1, 1])

    for K in (2, 4):
      io = jnp.zeros((4, K), jnp.int32).at[1, 0].set(1)
      ss = jnp.zeros((4, K), jnp.int32).at[1, 0].set(1)
      oo = jnp.zeros((4, K), jnp.int32).at[1, 0].set(1)
      rs = jnp.zeros((4, K), jnp.int32).at[0, K // 2].set(1)

      @jax.shard_map(mesh=mesh, in_specs=P('x'), out_specs=P('x'))
      def ra2a(x, out, io, ss, oo, rs):
        return lax.ragged_all_to_all(
            x[0], out[0], io[0], ss[0], oo[0], rs[0], axis_name='x',
            axis_index_groups=groups)[None]

      self.assertAllClose(ra2a(x, out, io, ss, oo, rs), ref(x, out))
      got_x, got_out = jax.grad(
          lambda x, out: jnp.sum(ra2a(x, out, io, ss, oo, rs) * w),
          argnums=(0, 1))(x, out)
      want_x, want_out = jax.grad(
          lambda x, out: jnp.sum(ref(x, out) * w),
          argnums=(0, 1))(x, out)
      self.assertAllClose(got_x, want_x)
      self.assertAllClose(got_out, want_out)

  def test_ragged_all_to_all_vmap_unbatched_offsets_and_jacobians(self):
    if jax.devices()[0].platform == 'tpu' and jtu.get_tpu_version() < 4:
      raise unittest.SkipTest('ragged-all-to-all requires TPU v4+')
    mesh = jtu.create_mesh((2,), ('x',))
    io = oo = jnp.zeros((2, 2), jnp.int32)
    ss = rs = jnp.array([[0, 1], [1, 0]], jnp.int32)
    x_batched = jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    out_batched = jnp.zeros((2, 2, 1))

    @jax.shard_map(mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def ra2a_vmap(x, out, io, ss, oo, rs):
      f = lambda x, out: lax.ragged_all_to_all(x, out, io[0], ss[0], oo[0], rs[0], axis_name='x')
      return jax.vmap(f)(x[0], out[0])[None]

    self.assertAllClose(ra2a_vmap(x_batched, out_batched, io, ss, oo, rs), x_batched[::-1])

    x = jnp.array([[1.0], [2.0]])
    out = jnp.zeros((2, 1))
    w = jnp.array([[10.0], [20.0]])

    @jax.shard_map(mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def ra2a(x, out, io, ss, oo, rs):
      return lax.ragged_all_to_all(x[0], out[0], io[0], ss[0], oo[0], rs[0], axis_name='x')[None]

    loss = lambda x: jnp.sum(ra2a(x, out, io, ss, oo, rs) * w)
    want_grad = jax.grad(lambda x: jnp.sum(x[::-1] * w))(x)
    self.assertAllClose(jax.jacfwd(loss)(x), want_grad)
    self.assertAllClose(jax.jacrev(loss)(x), want_grad)
    self.assertAllClose(jax.hessian(loss)(x), jnp.zeros(x.shape + x.shape))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
