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
    if jtu.test_device_matches(['tpu']) and jtu.get_tpu_version() < 4:
      self.skipTest('UNSUPPORTED: HLO opcode `ragged-all-to-all` is not '
                    f'supported by TPU v{jtu.get_tpu_version()}')

  @parameterized.product(axis_name=['x'], mesh_axes=[dict(x=2)],
                         check_vma=[True, False])
  def test_ragged_all_to_all(self, axis_name, mesh_axes, check_vma):
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
        check_vma=check_vma,
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

  @parameterized.product(axis_name=['x'], mesh_axes=[dict(x=2)],
                         check_vma=[True, False])
  def test_ragged_all_to_all_grad(self, axis_name, mesh_axes, check_vma):
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
        check_vma=check_vma,
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
    jtu.check_grads(lambda op: fwd(op, output, *args), (operand,), order=1)
    jtu.check_grads(lambda out: fwd(operand, out, *args), (output,), order=1)

    # The cotangent of output alone doesn't need a reverse ragged_all_to_all.
    y, f_vjp = jax.vjp(lambda out: fwd(operand, out, *args), output)
    self.assertNotIn('ragged_all_to_all', str(jax.make_jaxpr(f_vjp)(y)))

  @parameterized.product(check_vma=[True, False])
  def test_ragged_all_to_all_grad_empty_updates(self, check_vma):
    mesh = jtu.create_mesh((2,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'),
             check_vma=check_vma)
    def fwd(operand, output, input_offsets, send_sizes, output_offsets,
            recv_sizes):
      return lax.ragged_all_to_all(
          operand[0], output[0], input_offsets[0], send_sizes[0],
          output_offsets[0], recv_sizes[0], axis_name='x')[None]

    # Device 1 sends both its rows to device 0, and the other three updates are
    # empty. All four updates start at row 0 of their receivers.
    operand = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    output = jnp.array([[-1, -2], [-3, -4]], dtype=jnp.float32)
    input_offsets = jnp.array([[0, 0], [0, 0]], dtype=jnp.int32)
    send_sizes = jnp.array([[0, 0], [2, 0]], dtype=jnp.int32)
    output_offsets = jnp.array([[0, 0], [0, 0]], dtype=jnp.int32)
    recv_sizes = jnp.array([[0, 2], [0, 0]], dtype=jnp.int32)
    args = input_offsets, send_sizes, output_offsets, recv_sizes

    self.assertAllClose(fwd(operand, output, *args),
                        jnp.array([[3, 4], [-3, -4]], dtype=jnp.float32))
    jtu.check_grads(lambda op, out: fwd(op, out, *args), (operand, output),
                    order=1)

  @parameterized.product(updates_per_peer=[1, 2], check_vma=[True, False])
  def test_ragged_all_to_all_grad_axis_index_groups(self, updates_per_peer,
                                                    check_vma):
    mesh = jtu.create_mesh((4,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'),
             check_vma=check_vma)
    def fwd(operand, output, input_offsets, send_sizes, output_offsets,
            recv_sizes):
      return lax.ragged_all_to_all(
          operand[0], output[0], input_offsets[0], send_sizes[0],
          output_offsets[0], recv_sizes[0], axis_name='x',
          axis_index_groups=((0, 1), (2, 3)))[None]

    # In each group, slot k of the device at position p sends operand row k to
    # position k // n, which writes it at row p * n + k % n, where n is
    # updates_per_peer. That is a tiled all_to_all within each group.
    n = updates_per_peer
    position = jnp.array([[0], [1], [0], [1]], dtype=jnp.int32)
    k = jnp.arange(2 * n, dtype=jnp.int32)
    input_offsets = jnp.broadcast_to(k, (4, 2 * n))
    send_sizes = recv_sizes = jnp.ones((4, 2 * n), dtype=jnp.int32)
    output_offsets = position * n + k[None] % n
    args = input_offsets, send_sizes, output_offsets, recv_sizes
    f = lambda operand, output: fwd(operand, output, *args)
    def f_ref(operand, output):
      del output  # every row is overwritten
      return operand.reshape(2, 2, 2, n).transpose(0, 2, 1, 3).reshape(4, 2 * n)

    operand = jnp.linspace(0, 1, 8 * n, dtype=jnp.float32).reshape(4, 2 * n)
    output = jnp.zeros((4, 2 * n), dtype=jnp.float32)
    self.assertAllClose(f(operand, output), f_ref(operand, output))
    jtu.check_grads(f, (operand, output), order=1)
    for jac in [jax.jacfwd, jax.jacrev]:
      self.assertAllClose(jac(f, (0, 1))(operand, output),
                          jac(f_ref, (0, 1))(operand, output))

  @parameterized.product(axis_name=['x'], mesh_axes=[dict(x=4)],
                         check_vma=[True, False])
  def test_ragged_all_to_all_axis_index_groups(self, axis_name, mesh_axes,
                                               check_vma):
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
        check_vma=check_vma,
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

  @parameterized.product(axis_name=['x'], mesh_axes=[dict(x=2)],
                         check_vma=[True, False])
  def test_ragged_all_to_all_degenerate_groups(self, axis_name, mesh_axes,
                                               check_vma):
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
        check_vma=check_vma,
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

  @parameterized.product(check_vma=[True, False])
  def test_ragged_all_to_all_vmap_multi_dim_operand(self, check_vma):
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
        check_vma=check_vma,
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

  @parameterized.product(
      [dict(vmap_batch_axis=0, data_shard_axis=0, input_config=0),
       dict(vmap_batch_axis=0, data_shard_axis=1, input_config=0),
       dict(vmap_batch_axis=1, data_shard_axis=0, input_config=1),
       dict(vmap_batch_axis=1, data_shard_axis=1, input_config=1)],
      axis_name=['x'], vmap_axis_name=['y'], mesh_axes=[dict(x=2, y=2)],
      check_vma=[True, False],
  )
  def test_ragged_all_to_all_vmap(
      self,
      axis_name,
      vmap_axis_name,
      mesh_axes,
      vmap_batch_axis,
      data_shard_axis,
      input_config,
      check_vma,
  ):
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
        check_vma=check_vma,
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

  @parameterized.product(check_vma=[True, False])
  def test_ragged_all_to_all_vmap_axis_index_groups(self, check_vma):
    mesh = jtu.create_mesh((4,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'),
             check_vma=check_vma)
    def fwd(operand, output, input_offsets, send_sizes, output_offsets,
            recv_sizes):
      return lax.ragged_all_to_all(
          operand[0], output[0], input_offsets[0], send_sizes[0],
          output_offsets[0], recv_sizes[0], axis_name='x',
          axis_index_groups=((0, 1), (2, 3)))[None]

    # The docstring example within each group, for a batch of two operands.
    operand = jnp.array([[1, 2, 2], [3, 4, 0], [10, 20, 20], [30, 40, 0]],
                        dtype=jnp.int32)
    output = jnp.zeros((4, 4), dtype=jnp.int32)
    input_offsets = jnp.array([[0, 1]] * 4, dtype=jnp.int32)
    send_sizes = jnp.array([[1, 2], [1, 1]] * 2, dtype=jnp.int32)
    output_offsets = jnp.array([[0, 0], [1, 2]] * 2, dtype=jnp.int32)
    recv_sizes = jnp.array([[1, 1], [2, 1]] * 2, dtype=jnp.int32)
    args = output, input_offsets, send_sizes, output_offsets, recv_sizes

    res = vmap(fwd)(jnp.stack([operand, 2 * operand]),
                    *(jnp.stack([x, x]) for x in args))
    expected = jnp.array([[1, 3, 0, 0], [2, 2, 4, 0],
                          [10, 30, 0, 0], [20, 20, 40, 0]], dtype=jnp.int32)
    self.assertAllClose(res, jnp.stack([expected, 2 * expected]))

  @parameterized.product(check_vma=[True, False])
  def test_ragged_all_to_all_vmap_unbatched_offsets(self, check_vma):
    mesh = jtu.create_mesh((2,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'),
             check_vma=check_vma)
    def fwd(operand, output, input_offsets, send_sizes, output_offsets,
            recv_sizes):
      return lax.ragged_all_to_all(
          operand[0], output[0], input_offsets[0], send_sizes[0],
          output_offsets[0], recv_sizes[0], axis_name='x')[None]

    # A batch of two operands, sent with the docstring example's offsets and
    # sizes, which are not batched.
    operand = jnp.array([[[1, 2, 2], [3, 4, 0]],
                         [[5, 6, 6], [7, 8, 0]]], dtype=jnp.int32)
    output = jnp.zeros((2, 4), dtype=jnp.int32)
    input_offsets = jnp.array([[0, 1], [0, 1]], dtype=jnp.int32)
    send_sizes = jnp.array([[1, 2], [1, 1]], dtype=jnp.int32)
    output_offsets = jnp.array([[0, 0], [1, 2]], dtype=jnp.int32)
    recv_sizes = jnp.array([[1, 1], [2, 1]], dtype=jnp.int32)

    res = vmap(fwd, in_axes=(0, None, None, None, None, None))(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes)
    self.assertAllClose(
        res, jnp.array([[[1, 3, 0, 0], [2, 2, 4, 0]],
                        [[5, 7, 0, 0], [6, 6, 8, 0]]], dtype=jnp.int32))

  @parameterized.product(check_vma=[True, False])
  def test_ragged_all_to_all_jacfwd_jacrev(self, check_vma):
    mesh = jtu.create_mesh((2,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'),
             check_vma=check_vma)
    def fwd(operand, output, input_offsets, send_sizes, output_offsets,
            recv_sizes):
      return lax.ragged_all_to_all(
          operand[0], output[0], input_offsets[0], send_sizes[0],
          output_offsets[0], recv_sizes[0], axis_name='x')[None]

    # The docstring example. jacfwd and jacrev batch the tangents or
    # cotangents, but not the offsets and sizes.
    input_offsets = jnp.array([[0, 1], [0, 1]], dtype=jnp.int32)
    send_sizes = jnp.array([[1, 2], [1, 1]], dtype=jnp.int32)
    output_offsets = jnp.array([[0, 0], [1, 2]], dtype=jnp.int32)
    recv_sizes = jnp.array([[1, 1], [2, 1]], dtype=jnp.int32)
    f = lambda operand, output: fwd(operand, output, input_offsets, send_sizes,
                                    output_offsets, recv_sizes)
    def f_ref(operand, output):
      return (output.at[0, 0].set(operand[0, 0]).at[0, 1].set(operand[1, 0])
              .at[1, :2].set(operand[0, 1:]).at[1, 2].set(operand[1, 1]))

    operand = jnp.array([[1, 2, 2], [3, 4, 0]], dtype=jnp.float32)
    output = jnp.zeros((2, 4), dtype=jnp.float32)
    self.assertAllClose(f(operand, output), f_ref(operand, output))
    for jac in [jax.jacfwd, jax.jacrev]:
      self.assertAllClose(jac(f, (0, 1))(operand, output),
                          jac(f_ref, (0, 1))(operand, output))
    hess = lambda f: jax.hessian(lambda x: (f(x, output) ** 2).sum())(operand)
    self.assertAllClose(hess(f), hess(f_ref))

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

  @parameterized.product(check_vma=[True, False])
  def test_vmap_basic(self, check_vma):
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
        check_vma=check_vma,
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

  def test_ragged_all_to_all_vma_invariant_output(self):
    mesh = jtu.create_mesh((2,), ('x',))

    def f(operand, input_offsets, send_sizes, output_offsets, recv_sizes):
      output = jnp.zeros((1,), dtype=operand.dtype)  # device-invariant
      result = lax.ragged_all_to_all(
          operand[0], output, input_offsets[0], send_sizes[0],
          output_offsets[0], recv_sizes[0], axis_name='x')
      self.assertEqual(jax.typeof(result).mat.varying, {'x'})
      return result[None]

    # Each device sends its row to the other device.
    input_offsets = output_offsets = jnp.zeros((2, 2), dtype=jnp.int32)
    send_sizes = recv_sizes = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    args = input_offsets, send_sizes, output_offsets, recv_sizes
    operand = jnp.array([[1.], [2.]], dtype=jnp.float32)

    with self.assertRaisesRegex(ValueError, 'require replication'):
      shard_map(f, mesh=mesh, in_specs=P('x'), out_specs=P())(operand, *args)
    g = shard_map(f, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    self.assertAllClose(g(operand, *args), operand[::-1])
    w = jnp.array([[10.], [20.]], dtype=jnp.float32)
    grad = jax.grad(lambda x: (g(x, *args) * w).sum())(operand)
    self.assertAllClose(grad, w[::-1])
    _, y_dot = jax.jvp(lambda x: g(x, *args), (operand,), (w,))
    self.assertAllClose(y_dot, w[::-1])

  def test_ragged_all_to_all_vma_offsets_vary_over_other_axis(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))

    def f(operand, output, input_offsets, send_sizes, output_offsets,
          recv_sizes):
      result = lax.ragged_all_to_all(
          operand[0], output[0], input_offsets[0, 0], send_sizes[0, 0],
          output_offsets[0, 0], recv_sizes[0, 0], axis_name='x')
      self.assertEqual(jax.typeof(result).mat.varying, {'x', 'y'})
      return result[None, None]

    # The operand and output vary only over 'x', and the offsets and sizes also
    # vary over 'y': at y=0 the devices along 'x' swap their rows, and at y=1
    # each one keeps its own row.
    input_offsets = output_offsets = jnp.zeros((2, 2, 2), dtype=jnp.int32)
    send_sizes = recv_sizes = jnp.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]],
                                        dtype=jnp.int32)
    args = input_offsets, send_sizes, output_offsets, recv_sizes
    in_specs = (P('x'), P('x')) + (P('x', 'y'),) * 4
    operand = jnp.array([[1.], [2.]], dtype=jnp.float32)
    output = jnp.zeros((2, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(ValueError, 'require replication'):
      shard_map(lambda *args: f(*args)[:, 0], mesh=mesh, in_specs=in_specs,
                out_specs=P('x'))(operand, output, *args)
    g = shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=P('x', 'y'))
    swap_then_keep = lambda x: jnp.stack([x[::-1], x], axis=1)
    self.assertAllClose(g(operand, output, *args), swap_then_keep(operand))
    w = jnp.array([[[1.], [2.]], [[4.], [8.]]], dtype=jnp.float32)
    grad = jax.grad(lambda x: (g(x, output, *args) * w).sum())(operand)
    self.assertAllClose(grad, jnp.array([[6.], [9.]], dtype=jnp.float32))
    t = jnp.array([[3.], [5.]], dtype=jnp.float32)
    _, y_dot = jax.jvp(lambda x: g(x, output, *args), (operand,), (t,))
    self.assertAllClose(y_dot, swap_then_keep(t))

  def test_ragged_all_to_all_vma_replicated_operand(self):
    mesh = jtu.create_mesh((2,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=(P(),) + (P('x'),) * 5,
             out_specs=P('x'))
    def f(operand, output, input_offsets, send_sizes, output_offsets,
          recv_sizes):
      return lax.ragged_all_to_all(
          operand, output[0], input_offsets[0], send_sizes[0],
          output_offsets[0], recv_sizes[0], axis_name='x')[None]

    # Both devices hold the same operand. Device 0 sends its row 0 to device 1,
    # and device 1 sends its row 1 to device 0.
    input_offsets = jnp.array([[0, 0], [1, 0]], dtype=jnp.int32)
    output_offsets = jnp.zeros((2, 2), dtype=jnp.int32)
    send_sizes = recv_sizes = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    args = input_offsets, send_sizes, output_offsets, recv_sizes
    operand = jnp.array([1., 2.], dtype=jnp.float32)
    output = jnp.zeros((2, 1), dtype=jnp.float32)

    self.assertAllClose(f(operand, output, *args), operand[::-1, None])
    w = jnp.array([[10.], [20.]], dtype=jnp.float32)
    grad = jax.grad(lambda x: (f(x, output, *args) * w).sum())(operand)
    self.assertAllClose(grad, w[::-1, 0])
    t = jnp.array([3., 5.], dtype=jnp.float32)
    _, y_dot = jax.jvp(lambda x: f(x, output, *args), (operand,), (t,))
    self.assertAllClose(y_dot, t[::-1, None])

  @parameterized.product(arg=['operand', 'output'],
                         kind=['unreduced', 'reduced'])
  @jtu.with_explicit_mesh((2,), ('x',))
  def test_ragged_all_to_all_unreduced_and_reduced_args(self, arg, kind, mesh):
    cast = {
        'unreduced': lambda x: lax.pcast(x, 'x', to='unreduced'),
        'reduced': lambda x: lax.pcast(jnp.zeros_like(x), 'x', to='reduced'),
    }[kind]

    def f(x, input_offsets, send_sizes, output_offsets, recv_sizes):
      operand = cast(x[0]) if arg == 'operand' else x[0]
      output = cast(x[0]) if arg == 'output' else x[0]
      return lax.ragged_all_to_all(
          operand, output, input_offsets[0], send_sizes[0], output_offsets[0],
          recv_sizes[0], axis_name='x')[None]

    x = jnp.array([[1.], [2.]], dtype=jnp.float32)
    offsets = jnp.zeros((2, 2), dtype=jnp.int32)
    sizes = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    args = [jax.device_put(a, P('x')) for a in (x, offsets, sizes, offsets,
                                                sizes)]

    # With auto_pcast, the pvary that ragged_all_to_all inserts rejects these.
    msg = f'ragged_all_to_all cannot accept args which are {kind}'
    with config.auto_pcast(False), self.assertRaisesRegex(ValueError, msg):
      jax.make_jaxpr(shard_map(f, in_specs=P('x'), out_specs=P('x')))(*args)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
