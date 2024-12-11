# Copyright 2024 The JAX Authors.
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

"""Tests the simple ragged all-to-all kernel."""

import functools

from absl.testing import absltest

import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu import ragged_all_to_all
import jax.numpy as jnp
import numpy.testing as npt

jax.config.parse_flags_with_absl()
P = jax.sharding.PartitionSpec


class RaggedAllToAllTest(jtu.JaxTestCase):
  def test_ragged_all_to_all(self):
    num_devices = jax.device_count()
    if num_devices < 4:
      self.skipTest("Need at least 4 devices, found {}".format(num_devices))

    if num_devices >= 4:
      num_devices = 4
    mesh = jax.make_mesh(
        (num_devices,), ("x",), devices=jax.devices()[:num_devices])

    x = jnp.array([
      0., 0., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7., 7.,
      0., 1., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7., -1.,
      0., 0., 0., 1., 2., 3., 3., 3., 4., 4., 5., 5., 6., 6., -1., -1.,
      0., 0., 1., 1., 2., 2., 2., 3., 4., 4., 5., 5., 6., -1., -1., -1.,
    ])
    input_offsets = jnp.array([
      [0, 2, 4, 6, 8, 10, 12, 14],
      [0, 1, 4, 6, 8, 10, 12, 14],
      [0, 3, 4, 5, 8, 10, 12, 14],
      [0, 2, 4, 7, 8, 10, 12, 13],
    ])
    input_sizes = jnp.array([
      [2, 2, 2, 2, 2, 2, 2, 2],
      [1, 3, 2, 2, 2, 2, 2, 1],
      [3, 1, 1, 3, 2, 2, 2, 0],
      [2, 2, 3, 1, 2, 2, 1, 0],
    ])
    expected_output = jnp.array([
      0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
      2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
      4., 4., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 5., 5., 5., 5.,
      6., 6., 6., 6., 6., 6., 6., 7., 7., 7., 0., 0., 0., 0., 0., 0.,
    ])

    output = jax.jit(jax.experimental.shard_map.shard_map(
      functools.partial(ragged_all_to_all.ragged_all_to_all,
                        mesh=mesh, axis_name="x"),
      mesh=mesh,
      in_specs=(P("x"), None, None, P("x")),
      out_specs=P("x"),
      check_rep=False
    ))(
        jnp.tile(x[:, jnp.newaxis, jnp.newaxis], (1, 1, 128)),
        input_offsets,
        input_sizes,
        jnp.zeros((x.shape[0], 1, 128))
    )[..., 0, 0]

    npt.assert_array_equal(output, expected_output)

    output2 = jax.jit(jax.experimental.shard_map.shard_map(
      functools.partial(ragged_all_to_all.ragged_all_to_all,
                        mesh=mesh, axis_name="x", transpose=True),
      mesh=mesh,
      in_specs=(P("x"), None, None, P("x")),
      out_specs=P("x"),
      check_rep=False
    ))(
        jnp.tile(output[:, jnp.newaxis, jnp.newaxis], (1, 1, 128)),
        input_offsets,
        input_sizes,
        jnp.full((x.shape[0], 1, 128), -1., output.dtype)
    )[..., 0, 0]

    npt.assert_array_equal(output2, x)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
