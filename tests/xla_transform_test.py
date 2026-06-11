# Copyright 2026 The JAX Authors.
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

"""Tests for jax.extend.xla compiler pass registration."""

from __future__ import annotations

import functools
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import lib
from jax._src import test_util as jtu
from jax._src.lib import hlo as _hlo
import jax.extend.xla as jex_xla
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


@jtu.thread_unsafe_test_class()
@unittest.skipIf(
    lib.jaxlib_extension_version < 461,
    "Requires jaxlib_extension_version >= 461",
)
class XlaTransformTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "cpu":
      self.skipTest("Skipping test for non-CPU devices")
    self._registered_transforms = []

  def tearDown(self):
    for name, stage in self._registered_transforms:
      jex_xla.clear_hlo_module_transformation(name, stage)
    super().tearDown()

  @parameterized.parameters(
      jex_xla.PipelineStage.PRE_SCHEDULER,
      jex_xla.PipelineStage.POST_SCHEDULER,
  )
  def test_sin_to_cos(self, stage):
    """Register a pass that replaces sin ops with cos ops."""

    def sin_to_cos(serialized_hlo: bytes) -> bytes | None:
      """Replace all sin ops with cos ops in an HLO module."""
      module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
      changed = False

      for comp in module.computations():
        for inst in comp.instructions():
          if inst.opcode == _hlo.HloOpcode.kSin:
            operand = inst.operands()[0]
            new_inst = comp.create_unary_instruction(
                _hlo.HloOpcode.kCos, operand
            )
            comp.replace_instruction(inst, new_inst)
            changed = True

      if not changed:
        return None

      return module.as_serialized_hlo_module_proto()

    name = f"sin_to_cos_{stage.name}_test"
    jex_xla.register_hlo_module_transformation(
        sin_to_cos,
        name=name,
        stage=stage,
    )
    self._registered_transforms.append((name, stage))

    @jax.jit
    def f(x):
      return jnp.sin(x)

    x = jnp.array([0.0, 1.0, 2.0])
    result = f(x)
    expected = jnp.cos(x)
    self.assertAllClose(result, expected, atol=1e-5)

  def test_sin_to_cos_platform_filtering(self):
    """Register a pass for tpu only, compiling on cpu should not apply it."""

    def sin_to_cos(serialized_hlo: bytes) -> bytes | None:
      module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
      changed = False

      for comp in module.computations():
        for inst in comp.instructions():
          if inst.opcode == _hlo.HloOpcode.kSin:
            operand = inst.operands()[0]
            new_inst = comp.create_unary_instruction(
                _hlo.HloOpcode.kCos, operand
            )
            comp.replace_instruction(inst, new_inst)
            changed = True

      if not changed:
        return None

      return module.as_serialized_hlo_module_proto()

    # Registering for "tpu" only. Since our test runs on "cpu", this pass
    # should NOT be executed.
    name = "sin_to_cos_tpu_only_test"
    stage = jex_xla.PipelineStage.PRE_SCHEDULER
    jex_xla.register_hlo_module_transformation(
        sin_to_cos,
        name=name,
        stage=stage,
        platforms="tpu",
    )
    self._registered_transforms.append((name, stage))

    @jax.jit
    def f(x):
      return jnp.sin(x)

    x = jnp.array([0.0, 1.0, 2.0])
    result = f(x)
    # Since it's compiled on CPU, it should still return sin(x), not cos(x).
    expected = jnp.sin(x)
    self.assertAllClose(result, expected, atol=1e-5)

  def test_clear_transform(self):
    """Register a pass, verify it applies, clear it, verify it doesn't apply."""

    def sin_to_cos(serialized_hlo: bytes) -> bytes | None:
      module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
      changed = False

      for comp in module.computations():
        for inst in comp.instructions():
          if inst.opcode == _hlo.HloOpcode.kSin:
            operand = inst.operands()[0]
            new_inst = comp.create_unary_instruction(
                _hlo.HloOpcode.kCos, operand
            )
            comp.replace_instruction(inst, new_inst)
            changed = True

      if not changed:
        return None

      return module.as_serialized_hlo_module_proto()

    name = "sin_to_cos_clear_test"
    stage = jex_xla.PipelineStage.PRE_SCHEDULER

    jex_xla.register_hlo_module_transformation(
        sin_to_cos,
        name=name,
        stage=stage,
    )
    self._registered_transforms.append((name, stage))

    @jax.jit
    def f(x):
      return jnp.sin(x)

    x = jnp.array([0.0, 1.0, 2.0])

    # 1. Verify it applies.
    self.assertAllClose(f(x), jnp.cos(x), atol=1e-5)

    # 2. Clear it.
    cleared = jex_xla.clear_hlo_module_transformation(name, stage)
    self.assertTrue(cleared)
    self._registered_transforms.remove((name, stage))

    # Clear JIT cache to force recompilation.
    f.clear_cache()

    # 3. Verify it does NOT apply anymore.
    self.assertAllClose(f(x), jnp.sin(x), atol=1e-5)


def layernorm(x, eps=1e-5):
  mean = jnp.mean(x, axis=-1, keepdims=True)
  var = jnp.var(x, axis=-1, keepdims=True)
  return (x - mean) / jnp.sqrt(var + eps)


def mlp_layer(x, params, mesh, activation):
  w0, b0, w1, b1 = params
  in_features = w0.shape[0]

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          jax.sharding.PartitionSpec("x", None, None),  # x
          jax.sharding.PartitionSpec(None, "x"),  # w0
          jax.sharding.PartitionSpec("x"),  # b0
          jax.sharding.PartitionSpec("x", None),  # w1
          jax.sharding.PartitionSpec(),  # b1
      ),
      out_specs=jax.sharding.PartitionSpec("x", None, None),
  )
  def mlp_shard_map(x_shard, w0_shard, b0_shard, w1_shard, b1):
    mesh_size = mesh.shape["x"]
    hidden_features = w0_shard.shape[1] * mesh_size

    w0_gathered = jax.lax.all_gather(w0_shard, "x", axis=1)
    w0_gathered = w0_gathered.reshape(in_features, hidden_features)
    b0_gathered = jax.lax.all_gather(b0_shard, "x", axis=0)
    b0_gathered = b0_gathered.reshape(hidden_features)

    h = (
        jnp.matmul(x_shard, w0_gathered)
        + b0_gathered[jnp.newaxis, jnp.newaxis, :]
    )
    h = activation(h)

    w1_gathered = jax.lax.all_gather(w1_shard, "x", axis=0)
    w1_gathered = w1_gathered.reshape(hidden_features, in_features)

    out = jnp.matmul(h, w1_gathered) + b1[jnp.newaxis, jnp.newaxis, :]
    return out

  return mlp_shard_map(x, w0, b0, w1, b1)


def sharded_mlp_model(x, params, mesh, activation):
  for layer_params in params:
    mlp_out = mlp_layer(x, layer_params, mesh, activation)
    x = x + mlp_out
    x = layernorm(x)
  return x


@jtu.thread_unsafe_test_class()
@unittest.skipIf(
    lib.jaxlib_extension_version < 461,
    "Requires jaxlib_extension_version >= 461",
)
class XlaTransformE2ETest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self._registered_transforms = []

  def tearDown(self):
    for name, stage in self._registered_transforms:
      jex_xla.clear_hlo_module_transformation(name, stage)
    super().tearDown()

  # Pre-scheduler XLA transformation: replaces all sin ops with cos.
  def sin_to_cos(self, serialized_hlo: bytes) -> bytes | None:
    module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
    changed = False

    for comp in module.computations():
      for inst in comp.instructions():
        if inst.opcode == _hlo.HloOpcode.kSin:
          operand = inst.operands()[0]
          new_inst = comp.create_unary_instruction(_hlo.HloOpcode.kCos, operand)
          comp.replace_instruction(inst, new_inst)
          changed = True

    if not changed:
      return None

    sched = module.schedule()
    if sched is not None:
      sched.update()
      module.set_schedule(sched)

    return module.as_serialized_hlo_module_proto()

  # Post-scheduler XLA transformation: custom list scheduler for async ops.
  def schedule_async_ops(self, serialized_hlo: bytes) -> bytes | None:
    module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
    schedule = module.schedule()
    assert schedule is not None

    def is_async_start(inst):
      async_start_opcodes = {
          _hlo.HloOpcode.kAsyncStart,
          _hlo.HloOpcode.kAllGatherStart,
          _hlo.HloOpcode.kAllReduceStart,
          _hlo.HloOpcode.kCollectivePermuteStart,
          _hlo.HloOpcode.kCopyStart,
          _hlo.HloOpcode.kSend,
          _hlo.HloOpcode.kRecv,
      }
      return inst.opcode in async_start_opcodes

    def is_async_done(inst):
      async_done_opcodes = {
          _hlo.HloOpcode.kAsyncDone,
          _hlo.HloOpcode.kAllGatherDone,
          _hlo.HloOpcode.kAllReduceDone,
          _hlo.HloOpcode.kCollectivePermuteDone,
          _hlo.HloOpcode.kCopyDone,
          _hlo.HloOpcode.kSendDone,
          _hlo.HloOpcode.kRecvDone,
      }
      return inst.opcode in async_done_opcodes

    def schedule_next_op(worklist):
      async_start_ops = []
      async_done_ops = []
      sync_ops = []
      for inst in worklist:
        if is_async_start(inst):
          async_start_ops.append(inst)
        elif is_async_done(inst):
          async_done_ops.append(inst)
        else:
          sync_ops.append(inst)

      if async_start_ops:
        return async_start_ops[0]
      if sync_ops:
        return sync_ops[0]
      if async_done_ops:
        return async_done_ops[0]
      return None

    for comp in module.make_nonfusion_computations():
      new_seq = []
      worklist = []
      visited = set()

      # Initialize worklist with instructions that have no operands.
      for inst in comp.instructions():
        if not inst.operands():
          worklist.append(inst)
          visited.add(inst)

      # Process worklist until empty, visiting graph nodes in post-order.
      while worklist:
        inst = schedule_next_op(worklist)
        assert inst is not None
        worklist.remove(inst)
        new_seq.append(inst)

        # Add any users of the scheduled instruction, if they are now ready.
        for user in inst.users():
          if user not in visited and all(
              op in visited for op in user.operands()
          ):
            worklist.append(user)
            visited.add(user)

      # Update the schedule for the computation.
      schedule.set_sequence(comp, new_seq)

    schedule.update()
    schedule.verify()
    module.set_schedule(schedule)
    self.assertIsNotNone(module.schedule())

    return module.as_serialized_hlo_module_proto()

  @jtu.run_on_devices("tpu")
  def test_transformer_schedule_async_ops_sharded(self):
    if not jtu.is_cloud_tpu_at_least(2026, 6, 10):
      self.skipTest("Requires newer libtpu")

    # Register the pre-scheduler transformation.
    name_pre = "schedule_async_ops_test_pre_scheduler_transformation"
    stage_pre = jex_xla.PipelineStage.PRE_SCHEDULER
    jex_xla.register_hlo_module_transformation(
        self.sin_to_cos,
        name=name_pre,
        stage=stage_pre,
        platforms="tpu",
    )
    self._registered_transforms.append((name_pre, stage_pre))

    # Register the post-scheduler transformation.
    name_post = "schedule_async_ops_test_post_scheduler_transformation"
    stage_post = jex_xla.PipelineStage.POST_SCHEDULER
    jex_xla.register_hlo_module_transformation(
        self.schedule_async_ops,
        name=name_post,
        stage=stage_post,
        platforms="tpu",
    )
    self._registered_transforms.append((name_post, stage_post))

    # Setup Mesh
    num_devices = jax.local_device_count()
    devices = np.array(jax.local_devices())
    mesh = jax.sharding.Mesh(devices, ("x",))

    # Initialize models
    key = jax.random.key(0)
    batch_size = num_devices * 2
    in_features = 8
    hidden_features = 32
    num_layers = 2

    x = jax.random.normal(
        key, (batch_size, 4, in_features)
    )  # batch, seq_len, features

    # Initialize parameters
    params = []
    for _ in range(num_layers):
      key, k0, k1 = jax.random.split(key, 3)
      w0 = jax.random.normal(k0, (in_features, hidden_features)) * 0.02
      b0 = jnp.zeros((hidden_features,))
      w1 = jax.random.normal(k1, (hidden_features, in_features)) * 0.02
      b1 = jnp.zeros((in_features,))
      params.append((w0, b0, w1, b1))

    # Shard input
    in_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x", None, None)
    )
    x_sharded = jax.device_put(x, in_sharding)

    # Shard parameters
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    w0_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, "x")
    )
    b0_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x")
    )
    w1_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x", None)
    )

    params_sharded = []
    for w0, b0, w1, b1 in params:
      w0_s = jax.device_put(w0, w0_sharding)
      b0_s = jax.device_put(b0, b0_sharding)
      w1_s = jax.device_put(w1, w1_sharding)
      b1_s = jax.device_put(b1, replicated_sharding)
      params_sharded.append((w0_s, b0_s, w1_s, b1_s))

    # Compile and run both models
    run_sin = jax.jit(
        lambda p, x: sharded_mlp_model(x, p, mesh, jnp.sin),
        out_shardings=in_sharding,
    )
    run_cos = jax.jit(
        lambda p, x: sharded_mlp_model(x, p, mesh, jnp.cos),
        out_shardings=in_sharding,
    )

    output_sin = run_sin(params_sharded, x_sharded)
    output_cos = run_cos(params_sharded, x_sharded)
    jax.block_until_ready((output_sin, output_cos))

    # Assert that the output of the Sin model (transformed to Cos)
    # is close to the output of the Cos model.
    self.assertAllClose(output_sin, output_cos, atol=1e-5, rtol=1e-5)
    self.assertEqual(output_sin.sharding, in_sharding)
    self.assertEqual(output_cos.sharding, in_sharding)


if __name__ == "__main__":
  absltest.main()
