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
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax import numpy as jnp
from jax._src import config
from jax._src import core as jax_core
from jax._src import hijax
from jax._src import test_util as jtu
from jax._src.pallas import cost_estimate
from jax._src.state import discharge


config.parse_flags_with_absl()


class PallasCostEstimateTest(jtu.JaxTestCase):

  def test_exp_add(self):
    def exp_add(x, y):
      return jnp.exp(x + y)
    cost = cost_estimate.estimate_cost(exp_add,
                                       jnp.ones(10, dtype=jnp.float32),
                                       jnp.ones(10, dtype=jnp.float32))
    self.assertEqual(cost.flops, 10)
    self.assertEqual(cost.transcendentals, 10)
    self.assertEqual(cost.bytes_accessed, 4 * 30)

  def test_very_large_matmul(self):
    def matmul(a, b):
      return a @ b
    m, k, n = 400_000, 800_000, 900_000
    cost = cost_estimate.estimate_cost(
        matmul,
        jax.ShapeDtypeStruct((m, k), jnp.bfloat16),
        jax.ShapeDtypeStruct((k, n), jnp.bfloat16))
    self.assertEqual(cost.flops, 2*m*k*n)
    self.assertEqual(cost.transcendentals, 0)
    self.assertEqual(cost.bytes_accessed, 2*(m*k + n*k + m*n))

  def test_batched_matmul(self):
    def matmul(a, b):
      return jnp.matmul(a, b)
    b, m, k, n = 7, 37, 91, 23
    cost = cost_estimate.estimate_cost(
        matmul,
        jax.ShapeDtypeStruct((b, m, k), jnp.float32),
        jax.ShapeDtypeStruct((b, k, n), jnp.float32))
    self.assertEqual(cost.flops, 2*b*m*k*n)
    self.assertEqual(cost.transcendentals, 0)
    self.assertEqual(cost.bytes_accessed, 4*(b*m*k + b*n*k + b*m*n))

  @parameterized.parameters(
      ((10, 11, 12), (11, 12), "abc,bc->a", 2640),
      ((10, 11, 12), (13, 11, 12), "abc,dbc->ad", 34320),
      ((10, 11, 12), (9, 10, 11, 12), "abc,dabc->d", 23760),
  )
  def test_einsum(self, a_shape, b_shape, pattern, expected_flops):
    def matmul(a, b):
      return jnp.einsum(pattern, a, b)
    cost = cost_estimate.estimate_cost(
        matmul,
        jax.ShapeDtypeStruct(a_shape, jnp.float32),
        jax.ShapeDtypeStruct(b_shape, jnp.float32))
    self.assertEqual(cost.flops, expected_flops)

  def test_attention(self):
    qk_dim = 16
    v_dim = 4
    kv_len = 128
    q_len = 64
    def attention(q, k, v):
      return jax.nn.softmax(q @ k.T, axis=-1) @ v
    cost = cost_estimate.estimate_cost(
        attention,
        jnp.zeros((q_len, qk_dim), dtype=jnp.float32),
        jnp.zeros((kv_len, qk_dim), dtype=jnp.float32),
        jnp.zeros((kv_len, v_dim), dtype=jnp.float32))
    qk_cost = 2 * q_len * kv_len * qk_dim
    v_cost = 2 * q_len * kv_len * v_dim
    softmax_flops = kv_len * q_len
    self.assertEqual(cost.flops, qk_cost + v_cost + 2 * softmax_flops + q_len)
    self.assertEqual(cost.transcendentals, softmax_flops)
    input_bytes = q_len * qk_dim + kv_len * qk_dim + kv_len * v_dim
    output_bytes = q_len * v_dim
    self.assertEqual(cost.bytes_accessed, 4 * (input_bytes + output_bytes))

  @parameterized.parameters(
      (1, 0), (7, 5), (8, 4), (9, 5)
  )
  def test_integer_pow(self, power, expected_flops_per_element):
    cost = cost_estimate.estimate_cost(lambda x: lax.integer_pow(x, power),
                                       jnp.ones(10, dtype=jnp.float32))
    self.assertEqual(cost.flops, 10 * expected_flops_per_element)
    self.assertEqual(cost.transcendentals, 0)
    self.assertEqual(cost.bytes_accessed, 80)

  def test_run_state(self):
    def add_refs(refs):
      x_ref, y_ref, z_ref = refs
      x = x_ref[:]
      y = y_ref[:]
      z = x + y
      z_ref[:] = z
    input_shape = jax.ShapeDtypeStruct((100,), jnp.float32)
    cost = cost_estimate.estimate_cost(
        discharge.run_state(add_refs),
        (input_shape, input_shape, input_shape))
    self.assertEqual(cost.flops, 100)
    self.assertEqual(cost.transcendentals, 0)
    # TODO(justinfu): This is off by a factor of 2 because run_state
    # has all inputs/outputs as both arguments and return values.
    self.assertEqual(cost.bytes_accessed / 2, 3 * 4 * 100)

  def test_call_hi_primitive(self):
    class DummyHiPrim(hijax.HiPrim):

      def __init__(self, x_aval, y_aval):
        self.in_avals = (x_aval, y_aval)
        self.out_aval = jax_core.ShapedArray(
            (x_aval.shape[0], y_aval.shape[1]), x_aval.dtype
        )
        self.params = {}
        super().__init__()

      def expand(self, x, y):
        dimension_numbers = (((1,), (0,)), ((), ()))
        return lax.dot_general(x, y, dimension_numbers)

    def fun(x, y):
      x_aval = jax_core.ShapedArray(x.shape, x.dtype)
      y_aval = jax_core.ShapedArray(y.shape, y.dtype)
      return DummyHiPrim(x_aval, y_aval)(x, y)

    m, k, n = 10, 20, 30
    cost = cost_estimate.estimate_cost(
        fun,
        jax.ShapeDtypeStruct((m, k), jnp.float32),
        jax.ShapeDtypeStruct((k, n), jnp.float32),
    )
    self.assertEqual(cost.flops, 2 * m * k * n)
    self.assertEqual(cost.transcendentals, 0)
    self.assertEqual(cost.bytes_accessed, 4 * (m * k + k * n + m * n))

  def test_symbolic_cost_estimate(self):
    from jax.experimental import pallas as pl
    from jax.export import symbolic_shape

    (s,) = symbolic_shape('s')
    cost = pl.CostEstimate(
        flops=2 * s * 128,
        transcendentals=s,
        bytes_accessed=4 * s * 64,
        remote_bytes_transferred=0,
    )
    self.assertTrue(jax_core.is_dim(cost.flops))
    self.assertTrue(jax_core.is_dim(cost.transcendentals))
    self.assertTrue(jax_core.is_dim(cost.bytes_accessed))
    json_bytes = cost.to_json()
    self.assertIn(b'"flops": "256*s"', json_bytes)
    self.assertIn(b'"transcendentals": "s"', json_bytes)
    self.assertIn(b'"remote_bytes_transferred": 0', json_bytes)

    # Test concrete CostEstimate still outputs unquoted numbers
    concrete_cost = pl.CostEstimate(
        flops=100, transcendentals=10, bytes_accessed=20
    )
    concrete_json = concrete_cost.to_json()
    self.assertIn(b'"flops": 100', concrete_json)
    self.assertIn(b'"transcendentals": 10', concrete_json)
    self.assertIn(b'"bytes_accessed": 20', concrete_json)
    self.assertIn(b'"remote_bytes_transferred": 0', concrete_json)

    # Test unzip_dynamic_grid_bounds with symbolic dimensions
    from jax._src.pallas.core import unzip_dynamic_grid_bounds, GridSpec

    grid_spec = GridSpec(grid=(16, s // 128))
    static_spec, dynamic_bounds = unzip_dynamic_grid_bounds(grid_spec)
    self.assertEqual(static_spec.grid, (16, None))
    self.assertEqual(len(dynamic_bounds), 1)
    self.assertEqual(dynamic_bounds[0], s // 128)

  def test_traced_specialize_pallas_call(self):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.export import symbolic_shape

    (b, s) = symbolic_shape("b, s", constraints=["b >= 8", "s >= 128"])

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...] * 2.0

    def f(x):
      return pl.pallas_call(
          kernel,
          grid_spec=pl.GridSpec(
              grid=(x.shape[0] // 8, x.shape[1] // 128),
              in_specs=[
                  pl.BlockSpec(
                      (8, 128),
                      lambda i, j: (i * 8, j * 128),
                  )
              ],
              out_specs=pl.BlockSpec(
                  (8, 128),
                  lambda i, j: (i * 8, j * 128),
              ),
          ),
          out_shape=x,
          cost_estimate=pl.CostEstimate(
              flops=2 * x.shape[0] * x.shape[1],
              transcendentals=0,
              bytes_accessed=8 * x.shape[0] * x.shape[1],
          ),
      )(x)

    traced = jax.jit(f).trace(jax.ShapeDtypeStruct((b, s), jnp.float32))

    # Specialize to concrete shape (8, 256)
    spec_traced_256 = traced.specialize(
        jax.ShapeDtypeStruct((8, 256), jnp.float32)
    )
    pallas_eqn_256 = [
        eqn
        for eqn in spec_traced_256.jaxpr.eqns
        if eqn.primitive.name == "pallas_call"
    ][0]
    gm_256 = pallas_eqn_256.params["grid_mapping"]
    self.assertEqual(gm_256.grid, (1, 2))
    self.assertEqual(
        gm_256.block_mappings[0].block_shape, (pl.Blocked(8), pl.Blocked(128))
    )
    self.assertEqual(gm_256.block_mappings[0].array_aval.shape, (8, 256))
    ce_256 = pallas_eqn_256.params["cost_estimate"]
    self.assertEqual(ce_256.flops, 2 * 8 * 256)
    self.assertEqual(ce_256.bytes_accessed, 8 * 8 * 256)

    # Specialize to another concrete shape (16, 512)
    spec_traced_512 = traced.specialize(
        jax.ShapeDtypeStruct((16, 512), jnp.float32)
    )
    pallas_eqn_512 = [
        eqn
        for eqn in spec_traced_512.jaxpr.eqns
        if eqn.primitive.name == "pallas_call"
    ][0]
    gm_512 = pallas_eqn_512.params["grid_mapping"]
    self.assertEqual(gm_512.grid, (2, 4))
    self.assertEqual(
        gm_512.block_mappings[0].block_shape, (pl.Blocked(8), pl.Blocked(128))
    )
    self.assertEqual(gm_512.block_mappings[0].array_aval.shape, (16, 512))
    ce_512 = pallas_eqn_512.params["cost_estimate"]
    self.assertEqual(ce_512.flops, 2 * 16 * 512)
    self.assertEqual(ce_512.bytes_accessed, 8 * 16 * 512)

    # Verify TPU Mosaic lowering on specialized stages
    lowered_256 = spec_traced_256.lower(lowering_platforms=("tpu",))
    self.assertIsNotNone(lowered_256)
    lowered_512 = spec_traced_512.lower(lowering_platforms=("tpu",))
    self.assertIsNotNone(lowered_512)

  def test_traced_specialize_pallas_symbolic_block_shape_and_scratch(self):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu  # pyrefly: ignore[missing-import]
    from jax.export import symbolic_shape

    (b, s) = symbolic_shape(
        "b, s", constraints=["b >= 8", "s >= 256", "mod(s, 2) == 0"]
    )

    def kernel(x_ref, y_ref, scratch_ref):
      scratch_ref[...] = x_ref[...] * 2.0
      y_ref[...] = scratch_ref[...] + 1.0

    def f(x):
      block_s = x.shape[1] // 2
      return pl.pallas_call(
          kernel,
          grid_spec=pl.GridSpec(
              grid=(x.shape[0] // 8, 2),
              in_specs=[
                  pl.BlockSpec(
                      (8, block_s),
                      lambda i, j: (i, j),
                  )
              ],
              out_specs=pl.BlockSpec(
                  (8, block_s),
                  lambda i, j: (i, j),
              ),
              scratch_shapes=[pltpu.VMEM((8, block_s), jnp.float32)],
          ),
          out_shape=x,
      )(x)

    with config.enable_checks(True):
      traced = jax.jit(f).trace(jax.ShapeDtypeStruct((b, s), jnp.float32))
      spec_256 = traced.specialize(jax.ShapeDtypeStruct((8, 256), jnp.float32))

      jax_core.check_jaxpr(spec_256.jaxpr)
      pallas_eqn = [
          eqn
          for eqn in spec_256.jaxpr.eqns
          if eqn.primitive.name == "pallas_call"
      ][0]
      kernel_jaxpr = pallas_eqn.params["jaxpr"]
      jax_core.check_jaxpr(kernel_jaxpr)
      for v in kernel_jaxpr.invars:
        self.assertEqual(v.aval.shape, (8, 128))
        self.assertTrue(
            all(isinstance(d, int) for d in v.aval.shape), v.aval.shape
        )

      gm = pallas_eqn.params["grid_mapping"]
      gm.check_invariants()
      self.assertEqual(gm.scratch_avals[0].shape, (8, 128))
      self.assertTrue(all(isinstance(d, int) for d in gm.scratch_avals[0].shape))
      for bm in gm.block_mappings:
        bm.check_invariants()
        self.assertEqual(bm.transformed_block_aval.shape, (8, 128))
        self.assertTrue(
            all(isinstance(d, int) for d in bm.transformed_block_aval.shape)
        )

      lowered_256 = spec_256.lower(lowering_platforms=("tpu",))
      self.assertIsNotNone(lowered_256)

  @parameterized.named_parameters(
      ("x64_false", False),
      ("x64_true", True),
  )
  def test_traced_specialize_pallas_symbolic_index_map_dtype(self, enable_x64):
    from jax.experimental import pallas as pl
    from jax.export import symbolic_shape

    with config.enable_x64(enable_x64), config.enable_checks(True):
      (b, s) = symbolic_shape(
          "b, s", constraints=["b >= 8", "s >= 256", "mod(s, 128) == 0"]
      )

      def kernel(x_ref, y_ref):
        y_ref[...] = x_ref[...]

      def f(x):
        num_s_blocks = x.shape[1] // 128
        return pl.pallas_call(
            kernel,
            grid_spec=pl.GridSpec(
                grid=(x.shape[0] // 8, num_s_blocks),
                in_specs=[
                    pl.BlockSpec(
                        (8, 128),
                        lambda i, j: (i, (num_s_blocks - 1) - j),
                    )
                ],
                out_specs=pl.BlockSpec(
                    (8, 128),
                    lambda i, j: (i, j),
                ),
            ),
            out_shape=x,
        )(x)

      traced = jax.jit(f).trace(jax.ShapeDtypeStruct((b, s), jnp.float32))

      for specialize_x64 in (False, True):
        with config.enable_x64(specialize_x64):
          spec = traced.specialize(jax.ShapeDtypeStruct((8, 512), jnp.float32))
          jax_core.check_jaxpr(spec.jaxpr)
          pallas_eqn = [
              eqn
              for eqn in spec.jaxpr.eqns
              if eqn.primitive.name == "pallas_call"
          ][0]
          gm = pallas_eqn.params["grid_mapping"]
          gm.check_invariants()
          bm_in = gm.block_mappings[0]
          bm_in.check_invariants()
          jax_core.check_jaxpr(bm_in.index_map_jaxpr)
          self.assertFalse(
              any(
                  eqn.primitive.name == "dim_as_value"
                  for eqn in bm_in.index_map_jaxpr.eqns
              )
          )
          for eqn in bm_in.index_map_jaxpr.eqns:
            for atom in eqn.invars:
              if isinstance(atom, jax_core.Literal):
                self.assertEqual(atom.val.dtype, atom.aval.dtype)
                self.assertEqual(atom.aval.dtype, jnp.int32)
          for ov in bm_in.index_map_jaxpr.out_avals:
            self.assertEqual(ov.dtype, jnp.int32)
          self.assertEqual(
              bm_in.compute_start_indices_interpret(
                  (jnp.int32(0), jnp.int32(0))
              ),
              (0, 384),
          )
          lowered = spec.lower(lowering_platforms=("tpu",))
          self.assertIsNotNone(lowered)

  def test_pallas_post_lowering_specialize_and_benchmark(self):
    import time
    from jax.experimental import pallas as pl
    from jax.export import symbolic_shape

    (b, s) = symbolic_shape(
        "b, s", constraints=["b >= 8", "s >= 128", "mod(s, 128) == 0"]
    )

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...] * 2.0

    def f(x):
      return pl.pallas_call(
          kernel,
          grid_spec=pl.GridSpec(
              grid=(x.shape[0] // 8, x.shape[1] // 128),
              in_specs=[
                  pl.BlockSpec(
                      (8, 128),
                      lambda i, j: (i * 8, j * 128),
                  )
              ],
              out_specs=pl.BlockSpec(
                  (8, 128),
                  lambda i, j: (i * 8, j * 128),
              ),
          ),
          out_shape=x,
          cost_estimate=pl.CostEstimate(
              flops=2 * x.shape[0] * x.shape[1],
              transcendentals=0,
              bytes_accessed=8 * x.shape[0] * x.shape[1],
          ),
      )(x)

    traced = jax.jit(f).trace(jax.ShapeDtypeStruct((b, s), jnp.float32))
    sym_lowered = traced.lower(lowering_platforms=("tpu",))

    # 7 buckets: 1024, 2048, 4096, 8192, 16384, 32768, 65536
    bucket_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    # Verify structural and signature custom call parity across all 7 buckets
    for seq_len in bucket_lengths:
      spec_traced = traced.specialize(
          jax.ShapeDtypeStruct((8, seq_len), jnp.float32)
      )
      lowered_pre = spec_traced.lower(lowering_platforms=("tpu",))
      lowered_post = sym_lowered.specialize(
          jax.ShapeDtypeStruct((8, seq_len), jnp.float32)
      )
      # Both produce valid lowered TPU custom calls with correct concrete shapes
      self.assertIn(f"tensor<8x{seq_len}xf32>", lowered_post.as_text())
      self.assertIn("@tpu_custom_call", lowered_post.as_text())
      self.assertIn(f"tensor<8x{seq_len}xf32>", lowered_pre.as_text())
      self.assertIn("@tpu_custom_call", lowered_pre.as_text())

    # Benchmark: 7x Trace + 7x Lower vs 1x Trace + 7x Specialize + 7x Lower vs 1x Trace + 1x Lower + 7x Lowered.specialize
    # Warmup
    for seq_len in bucket_lengths:
      _ = traced.specialize(jax.ShapeDtypeStruct((8, seq_len), jnp.float32)).lower(lowering_platforms=("tpu",))
      _ = sym_lowered.specialize(jax.ShapeDtypeStruct((8, seq_len), jnp.float32))

    t0 = time.perf_counter()
    for _ in range(5):
      for seq_len in bucket_lengths:
        t = jax.jit(f).trace(jax.ShapeDtypeStruct((8, seq_len), jnp.float32))
        _ = t.lower(lowering_platforms=("tpu",))
    t_baseline = (time.perf_counter() - t0) / 5.0

    t0 = time.perf_counter()
    for _ in range(5):
      t_sym = jax.jit(f).trace(jax.ShapeDtypeStruct((b, s), jnp.float32))
      for seq_len in bucket_lengths:
        spec_t = t_sym.specialize(jax.ShapeDtypeStruct((8, seq_len), jnp.float32))
        _ = spec_t.lower(lowering_platforms=("tpu",))
    t_pre_lowering = (time.perf_counter() - t0) / 5.0

    t0 = time.perf_counter()
    for _ in range(5):
      t_sym = jax.jit(f).trace(jax.ShapeDtypeStruct((b, s), jnp.float32))
      low_sym = t_sym.lower(lowering_platforms=("tpu",))
      for seq_len in bucket_lengths:
        _ = low_sym.specialize(jax.ShapeDtypeStruct((8, seq_len), jnp.float32))
    t_post_lowering = (time.perf_counter() - t0) / 5.0

    print(
        f"\n[BENCHMARK] 7-Bucket Pallas Specialization Timings (5-run avg):\n"
        f"  (1) Baseline (7x Trace + 7x Lower): {t_baseline*1000:.2f} ms\n"
        f"  (2) Pre-Lowering Specialize (1x Trace + 7x Traced.spec + 7x Lower): {t_pre_lowering*1000:.2f} ms\n"
        f"  (3) Post-Lowering Specialize (1x Trace + 1x Lower + 7x Lowered.spec): {t_post_lowering*1000:.2f} ms\n"
        f"  Speedup vs Baseline: {t_baseline / t_post_lowering:.2f}x\n"
        f"  Speedup vs Pre-Lowering: {t_pre_lowering / t_post_lowering:.2f}x\n"
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
