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
from absl.testing import absltest

import re
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec, NamedSharding
from jax._src import config
from jax._src import dtypes as _dtypes
from jax._src import test_util as jtu
from jax._src.cudnn.fused_attention_stablehlo import check_cudnn_version
from jax._src.cudnn.scaled_matmul_stablehlo import (
    scaled_matmul_wrapper,
    scaled_dot_general_wrapper,
    shape_normalization,
    quantize,
    BlockScaleConfig,
)

config.parse_flags_with_absl()
input_shardings = [
    (("dp", None, "tp"), ("dp", None, "tp")),
    (("dp", None, "tp"), ("dp", None, None)),
    (("dp", None, "tp"), ("dp", "tp", None)),
    (("dp", None, None), ("dp", "tp", None)),
    (("dp", "tp", None), ("dp", "tp", None)),
    ((None, "dp", "tp"), (None, "dp", "tp")),
    ((None, "tp", None), (None, "tp", None)),
    ((None, None, "tp"), (None, "tp", None)),
]
c_name = "__cudnn$blockScaledDot"
expected_hlos = [
    (c_name, "all-reduce", "f32[1,512,512]", "replica_groups={{0,1},{2,3}}"),
    ("all-gather", "f8e4m3fn[1,512,512]", "replica_groups=[2,2]<=[4]", c_name),
    ("all-gather", "f8e4m3fn[1,512,512]", "replica_groups=[2,2]<=[4]", c_name),
    (c_name,),
    ("all-gather", "f8e4m3fn[1,256,1024]", "replica_groups=[2,2]<=[4]", c_name),
    (c_name, "reduce-scatter", "f32[2,256,512]", "replica_groups={{0,1},{2,3}}"),
    ("all-gather", "f8e4m3fn[2,512,1024]", "replica_groups=[2,2]<=[4]", c_name),
    ("all-gather", "f8e4m3fn[2,512,512]", "replica_groups=[2,2]<=[4]", c_name),
]
expected_output_spec = [
    PartitionSpec('dp',),
    PartitionSpec('dp',),
    PartitionSpec('dp', None, 'tp'),
    PartitionSpec('dp', None, 'tp'),
    PartitionSpec('dp', 'tp', None),
    PartitionSpec(None, 'dp', 'tp'),
    PartitionSpec(None, 'tp', None),
    PartitionSpec(None, None, 'tp'),
]
sharding_configs = {
    input_sharding: (hlo, output_spec)
    for input_sharding, hlo, output_spec in zip(input_shardings,
                                                expected_hlos,
                                                expected_output_spec)
}

def quantize_to_qtype(x, q_dtype, compute_dtype, scale):
  # Explicitly cast the max values to the compute dtype to avoid unnecessary
  # casting to FP32 during the subsequent math operations."
  assert q_dtype in (jnp.float8_e4m3fn, jnp.float4_e2m1fn)
  dtype_max = jnp.finfo(q_dtype).max.astype(compute_dtype)
  scaled_x = x / jnp.broadcast_to(
      jnp.asarray(scale, dtype=compute_dtype), x.shape
  )
  clipped_x = jnp.clip(scaled_x, -dtype_max, dtype_max)
  return clipped_x.astype(q_dtype)

def quantize_dequantize(x, q_dtype, scale, compute_dtype):
  qx = quantize_to_qtype(x, q_dtype, compute_dtype, scale)
  out = qx.astype(x.dtype) * jnp.broadcast_to(
      jnp.asarray(scale, dtype=x.dtype), qx.shape
  )
  return out

def generate_quantized_tensors(
    batch, lhs_non_contract, contract, rhs_non_contract,
    configs, dtype=jnp.float32,
  ):
  cast_to_representable = partial(
      quantize_dequantize,
      scale=jnp.ones((1,)),
      compute_dtype=dtype,
  )

  k1, k2 = jax.random.split(jax.random.key(123), 2)

  a = cast_to_representable(
      jax.random.uniform(
          k1, (batch, lhs_non_contract, contract), minval=-1.0, dtype=dtype
      ),
      configs[0].data_type,
  )
  b = cast_to_representable(
      jax.random.uniform(
          k2, (batch, rhs_non_contract, contract), minval=-1.0, dtype=dtype
      ),
      configs[1].data_type,
  )

  dn = ((2,), (0,))
  a_3d = shape_normalization(a, dn)
  b_3d = shape_normalization(b, dn)
  a_q, a_scales = quantize(a, configs[0])
  b_q, b_scales = quantize(b, configs[1])

  return a, b, a_q, b_q, a_scales, b_scales


def shard_and_device_put(
    mesh, a_sharding, b_sharding, a, b, a_scales=None, b_scales=None
  ):
  a_spec = PartitionSpec(*a_sharding)
  b_spec = PartitionSpec(*b_sharding)

  a_named_sharding = NamedSharding(mesh, a_spec)
  b_named_sharding = NamedSharding(mesh, b_spec)

  a = jax.device_put(a, a_named_sharding)
  b = jax.device_put(b, b_named_sharding)
  if a_scales is not None:
    a_scales = jax.device_put(a_scales, a_named_sharding)
  if b_scales is not None:
    b_scales = jax.device_put(b_scales, b_named_sharding)

  in_shardings = (
      a_named_sharding,
      b_named_sharding,
  )
  if a_scales is not None and b_scales is not None:
    in_shardings = (
        a_named_sharding,
        b_named_sharding,
        a_named_sharding,
        b_named_sharding,
    )
    return a, b, a_scales, b_scales, in_shardings

  return a, b, in_shardings

def create_nvfp4_configs(global_scale=None):
  if _dtypes.float4_e2m1fn is None:
    return None
  g_one_scale = jnp.ones((1, ), dtype=jnp.float32)
  nvfp4_config = BlockScaleConfig(
        mode='nvfp4',
        block_size=16,
        data_type=jnp.float4_e2m1fn,
        scale_type=jnp.float8_e4m3fn,
        global_scale=g_one_scale if global_scale is None else global_scale,
        infer_only=False
  )

  return [nvfp4_config for _ in range(3)]

def update_global_scale(config, new_global_scale):
  config.global_scale = new_global_scale
  return config

def generate_nvfp4_quantized_tensors(dot_config, output_type, enable_grad_clip=False):
  k1, k2 = jax.random.split(jax.random.key(0), 2)

  a_shape, b_shape, dimension_numbers = dot_config
  (a_contract, b_contract), (a_batch, b_batch) = dimension_numbers
  a_dn = (a_contract, a_batch)
  b_dn = (b_contract, b_batch)

  a_raw = jax.random.uniform(k1, a_shape, minval=-1.0, dtype=output_type)
  b_raw = jax.random.uniform(k2, b_shape, minval=-1.0, dtype=output_type)
  a = shape_normalization(a_raw, a_dn)
  b = shape_normalization(b_raw, b_dn)

  # Initialize NVFP4 configurations
  block_scale_configs_nvfp4 = create_nvfp4_configs()

  # Compute maximum absolute values for scaling
  amax_a = jnp.max(jnp.abs(a)).astype(jnp.float32)
  amax_b = jnp.max(jnp.abs(b)).astype(jnp.float32)

  # To emulate calibrated amax
  amax_sf = 0.9 if enable_grad_clip else 1.0
  amax_a *= amax_sf
  amax_b *= amax_sf

  # Update global scales
  data_max = jnp.finfo(block_scale_configs_nvfp4[0].data_type).max.astype(
      jnp.float32
  )
  scale_max = jnp.finfo(block_scale_configs_nvfp4[0].scale_type).max.astype(
      jnp.float32
  )

  block_scale_configs_nvfp4[0] = update_global_scale(
      block_scale_configs_nvfp4[0], amax_a / (data_max * scale_max))
  block_scale_configs_nvfp4[1] = update_global_scale(
      block_scale_configs_nvfp4[1], amax_b / (data_max * scale_max))

  # Quantize tensors
  a_nvfp4, a_scale = quantize(a, block_scale_configs_nvfp4[0])
  b_nvfp4, b_scale = quantize(b, block_scale_configs_nvfp4[1])

  # Reshape and scale quantized tensors
  def reshape_and_scale(x, scale, global_scale, bs, k):
    reshaped = x.astype(output_type).reshape(*bs, k // 16, 16)
    scaled = reshaped * jnp.expand_dims(scale.astype(output_type), -1)
    return scaled.reshape(*bs, k) * global_scale.astype(output_type)

  *bs_a, k_a = a_nvfp4.shape
  *bs_b, k_b = b_nvfp4.shape
  assert k_a == k_b

  a_dequantized = reshape_and_scale(
      a_nvfp4, a_scale, block_scale_configs_nvfp4[0].global_scale, bs_a, k_a)
  b_dequantized = reshape_and_scale(
      b_nvfp4, b_scale, block_scale_configs_nvfp4[1].global_scale, bs_b, k_b)

  return (
      (a_raw, b_raw),
      (a_dequantized, b_dequantized),
      (a_nvfp4, b_nvfp4, a_scale, b_scale),
      block_scale_configs_nvfp4
  )

def create_mxfp8_configs():
  if _dtypes.float8_e8m0fnu is None:
    return None

  mxfp8_config = BlockScaleConfig(
        mode='mxfp8',
        block_size=32,
        data_type=jnp.float8_e4m3fn,
        scale_type=jnp.float8_e8m0fnu,
        global_scale=None,
        infer_only=False
  )

  return [mxfp8_config for _ in range(3)]

def get_hlo_text(in_shardings, block_scale_configs):
  mesh_names = ("dp", "tp")
  devices = np.array(jax.local_devices()[:4]).reshape((2, 2))
  mesh = Mesh(devices, mesh_names)
  _, _, a_q, b_q, a_scales, b_scales = generate_quantized_tensors(
      2, 512, 1024, 512, block_scale_configs,
  )

  with mesh:
    a_q, b_q, a_scales, b_scales, in_shardings = shard_and_device_put(
        mesh, in_shardings[0], in_shardings[1], a_q, b_q, a_scales, b_scales
    )
    pjit_fn = jax.jit(scaled_matmul_wrapper, in_shardings=in_shardings)
    hlo = pjit_fn.lower(a_q, b_q, a_scales, b_scales).compile()
  return hlo.as_text()

@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ScaledMatmulTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if _dtypes.float8_e8m0fnu is None:
      self.skipTest("Requires >= ml_dtypes 0.5.0 to support float8_e8m0fnu")
    if _dtypes.float4_e2m1fn is None:
      self.skipTest("Requires >= ml_dtypes 0.5.0 to support float4_e2m1fn")
    if cudnn_version < 90700:
      self.skipTest("Requires >= cuDNN 9.7.0")
    if not jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest("Requires at least Blackwell arch")

  mxfp8_configs = create_mxfp8_configs()

  @jtu.sample_product(
      in_shardings=sharding_configs,
      block_scale_configs=[mxfp8_configs,],
  )
  @jtu.run_on_devices("cuda")
  def test_collectives(self, in_shardings, block_scale_configs):
    if jtu.device_under_test() != "gpu" or len(jax.local_devices()) < 4:
      self.skipTest("Partition Test enabled for at least 4 GPUs")

    expected_hlo = sharding_configs[in_shardings][0]
    hlo_text = get_hlo_text(in_shardings, block_scale_configs)

    hlo_pattern = re.compile(
        r".*".join([re.escape(x) for x in expected_hlo]), flags=re.DOTALL
    )
    self.assertRegex(
        hlo_text, hlo_pattern, msg=f"Failed to find pattern: {expected_hlo}"
    )

  @jtu.sample_product(
      contract=[160, 96],
      lhs_non_contract=[240, 100],
      dtype=[jnp.float32, jnp.bfloat16, jnp.float16],
  )
  @jtu.run_on_devices("cuda")
  def test_scaled_matmul_nvfp4(
      self, contract, lhs_non_contract, dtype,
  ):
    batch, rhs_non_contract = 2, 128
    dot_config = (
        (batch, lhs_non_contract, contract),
        (batch, rhs_non_contract, contract),
        (([2], [2]), ([0], [0]))
    )
    _, (a_dq, b_dq), (a_q, b_q, a_s, b_s), block_scale_configs = (
        generate_nvfp4_quantized_tensors(dot_config, dtype)
    )
    a_gs = block_scale_configs[0].global_scale
    b_gs = block_scale_configs[1].global_scale

    def wrapper(lhs, rhs, lhs_scales, rhs_scales, out_type):
      out = scaled_matmul_wrapper(
          lhs,
          rhs,
          lhs_scales,
          rhs_scales,
          preferred_element_type=jnp.float32,
      )
      gs = a_gs * b_gs
      return (out * gs).astype(out_type)

    j_scaled_matmul = jax.jit(partial(wrapper, out_type=dtype))
    hlo_text = (
        j_scaled_matmul.lower(a_q, b_q, a_s, b_s)
        .compile()
        .as_text()
    )
    hlo_pattern = re.compile(
        r".*".join([re.escape(x) for x in ("custom-call", c_name)])
    )
    self.assertRegex(hlo_text, hlo_pattern)

    out = j_scaled_matmul(a_q, b_q, a_s, b_s)
    out_ref = jnp.einsum(
        "BMK,BNK->BMN", a_dq, b_dq
    )
    self.assertArraysAllClose(
        out, out_ref.astype(dtype), rtol=1e-2, atol=5e-2
    )

  @jtu.sample_product(
      contract=[160, 96],
      lhs_non_contract=[240, 100],
      dtype=[jnp.float16, jnp.bfloat16, jnp.float32],
      block_scale_configs=[mxfp8_configs,],
  )
  @jtu.run_on_devices("cuda")
  def test_scaled_matmul(
      self, contract, lhs_non_contract, dtype, block_scale_configs,
  ):
    batch, rhs_non_contract = 2, 128
    a, b, a_q, b_q, a_scales, b_scales = generate_quantized_tensors(
        batch, lhs_non_contract, contract, rhs_non_contract,
        block_scale_configs, dtype=dtype,
    )

    def wrapper(lhs, rhs, lhs_scales, rhs_scales, out_type):
      return scaled_matmul_wrapper(
          lhs,
          rhs,
          lhs_scales,
          rhs_scales,
          preferred_element_type=out_type,
      )

    j_scaled_matmul = jax.jit(partial(wrapper, out_type=dtype))
    hlo_text = (
        j_scaled_matmul.lower(a_q, b_q, a_scales, b_scales)
        .compile()
        .as_text()
    )
    hlo_pattern = re.compile(
        r".*".join([re.escape(x) for x in ("custom-call", c_name)])
    )
    self.assertRegex(hlo_text, hlo_pattern)

    out = j_scaled_matmul(a_q, b_q, a_scales, b_scales)
    out_ref = np.einsum(
        "BMK,BNK->BMN", a.astype(jnp.float32), b.astype(jnp.float32)
    )
    self.assertArraysAllClose(
        out, out_ref.astype(dtype), rtol=1e-3, atol=1e-3
    )

  @jtu.sample_product(
        in_shardings=sharding_configs,
        block_scale_configs=[mxfp8_configs,],
  )
  @jtu.run_on_devices("cuda")
  def test_scaled_matmul_sharded(self, in_shardings, block_scale_configs):
    if len(jax.local_devices()) < 4:
      self.skipTest("Require at least 4 devices to run sharding tests.")
    batch, contract, non_contract = 2, 1024, 256
    a, b, a_q, b_q, a_scales, b_scales = generate_quantized_tensors(
        batch, non_contract, contract, non_contract, block_scale_configs,
    )

    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    expected_output_spec = sharding_configs[in_shardings][1]

    with Mesh(devices, ("dp", "tp")) as mesh:
      a_q, b_q, a_scales, b_scales, input_shardings = (
          shard_and_device_put(
              mesh,
              in_shardings[0],
              in_shardings[1],
              a_q,
              b_q,
              a_scales,
              b_scales,
          )
      )

      args = [a_q, b_q, a_scales, b_scales]
      j_scaled_matmul = jax.jit(
          scaled_matmul_wrapper, in_shardings=input_shardings
      )
      hlo_compiled = j_scaled_matmul.lower(*args).compile()
      hlo_pattern = re.compile(
          r".*".join([re.escape(x) for x in ("custom-call", c_name)])
      )
      self.assertRegex(hlo_compiled.as_text(), hlo_pattern)

      j_ref = jax.jit(
          partial(
              jax.lax.dot_general,
              dimension_numbers=(([2], [2]), ([0], [0])),
          ),
          in_shardings=input_shardings[:2],
      )

      out = j_scaled_matmul(*args)
      out_ref = j_ref(a, b)
      expected_output_sharding = NamedSharding(
          mesh=mesh, spec=expected_output_spec
      )
      self.assertArraysAllClose(out, out_ref, rtol=1e-3, atol=1e-3)
      self.assertTrue(
          out.sharding.is_equivalent_to(expected_output_sharding, out.ndim)
      )

@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ScaledDotGeneralTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if _dtypes.float8_e8m0fnu is None:
      self.skipTest("Requires >= ml_dtypes 0.5.0 to support float8_e8m0fnu")
    if cudnn_version < 90700:
      self.skipTest("Requires >= cuDNN 9.7.0")
    if not jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest("Requires at least Blackwell arch")

  block_scale_configs = create_mxfp8_configs()

  @jtu.sample_product(
      shape=[
          (1, 128, 128),
          (64, 32),
          (1024, 2048),
      ],
  )
  @jtu.run_on_devices("cuda")
  def test_quantize_nvfp4(self, shape):
    # To test the q-dq logic is valid with XLA
    output_type = jnp.float32
    k1, k2 = jax.random.split(jax.random.key(0), 2)

    a = jax.random.uniform(k1, shape, minval=-1.0, dtype=output_type)

    block_scale_configs_nvfp4 = create_nvfp4_configs()
    data_max = jnp.finfo(jnp.float4_e2m1fn).max.astype(jnp.float32)
    scale_max = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)
    amax_a = jnp.max(jnp.abs(a)).astype(jnp.float32) / (data_max * scale_max)
    block_scale_configs_nvfp4[0] = update_global_scale(
        block_scale_configs_nvfp4[0], jnp.asarray(amax_a, jnp.float32)
    )

    def fn(a):
      a_nvfp4, a_scale = quantize(a, block_scale_configs_nvfp4[0])
      return a_nvfp4, a_scale

    out_q, scale = jax.jit(fn)(a)
    out_q_ref, scale_ref = fn(a)
    self.assertArraysAllClose(out_q, out_q_ref, rtol=1e-5, atol=1e-5)
    self.assertArraysAllClose(scale, scale_ref, rtol=1e-5, atol=1e-5)

  @jtu.sample_product(
      enable_grad_clip=[True, False],
      configs=[
          # a_shape, b_shape, dimension_numbers
          ((1, 128, 128), (1, 128, 128), (([2], [2]), ([0], [0]))),
          ((30, 64), (100, 64), (([1], [1]), ([], []))),
      ]
  )
  @jtu.run_on_devices("cuda")
  def test_nvfp4_gradient_clip(self, enable_grad_clip, configs):
    output_type = jnp.float32
    (a_raw, b_raw), (a_dq, b_dq), _, block_scale_configs = (
        generate_nvfp4_quantized_tensors(configs, output_type, enable_grad_clip)
    )
    a_gs = block_scale_configs[0].global_scale
    b_gs = block_scale_configs[1].global_scale
    dimension_numbers = configs[2]

    scaled_dot_general = partial(
        scaled_dot_general_wrapper,
        configs=block_scale_configs
    )

    def fwd(a, b, use_normalized=False):
      y = scaled_dot_general(
          a, b, dimension_numbers,
          preferred_element_type=output_type
      )
      return jnp.sum(y)

    j_train = jax.jit(jax.value_and_grad(fwd, argnums=[0, 1]))
    _, (x_grad, w_grad) = j_train(a_raw, b_raw)

    data_max = jnp.finfo(jnp.float4_e2m1fn).max.astype(output_type)
    scale_max = jnp.finfo(jnp.float8_e4m3fn).max.astype(output_type)
    prev_amax_a = a_gs * data_max * scale_max
    prev_amax_b = b_gs * data_max * scale_max

    # Use a large value to ensure no clipping
    threshold_a = prev_amax_a if enable_grad_clip else 1e9
    threshold_b = prev_amax_b if enable_grad_clip else 1e9

    # Verify gradients are clipped to 0 where |input| > global_scale * MAX * SCALE_MAX
    self.assertArraysEqual(
        jnp.where(jnp.abs(a_raw) > threshold_a, x_grad, 0),
        jnp.zeros_like(x_grad),
    )
    self.assertArraysEqual(
        jnp.where(jnp.abs(b_raw) > threshold_b, w_grad, 0),
        jnp.zeros_like(w_grad),
    )
    if enable_grad_clip:
      # Verify gradients are preserved where |input| <= global_scale * MAX * SCALE_MAX
      self.assertArraysEqual(
          jnp.where(jnp.abs(a_raw) <= prev_amax_a, x_grad, 0),
          x_grad,
      )
      self.assertArraysEqual(
          jnp.where(jnp.abs(b_raw) <= prev_amax_b, w_grad, 0),
          w_grad,
      )

  @jtu.sample_product(
      configs=[
          # a_shape, b_shape, dimension_numbers, is_training
          ((1, 128, 128), (1, 128, 128), (([2], [2]), ([0], [0])), False),
          ((30, 64), (100, 64), (([1], [1]), ([], [])), False),
          ((192, 96), (160, 96), (([1], [1]), ([], [])), True),
          ((64, 128, 4), (128, 128), (([1], [0]), ([], [])), True),
          ((1, 128, 1024), (1, 1024, 128), (([2], [1]), ([0], [0])), True),
          (
              (1, 128, 128, 2),
              (128, 1, 2, 128),
              (([2], [0]), ([0, 3], [1, 2])),
              True,
          ),
      ],
      output_type=[jnp.float32, jnp.float16, jnp.bfloat16],
  )
  @jtu.run_on_devices("cuda")
  def test_dot_general_nvfp4(self, configs, output_type):
    (a_raw, b_raw), (a_dq, b_dq), _, block_scale_configs = (
        generate_nvfp4_quantized_tensors(configs[:-1], output_type)
    )
    a_gs = block_scale_configs[0].global_scale
    b_gs = block_scale_configs[1].global_scale

    scaled_dot_general = partial(
        scaled_dot_general_wrapper,
        configs=block_scale_configs
    )

    dimension_numbers = configs[2]
    is_training = configs[-1]
    def fwd(a, b, is_ref=False, use_normalized=False):
      fn = jax.lax.dot_general if is_ref else scaled_dot_general
      if is_ref and use_normalized:
        dms = (([2], [2]), ([0], [0]))
      else:
        dms = dimension_numbers

      y = fn(a, b, dms,
             preferred_element_type=output_type)

      return jnp.sum(y) if is_training else y

    if is_training:
      j_train = jax.jit(jax.value_and_grad(fwd, argnums=[0, 1]))
      j_train_ref = jax.jit(
          jax.value_and_grad(partial(fwd, is_ref=True), argnums=[0, 1])
      )
      j_train_fwd_ref = jax.jit(
          jax.value_and_grad(
              partial(fwd, is_ref=True, use_normalized=True), argnums=[0, 1]
          )
      )
      out, (x_grad, w_grad) = j_train(a_raw, b_raw)
      _, (x_grad_ref, w_grad_ref) = j_train_ref(a_raw, b_raw)
      out_ref, _ = j_train_fwd_ref(a_dq, b_dq)

      self.assertArraysAllClose(out, out_ref, rtol=1e-2, atol=1e-2)
      def _grad_clip(amax, x, grad):
        return jnp.where(jnp.abs(x) <= amax, grad, 0)

      data_max = jnp.finfo(jnp.float4_e2m1fn).max.astype(output_type)
      scale_max = jnp.finfo(jnp.float8_e4m3fn).max.astype(output_type)
      prev_amax_a = a_gs * data_max * scale_max
      prev_amax_b = b_gs * data_max * scale_max

      x_grad_ref = _grad_clip(prev_amax_a, a_raw, x_grad_ref)
      w_grad_ref = _grad_clip(prev_amax_b, b_raw, w_grad_ref)
      self.assertArraysAllClose(x_grad, x_grad_ref, rtol=1e-2, atol=1e1)
      self.assertArraysAllClose(w_grad, w_grad_ref, rtol=1e-2, atol=1e1)
    else:
      j_inference = jax.jit(fwd)
      j_inference_ref = jax.jit(partial(fwd, is_ref=True, use_normalized=True))
      out = j_inference(a_raw, b_raw)
      out_ref = jnp.reshape(j_inference_ref(a_dq, b_dq), out.shape)
      self.assertArraysAllClose(out, out_ref, rtol=1e-2, atol=2e-1)

  @jtu.sample_product(
      configs=[
          # a_shape, b_shape, dimension_numbers, is_training
          ((1, 32), (2, 32), (([1], [1]), ([], [])), False),
          ((30, 64), (100, 64), (([1], [1]), ([], [])), False),
          ((192, 96), (160, 96), (([1], [1]), ([], [])), True),
          ((64, 128, 4), (128, 128), (([1], [0]), ([], [])), True),
          ((1, 128, 1024), (1, 1024, 128), (([2], [1]), ([0], [0])), True),
          (
              (1, 128, 128, 2),
              (128, 1, 2, 128),
              (([2], [0]), ([0, 3], [1, 2])),
              True,
          ),
      ],
      output_type=[jnp.float16, jnp.bfloat16, jnp.float32],
  )
  @jtu.run_on_devices("cuda")
  def test_dot_general(self, configs, output_type):
    cast_to_representable = partial(
        quantize_dequantize,
        scale=jnp.ones((1,)),
        compute_dtype=jnp.float32,
    )
    k1, k2 = jax.random.split(jax.random.key(0), 2)

    a_shape, b_shape, dimension_numbers, is_training = configs
    a = cast_to_representable(
        jax.random.uniform(k1, a_shape, minval=-1.0, dtype=output_type),
        self.block_scale_configs[0].data_type,
    )
    b = cast_to_representable(
        jax.random.uniform(k2, b_shape, minval=-1.0, dtype=output_type),
        self.block_scale_configs[1].data_type,
    )

    scaled_dot_general = partial(
        scaled_dot_general_wrapper,
        configs=self.block_scale_configs
    )
    def fwd(a, b, is_ref=False):
      fn = jax.lax.dot_general if is_ref else scaled_dot_general
      y = fn(a, b, dimension_numbers,
             preferred_element_type=output_type)
      return jnp.sum(y)

    if is_training:
      j_train = jax.jit(jax.value_and_grad(fwd, argnums=[0, 1]))

      j_train_ref = jax.jit(
          jax.value_and_grad(partial(fwd, is_ref=True), argnums=[0, 1])
      )
      out, (x_grad, w_grad) = j_train(a, b)
      out_ref, (x_grad_ref, w_grad_ref) = j_train_ref(a, b)

      self.assertArraysAllClose(out, out_ref, rtol=1e-2, atol=1e-2)
      self.assertArraysAllClose(x_grad, x_grad_ref, rtol=1e-2, atol=1e1)
      self.assertArraysAllClose(w_grad, w_grad_ref, rtol=1e-2, atol=1e1)
    else:
      j_inference = jax.jit(fwd)
      j_inference_ref = jax.jit(partial(fwd, is_ref=True))
      out = j_inference(a, b)
      out_ref = j_inference_ref(a, b)
      self.assertArraysAllClose(out, out_ref, rtol=1e-2, atol=1e-2)

  @jtu.sample_product(in_shardings=sharding_configs)
  @jtu.run_on_devices("cuda")
  def test_dot_general_sharded(self, in_shardings):
    if len(jax.local_devices()) < 4:
      self.skipTest("Require at least 4 devices to run sharding tests.")

    cast_to_representable = partial(
        quantize_dequantize,
        scale=jnp.ones((1,)),
        compute_dtype=jnp.float32,
    )

    dimension_numbers = (([2], [2]), ([0], [0]))
    a_shape = (2, 128, 512)
    b_shape = (2, 256, 512)

    k1, k2 = jax.random.split(jax.random.key(0), 2)
    a = cast_to_representable(
        jax.random.uniform(k1, a_shape, minval=-1.0, dtype=jnp.float32),
        self.block_scale_configs[0].data_type,
    )
    b = cast_to_representable(
        jax.random.uniform(k2, b_shape, minval=-1.0, dtype=jnp.float32),
        self.block_scale_configs[1].data_type,
    )

    scaled_dot_general = partial(
        scaled_dot_general_wrapper,
        configs=self.block_scale_configs
    )
    def fwd(a, b, is_ref=False):
      fn = jax.lax.dot_general if is_ref else scaled_dot_general
      y = fn(a, b, dimension_numbers)
      # Use a little complex loss function to avoid constant grads, whose
      # sharding info might be optimized off and then cause issue with the
      # custom scaled_matmul op.
      return jnp.sum(jnp.tanh(y))

    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ("dp", "tp")) as mesh:
      a, b, input_shardings = (
          shard_and_device_put(
              mesh,
              in_shardings[0],
              in_shardings[1],
              a,
              b,
          )
      )

      j_train = jax.jit(jax.value_and_grad(partial(fwd), argnums=[0, 1]),
                        in_shardings=input_shardings)

      j_train_ref = jax.jit(
          jax.value_and_grad(partial(fwd, is_ref=True), argnums=[0, 1]),
          in_shardings=input_shardings
      )
      out, (x_grad, w_grad) = j_train(a, b)
      out_ref, (x_grad_ref, w_grad_ref) = j_train_ref(a, b)
      self.assertArraysAllClose(out, out_ref, rtol=1e-2, atol=1e-2)
      self.assertArraysAllClose(x_grad, x_grad_ref, rtol=1e-2, atol=1e1)
      self.assertArraysAllClose(w_grad, w_grad_ref, rtol=1e-2, atol=1e1)


  @jtu.sample_product(
      configs=[
          ((1, 128, 256), (1, 128, 256), (0, 0, 0)),
          ((2, 128, 128), (2, 128, 128), (0, 0, 0)),
          ((2, 128, 128), (128, 2, 128), (0, 1, 2)),
      ]
  )
  @jtu.run_on_devices("cuda")
  def test_dot_general_vmap(self, configs):
    cast_to_representable = partial(
        quantize_dequantize,
        scale=jnp.ones((1,)),
        compute_dtype=jnp.float32,
    )
    k1, k2 = jax.random.split(jax.random.key(0), 2)

    a_shape, b_shape, vmap_axes = configs
    a_axis, b_axis, o_axis = vmap_axes
    dimension_numbers = (([1], [1]), ([], []))

    a = cast_to_representable(
        jax.random.uniform(k1, a_shape, minval=-1.0, dtype=jnp.float32),
        self.block_scale_configs[0].data_type,
    )
    b = cast_to_representable(
        jax.random.uniform(k2, b_shape, minval=-1.0, dtype=jnp.float32),
        self.block_scale_configs[1].data_type,
    )

    scaled_dot_general = partial(
        scaled_dot_general_wrapper,
        configs=self.block_scale_configs
    )
    def fwd(a, b, is_ref=False):
      fn = jax.vmap(
          jax.lax.dot_general if is_ref else scaled_dot_general,
          in_axes=(a_axis, b_axis, None),
          out_axes=o_axis,
      )
      y = fn(a, b, dimension_numbers)
      return jnp.sum(y)

    j_train = jax.jit(jax.value_and_grad(fwd, argnums=[0, 1]))
    j_train_ref = jax.jit(
        jax.value_and_grad(partial(fwd, is_ref=True), argnums=[0, 1])
    )
    out, (x_grad, w_grad) = j_train(a, b)
    out_ref, (x_grad_ref, w_grad_ref) = j_train_ref(a, b)

    self.assertArraysAllClose(out, out_ref, rtol=1e-2, atol=1e2)
    self.assertArraysAllClose(x_grad, x_grad_ref, rtol=1e-2, atol=1e1)
    self.assertArraysAllClose(w_grad, w_grad_ref, rtol=1e-2, atol=1e1)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
