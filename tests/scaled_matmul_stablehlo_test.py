from functools import partial
from absl.testing import absltest

import re
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec, NamedSharding
from jax._src import config
from jax._src import test_util as jtu
from jax._src.cudnn.fused_attention_stablehlo import check_cudnn_version
from jax._src.cudnn.scaled_matmul_stablehlo import (
    scaled_matmul,
    mxfp8_dot_general,
    quantize,
    shape_normalization,
)


config.parse_flags_with_absl()
input_sharding_configs = [
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
sharding_configs = [[i, j] for i, j in zip(input_sharding_configs, expected_hlos)]

def quantize_to_fp8(x, q_dtype, compute_dtype, scale):
  # Explicitly cast the max values to the compute dtype to avoid unnecessary
  # casting to FP32 during the subsequent math operations."
  assert q_dtype in (jnp.float8_e4m3fn, )
  dtype_max = jnp.finfo(q_dtype).max.astype(compute_dtype)
  scaled_x = x / jnp.broadcast_to(
      jnp.asarray(scale, dtype=compute_dtype), x.shape
  )
  clipped_x = jnp.clip(scaled_x, -dtype_max, dtype_max)
  return clipped_x.astype(q_dtype)

def quantize_dequantize_fp8(x, q_dtype, scale, compute_dtype):
  qx = quantize_to_fp8(x, q_dtype, compute_dtype, scale)
  out = qx.astype(x.dtype) * jnp.broadcast_to(
      jnp.asarray(scale, dtype=x.dtype), qx.shape
  )
  return out

def generate_quantized_tensors(
    batch, lhs_non_contract, contract, rhs_non_contract, dtype=jnp.float32
  ):
  cast_to_representable = partial(
      quantize_dequantize_fp8,
      scale=jnp.ones((1,)),
      compute_dtype=dtype,
  )

  k1, k2 = jax.random.split(jax.random.key(123), 2)

  f8_dtype = jnp.float8_e4m3fn

  a = cast_to_representable(
      jax.random.uniform(
          k1, (batch, lhs_non_contract, contract), minval=-1.0, dtype=dtype
      ),
      f8_dtype,
  )
  b = cast_to_representable(
      jax.random.uniform(
          k2, (batch, rhs_non_contract, contract), minval=-1.0, dtype=dtype
      ),
      f8_dtype,
  )

  dn = ((2,), (0,))
  a_3d = shape_normalization(a, dn)
  b_3d = shape_normalization(b, dn)
  a_q, a_scales = quantize(a, f8_dtype)
  b_q, b_scales = quantize(b, f8_dtype)

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


def get_hlo_text(in_shardings):
  mesh_names = ("dp", "tp")
  devices = np.array(jax.local_devices()[:4]).reshape((2, 2))
  mesh = Mesh(devices, mesh_names)
  _, _, a_q, b_q, a_scales, b_scales = generate_quantized_tensors(
      2, 512, 1024, 512
  )

  with mesh:
    a_q, b_q, a_scales, b_scales, in_shardings = shard_and_device_put(
        mesh, in_shardings[0], in_shardings[1], a_q, b_q, a_scales, b_scales
    )
    pjit_fn = jax.jit(scaled_matmul, in_shardings=in_shardings)
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
    if cudnn_version < 90700:
      self.skipTest("Requires >= cuDNN 9.7.0")
    if not jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest("Requires at least Blackwell arch")

  @jtu.sample_product(
      sharding_config=sharding_configs,
  )
  @jtu.run_on_devices("cuda")
  def test_collectives(self, sharding_config):
    if jtu.device_under_test() != "gpu" or len(jax.local_devices()) < 4:
      self.skipTest("Partition Test enabled for at least 4 GPUs")

    input_sharding, expected_hlo = sharding_config[0], sharding_config[1]
    hlo_text = get_hlo_text(input_sharding)

    hlo_pattern = re.compile(
        r".*".join([re.escape(x) for x in expected_hlo]), flags=re.DOTALL
    )
    self.assertRegex(
        hlo_text, hlo_pattern, msg=f"Failed to find pattern: {expected_hlo}"
    )

  @jtu.sample_product(
      contract=[160, 96],
      lhs_non_contract=[240, 100],
      dtype=[jnp.float16, jnp.bfloat16, jnp.float32],
  )
  @jtu.run_on_devices("cuda")
  def test_scaled_matmul(self, contract, lhs_non_contract, dtype):
    batch, rhs_non_contract = 2, 128
    a, b, a_q, b_q, a_scales, b_scales = generate_quantized_tensors(
        batch, lhs_non_contract, contract, rhs_non_contract, dtype=dtype
    )

    def wrapper(lhs, rhs, lhs_scales, rhs_scales, out_type):
      return scaled_matmul(
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

  @jtu.sample_product(sharding_config=sharding_configs)
  @jtu.run_on_devices("cuda")
  def test_scaled_matmul_sharded(self, sharding_config):
    if len(jax.local_devices()) < 4:
      self.skipTest("Require at least 4 devices to run sharding tests.")
    batch, contract, non_contract = 2, 1024, 256

    a, b, a_q, b_q, a_scales, b_scales = generate_quantized_tensors(
        batch, non_contract, contract, non_contract
    )

    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    in_shardings = sharding_config[0]

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
          scaled_matmul, in_shardings=input_shardings
      )
      hlo_text = j_scaled_matmul.lower(*args).compile().as_text()
      hlo_pattern = re.compile(
          r".*".join([re.escape(x) for x in ("custom-call", c_name)])
      )
      self.assertRegex(hlo_text, hlo_pattern)

      j_ref = jax.jit(
          partial(
              jax.lax.dot_general,
              dimension_numbers=(([2], [2]), ([0], [0])),
          ),
          in_shardings=input_shardings[:2],
      )

      out = j_scaled_matmul(*args)
      out_ref = j_ref(a, b)
      self.assertArraysAllClose(out, out_ref, rtol=1e-3, atol=1e-3)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class Mxfp8DotGeneralTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if cudnn_version < 90700:
      self.skipTest("Requires >= cuDNN 9.7.0")
    if not jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest("Requires at least Blackwell arch")

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
        quantize_dequantize_fp8,
        scale=jnp.ones((1,)),
        compute_dtype=jnp.float32,
    )
    k1, k2 = jax.random.split(jax.random.key(0), 2)

    a_shape, b_shape, dimension_numbers, is_training = configs
    a = cast_to_representable(
        jax.random.uniform(k1, a_shape, minval=-1.0, dtype=output_type),
        jnp.float8_e4m3fn,
    )
    b = cast_to_representable(
        jax.random.uniform(k2, b_shape, minval=-1.0, dtype=output_type),
        jnp.float8_e4m3fn,
    )

    def fwd(a, b, is_ref=False):
      fn = jax.lax.dot_general if is_ref else mxfp8_dot_general
      y = fn(a, b, dimension_numbers, preferred_element_type=output_type)
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

  @jtu.sample_product(sharding_config=input_sharding_configs)
  @jtu.run_on_devices("cuda")
  def test_dot_general_sharded(self, sharding_config):
    if len(jax.local_devices()) < 4:
      self.skipTest("Require at least 4 devices to run sharding tests.")

    cast_to_representable = partial(
        quantize_dequantize_fp8,
        scale=jnp.ones((1,)),
        compute_dtype=jnp.float32,
    )

    dimension_numbers = (([2], [2]), ([0], [0]))
    a_shape = (2, 128, 512)
    b_shape = (2, 256, 512)

    k1, k2 = jax.random.split(jax.random.key(0), 2)
    a = cast_to_representable(
        jax.random.uniform(k1, a_shape, minval=-1.0), jnp.float8_e4m3fn
    )
    b = cast_to_representable(
        jax.random.uniform(k2, b_shape, minval=-1.0), jnp.float8_e4m3fn
    )

    def fwd(a, b, is_ref=False):
      fn = jax.lax.dot_general if is_ref else mxfp8_dot_general
      y = fn(a, b, dimension_numbers)
      # Use a little complex loss function to avoid constant grads, whose
      # sharding info might be optimized off and then cause issue with the
      # custom scaled_matmul op.
      return jnp.sum(jnp.tanh(y))

    in_shardings = sharding_config
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
      hlo_text = j_train.lower(a, b).compile().as_text()
      hlo_pattern = re.compile(
          r".*".join([re.escape(x) for x in ("custom-call", c_name)])
      )

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
        quantize_dequantize_fp8,
        scale=jnp.ones((1,)),
        compute_dtype=jnp.float32,
    )
    k1, k2 = jax.random.split(jax.random.key(0), 2)

    a_shape, b_shape, vmap_axes = configs
    a_axis, b_axis, o_axis = vmap_axes
    dimension_numbers = (([1], [1]), ([], []))

    a = cast_to_representable(
        jax.random.uniform(k1, a_shape, minval=-1.0), jnp.float8_e4m3fn
    )
    b = cast_to_representable(
        jax.random.uniform(k2, b_shape, minval=-1.0), jnp.float8_e4m3fn
    )

    def fwd(a, b, is_ref=False):
      fn = jax.vmap(
          jax.lax.dot_general if is_ref else mxfp8_dot_general,
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


