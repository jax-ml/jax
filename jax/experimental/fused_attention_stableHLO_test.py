import argparse
from functools import partial
from typing import Any, Optional
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.fused_attention_stableHLO import dot_product_attention

from jax import core, vmap
import os

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray


def f(input_q: Array,
    input_k: Array,
    input_v: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    mask_type: Optional[str] = None,
    scale: float = 0.5) -> Array:

    output = dot_product_attention(
        input_q,
        input_k,
        input_v,
        scale=scale,
        bias=bias,
        mask=mask,
        is_cauasl_mask=False,
        dropout_rate=0.1)

    return output

def train_step(input_q, input_k, input_v, bias, mask, grad):
    out, f_vjp = jax.vjp(
        vmap(partial(f, scale=4.0, mask_type=""), in_axes=(0, 0, 0, 0, 0)),
        input_q, input_k, input_v, bias, mask
    )
    input_q_grad, input_k_grad, input_v_grad, bias_grad, _ = f_vjp(grad)
    return out, (input_q_grad, input_k_grad, input_v_grad, bias_grad, _)

def main():
    parser = argparse.ArgumentParser(description='T5X MHA Unit Test')
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--q_seq_len", dest="q_seq_len", type=int, default=256)
    parser.add_argument("--kv_seq_len", dest="kv_seq_len", type=int, default=256)
    parser.add_argument("--num_attn_heads", dest="num_attn_heads", type=int, default=16)
    parser.add_argument("--head_dim", dest="head_dim", type=int, default=64)
    parser.add_argument("--scale", dest="scale", type=float, default=1.0)
    parser.add_argument("--dropout_rate", dest="dropout_rate", type=float, default=0.)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--mask_type", dest="mask_type", type=str, default=None)
    parser.add_argument("--fwd_only", dest='fwd_only', action="store_true")
    parser.add_argument("--xla_dump_to", dest="xla_dump_to", type=str, default=None)
    args = parser.parse_args()
   
    # Set XLA flags with additional flags when xla_dump_to is provided
    xla_flags = '--xla_gpu_enable_triton_gemm=false --xla_gpu_simplify_all_fp_conversions --xla_gpu_fused_attention_use_cudnn_rng=true --xla_gpu_enable_cudnn_fmha=true --xla_gpu_enable_xla_runtime_executable=false --xla_gpu_fused_attention_use_cudnn_rng=true'

    if args.xla_dump_to:
        xla_flags += f' --xla_dump_disable_metadata=true --xla_dump_hlo_pass_re=.* --xla_dump_hlo_as_text --xla_dump_to={args.xla_dump_to}'

    os.environ['XLA_FLAGS'] = xla_flags

    dtype = jnp.bfloat16
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key, 2)
    input_q = jax.random.uniform(key2, (4, args.batch_size, args.q_seq_len, args.num_attn_heads, args.head_dim), dtype=dtype)
    input_k = jax.random.uniform(key2, (4, args.batch_size, args.kv_seq_len, args.num_attn_heads, args.head_dim), dtype=dtype)
    input_v = jax.random.uniform(key2, (4, args.batch_size, args.kv_seq_len, args.num_attn_heads, args.head_dim), dtype=dtype)
    bias = jax.random.uniform(key2, (4, args.batch_size, args.num_attn_heads, args.q_seq_len, args.kv_seq_len), dtype=dtype) if args.bias else None
    mask = jax.random.uniform(key2, (4, args.batch_size, args.num_attn_heads, args.q_seq_len, args.kv_seq_len), dtype=dtype) if args.mask else None
    grad = jax.random.uniform(key2, (4, args.batch_size, args.q_seq_len, args.num_attn_heads, args.head_dim), dtype=dtype)

    if args.fwd_only:
        jitted_fwd_step = jax.jit(partial(f, mask_type=args.mask_type, scale=args.scale))
        out = jitted_fwd_step(input_q, input_k, input_v, bias, mask)
        return out
    else:
        jitted_train_step = jax.jit(train_step)
        # out, input_grad = jitted_train_step(args.mask_type, args.scale, input_q, input_k, input_v, bias, mask, grad)
        # out, input_grad = jitted_train_step(input_q, input_k, input_v, bias, mask, grad)
        # print(input_grad[0][0,0,0,:20])
        # return out, input_grad
        import numpy as np
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec, NamedSharding
        from jax.experimental.pjit import pjit
        devices = np.array(jax.local_devices())
        devices = devices.reshape((2, 4, 1, 1, 1))
        with Mesh(devices, ('p', 'b', 's', 'n', 'd')) as mesh:
            input_q = jax.device_put(input_q, NamedSharding(mesh, PartitionSpec('p', 'b', None, None, None)))
            input_k = jax.device_put(input_k, NamedSharding(mesh, PartitionSpec('p', 'b', None, None, None)))
            input_v = jax.device_put(input_v, NamedSharding(mesh, PartitionSpec('p', 'b', None, None, None)))
            bias = jax.device_put(bias, NamedSharding(mesh, PartitionSpec('p', 'b', None, None, None)))
            mask = jax.device_put(mask, NamedSharding(mesh, PartitionSpec('p', 'b', None, None, None)))
            grad = jax.device_put(grad, NamedSharding(mesh, PartitionSpec('p', 'b', None, None, None)))

            pjitter = pjit(jitted_train_step,
                in_shardings=(
                            PartitionSpec('p', 'b', None, None, None),
                            PartitionSpec('p', 'b', None, None, None),
                            PartitionSpec('p', 'b', None, None, None),
                            PartitionSpec('p', 'b', None, None, None),
                            PartitionSpec('p', 'b', None, None, None),
                            PartitionSpec('p', 'b', None, None, None)),
                out_shardings=(None, 
                            (PartitionSpec('p', 'b', None, None, None),
                            PartitionSpec('p', 'b', None, None, None),
                            PartitionSpec('p', 'b', None, None, None),
                            None, 
                            None))
            )

            out, grads = pjitter(input_q, input_k, input_v, bias, mask, grad)
            print(grads[0][0,0,0,0,:20])

if __name__ == "__main__":
    main()