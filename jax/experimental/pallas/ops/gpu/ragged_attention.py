"""Kernels for ragged attention."""

import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


@functools.partial(jax.jit, static_argnames=["mask_value"])
def mqa_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> jax.Array:
  """Multi query attention reference.

  Args:
    q: A [batch_size, num_heads, head_dim] jax.Array.
    k: A [batch_size, seq_len, head_dim] jax.Array.
    v: A [batch_size, seq_len, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.

  Returns:
    The output of attention([batch_size, num_heads, head_dim]).
  """
  seq_len = k.shape[1]
  logits = jnp.einsum(
      "bhd,btd->bht", q.astype(jnp.float32), k.astype(jnp.float32)
  )

  mask = jnp.arange(seq_len)[None, :] < lengths[:, None]
  mask = jnp.expand_dims(mask, 1)  # [batch_size, 1, seq_len]
  logits = logits + jnp.where(mask, 0.0, mask_value)

  logits = jax.nn.softmax(logits, axis=-1)
  o = jnp.einsum("bht,btd->bhd", logits, v.astype(jnp.float32))

  return o.astype(q.dtype)


@functools.partial(jax.jit, static_argnames=["mask_value"])
def gqa_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> jax.Array:
  """Grouped query attention reference.

  Args:
    q: A [batch_size, num_q_heads, head_dim] jax.Array.
    k: A [batch_size, seq_len, num_kv_heads, head_dim] jax.Array.
    v: A [batch_size, seq_len, num_kv_heads, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.

  Returns:
    The output of attention([batch_size, num_q_heads, head_dim]).
  """

  batch_size, num_q_heads, head_dim = q.shape
  _, seq_len, num_kv_heads, _ = k.shape

  if num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"num_q_heads {num_q_heads} must be a multiple of num_kv_heads"
        f" {num_kv_heads}"
    )

  orig_q_shape = q.shape
  num_groups = num_q_heads // num_kv_heads

  q = jnp.reshape(
      q, [batch_size, num_kv_heads, num_q_heads // num_kv_heads, head_dim]
  )
  logits = jnp.einsum(
      "bhgd,bthd->bhgt", q.astype(jnp.float32), k.astype(jnp.float32)
  )

  mask = jnp.arange(seq_len)[None, :] < lengths[:, None]
  mask = jnp.expand_dims(mask, (1, 2))  # (batch_size, 1, 1, seq_len)
  mask = jnp.broadcast_to(mask, (batch_size, num_kv_heads, num_groups, seq_len))

  logits = logits + jnp.where(mask, 0.0, mask_value)
  logits = jax.nn.softmax(logits)

  o = jnp.einsum("bhgt,bthd->bhgd", logits, v.astype(jnp.float32))
  o = jnp.reshape(o, orig_q_shape)

  return o.astype(q.dtype)


def flash_attention_kernel(
    lengths_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    *,
    block_k: int,
    mask_value: float,
):
  b = pl.program_id(0)
  num_heads, head_dim = q_ref.shape

  m_i = jnp.zeros(num_heads, dtype=jnp.float32) - jnp.finfo(jnp.float32).max
  l_i = jnp.zeros(num_heads, dtype=jnp.float32)
  o = jnp.zeros((num_heads, head_dim), dtype=jnp.float32)

  length = lengths_ref[b]

  q = q_ref[...].astype(jnp.float32)  # (block_h, head_dim)

  def body(start_k, carry):
    o_prev, m_prev, l_prev = carry

    curr_kv_slice = pl.dslice(start_k * block_k, block_k)
    k = pl.load(k_ref, (curr_kv_slice, slice(None))).astype(jnp.float32)
    v = pl.load(v_ref, (curr_kv_slice, slice(None))).astype(jnp.float32)

    def _dot(a, b):
      if a.shape[0] == 1:
        # Use matrix vector product
        return (a.T * b).sum(axis=0, keepdims=True)

      return pl.dot(a, b)

    qk = _dot(q, k.T)

    mask = (start_k * block_k + jnp.arange(block_k)) < length  # (block_k,)
    mask = jax.lax.broadcast_in_dim(mask, (num_heads, block_k), (1,))
    qk = qk + jnp.where(mask, 0.0, mask_value)

    m_curr = qk.max(axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    l_prev_corr = alpha * l_prev
    s_curr = jnp.exp(qk - m_next[:, None])
    l_curr = s_curr.sum(axis=-1)
    l_next = l_prev_corr + l_curr
    o_prev_corr = alpha[:, None] * o_prev
    o_curr = _dot(s_curr.astype(v.dtype), v)
    o_next = o_prev_corr + o_curr

    return o_next, m_next, l_next

  o, _, l_i = lax.fori_loop(0, pl.cdiv(length, block_k), body, (o, m_i, l_i))
  o /= l_i[:, None]

  o_ref[...] = o.astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=[
        "block_h",
        "block_k",
        "num_warps",
        "num_stages",
        "mask_value",
        "interpret",
        "debug",
    ],
)
def mqa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    *,
    block_h: int = 16,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 2,
    mask_value: float = DEFAULT_MASK_VALUE,
    interpret: bool = False,
    debug: bool = False,
) -> jax.Array:
  """Ragged multi query attention."""
  batch_size, num_heads, head_dim = q.shape
  seq_len = k.shape[1]

  if lengths.shape != (batch_size,):
    raise ValueError(
        f"lengths should have shape: {(batch_size,)}, but got {lengths.shape}."
    )

  if lengths.dtype != jnp.int32:
    raise ValueError(
        f"lengths should have dtype: {jnp.int32}, but got {lengths.dtype}."
    )

  block_k = min(block_k, seq_len)
  if seq_len % block_k != 0:
    raise ValueError(
        f"Sequence length {seq_len} must be a multiple of block_k {block_k}"
    )

  _num_heads = num_heads
  if _num_heads == 1:
    # We will use a matrix vector product instead of a  matmul.
    block_h = 1
  elif _num_heads < 16:
    q = jnp.pad(q, ((0, 0), (0, 16 - _num_heads), (0, 0)))
    _num_heads = 16

  if _num_heads % block_h != 0:
    raise ValueError(
        f"Number of heads {_num_heads} must be a multiple of block_h {block_h}"
    )

  _grid = (batch_size, _num_heads // block_h)

  out = pl.pallas_call(
      functools.partial(
          flash_attention_kernel, block_k=block_k, mask_value=mask_value
      ),
      in_specs=[
          pl.BlockSpec((batch_size,), lambda b, i: (0,)),
          pl.BlockSpec((None, block_h, head_dim), lambda b, i: (b, i, 0)),
          pl.BlockSpec((None, seq_len, head_dim), lambda b, i: (b, 0, 0)),
          pl.BlockSpec((None, seq_len, head_dim), lambda b, i: (b, 0, 0)),
      ],
      out_specs=pl.BlockSpec((None, block_h, head_dim), lambda b, i: (b, i, 0)),
      grid=_grid,
      out_shape=q,
      name=f"ragged_attention_{block_k}block_k_{block_h}block_h",
      debug=debug,
      interpret=interpret,
      compiler_params=dict(
          triton=dict(num_warps=num_warps, num_stages=num_stages)
      ),
  )(lengths, q, k, v)


  out = out[:, :num_heads, :]

  return out


@functools.partial(
    jax.jit,
    static_argnames=[
        "block_h",
        "block_k",
        "num_warps",
        "num_stages",
        "mask_value",
        "interpret",
        "debug",
    ],
)
def gqa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    *,
    block_h: int = 16,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 2,
    mask_value: float = DEFAULT_MASK_VALUE,
    interpret: bool = False,
    debug: bool = False,
) -> jax.Array:
  """Ragged grouped query attention."""
  batch_size, num_q_heads, head_dim = q.shape
  _, seq_len, num_kv_heads, _ = k.shape

  kv_shape = (batch_size, seq_len, num_kv_heads, head_dim)

  if k.shape != kv_shape:
    raise ValueError(f"key should have shape: {kv_shape}, but got {k.shape}.")

  if v.shape != kv_shape:
    raise ValueError(f"value should have shape: {kv_shape}, but got {v.shape}.")

  if lengths.shape != (batch_size,):
    raise ValueError(
        f"lengths should have shape: {(batch_size,)}, but got {lengths.shape}."
    )

  if lengths.dtype != jnp.int32:
    raise ValueError(
        f"lengths should have dtype: {jnp.int32}, but got {lengths.dtype}."
    )

  if num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"num_q_heads {num_q_heads} must be a multiple of num_kv_heads"
        f" {num_kv_heads}"
    )

  orig_q_shape = q.shape

  lengths = jax.lax.broadcast_in_dim(lengths, (batch_size, num_kv_heads), (0,))
  lengths = jnp.reshape(lengths, [batch_size * num_kv_heads])

  q = jnp.reshape(q, [-1, num_q_heads // num_kv_heads, head_dim])
  k = jnp.reshape(jnp.swapaxes(k, 1, 2), [-1, seq_len, head_dim])
  v = jnp.reshape(jnp.swapaxes(v, 1, 2), [-1, seq_len, head_dim])

  out = mqa(
      q,
      k,
      v,
      lengths,
      block_h=block_h,
      block_k=block_k,
      num_warps=num_warps,
      num_stages=num_stages,
      mask_value=mask_value,
      interpret=interpret,
      debug=debug,
  )  # (batch_size * head_groups, num_kv_heads, head_dim)

  out = jnp.reshape(out, orig_q_shape)  # (batch_size, num_q_heads, head_dim)
  return out
