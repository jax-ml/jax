"""Minimal model definition."""

# CCC: Branched from https://github.com/sholtodouglas/minformer/blob/main/minformer/model_test.py
# All the changes from the original are marked with a CCC if it is a benign
# change or with TTT if it is a temporary workaround that requires some
# improvements in the symbolic shape support.
#
"""Minimal model definition."""

import dataclasses
import math
from collections import namedtuple
from dataclasses import field
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import struct
from jax.experimental import mesh_utils
from jax.experimental.pallas.ops.tpu import flash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

jax.config.parse_flags_with_absl()

# TTT: introduce a global flag to workaround the fact that some of the code
# cannot be processed symbolically.
TTT = True

# CCC: I would like to use a device mesh with a symbolic number of devices
#  but don't know how.
def create_mesh():
  """Always 1D because only care about FSDP."""
  devices = jax.devices()
  mesh_shape = (len(devices),)
  # Create a 1D mesh with all devices along the 'x' axis
  mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh(mesh_shape, devices), ("x",))
  return mesh


ShardingRules = namedtuple(
    "FSDPRules",
    [
        "batch",
        "sequence",
        "d_model",
        "query_heads",
        "key_heads",
        "key_dim",
        "ffw",
        "vocab",
    ],
)

# Define sharding rules for Fully Sharded Data Parallelism (FSDP)
fsdp_rules = ShardingRules(
    batch="x",  # Shard batch dimension
    sequence=None,  # Don't shard sequence dimension
    d_model="x",  # Shard model dimension
    query_heads=None,
    key_heads=None,
    key_dim=None,
    ffw=None,
    vocab=None,
)

# Define sharding rules for model parallelism
mdl_parallel_rules = ShardingRules(
    batch=None,
    sequence=None,
    d_model=None,
    query_heads="x",  # Shard query heads
    key_heads="x",  # Shard key heads
    key_dim=None,
    ffw="x",  # Shard feed-forward layer
    vocab=None,
)


def _logical_to_physical(logical: P, rules: ShardingRules):
  """Converts logical to physical pspec."""
  return P(*(getattr(rules, axis) for axis in logical))


def _logical_to_sharding(logical: P, mesh: jax.sharding.Mesh, rules: ShardingRules):
  """Converts logical to sharding."""
  return jax.sharding.NamedSharding(mesh, _logical_to_physical(logical, rules))


@struct.dataclass
class Config:
  d_model: int
  ffw_multiplier: int
  query_heads: int
  key_heads: int
  num_layers: int
  key_dim: int
  vocab_size: int
  # Max seq len here can be a source of nasty bugs in incremental prefill
  # if we overflow (since dynamic slice will shunt left instead of erroring. Fix?
  max_seq_len: int
  causal: bool
  use_attn_kernel: bool
  weight_dtype_at_rest: jnp.float32
  active_weight_dtype: jnp.bfloat16
  # Sharding rules
  rules: ShardingRules
  mesh: jax.sharding.Mesh | None
  # Optimizer config
  max_lr: float = 3e-4
  min_lr: float = 1e-5
  weight_decay: float = 1e-2
  beta1: float = 0.9
  beta2: float = 0.999
  eps: float = 1e-8
  amsgrad: bool = False
  warmup_steps: int = 50
  total_steps: int = 10000
  # Rescale gradients which spike.
  grad_norm_clip: float = 1


@struct.dataclass
class TensorInfo:
  shape: jax.ShapeDtypeStruct
  logical_axes: tuple
  initializer: Callable | None = None
  metadata: dict = field(default_factory=dict)


@struct.dataclass
class Layer:
  q: jax.Array | TensorInfo
  k: jax.Array | TensorInfo
  v: jax.Array | TensorInfo
  proj: jax.Array | TensorInfo
  w1: jax.Array | TensorInfo
  w2: jax.Array | TensorInfo
  attn_gamma: jax.Array | TensorInfo
  ffn_gamma: jax.Array | TensorInfo
  q_gamma: jax.Array | TensorInfo
  k_gamma: jax.Array | TensorInfo

  @classmethod
  def abstract(cls, cfg: Config):
    return Layer(
        q=TensorInfo(
            jax.ShapeDtypeStruct((cfg.d_model, cfg.query_heads, cfg.key_dim), cfg.weight_dtype_at_rest),
            ("d_model", "query_heads", "key_dim"),
            jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2)),
        ),
        k=TensorInfo(
            jax.ShapeDtypeStruct((cfg.d_model, cfg.key_heads, cfg.key_dim), cfg.weight_dtype_at_rest),
            ("d_model", "key_heads", "key_dim"),
            jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2)),
        ),
        v=TensorInfo(
            jax.ShapeDtypeStruct((cfg.d_model, cfg.key_heads, cfg.key_dim), cfg.weight_dtype_at_rest),
            ("d_model", "key_heads", "key_dim"),
            jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2)),
        ),
        proj=TensorInfo(
            jax.ShapeDtypeStruct((cfg.query_heads, cfg.key_dim, cfg.d_model), cfg.weight_dtype_at_rest),
            ("query_heads", "key_dim", "d_model"),
            jax.nn.initializers.zeros,
        ),
        w1=TensorInfo(
            jax.ShapeDtypeStruct((cfg.d_model, cfg.d_model * cfg.ffw_multiplier), cfg.weight_dtype_at_rest),
            ("d_model", "ffw"),
            jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
        ),
        w2=TensorInfo(
            jax.ShapeDtypeStruct((cfg.d_model * cfg.ffw_multiplier, cfg.d_model), cfg.weight_dtype_at_rest),
            ("ffw", "d_model"),
            jax.nn.initializers.zeros,
        ),
        attn_gamma=TensorInfo(
            jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
            ("d_model",),
            jax.nn.initializers.constant(1.0),
        ),
        ffn_gamma=TensorInfo(
            jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
            ("d_model",),
            jax.nn.initializers.constant(1.0),
        ),
        q_gamma=TensorInfo(
            jax.ShapeDtypeStruct((cfg.key_dim,), cfg.weight_dtype_at_rest),
            ("key_dim",),
            jax.nn.initializers.constant(1.0),
        ),
        k_gamma=TensorInfo(
            jax.ShapeDtypeStruct((cfg.key_dim,), cfg.weight_dtype_at_rest),
            ("key_dim",),
            jax.nn.initializers.constant(1.0),
        ),
    )


@struct.dataclass
class Weights:
  layers: list[Layer]
  embedding: jax.Array | TensorInfo
  gamma_final: jax.Array | TensorInfo

  @classmethod
  def abstract(cls, cfg: Config):
    return Weights(
        layers=[Layer.abstract(cfg) for _ in range(cfg.num_layers)],
        embedding=TensorInfo(
            jax.ShapeDtypeStruct((cfg.vocab_size, cfg.d_model), cfg.weight_dtype_at_rest),
            ("vocab", "d_model"),
            jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
        ),
        gamma_final=TensorInfo(
            jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
            ("d_model",),
            jax.nn.initializers.constant(1.0),
        ),
    )

  @classmethod
  def shardings(cls, cfg: Config, mesh: jax.sharding.Mesh, rules: dict):
    abstract = cls.abstract(cfg)
    return jax.tree.map(
        lambda info: _logical_to_sharding(info.logical_axes, mesh, rules),
        abstract,
        is_leaf=lambda x: isinstance(x, TensorInfo),
    )

  @classmethod
  def init(
      cls, cfg: Config, key: jax.random.PRNGKey, mesh: jax.sharding.Mesh, rules: dict, use_low_mem_init: bool = True
  ):
    def _init():
      abstract = cls.abstract(cfg)
      # Create one new RNG key per tensor.
      num_leaves = len(jax.tree_util.tree_leaves(abstract))
      key_iter = iter(jax.random.split(key, num_leaves))
      return jax.tree.map(
          lambda info: info.initializer(next(key_iter), info.shape.shape, info.shape.dtype),
          abstract,
          is_leaf=lambda x: isinstance(x, TensorInfo),
      )

    if use_low_mem_init:
      _init = jax.jit(_init, out_shardings=cls.shardings(cfg, mesh, rules))
    return jax.device_put(_init(), cls.shardings(cfg, mesh, rules))


@struct.dataclass
class KVCache:
  k: list[jax.Array]  # (batch_size, key_heads, max_seq_len, key_dim)
  v: list[jax.Array]  # (batch_size, key_heads, max_seq_len, key_dim)
  lengths: jax.Array  # [batch_size]

  @classmethod
  def abstract(cls, cfg: Config, batch_size: int, max_seq_len: int):
    return KVCache(
        k=[
            TensorInfo(
                jax.ShapeDtypeStruct((batch_size, cfg.key_heads, max_seq_len, cfg.key_dim), jnp.bfloat16),
                ("batch", "key_heads", "sequence", "key_dim"),
            )
            for _ in range(cfg.num_layers)
        ],
        v=[
            TensorInfo(
                jax.ShapeDtypeStruct((batch_size, cfg.key_heads, max_seq_len, cfg.key_dim), jnp.bfloat16),
                ("batch", "key_heads", "sequence", "key_dim"),
            )
            for _ in range(cfg.num_layers)
        ],
        lengths=TensorInfo(
            jax.ShapeDtypeStruct((batch_size,), jnp.int32),
            ("batch",),
        ),
    )

  @classmethod
  def shardings(cls, cfg: Config, mesh: jax.sharding.Mesh, rules: ShardingRules):
    abstract = cls.abstract(
        cfg, batch_size=1, max_seq_len=cfg.max_seq_len
    )  # Batch size 1, since we just want the axes names.
    return jax.tree.map(
        lambda info: _logical_to_sharding(info.logical_axes, mesh, rules),
        abstract,
        is_leaf=lambda x: isinstance(x, TensorInfo),
    )

  @classmethod
  def init(cls, cfg: Config, batch_size: int, max_seq_len: int):
    abstract = cls.abstract(cfg, batch_size, max_seq_len)
    return jax.tree.map(
        lambda info: jnp.zeros(info.shape.shape, info.shape.dtype),
        abstract,
        is_leaf=lambda x: isinstance(x, TensorInfo),
    )

  @property
  def time_axis(self) -> int:
    return 2


def segment_ids_to_positions(segment_ids):
  """Counts positions for segment ids."""

  def scan_fun(a, b):
    return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

  vals = (jnp.zeros_like(segment_ids), segment_ids)
  # TTT: lax.associative_scan is not supported for symbolic shapes
  if TTT:
    # We only care about shapes
    return jnp.zeros_like(segment_ids)
  return jnp.array(jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32")


def _generate_pos_embeddings(
    positions: jax.Array, features: int, min_timescale=1.0, max_timescale=16384.0
) -> tuple[jax.Array, jax.Array]:
  """Generate Sin/Cos for Rotary Embeddings.

  Generates sinusoids at (features//2) different timescales, where the
  timescales form a geometric series from min_timescale to max_timescale
  (max_timescale is not included, but would be the next element in the series).

  Sinusoids are evaluated at integer positions i in [0, length).

  The outputs are computed as:


  sin[b, t, j] = sin(rope_pos[b, t] / timescale[j])
  cos[b, t, j] = cos(rope_pos[b, t] / timescale[j])

  Args:
      postions: [batch, time]
      features: d_head.
      min_timescale: an optional float
      max_timescale: an optional float

  Returns:
      output_sin: a float32 Tensor with shape [length, features // 2]
      output_cos: a float32 Tensor with shape [length, features // 2]
  """
  # Forked from
  # flaxformer/components/embedding.py;l=592
  fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
  timescale = min_timescale * (max_timescale / min_timescale) ** fraction
  rotational_frequency = 1.0 / timescale
  # Must use high precision einsum here, since rounding off to a bfloat16 is
  # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
  # from sin(256).
  sinusoid_inp = jnp.einsum(
      "BT,k->BTk",
      positions,
      rotational_frequency,
      precision=jax.lax.Precision.HIGHEST,
  )
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rotary_embedding(x, sin, cos):
  assert x.ndim == 4
  assert sin.ndim == 3 and cos.ndim == 3
  x1, x2 = jnp.split(x, 2, axis=-1)
  sin, cos = (
      sin[:, None, :, :],
      cos[:, None, :, :],
  )  # [B, T, head_dim] -> [B, h, T, head_dim]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal: bool):
  # [B, t, T]
  segment_mask = q_segment_ids[:, :, None] == k_segment_ids[:, None, :]
  # [B, t, T] -> [B, 1, t, T]
  segment_mask = segment_mask[:, None, :, :]

  if causal:
    # [b, h, t, T]
    qk = (1, 1, q_len, k_len)
    q_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 2)
    k_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 3)
    q_positions = q_iota + q_offset[:, None, None, None]
    causal_mask = q_positions >= k_iota
    combined_mask = jnp.logical_and(segment_mask, causal_mask)
    return combined_mask
  else:
    return segment_mask


def attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_segment_ids: jax.Array,
    k_segment_ids: jax.Array,
    q_offset: jax.Array,
    cfg: Config,
) -> jax.Array:
  """
  Compute attention.

  Args:
  q: Query tensor of shape (batch_size, num_heads, q_len, head_dim)
  k: Key tensor of shape (batch_size, num_heads, k_len, head_dim)
  v: Value tensor of shape (batch_size, num_heads, k_len, head_dim)
  q_segment_ids: Query segment IDs of shape (batch_size, q_len)
  k_segment_ids: Key segment IDs of shape (batch_size, k_len)
  q_offset: Query offset of shape (batch_size,)
  cfg: Configuration object

  Returns:
  Attention output of shape (batch_size, num_heads, q_len, head_dim)
  """
  # Div sqrt(key_dim)
  # CCC: symbolic dimensions cannot be raised to non-integer powers
  # Added jnp.array(...) to turn the dimension into an array
  scale = jnp.array(q.shape[-1]) ** -0.5
  qk = jnp.einsum("bhtd,bhTd->bhtT", q, k) * scale
  mask = make_attention_mask(q.shape[2], k.shape[2], q_segment_ids, k_segment_ids, q_offset, cfg.causal)
  # Apply the combined mask
  qk = jnp.where(mask, qk, -1e30)
  # Jax softmax impl includes max subtraction for numerical stability, no need to
  # do it outside.
  attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)
  return jnp.einsum("bhtT,bhTd->bhtd", attn, v).astype(jnp.bfloat16)


def attention_kernel(q, k, v, q_segment_ids, kv_segment_ids, cfg: Config):
  """Flash attention kernel!"""

  # On TPUv3, pallas seems to only work with float32.
  q, k, v = jnp.float32(q), jnp.float32(k), jnp.float32(v)
  if TTT:
    # TTT: the flashattention needs scale to be a static, but we cannot
    # raise a symbolic expression to a non-integer power, unless we turn it
    # into a jax.Array, but at that point it cannot be a static jit arg.
    scale = 8 ** -0.5
  else:
    scale = q.shape[-1] ** -0.5

  @partial(
      shard_map,
      mesh=cfg.mesh,
      in_specs=(
          _logical_to_physical(P("batch", "query_heads", "sequence", "key_dim"), cfg.rules),
          _logical_to_physical(P("batch", "key_heads", "sequence", "key_dim"), cfg.rules),
          _logical_to_physical(P("batch", "key_heads", "sequence", "key_dim"), cfg.rules),
          _logical_to_physical(P("batch", "sequence"), cfg.rules),
          _logical_to_physical(P("batch", "sequence"), cfg.rules),
      ),
      out_specs=_logical_to_physical(P("batch", "query_heads", "sequence", "key_dim"), cfg.rules),
      check_rep=False,
  )
  def _f(q, k, v, q_segment_ids, kv_segment_ids):
    segment_ids = flash_attention.SegmentIds(q_segment_ids, kv_segment_ids)
    return flash_attention.flash_attention(
        q,
        k,
        v,
        segment_ids=segment_ids,
        causal=True,
        sm_scale=scale,
        block_sizes=flash_attention.BlockSizes(
            block_q=512,
            block_k_major=512,
            block_k=512,
            block_b=1,
            block_q_major_dkv=512,
            block_k_major_dkv=512,
            block_k_dkv=512,
            block_q_dkv=512,
            block_k_major_dq=512,
            block_k_dq=512,
            block_q_dq=512,
        ),
    )

  return _f(q, k, v, q_segment_ids, kv_segment_ids).astype(jnp.bfloat16)


def rms_norm(x: jax.Array, gamma: jax.Array) -> jax.Array:
  """Apply RMS normalization."""
  rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
  return jnp.astype(gamma * x / rms, jnp.bfloat16)


def attention_block(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: Layer,
    sin: jax.Array,
    cos: jax.Array,
    cfg: Config,
    cache: KVCache | None = None,
    idx: int | None = None,
):
  # Multi-head attention
  with jax.named_scope("qkv_matmul"):
    q = jnp.einsum("btd,dhq->bhtq", x, layer.q)
    k = jnp.einsum("btd,dhk->bhtk", x, layer.k)
    v = jnp.einsum("btd,dhv->bhtv", x, layer.v)

  # Apply rotary embeddings
  with jax.named_scope("rope"):
    q = apply_rotary_embedding(q, sin, cos)
    k = apply_rotary_embedding(k, sin, cos)

  # QKNorm
  with jax.named_scope("qk_norm"):
    q = rms_norm(q, layer.q_gamma)
    k = rms_norm(k, layer.k_gamma)

  with jax.named_scope("cache_update"):
    if cache is not None:
      cache_k, cache_v = cache.k[idx], cache.v[idx]

      def update(original, update, at):
        # Axis -1 because we are in vmap.
        return jax.lax.dynamic_update_slice_in_dim(original, update, at, axis=cache.time_axis - 1)

      # TODO(sholto): Guaranteed this introduces a gather :)
      k, v = jax.vmap(update, in_axes=(0, 0, 0))(cache_k, k.astype(cache_k.dtype), cache.lengths), jax.vmap(
          update, in_axes=(0, 0, 0)
      )(cache_v, v.astype(cache_v.dtype), cache.lengths)
      q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
      time_indices = jnp.arange(0, v.shape[cache.time_axis])[None, :]  # [1, T]
      incremental_positions = jnp.sum(segment_ids != 0, axis=-1)  # [B,]
      # I.e. valid below where we've written things [B, T]
      k_segment_ids = jnp.where(time_indices < (cache.lengths + incremental_positions)[:, None], 1, 0)
      # Mask our new k and v so that its very visible and easy to test kv values being entered. Tiny perf hit b/c it is unnecessary.
      k, v = k * k_segment_ids[:, None, :, None], v * k_segment_ids[:, None, :, None]
      q_offset = cache.lengths
    else:
      q_segment_ids = segment_ids
      k_segment_ids = segment_ids
      q_offset = jnp.zeros(x.shape[0], dtype=jnp.int32)

  # Compute attention
  with jax.named_scope("attention"):
    if cfg.use_attn_kernel:
      if cache is not None:
        raise ValueError("Kernel is only for training.")
      attn_out = attention_kernel(q, k, v, q_segment_ids, k_segment_ids, cfg)
    else:
      attn_out = attention(q, k, v, q_segment_ids, k_segment_ids, q_offset, cfg)

  # Project attention output
  with jax.named_scope("projection"):
    attn_out = jnp.einsum("bhtq,hqd->btd", attn_out, layer.proj)

  return attn_out, k, v


def ffn_block(x: jax.Array, layer: Layer):
  with jax.named_scope("ffn"):
    ff_out = jnp.einsum("btd,df->btf", x, layer.w1)
    ff_out = jax.nn.relu(ff_out) ** 2  # https://arxiv.org/abs/2109.08668v2
    ff_out = jnp.einsum("btf,fd->btd", ff_out, layer.w2)

  return ff_out


def forward_layer(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: Layer,
    sin: jax.Array,
    cos: jax.Array,
    idx: int,
    cfg: Config,
    cache: KVCache | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  # Cast non-norms to bfloat16 for faster operations.
  layer = dataclasses.replace(layer,
                              q=cfg.active_weight_dtype(layer.q),
                              k=cfg.active_weight_dtype(layer.k),
                              v=cfg.active_weight_dtype(layer.v),
                              w1=cfg.active_weight_dtype(layer.w1),
                              w2=cfg.active_weight_dtype(layer.w2),
                              )

  # Attention block
  with jax.named_scope("attn_pre_norm"):
    attn_in = rms_norm(x, layer.attn_gamma)
  attn_out, k, v = attention_block(attn_in, segment_ids, layer, sin, cos, cfg, cache, idx)
  with jax.named_scope("residual"):
    x = x + attn_out

  # FFN block
  with jax.named_scope("ffn_pre_norm"):
    ff_in = rms_norm(x, layer.ffn_gamma)
  ff_out = ffn_block(ff_in, layer)
  with jax.named_scope("residual"):
    x = x + ff_out

  return x, k, v


def forward(
    x: jax.Array,
    segment_ids: jax.Array,
    weights: Weights,
    cfg: Config,
    cache: KVCache | None = None,
):
  internals = {}
  # Embed input tokens [B, T] -> [B, T D]
  x = weights.embedding[x, :]
  batch = x.shape[0]
  positions = segment_ids_to_positions(segment_ids)
  # Apply rotary embeddings: [B, T, head_dim]
  if cache is not None:
    # For inference with cache, we need to index the positional embeddings
    start_indices = cache.lengths
  else:
    start_indices = jnp.zeros((batch,), dtype=jnp.int32)
  # NOTE: At inference time this only works for UNPACKED sequences.
  positions = start_indices[:, None] + positions
  # [B, T, head_dim]
  sin, cos = _generate_pos_embeddings(positions, cfg.key_dim, min_timescale=1.0, max_timescale=cfg.max_seq_len)

  for idx, layer in enumerate(weights.layers):
    x, k, v = forward_layer(x, segment_ids, layer, sin, cos, idx, cfg, cache)
    if cache is not None:
      cache.k[idx] = k
      cache.v[idx] = v

  # Final layer norm.
  x = rms_norm(x, weights.gamma_final)
  # Project to vocabulary size
  logits = jnp.einsum("btd,vd->btv", x, weights.embedding)
  if cache is not None:
    # Sum where there is a valid segment id (i.e. non padding tokens) [B, T] -> [B,]
    cache = dataclasses.replace(cache, lengths=cache.lengths + jnp.sum(segment_ids != 0, axis=-1))
    return logits, cache, internals
  return logits, internals


# Training.


def get_lr_with_cosine_decay_and_warmup(step: int, total_steps: int, max_lr: float, min_lr: float, warmup_steps: int):
  """Calculate learning rate using cosine decay with linear warmup."""

  def warmup(s):
    return max_lr * (s / warmup_steps)

  def cosine_decay(s):
    progress = (s - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + jnp.cos(jnp.pi * progress))

  return jax.lax.cond(step < warmup_steps, warmup, cosine_decay, step)


def adamw_update(
    param: jax.Array,
    grad: jax.Array,
    m: jax.Array,
    v: jax.Array,
    v_max: jax.Array,
    lr: float,
    t: int,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    amsgrad: bool = False,
):
  # Momentum.
  m = beta1 * m + (1 - beta1) * grad
  # Grad variance.
  v = beta2 * v + (1 - beta2) * jnp.square(grad)
  # Debiasing (helps with early training).
  m_hat = m / (1 - beta1 ** (t + 1))
  v_hat = v / (1 - beta2 ** (t + 1))

  # Adjusts the gradient update w/ momentum by the variance. Effectively
  # high variance = more cautious step, low variance = more aggressive step.
  if amsgrad:
    # AMSGrad: Keep the maximum of past and current v_hat
    v_max = jnp.maximum(v_max, v_hat)
    # Use v_max for the update
    update = lr * m_hat / (jnp.sqrt(v_max) + eps)
  else:
    update = lr * m_hat / (jnp.sqrt(v_hat) + eps)

  # AdamW: Apply weight decay after the main gradient-based update
  param_update = param - update - lr * weight_decay * param
  return param_update, m, v, v_max


def init_optimizer_state(weights: Weights):
  def _zeros_like(old):
    # TTT: we need this because _zeros_like does not work inside a jit; old.sharding is not defined for DynamicJaxprTracer
    if TTT:
      return jnp.zeros_like(old)
    elif isinstance(old, jax.ShapeDtypeStruct):
      return jax.ShapeDtypeStruct(old.shape, old.dtype, sharding=old.sharding)
    else:
      return jax.device_put(jnp.zeros_like(old), old.sharding)

  return jax.tree.map(lambda p: (_zeros_like(p), _zeros_like(p), _zeros_like(p)), weights)


def cross_entropy_loss(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
    internals: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, Any]:
  num_classes = logits.shape[-1]
  labels_one_hot = jax.nn.one_hot(labels, num_classes)
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  loss = -jnp.sum(labels_one_hot * log_probs, axis=-1)
  loss *= mask

  if internals is not None:
    internals["per_token_loss"] = loss

  valid_tokens = jnp.sum(mask)
  # Compute mean over valid values.
  loss = loss.sum() / valid_tokens

  predictions = jnp.argmax(logits, axis=-1)
  correct_predictions = jnp.sum((predictions == labels) * mask)
  accuracy = correct_predictions / valid_tokens

  return (loss, accuracy) if internals is None else (loss, accuracy, internals)


def compute_loss(
    weights: Weights, x: jax.Array, segment_ids: jax.Array, y: jax.Array, cfg: Config
) -> tuple[jax.Array, Any]:
  logits, internals = forward(x, segment_ids, weights, cfg)
  # Important assumption that segment_ids 0 is 'padding'.
  loss_mask = jnp.where(segment_ids == 0, 0, 1)
  loss, accuracy, internals = cross_entropy_loss(logits, y, loss_mask, internals)
  internals["accuracy"] = accuracy
  return loss, internals


def update_weights(weights: Weights, grads: Weights, state: Any, lr: float, t: int, cfg: Config, internals: Any):
  def update_fn(param, grad, state, grad_norm):
    m, v, v_max = state
    # Clip and rescale gradients when they exceed a specified value, useful for training instabilities.
    # This clips per parameter - rather than globally, but that lets us overlap weights sync on the bwd pass.
    # Bit hacky? TBD if needed.
    scale_factor = jnp.maximum(grad_norm, cfg.grad_norm_clip)
    grad = grad / scale_factor.astype(grad.dtype) * cfg.grad_norm_clip
    param_update, m_new, v_new, v_max_new = adamw_update(
        param, grad, m, v, v_max, lr, t, cfg.weight_decay, cfg.beta1, cfg.beta2, cfg.eps, cfg.amsgrad
    )
    return param_update, (m_new, v_new, v_max_new)

  grad_norms = jax.tree.map(jnp.linalg.norm, grads)
  internals["grad_norms"] = grad_norms
  updated = jax.tree.map(update_fn, weights, grads, state, grad_norms)
  # Use weights for it's tree prefix.
  new_weights = jax.tree.map(lambda _, u: u[0], weights, updated)
  new_state = jax.tree.map(lambda _, u: u[1], weights, updated)
  return new_weights, new_state, internals


def update_step(
    weights: Weights,
    x: jax.Array,
    segment_ids: jax.Array,
    y: jax.Array,
    opt_state: Any,
    step: int,
    cfg: Config,
):
  (loss, internals), grads = jax.value_and_grad(compute_loss, has_aux=True)(weights, x, segment_ids, y, cfg)
  lr = get_lr_with_cosine_decay_and_warmup(step, cfg.total_steps, cfg.max_lr, cfg.min_lr, cfg.warmup_steps)
  weights, opt_state, internals = update_weights(weights, grads, opt_state, lr, step, cfg, internals)
  internals["lr"] = lr
  return loss, weights, opt_state, internals


def input_shardings(
    mesh, rules
) -> tuple[jax.sharding.NamedSharding, jax.sharding.NamedSharding, jax.sharding.NamedSharding]:
  logical_axes = {
      "x": P("batch", "sequence"),
      "segment_ids": P("batch", "sequence"),
      "y": P("batch", "sequence"),
  }
  return jax.tree.map(partial(_logical_to_sharding, mesh=mesh, rules=rules), logical_axes)


# Checkpointing logic
def make_mngr(path="/tmp/checkpoint_manager_sharded", erase: bool = False):
  if erase:
    path = ocp.test_utils.erase_and_create_empty(path)
  options = ocp.CheckpointManagerOptions(max_to_keep=3)
  mngr = ocp.CheckpointManager(path, options=options)
  return mngr


def save(mngr: ocp.CheckpointManager, weights: Weights, opt_state: Any, step: int):
  mngr.save(step, args=ocp.args.StandardSave({"weights": weights, "opt_state": opt_state}))
  mngr.wait_until_finished()


def load(mngr: ocp.CheckpointManager, cfg: Config, step: int | None = None):
  abstract_weights = Weights.abstract(cfg)
  weights_shapes_shardings = jax.tree.map(
      lambda info: jax.ShapeDtypeStruct(
          info.shape.shape,
          info.shape.dtype,
          sharding=jax.sharding.NamedSharding(cfg.mesh, _logical_to_physical(info.logical_axes, cfg.rules)),
      ),
      abstract_weights,
      is_leaf=lambda x: isinstance(x, TensorInfo),
  )
  opt_shapes_shardings = init_optimizer_state(weights_shapes_shardings)
  restored = mngr.restore(
      mngr.latest_step() if step is None else step,
      args=ocp.args.StandardRestore({"weights": weights_shapes_shardings, "opt_state": opt_shapes_shardings}),
  )
  return restored["weights"], restored["opt_state"]


# Inference.
def prepare_chunk(chunk, pad_to: int, pad_id: int):
  # [length] -> [1, padded]
  chunk = jnp.pad(chunk, (0, pad_to - len(chunk)))[None, :]
  segment_ids = jnp.where(chunk != pad_id, 1, 0).astype(jnp.int32)
  return chunk, segment_ids


def sample_next_token(logits, temperature=1.0, greedy: bool = True):
  if greedy:
    return jnp.argmax(logits, -1)
  else:
    # Apply temperature
    logits = logits / temperature
    # Convert to probabilities
    probs = jax.nn.softmax(logits, axis=-1)
    # Sample from the distribution
    return jax.random.categorical(jax.random.PRNGKey(0), probs, axis=-1)


def sample_from_prompt(
    tokens: jax.Array,
    weights: Weights,
    cache: KVCache,
    cfg: Config,
    batch_idx: int = 0,
    num_steps: int = 20,
    greedy: bool = True,
):
  """Samples from a prompt."""

  # Calculate the next power of 2 for padding, up to cfg.max_seq.
  # CCC: in presence of symbolic shapes, use tokens.shape[0] instead of len(tokens)
  assert tokens.shape[0] <= cfg.max_seq_len
  # TTT: the following is not supported with symbolic shapes, i.e., we can
  # compute it as a jax.Array but `pad_to` needs to be a symbolic expression
  # below.
  pad_to = 2 ** math.ceil(math.log2((tokens.shape[0])))
  prompt, prompt_segment_ids = prepare_chunk(tokens, pad_to=pad_to, pad_id=0)
  cache = dataclasses.replace(
      cache,
      lengths=jax.lax.dynamic_update_index_in_dim(cache.lengths, 0, batch_idx, axis=0),
  )
  logits, cache, _ = jax.jit(forward, static_argnames="cfg")(prompt, prompt_segment_ids, weights, cfg, cache)
  next_token_logit = logits[batch_idx, cache.lengths[batch_idx] - 1, :]

  tokens = []
  for _ in range(0, num_steps):
    next_token = sample_next_token(next_token_logit, greedy=greedy)[None]
    tokens.append(next_token[0])
    prompt, prompt_segment_ids = prepare_chunk(next_token, pad_to=1, pad_id=0)
    logits, cache, _ = jax.jit(forward, static_argnames="cfg")(prompt, prompt_segment_ids, weights, cfg, cache)
    next_token_logit = logits[batch_idx, 0, :]
  return tokens, cache

#
# CCC: everything below is added to exercise different configurations
#
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from jax import export


class MinformerSymbolicShapesTest(parameterized.TestCase):

  def setup_config(self, causal: bool):
    B, T, N, K, V, D, F, H, S = export.symbolic_shape(
        "B, T, N, K, V, D, F, H, S",
        # We need constraints because symbolic dimension tracing currently is
        # very strict: it needs to resolve all the Python conditionals, e.g.,
        # those in the implementation of jax.numpy and all those in the JAX
        # shape rules.
        constraints=[
            "mod(H, 2) == 0",  # needed because we do .split(2) in apply_rotary_embedding
            "S >= 6",  # In tests we initialize the input to 6 tokens
            "T >= S",  # max_seq_len >= seq_len
            "S >= 512",  # when using attn_kernel the block size is 512
            "mod(S, 512) == 0",
            # CCC or TTT: The following follows from mod(H, 2) == 0, but the
            # reasoning engine is not powerful enough.
            # Needed for some element wise op.
            "floordiv(H + 1, 2) == floordiv(H , 2)",
        ])
    K = N  # TODO: the code seems to be written with the assumption that query_heads == key_heads

    cfg = Config(
        d_model=D,
        ffw_multiplier=F,
        query_heads=N,
        key_heads=K,
        # TTT: the code uses a Python for loop over range(num_layers) to initialize
        # the layers, so the number of layers cannot be symbolic
        num_layers=4,
        key_dim=H,
        vocab_size=V,
        max_seq_len=T,
        causal=causal,
        use_attn_kernel=False,  # Kernel is only for training
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.float32,
        rules=mdl_parallel_rules,  # For inference
        mesh=create_mesh(),
    )
    return cfg, B, S

  @parameterized.named_parameters(
      dict(testcase_name=f"causal={causal}",
           causal=causal)
      for causal in [True, False]
  )
  # Based on model_test.py::test_incremental_prefil
  def test_inference(self, causal: bool = False):
    global TTT
    TTT = True
    cfg, batch_size, seq_len = self.setup_config(causal)

    def harness():
      key = jax.random.PRNGKey(2)
      weights = Weights.init(cfg, key, cfg.mesh, cfg.rules)
      print(f"\nweights shapes: {jax.tree.map(np.shape, weights)}")
      prefill_cache = KVCache.init(cfg=cfg, batch_size=1, max_seq_len=cfg.max_seq_len)
      print(f"prefill_cache shapes: {jax.tree.map(np.shape, prefill_cache)}")

      chunk = jnp.array([1, 2, 3, 4, 5, 6])
      chunk, segment_ids = prepare_chunk(chunk, pad_to=seq_len, pad_id=0)
      print(f"chunk_a shapes: {jax.tree.map(np.shape, chunk)}")
      print(f"segment_ids shapes: {jax.tree.map(np.shape, segment_ids)}")

      logits, prefill_cache_1, _ = forward(chunk, segment_ids, weights, cfg, prefill_cache)
      print(f"logits shapes: {jax.tree.map(np.shape, logits)}")

    jax.eval_shape(harness)


  @parameterized.named_parameters(
      dict(testcase_name=f"use_attn_kernel={use_attn_kernel}",
           use_attn_kernel_training=use_attn_kernel)
      for use_attn_kernel in [True, False]
  )
  def test_training(self, use_attn_kernel_training: bool = True):
    global TTT
    TTT = True
    cfg, batch_size, seq_len = self.setup_config(causal=False)
    if use_attn_kernel_training:
      # TTT: currently, the attn kernel requires the key_dim to be non-symbolic
      # because a kernel block dimension is equal to key_dim, and currently
      # Pallas does not support dynamically-shaped blocks.
      cfg = dataclasses.replace(cfg, key_dim=8)

    def harness():
      key = jax.random.PRNGKey(2)
      training_cfg = dataclasses.replace(cfg, use_attn_kernel=use_attn_kernel_training,
                                         rules=fsdp_rules)
      weights = Weights.init(training_cfg, key, training_cfg.mesh, training_cfg.rules)
      print(f"\nweights shapes: {jax.tree.map(np.shape, weights)}")

      opt_state = init_optimizer_state(weights)
      print(f"opt_state shapes: {jax.tree.map(np.shape, opt_state)}")

      # TTT: in the original this was 256, but results in error when using the
      # attention kernel with blocks of size 512
      test_batch = jnp.arange(1, seq_len + 2)[None, :]
      test_batch = jnp.repeat(test_batch, repeats=batch_size, axis=0)
      batch = {
          "x": test_batch[:, :-1],
          "y": test_batch[:, 1:],
          "segment_ids": jnp.ones((batch_size, seq_len)),
      }
      batch = jax.device_put(batch, input_shardings(training_cfg.mesh, training_cfg.rules))

      prefill_cache = KVCache.init(cfg=training_cfg, batch_size=batch_size, max_seq_len=training_cfg.max_seq_len)
      print(f"prefill_cache shapes: {jax.tree.map(np.shape, prefill_cache)}")

      loss, weights, opt_state, _ = update_step(weights, batch["x"], batch["segment_ids"], batch["y"],
                                                opt_state, 0,
                                                cfg=training_cfg)
      print(f"loss shapes: {jax.tree.map(np.shape, loss)}")

      # For inference we use a batch size of 1
      prompt = jnp.arange(0, 6)  # TTT: we would like to use input_size instead
                                 # of "6" here, but `sample_from_prompt` will want
                                 # to pad this to the nearest power of 2, which is
                                 # not supported with symbolic shapes
      cache = KVCache.init(cfg=cfg, batch_size=1, max_seq_len=cfg.max_seq_len)
      print(f"cache shapes: {jax.tree.map(np.shape, cache)}")

      tokens, cache = sample_from_prompt(prompt, weights, cache, cfg, batch_idx=0, num_steps=1)

    jax.eval_shape(harness)

  # CCC: This is the actual test from the original minformer
  def test_overtrain_and_sample_simple_sequence(self):
    self.skipTest("Works only on TPU")
    global TTT
    TTT = False
    # TODO(sholto): Extend with multiple sequence ids?
    cfg = Config(
        d_model=256,
        ffw_multiplier=4,
        query_heads=8,
        key_heads=8,
        num_layers=4,
        key_dim=128,
        vocab_size=256,
        max_seq_len=8192,
        causal=True,
        use_attn_kernel=True,
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.float32,
        rules=fsdp_rules,
        mesh=create_mesh(),
        max_lr=3e-4,
        min_lr=1e-5,
        warmup_steps=10,
        total_steps=100,
    )
    inference_config = Config(
        d_model=256,
        ffw_multiplier=4,
        query_heads=8,
        key_heads=8,
        num_layers=4,
        key_dim=128,
        vocab_size=256,
        max_seq_len=8192,
        causal=True,
        use_attn_kernel=False,
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.float32,
        rules=mdl_parallel_rules,
        mesh=create_mesh(),
    )
    weights = Weights.init(cfg, jax.random.PRNGKey(0), cfg.mesh, fsdp_rules)
    opt_state = init_optimizer_state(weights)
    step = jax.jit(update_step, static_argnames="cfg")
    step = partial(step, cfg=cfg)

    # CCC: the original test had 256 here, but that does not work with blocks
    # that are 512
    test_batch = jnp.arange(1, 512 + 2)[None, :]
    test_batch = jnp.repeat(test_batch, repeats=8, axis=0)
    batch = {
        "x": test_batch[:, :-1],
        "y": test_batch[:, 1:],
        "segment_ids": jnp.ones((8, 512)),
    }
    batch = jax.device_put(batch, input_shardings(cfg.mesh, cfg.rules))


    loss, weights, opt_state, _ = step(weights, batch["x"], batch["segment_ids"], batch["y"], opt_state, 0)

    prompt = jnp.arange(1, 60)
    cache = KVCache.init(cfg=inference_config, batch_size=1, max_seq_len=2048)
    tokens, cache = sample_from_prompt(prompt, weights, cache, inference_config, batch_idx=0, num_steps=1)


if __name__ == "__main__":
  absltest.main()
