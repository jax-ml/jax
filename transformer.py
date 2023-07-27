from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import ijit

# jax.config.update('jax_platform_name', 'cpu')

class RmsNorm(nn.Module):
  model_dims: int
  eps: float = 1e-6

  def setup(self):
    self.scale = self.param('scale', nn.initializers.zeros, (self.model_dims,))

  def __call__(self, x):
    var = jnp.mean(jnp.square(x), axis=[-1], keepdims=True)
    normed_x = x * jax.lax.rsqrt(var + self.eps)
    return normed_x * (1.0 + self.scale)

class FeedForwardResidualBlock(nn.Module):
  model_dims: int
  hidden_dims: int

  def setup(self) -> None:
    self.pre_layer_norm = RmsNorm(self.model_dims)
    self.post_layer_norm = RmsNorm(self.model_dims)
    self.ffn_layer1 = nn.Dense(self.hidden_dims)
    self.ffn_layer2 = nn.Dense(self.model_dims)

  def __call__(self, x):
    residual = x
    x = self.pre_layer_norm(x)
    x = self.ffn_layer1(x)
    x = jax.nn.relu(x)
    x = self.ffn_layer2(x)
    x = self.post_layer_norm(x)
    x = residual + x
    return x


class AttentionResidualBlock(nn.Module):
  model_dims: int
  hidden_dims: int
  num_heads: int

  def setup(self) -> None:
    self.dims_per_head = self.model_dims // self.num_heads
    D, N, H = self.model_dims, self.num_heads, self.dims_per_head
    self.pre_layer_norm = RmsNorm(self.model_dims)
    self.post_layer_norm = RmsNorm(self.model_dims)
    self.wq = self.param('wq', nn.initializers.zeros, (D, N, H))
    self.wk = self.param('wk', nn.initializers.zeros, (D, N, H))
    self.wv = self.param('wv', nn.initializers.zeros, (D, N, H))
    self.wo = self.param('wo', nn.initializers.zeros, (D, N, H))

  def __call__(self, x):
    residual = x
    x = self.pre_layer_norm(x)
    q = jnp.einsum('BTD,DNH->BTNH', x, self.wq)
    k = jnp.einsum('BTD,DNH->BTNH', x, self.wk)
    v = jnp.einsum('BTD,DNH->BTNH', x, self.wv)
    # logits will be computed in bf16 on TPU but accumulated in f32
    # TODO: find out logits on GPU
    logits = jnp.einsum('BTNH,BSNH->BNTS', q, k, preferred_element_type=jnp.float32)
    logits /= jnp.sqrt(self.dims_per_head)
    # softmax computes in f32, then cast back to inputs dtype (bf16)
    probs = jax.nn.softmax(logits).astype(residual.dtype)
    context = jnp.einsum('BNTS,BSNH->BTNH', probs, v)
    o = jnp.einsum('BTNH,DNH->BTD', context, self.wo)
    o = self.post_layer_norm(o)
    x = residual + o

    return x


class TransformerBlock(nn.Module):
  model_dims: int
  hidden_dims: int
  num_heads: int

  def setup(self):
    self.self_attention = AttentionResidualBlock(self.model_dims, self.hidden_dims, self.num_heads)
    self.ffw = FeedForwardResidualBlock(self.model_dims, self.hidden_dims)

  def __call__(self, x):
    x = self.self_attention(x)
    x = self.ffw(x)
    return x


def apply_position_embedding(x):
  _, T, D = x.shape
  freqs = np.arange(0, D, 2)
  inv_freq = 10_000**(-freqs / D)
  pos_seq = np.arange(T, 0, -1.0)
  sinusoids = np.einsum('i,j->ij', pos_seq, inv_freq)
  pos_emb = np.concatenate([np.sin(sinusoids), np.cos(sinusoids)], axis=-1)
  return x + pos_emb.astype(x.dtype)

class Transformer(nn.Module):
  num_layers: int
  model_dims: int
  hidden_dims: int
  num_heads: int

  def setup(self):
    self.blocks = [TransformerBlock(self.model_dims, self.hidden_dims, self.num_heads) for _ in range(self.num_layers)]

  def __call__(self, x):
    for block in self.blocks:
      x = block(x)
    return x

class LanguageModel(nn.Module):
  num_layers: int
  vocab_size: int
  model_dims: int
  hidden_dims: int
  num_heads: int

  def setup(self):
    self.embed = self.param('embed', nn.initializers.zeros, (self.vocab_size, self.model_dims))
    self.transformer = Transformer(self.num_layers, self.model_dims, self.hidden_dims, self.num_heads)


  def __call__(self, tokens):
    B, T = tokens.shape
    x = self.embed[tokens.ravel()].reshape((B, T, self.model_dims))
    x = apply_position_embedding(x)
    x = self.transformer(x)
    x = jnp.einsum('BTD,VD->BTV', x, self.embed)
    return x


# N, V, H, D, F = 2, 8, 4, 16, 64
# B, T = 2, 4
N, V, H, D, F = 12, 50257, 12, 768, 768*4
B, T = 8, 1024
lm = LanguageModel(num_layers=N, vocab_size=V, model_dims=D, hidden_dims=F, num_heads=H)

k = jax.random.PRNGKey(0)
x = jax.random.randint(k, (B, T), minval=0, maxval=V, dtype=jnp.int32)
w = lm.init(k, x)

def loss(w, x):
  return lm.apply(w, x)

w_bf16 = jax.tree_map(lambda x: x.astype(jnp.bfloat16), w)
# loss(w_bf16, x)

# from ijit import inductor_jit
#
# inductor_jit(loss)(w, x)

import torch
torch.set_float32_matmul_precision('high')
jaxpr = jax.make_jaxpr(loss)(w, x)
args = list(map(ijit.j2t, jax.tree_flatten((w, x))[0]))
exe = ijit.compile_fx(torch.fx.symbolic_trace(ijit.inductor_interpreter(jaxpr)), args)

def run_torch():
    outs = exe(*args)

jloss = jax.jit(loss)
def run_jax():
    return jloss(w, x)

jax.block_until_ready(run_jax())

import time
j = -time.perf_counter()
for i in range(20):
  jax.block_until_ready(run_jax())
j += time.perf_counter()

run_torch()
torch.cuda.synchronize(0)

t = -time.perf_counter()
for i in range(20):
  run_torch()
  torch.cuda.synchronize(0)
t += time.perf_counter()

print(j)
print(t)
