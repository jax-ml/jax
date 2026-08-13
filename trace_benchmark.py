"""Trace-time benchmark: python jax vs rustyjax on the same transformer.

The model is written ONCE, against a 13-op backend namespace; each backend
(jax, rustyjax) supplies those ops. That keeps the two benchmarks from
diverging: the only per-backend code is the op table itself, and numerics are
cross-checked (both executed through XLA CPU) at the end.

Usage: python trace_benchmark.py [jax|rusty|both]
"""
import argparse
import statistics
import time
from types import SimpleNamespace

import numpy as np

# Model size (bump N_LAYERS for a bigger trace).
VOCAB = 1024
SEQ = 128
D_MODEL = 512
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS
D_FF = 2048
N_LAYERS = 8

REPS = 10

ATTN_SCALE = float(np.sqrt(D_HEAD))
GELU_C = float(np.sqrt(2.0 / np.pi))

LAYER_KEYS = ["ln1_s", "ln1_b", "wq", "wk", "wv", "wo",
              "ln2_s", "ln2_b", "w1", "b1", "w2", "b2"]


def init_inputs():
  """Flat list of arrays: params, then tokens, then the causal mask."""
  rng = np.random.default_rng(0)
  p = lambda *shape: rng.normal(0.0, 0.02, shape).astype(np.float32)
  layer = lambda: [
      p(D_MODEL), p(D_MODEL),                              # ln1
      p(D_MODEL, D_MODEL), p(D_MODEL, D_MODEL),            # wq wk
      p(D_MODEL, D_MODEL), p(D_MODEL, D_MODEL),            # wv wo
      p(D_MODEL), p(D_MODEL),                              # ln2
      p(D_MODEL, D_FF), p(D_FF), p(D_FF, D_MODEL), p(D_MODEL),  # mlp
  ]
  flat = [p(VOCAB, D_MODEL), p(SEQ, D_MODEL)]              # embed, pos
  for _ in range(N_LAYERS):
    flat += layer()
  flat += [p(D_MODEL), p(D_MODEL)]                         # final ln
  flat.append(np.asarray(np.arange(SEQ) % VOCAB, np.int32))  # tokens
  mask = np.where(np.tril(np.ones((SEQ, SEQ), np.bool_)), 0.0, -np.inf)
  flat.append(mask.astype(np.float32))
  return flat


# === The model, written once against a backend op table B ===

def layer_norm(B, x, scale, bias):
  mu = B.div(B.reduce_sum(x, -1, True), float(D_MODEL))
  c = B.sub(x, mu)
  var = B.div(B.reduce_sum(B.mul(c, c), -1, True), float(D_MODEL))
  return B.add(B.mul(B.div(c, B.sqrt(B.add(var, 1e-6))), scale), bias)


def softmax(B, x):
  m = B.reduce_max(x, -1, True)
  e = B.exp(B.sub(x, m))
  return B.div(e, B.reduce_sum(e, -1, True))


def gelu(B, x):
  x3 = B.mul(B.mul(x, x), x)
  inner = B.tanh(B.mul(B.add(x, B.mul(x3, 0.044715)), GELU_C))
  return B.mul(B.mul(x, 0.5), B.add(inner, 1.0))


def attention(B, x, p, mask):
  split = lambda h: B.transpose(B.reshape(h, [SEQ, N_HEADS, D_HEAD]), [1, 0, 2])
  q, k, v = (split(B.matmul(x, p[w])) for w in ("wq", "wk", "wv"))
  scores = B.add(B.div(B.matmul(q, B.transpose(k, [0, 2, 1])), ATTN_SCALE), mask)
  probs = softmax(B, scores)
  out = B.reshape(B.transpose(B.matmul(probs, v), [1, 0, 2]), [SEQ, D_MODEL])
  return B.matmul(out, p["wo"])


def mlp(B, x, p):
  h = gelu(B, B.add(B.matmul(x, p["w1"]), p["b1"]))
  return B.add(B.matmul(h, p["w2"]), p["b2"])


def forward(B, *flat):
  it = iter(flat)
  embed, pos = next(it), next(it)
  layers = [{k: next(it) for k in LAYER_KEYS} for _ in range(N_LAYERS)]
  lnf_s, lnf_b, tokens, mask = it
  x = B.add(B.take(embed, tokens), pos)
  for p in layers:
    x = B.add(x, attention(B, layer_norm(B, x, p["ln1_s"], p["ln1_b"]), p, mask))
    x = B.add(x, mlp(B, layer_norm(B, x, p["ln2_s"], p["ln2_b"]), p))
  x = layer_norm(B, x, lnf_s, lnf_b)
  return B.matmul(x, B.transpose(embed, [1, 0]))


# === Backends: the only code that exists per-system ===

def jax_backend():
  import jax
  import jax.numpy as jnp
  B = SimpleNamespace(
      add=jnp.add, sub=jnp.subtract, mul=jnp.multiply, div=jnp.divide,
      exp=jnp.exp, tanh=jnp.tanh, sqrt=jnp.sqrt, matmul=jnp.matmul,
      take=lambda t, i: jnp.take(t, i, axis=0),
      reshape=jnp.reshape, transpose=jnp.transpose,
      reduce_sum=lambda x, axis, keepdims: jnp.sum(x, axis=axis, keepdims=keepdims),
      reduce_max=lambda x, axis, keepdims: jnp.max(x, axis=axis, keepdims=keepdims),
  )
  f = lambda *flat: forward(B, *flat)
  return SimpleNamespace(
      name=f"jax (cleared caches)",
      trace=lambda flat: jax.make_jaxpr(f)(*flat),
      n_eqns=lambda traced: len(traced.eqns),
      execute=lambda flat: np.asarray(f(*flat)),  # eager jnp -> XLA CPU
      pre_rep=jax.clear_caches,
  )


def rusty_backend():
  import rustyjax as rj
  B = SimpleNamespace(
      add=rj.add, sub=rj.sub, mul=rj.mul, div=rj.div,
      exp=rj.exp, tanh=rj.tanh, sqrt=rj.sqrt, matmul=rj.matmul,
      take=rj.take, reshape=rj.reshape, transpose=rj.transpose,
      reduce_sum=rj.reduce_sum, reduce_max=rj.reduce_max,
  )
  f = lambda *flat: forward(B, *flat)
  dtypes = {"float32": "f32", "int32": "i32"}
  types_cache = {}

  def types(flat):  # built once per input set, outside the timed region
    key = id(flat)
    if key not in types_cache:
      types_cache[key] = [rj.Type(list(a.shape), dtypes[str(a.dtype)]) for a in flat]
    return types_cache[key]

  return SimpleNamespace(
      name="rustyjax",
      trace=lambda flat: rj.trace(f, types(flat)),
      n_eqns=lambda traced: len(traced.jaxpr),
      execute=lambda flat: rj.trace(f, types(flat)).compile().eval(flat),
      pre_rep=None,
  )


def bench(label, f, pre_rep=None):
  f()  # warmup
  times = []
  for _ in range(REPS):
    if pre_rep:
      pre_rep()  # untimed: e.g. jax cache clearing
    t0 = time.perf_counter()
    f()
    times.append(time.perf_counter() - t0)
  ms = [t * 1e3 for t in times]
  print(f"{label:24s} min {min(ms):8.2f} ms   mean {statistics.mean(ms):8.2f} ms"
        f"   ({REPS} reps)")
  return min(ms)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("backend", nargs="?", default="both",
                      choices=["jax", "rusty", "both"])
  args = parser.parse_args()
  backends = {"jax": [jax_backend], "rusty": [rusty_backend],
              "both": [jax_backend, rusty_backend]}[args.backend]

  print(f"{N_LAYERS} layers, d_model={D_MODEL}, seq={SEQ}")
  flat = init_inputs()

  outs = {}
  for make in backends:
    b = make()
    traced = b.trace(flat)
    n = b.n_eqns(traced)
    best = bench(b.name, lambda: b.trace(flat), pre_rep=b.pre_rep)
    print(f"  {n} eqns -> {n / best * 1e3:,.0f} eqns/sec ({best / n * 1e3:.2f} us/eqn)")
    outs[b.name] = b.execute(flat)

  if len(outs) == 2:
    a, b = outs.values()
    diff, scale = np.max(np.abs(a - b)), np.max(np.abs(b))
    print(f"numeric check: max|diff| = {diff:.3g} (output scale {scale:.3g})")
    assert diff <= 1e-4 * max(scale, 1.0), "MISMATCH between backends"
    print("numeric check: PASS")


if __name__ == "__main__":
  main()
