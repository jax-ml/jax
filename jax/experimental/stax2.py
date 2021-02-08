from __future__ import print_function

import numpy as onp
import itertools as it

import jax
import jax.numpy as np
from jax import jit, grad, collect, inject, inject_add

# def unzip(f, *args, key_argnum=0, **kwargs):
#   # abstract_args = tuple(map(xla.abstractify, args)) # consider using Concrete
#   # del args
#   flat_args1, in_tree1 = tree_flatten((args, kwargs))
#   def init(key):
#     flat_fun, _ = flatten_fun(lu.wrap_init(f), in_tree1)
#     vals = flat_args1[:key_argnum] + [key] + flat_args1[key_argnum + 1:]
#     paths = (None,) * key_argnum + ((),) + (None,) * (len(flat_args1) - key_argnum - 1)
#     _, _, tree = tag_fun(flat_fun, vals, paths, None, dict())
#     return tree
#   def apply(tree, *args, **kwargs):
#     flat_fun, out_tree = flatten_fun(lu.wrap_init(f), in_tree1)
#     flat_args2, in_tree2 = tree_flatten((args, kwargs))
#     # cannot assert in_tree1 == in_tree2
#     # currently relying on order of args being preserved by flattening
#     vals = flat_args2[:key_argnum] + [flat_args1[key_argnum]] + flat_args2[key_argnum:]
#     paths = (None,) * key_argnum + ((),) + (None,) * (len(flat_args1) - key_argnum - 1)
#     out_vals, _, _ = tag_fun(flat_fun, vals, paths, _replace, tree)
#     return tree_unflatten(out_tree(), out_vals)
#   return init, apply

# The idea here is to have the user supply a function that describes both
# initialization and application of a network, and to "unzip" that function into
# its initialization part and its application part. The initialization part we
# execute to produce initial parameters, and the application part we return as a
# Python callable.

# In types, we start with a user function,
#
#   user_fun :: Key -> a -> b
#
# and apply our unzip function to it, given a key and example arguments encoding
# shapes,
#
#  unzip :: (Key -> a -> b) -> Key -> a -> (Params, Params -> a -> b)
#
# The second element of that tuple is the apply_fun.

# doesn't work yet because PRNGKey isn't a real type:
# random.PRNGKey.__getitem__ = random.fold_in



  ### Example 0: Sanity check

  def e(key, x):
    k = random.fold_in(key, "x")
    return lax.sample(random.uniform(k, (3,))) + x

  x = np.ones(3)
  key = random.PRNGKey(0)

  init_fun, apply_fun = unzip(e, key, x)
  param_tree = init_fun(key)
  print("Ex 0")
  print(param_tree)
  print(apply_fun(param_tree, x))

  ### Example 1: The basics

  # f :: Key -> a -> b
  def f(key, x):
    k1 = random.fold_in(key, 1)
    W = lax.sample(random.uniform(random.fold_in(k1, "W"), (3, 4)))
    b = lax.sample(random.uniform(random.fold_in(k1, "b"), (4,)))
    x = np.dot(x, W) + b

    k2 = random.fold_in(key, 2)
    W = lax.sample(random.uniform(random.fold_in(k2, "W"), (4, 2)))
    b = lax.sample(random.uniform(random.fold_in(k2, "b"), (2,)))
    x = np.dot(x, W) + b

    return x

  x = np.ones(3)
  key = random.PRNGKey(0)

  init_fun, apply_fun = unzip(f, key, x)
  param_tree = init_fun(key)
  print("Ex 1")
  print(param_tree)
  print(apply_fun(param_tree, x))
  print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))

  from jax import nn
  from jax.nn import initializers

  # ### Example 2: writing point-free combinator libraries

  # type Layer = Key -> Array -> Array
  # Dense :: Int -> Layer
  # Serial :: [Layer] -> Layer

  @curry
  def Serial(layers, key, x):
    for i, layer in enumerate(layers):
      x = layer(random.fold_in(key, i), x)
    return x

  @curry
  def Dense(num_features_out, key, x):
    num_features_in = x.shape[-1] # num_features_in, = x.shape
    W = initializers.normal()(random.fold_in(key, "W"), (num_features_in, num_features_out))
    b = initializers.normal()(random.fold_in(key, "b"), (num_features_out,))
    return np.dot(x, W) + b

  f = Serial([Dense(4), Dense(2)])

  x = np.ones(3)
  key = random.PRNGKey(0)

  init_fun, apply_fun = unzip(f, key, x)
  param_tree = init_fun(key)
  print("Ex 2")
  print(param_tree)
  print(apply_fun(param_tree, x))
  print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))

  ### Example 3: mixed pointy and point-free

  # Let's make a network with parameter sharing, using pointy fan-out of
  # parameters. One thing we could do is combine Ex 1 and Ex 2 and route
  # parameters themselves to express parameter reuse, but instead what we try here
  # is to handle parameter reuse by reusing keys. For that we use a slightly
  # different Serial combinator, the aptly-named Serial2.

  # type KeyedLayer = Array -> Array
  # Serial2 :: [KeyedLayer] -> KeyedLayer

  @curry
  def Serial2(layers, x):
    for layer in layers:
      x = layer(x)
    return x

  def g(key, x):
    layer_1 = Dense(2, random.fold_in(key, 0))
    layer_2 = Dense(x.shape[0], random.fold_in(key, 1))
    net = Serial2([layer_1, layer_2, layer_1])
    return net(x)

  x = np.ones(3)
  key = random.PRNGKey(0)
  init_fun, apply_fun = unzip(g, key, x)
  param_tree = init_fun(key)
  print("Ex 3")
  print(param_tree)
  print(apply_fun(param_tree, x))
  print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))

  # ----- Layers -----

  from jax import nn
  from jax.nn import initializers

  # def batch_norm(key, x):
  #   scale = initializers.ones(random.fold_in(key, 'scale'), x.shape)
  #   offset = initializers.zeros(random.fold_in(key, 'offset'), x.shape)
  #   normalize, _ = jax.api._papply(partial(nn.normalize, axis=0), 'batch')
  #   return normalize(x) * scale + offset

  # def model(key, x):
  #   return jax.soft_pmap(partial(batch_norm, key), 'batch')(x)

  # x = random.normal(random.PRNGKey(0), (4, 3))
  # init_fun, apply_fun = unzip(model, x)
  # param_tree = init_fun(random.PRNGKey(0))
  # print(param_tree)
  # print(apply_fun(param_tree, x))
  # print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))

  def adam(step_size, b1, b2, epsilon):
    def one_step(scope, x, g, i):
      m = lax.tag(np.zeros_like(x), scope, "first moment")
      v = lax.tag(np.zeros_like(x), scope, "second moment")
      m = (1 - b1) * g + b1 * m
      v = (1 - b2) * (g ** 2) + b2 * v
      mhat = m / (1 - b1 ** (i + 1))
      vhat = v / (1 - b2 ** (i + 1))
      x = x - step_size * mhat / (np.sqrt(vhat) + epsilon)
      return x, {"first moment": m, "second moment": v}
    return one_step

  def momentum(step_size, mass):
    def one_step(scope, x, g, i):
      velocity = lax.tag(np.zeros_like(x), scope, "velocity")
      velocity = mass * velocity - step_size * g
      x = x + velocity
      return x, {"velocity": velocity}
    return one_step

  from jax.random import fold_in

  @curry
  def BatchNorm(key, x):
    scale = initializers.ones(fold_in(key, "scale"), x.shape[1:])
    offset = initializers.zeros(fold_in(key, "offset"), x.shape[1:])
    stats_scope = fold_in(key, "stats")
    mean = lax.tag(np.mean(x, 0, keepdims=True), stats_scope, "mean")
    variance = lax.tag(np.mean(x**2 - mean**2, 0, keepdims=True), stats_scope, "variance")
    return nn.normalize(x, 0, mean, variance) * scale + offset

  def model(key, x):
    x = Dense(2, fold_in(key, "dense"))(x)
    x = BatchNorm(fold_in(key, "norm"))(x)
    return x

  x = random.normal(random.PRNGKey(0), (4, 3))
  key = random.PRNGKey(0)

  model_with_stats = collect(model, path_filter=lambda path: "stats" in path, return_value=True)
  init_fun = collect(model_with_stats, path_filter=lambda path: "stats" not in path)
  param_tree = init_fun(key, x)
  print("param_tree", param_tree)
  y, stats_tree = inject(model_with_stats, param_tree)(key, x)
  print("y", y)
  print("stats_tree", stats_tree)
  stats_tree["norm"]["stats"]["variance"] = stats_tree["norm"]["stats"]["variance"] / 10
  infer_y = inject(model, tree_join(param_tree, stats_tree))(key, x)
  print("infer_y", infer_y)
