from __future__ import print_function

import numpy as onp
import itertools as it

import jax
import jax.numpy as np
from jax import core, lax, random, linear_util as lu
from jax.interpreters import xla
from jax.util import curry, partial, unzip2, safe_map as map
from jax.api_util import flatten_fun
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

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

distributions = set([lax.sample_p])

class KeyTrace(core.Trace):

  def __init__(self, master, sublevel, storing, tree=None, path_filter=lambda path: True):
    super(KeyTrace, self).__init__(master, sublevel)
    self.tree = dict() if tree is None else tree
    self.storing = storing
    self.path_filter = path_filter

  def with_sublevel(self, sublevel):
    return type(self)(self.master, sublevel, self.storing, self.tree, self.path_filter)

  def pure(self, val):
    return KeyTracer(self, val, None)

  def lift(self, val):
    return KeyTracer(self, val, None)

  def sublift(self, val):
    return KeyTracer(self, val.val, val.path)

  def maybe_set(self, path, val):
    if self.path_filter(path):
      return maybe_set_tree(self.tree, path, val, self.storing)
    else:
      return val

  def process_primitive(self, primitive, tracers, params):
    # pick the shortest path!
    paths = [tracer.path for tracer in tracers if tracer.path is not None]
    if len(paths) > 0:
      # TODO more-well-defined tiebreaking
      out_path = min(paths, key=lambda path: len(path))
    else:
      out_path = None
    args = [tracer.val for tracer in tracers]
    out = primitive.bind(*args, **params)
    if primitive is lax.push_tag_p:
      out_path = out_path + (params["name"],)
    elif primitive is lax.tie_in_p:
      out_path = paths[0]
    elif primitive in distributions:
      assert not primitive.multiple_results
      out = self.maybe_set(out_path, out)
    if primitive.multiple_results:
      return [KeyTracer(self, val, out_path) for val in out]
    else:
      return KeyTracer(self, out, out_path)

  def process_call(self, call_primitive, f, tracers, params):

    in_vals, in_paths = unzip2((t.val, t.path) for t in tracers)
    f_key, out_paths = key_subtrace(f, self, in_paths)
    out_vals = call_primitive.bind(f_key, *in_vals, **params)
    return [KeyTracer(self, v, p) for v, p in zip(out_vals, out_paths())]

    # return call_primitive.impl(f, *tracers, **params)
    # let's inline it and see what happens
    # return f.call_wrapped(*tracers)

  def process_map(self, map_primitive, f, tracers, params):
    raise NotImplementedError

  def post_process_call(self, call_primitive, out_tracers, params):
    raise NotImplementedError

class KeyTracer(core.Tracer):
  __slots__ = ['trace', 'val', 'path']

  def __init__(self, trace, val, path):
    self.trace = trace
    self.val = val
    self.path = path

  def __repr__(self):
    return 'Traced<{}:{}>'.format(self.val, self.trace)

  @property
  def aval(self):
    return core.get_aval(self.val)

  def full_lower(self):
    if self.path is None:
      return self.val
    else:
      return self

@lu.transformation_with_aux
def key_subtrace(parent, in_paths, *in_vals):
  trace = parent.with_sublevel(core.cur_sublevel())
  in_tracers = map(partial(KeyTracer, trace), in_vals, in_paths)
  outs = yield in_tracers, {}
  assert isinstance(outs, (tuple, list))
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_paths = unzip2((t.val, t.path) for t in out_tracers)
  yield out_vals, out_paths

def key_fun(fun, in_vals, in_paths, storing, in_tree, path_filter=lambda path: True):
  with core.new_master(KeyTrace) as master:
    trace = KeyTrace(master, core.cur_sublevel(), storing, in_tree, path_filter)
    fun, out_paths = key_subtrace(fun, trace, in_paths)
    out_vals = fun.call_wrapped(*in_vals)
    del master
  return out_vals, out_paths(), trace.tree

def unzip(f, *args, key_arg=0, **kwargs):
  # abstract_args = tuple(map(xla.abstractify, args)) # consider using Concrete
  # del args
  flat_args1, in_tree1 = tree_flatten((args, kwargs))
  def init(key):
    flat_fun, _ = flatten_fun(lu.wrap_init(f), in_tree1)
    vals = flat_args1[:key_arg] + [key] + flat_args1[key_arg + 1:]
    paths = (None,) * key_arg + ((),) + (None,) * (len(flat_args1) - key_arg - 1)
    _, _, tree = key_fun(flat_fun, vals, paths, True, dict())
    return tree
  def apply(tree, *args, **kwargs):
    flat_fun, out_tree = flatten_fun(lu.wrap_init(f), in_tree1)
    flat_args2, in_tree2 = tree_flatten((args, kwargs))
    # cannot assert in_tree1 == in_tree2
    # currently relying on order of args being preserved by flattening
    vals = flat_args2[:key_arg] + [flat_args1[key_arg]] + flat_args2[key_arg:]
    paths = (None,) * key_arg + ((),) + (None,) * (len(flat_args1) - key_arg - 1)
    out_vals, _, _ = key_fun(flat_fun, vals, paths, False, tree)
    return tree_unflatten(out_tree(), out_vals)
  return init, apply

def iterpaths(tree, prefix=()):
  for key, val in tree.items():
    path = prefix + (key,)
    if isinstance(val, dict):
      yield from iterpaths(val, path)
    else:
      yield path, val

def maybe_set_tree(tree, path, val, storing):
  #print(tree, path, val, storing)
  if len(path) == 0:
    raise ValueError("cannot store samples without key path")
  subtree = tree
  for path_element in path[:-1]:
    if path_element not in subtree:
      if storing:
        subtree[path_element] = dict()
      else:
        return val
    subtree = subtree[path_element]
  if storing:
    # maybe when checks are enabled?
    # if path[-1] in subtree:
    #   assert subtree[path[-1]] == val, "found different samples for key path {}".format(path)
    subtree[path[-1]] = val
  if path[-1] in subtree:
    return subtree[path[-1]]
  else:
    return val

def as_tree(paths_and_values):
  tree = dict()
  for path, val in paths_and_values:
    maybe_set_tree(tree, path, val, True)
  return tree

def join_trees(tree1, tree2):
  return as_tree(it.chain(iterpaths(tree1), iterpaths(tree2)))

def collect(f, key_arg=0, path_filter=lambda path: True, return_value=True):
  fun = lu.wrap_init(f)
  def collect_fn(*args, **kwargs):
    flat_args, in_tree = tree_flatten((args, kwargs))
    flat_fun, out_tree = flatten_fun(fun, in_tree)
    paths = (None,) * key_arg + ((),) + (None,) * (len(flat_args) - key_arg - 1)
    out_flat, _, tree = key_fun(flat_fun, flat_args, paths, True, dict())
    tree = as_tree((path, val) for path, val in iterpaths(tree) if path_filter(path))
    if return_value:
      return tree_unflatten(out_tree(), out_flat), tree
    else:
      return tree
  return collect_fn

def inject(f, key_arg=0):
  fun = lu.wrap_init(f)
  def inject_fn(tree, *args, **kwargs):
    flat_args, in_tree = tree_flatten((args, kwargs))
    flat_fun, out_tree = flatten_fun(fun, in_tree)
    paths = (None,) * key_arg + ((),) + (None,) * (len(flat_args) - key_arg - 1)
    out_vals, _, _, = key_fun(flat_fun, flat_args, paths, False, tree)
    return tree_unflatten(out_tree(), out_vals)
  return inject_fn

# some differences:
# - metrics should run on unabstracted inputs
# - init should fail on data-dependence on abstracted inputs? well, 
#   not exactly abstracted inputs but ones that aren't provided as
#   inputs to the init fn.
# - 

# def unzip(f, *args):
#   def init(key):
#     with core.new_master(KeyTrace) as master:
#       trace = KeyTrace(master, core.cur_sublevel(), True)
#       args_tracers = [KeyTracer(trace, arg, other_data_dependence) for arg in args]
#       key_tracer = KeyTracer(trace, key, ())
#       out_tracer = f(key_tracer, *args_tracers)
#       return trace.tree
#   def apply(tree, *args):
#     with core.new_master(KeyTrace) as master:
#       trace = KeyTrace(master, core.cur_sublevel(), False, tree)
#       args_tracers = [KeyTracer(trace, arg, other_data_dependence) for arg in args]
#       key_tracer = KeyTracer(trace, random.PRNGKey(0), ())
#       out_tracer = f(key_tracer, *args_tracers)
#       return out_tracer.val
#   return init, apply

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

# TODOs
#   - figure out batch norm
#   - recurse into subjaxprs, especially initial-style (scan!), not clear how to
#     represent splits under a scan so maybe it's an error for now

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






# def adam(step_size, b1, b2, epsilon):
#   def one_step(key, x, g, i):
#     m = yield(tag(np.zeros_like(x), "first moment", key))
#     v = yield(tag(np.zeros_like(x), "second moment", key))
#     m = yield(tag((1 - b1) * g + b1 * m), "first moment", key2)
#     v = yield(tag((1 - b2) * (g ** 2) + b2 * v), "second moment", key2)
#     mhat = m / (1 - b1 ** (i + 1))
#     vhat = v / (1 - b2 ** (i + 1))
#     x = x - step_size * mhat / (np.sqrt(vhat) + epsilon)
#     return x

# def adam(step_size, b1, b2, epsilon):
#   def one_step(key, x, g, i):
#     m = yield(tag(np.zeros_like(x), "first moment", key))
#     v = yield(tag(np.zeros_like(x), "second moment", key))
#     m = yield(tag((1 - b1) * g + b1 * m), "first moment", key2)
#     v = yield(tag((1 - b2) * (g ** 2) + b2 * v), "second moment", key2)
#     mhat = m / (1 - b1 ** (i + 1))
#     vhat = v / (1 - b2 ** (i + 1))
#     x = x - step_size * mhat / (np.sqrt(vhat) + epsilon)
#     return x

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

# separate passes for "get" and "put"
# currently every sample point is both

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
# it's important to remove the sample primitives when the path matches
# and not when it doesn't...
# actually no, it's important to never remove the sample primitives.
# two reasons: if the primitive is an actual distribution, you can't remove it
# and also, while init maybe doesn't want to see it, inject does!
model_with_stats = collect(model, path_filter=lambda path: "stats" in path)
init_fun = collect(model_with_stats, return_value=False,
                   path_filter=lambda path: "stats" not in path)
apply_fun = inject(model)
apply_with_stats = inject(model_with_stats)
param_tree = init_fun(key, x)
print("param_tree", param_tree)
y, stats_tree = apply_with_stats(param_tree, key, x)
print("y", y)
print("stats_tree", stats_tree)
stats_tree["norm"]["stats"]["variance"] = stats_tree["norm"]["stats"]["variance"] / 10
infer_y = apply_fun(join_trees(param_tree, stats_tree), key, x)
print("infer_y", infer_y)

# init_fun, apply_fun = unzip(batch_norm, x)
# param_tree = init_fun(random.PRNGKey(0))
# print(param_tree)
# print(apply_fun(param_tree, 0, x))
# print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))
# compute_stats, continuation = unzip(apply_fun, param_tree, 0, x)
# stats = compute_stats(param_tree, x)
# print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))

  # TODO now
  # running_mean = lax.sample(running_mean * momentum + mean * (1 - momentum))
  # running_variance = lax.sample(running_variance * momentum + variance * (1 - momentum))

# def batch_norm(key, x, momentum=0.99):
#   scale = initializers.ones(random.fold_in(key, 'scale'), x.shape[1:])
#   offset = initializers.zeros(random.fold_in(key, 'offset'), x.shape[1:])
#   running_mean = initializers.zeros(random.fold_in(key, 'running_mean'), x.shape[1:])
#   running_variance = initializers.ones(random.fold_in(key, 'running_variance'), x.shape[1:])
#   mean = lax.tag(np.mean(x, axis=0, keepdims=True), "mean")
#   variance = lax.tag(np.mean(x**2, axis=0, keepdims=True) - mean**2, "variance")
#   running_mean = lax.sample(running_mean * momentum + mean * (1 - momentum))
#   running_variance = lax.sample(running_variance * momentum + variance * (1 - momentum))
#   return nn.normalize(x, axis=0, mean=mean, variance=variance) * scale + offset

# def batch_norm_init(key):
#   return {'scale': ones(key), 'offset': zeros(key)}

# def batch_norm_init_state(key):
#   return {'running_mean': zeros(key), 'running_variance': ones(key)}

# def batch_norm_apply(params, state, x):
#   scale = params['scale']
#   offset = params['offset']
#   mean = mean(x)
#   variance = variance(x)
#   info = {'mean': mean, 'variance': variance}
#   if isinstance(x, jax.interpreters.ad.JVPTracer):
#     return nn.normalize(x, mean, variance) * scale + offset, info
#   return nn.normalize(x, state['running_mean'], state['running_variance']) * scale + offset, info

# def batch_norm_update_state(state, info):
#   running_mean = state['running_mean']
#   running_variance = state['running_variance']
#   running_mean = 0.99 * running_mean + info['mean']
#   running_variance = 0.99 * running_variance + info['variance']
#   return {'running_mean': running_mean, 'running_variance': running_variance}

