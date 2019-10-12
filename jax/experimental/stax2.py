from __future__ import print_function

import numpy as onp
import operator as op

import jax
import jax.numpy as np
from jax import core, lax, random, linear_util as lu
from jax.interpreters import xla
from jax.util import curry, partial, unzip2

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

  def __init__(self, master, sublevel, storing, tree=None):
    super(KeyTrace, self).__init__(master, sublevel)
    self.tree = dict() if tree is None else tree
    self.storing = storing

  def with_sublevel(self, sublevel):
    return type(self)(self.master, sublevel, self.storing, self.tree)

  def pure(self, val):
    return KeyTracer(self, val, None)

  def lift(self, val):
    return KeyTracer(self, val, None)

  def sublift(self, val):
    return KeyTracer(self, val.val, val.path)

  def maybe_set(self, path, val):
    if len(path) == 0:
      raise ValueError("cannot store samples without key path")
    subtree = self.tree
    for path_element in path[:-1]:
      if path_element not in subtree:
        assert self.storing, "key path {} not in provided tree".format(path)
        subtree[path_element] = dict()
      subtree = subtree[path_element]
    if self.storing:
      subtree[path[-1]] = val
    assert path[-1] in subtree, "key path {} not in provided tree".format(path)
    return subtree[path[-1]]

  def process_primitive(self, primitive, tracers, params):
    if primitive is lax.tag_p:
      tracer, = tracers
      return KeyTracer(self, tracer.val, tracer.path + (params['name'],))
    elif primitive in distributions:
      tracer, = tracers
      val = self.maybe_set(tracer.path, tracer.val)
      return KeyTracer(self, val, tracer.path)
    else:
      # pick the shortest path!
      paths = [tracer.path for tracer in tracers if tracer.path is not None]
      if len(paths) > 0:
        # TODO more-well-defined tiebreaking
        out_path = min(paths, key=lambda path: len(path))
      else:
        out_path = None
      args = [tracer.val for tracer in tracers]
      out = primitive.bind(*args, **params)
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

def key_fun(fun, in_vals, in_paths, storing, in_tree):
  with core.new_master(KeyTrace) as master:
    trace = KeyTrace(master, core.cur_sublevel(), storing, in_tree)
    fun, out_paths = key_subtrace(fun, trace, in_paths)
    out_vals = fun.call_wrapped(*in_vals)
    del master
  return out_vals, out_paths(), trace.tree

def unzip(f, *args, key_arg=0, ):
  abstract_args = map(xla.abstractify, args) # consider using Concrete
  del args
  fun = lu.wrap_init(f)
  def init(key):
    vals = abstract_args[:key_arg] + (key,) + abstract_args[key_arg + 1:]
    paths = (None,) * key_arg + ((),) + (None,) * (len(args) - key_arg - 1)
    out_vals, _, tree = key_fun(fun, vals, paths, True, dict())
    return tree
  def apply(tree, *args):
    vals = args[:key_arg] + (abstract_args[key_arg],) + args[key_arg + 1:]
    paths = (None,) * key_arg + ((),) + (None,) * (len(args) - key_arg - 1)
    out_vals, _, _ = key_fun(fun, vals, paths, False, tree)
    return out_vals
  return init, apply

def metrics(f, key_arg=0):
  fun = lu.wrap_init(f)
  def metrics_fn(*args):
    paths = (None,) * key_arg + ((),) + (None,) * (len(args) - key_arg - 1)
    out_vals, _, tree = key_fun(fun, vals, paths, True, dict())
    return out_vals, tree
  return metrics_fn

# some differences:
# - metrics should run on unabstracted inputs
# - init should fail on data-dependence on abstracted inputs? well, 
#   not exactly abstracted inputs but ones that aren't provided as
#   inputs to the init fn.
# - 

@lu.transformation_with_aux
def key_subtrace(parent, in_paths, *in_vals):
  trace = parent.with_sublevel(core.cur_sublevel())
  in_tracers = map(partial(KeyTracer, trace), in_vals, in_paths)
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_paths = unzip2((t.val, t.path) for t in out_tracers)
  yield out_vals, out_paths

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

init_fun, apply_fun = unzip(e, x)
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

init_fun, apply_fun = unzip(f, x)
param_tree = init_fun(random.PRNGKey(0))
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
  num_features_in, = x.shape
  W = initializers.normal()(random.fold_in(key, "W"), (num_features_out, num_features_in))
  b = initializers.normal()(random.fold_in(key, "b"), (num_features_out,))
  return np.dot(W, x) + b

f = Serial([Dense(4), Dense(2)])

x = np.ones(3)
init_fun, apply_fun = unzip(f, x)
param_tree = init_fun(random.PRNGKey(0))
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

def f(key, x):
  layer_1 = Dense(2, random.fold_in(key, 0))
  layer_2 = Dense(x.shape[0], random.fold_in(key, 1))
  net = Serial2([layer_1, layer_2, layer_1])
  return net(x)

x = np.ones(3)
init_fun, apply_fun = unzip(f, x)
param_tree = init_fun(random.PRNGKey(0))
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
  def one_step(key, x, g, i):
    m = initializers.zeros(random.fold_in(key, "first moment"), x.shape)
    v = initializers.zeros(random.fold_in(key, "second moment"), x.shape)
    m = (1 - b1) * g + b1 * m
    v = (1 - b2) * (g ** 2) + b2 * v
    mhat = m / (1 - b1 ** (i + 1))
    vhat = v / (1 - b2 ** (i + 1))
    x = x - step_size * mhat / (np.sqrt(vhat) + epsilon)
    return x, {'first moment': m, 'second moment': v}

# separate passes for "get" and "put"
# currently every sample point is both

def batch_norm(key, x, momentum=0.99):
  scale = initializers.ones(random.fold_in(key, "scale"), x.shape[1:])
  offset = initializers.zeros(random.fold_in(key, "offset"), x.shape[1:])
  mean = lax.metric(np.mean(x, axis=0, keepdims=True),
                    lax.tag(key, "mean"),
                    accumulate=lambda old, new: )
  variance = lax.metric(np.mean(x**2 - mean**2, axis=0, keepdims=True), lax.tag(key, "variance"))
  return nn.normalize(x, axis=0, mean=mean, variance=variance) * scale + offset

x = random.normal(random.PRNGKey(0), (4, 3))
init_fun, apply_fun = unzip(batch_norm, x)
param_tree = init_fun(random.PRNGKey(0))
print(param_tree)
print(apply_fun(param_tree, 0, x))
print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))
compute_stats, continuation = unzip(apply_fun, param_tree, 0, x)
stats = compute_stats(param_tree, x)
print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))

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

