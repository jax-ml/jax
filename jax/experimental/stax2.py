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

class KeyTrace(core.Trace):

  def __init__(self, master, sublevel, storing, tree=None):
    super(KeyTrace, self).__init__(master, sublevel)
    self.tree = dict() if tree is None else tree
    self.storing = storing

  def with_sublevel(self, sublevel):
    return type(self)(self.master, sublevel, self.storing, self.tree)

  def pure(self, val):
    return KeyTracer(self, val, no_data_dependence)

  def lift(self, val):
    raise NotImplementedError

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
      val, path = tracers[0].val, tracers[0].path
      return KeyTracer(self, val, path + (params['name'],))
    else:
      # tracers are either:
      # (1) "candidate samples" with data dependence only on the key
      # (2) constants with no data dependence on an argument
      # (3) values with data dependence on at least one non-key argument
      # if an operation mixes one or more of (1) with (3), or mixes (1)s with
      # different paths, then the (1)s are stored in the tree as actual samples
      path_set = set(tracer.path for tracer in tracers)
      data_dependent_path_set = path_set - set([no_data_dependence])
      if len(data_dependent_path_set) > 1:
        out_path = other_data_dependence
        args = [self.maybe_set(tracer.path, tracer.val)
                if isinstance(tracer.path, tuple) else tracer.val
                for tracer in tracers]
      else:
        if len(data_dependent_path_set) == 1:
          out_path = data_dependent_path_set.pop()
        else:
          out_path = no_data_dependence
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

def unzip(f, *args):
  fun = lu.wrap_init(f)
  def init(key):
    vals = (key,) + args
    paths = ((),) + (other_data_dependence,) * len(args)
    _, _, tree = key_fun(fun, vals, paths, True, dict())
    return tree
  def apply(tree, *args):
    vals = (random.PRNGKey(0),) + args
    paths = ((),) + (other_data_dependence,) * len(args)
    out_vals, _, _ = key_fun(fun, vals, paths, False, tree)
    return out_vals
  return init, apply

@lu.transformation_with_aux
def key_subtrace(parent, in_paths, *in_vals):
  trace = parent.with_sublevel(core.cur_sublevel())
  in_tracers = map(partial(KeyTracer, trace), in_vals, in_paths)
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_paths = unzip2((t.val, t.path) for t in out_tracers)
  yield out_vals, out_paths

class NoDataDependence(object): pass
no_data_dependence = NoDataDependence()

class OtherDataDependence(object): pass
other_data_dependence = OtherDataDependence()

abstract_key = xla.abstractify(random.PRNGKey(0))

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
  return random.uniform(k, (3,)) + x

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
  W = random.uniform(random.fold_in(k1, "W"), (3, 4))
  b = random.uniform(random.fold_in(k1, "b"), (4,))
  x = np.dot(x, W) + b

  k2 = random.fold_in(key, 2)
  W = random.uniform(random.fold_in(k2, "W"), (4, 2))
  b = random.uniform(random.fold_in(k2, "b"), (2,))
  x = np.dot(x, W) + b

  return x

x = np.ones(3)

init_fun, apply_fun = unzip(f, x)
param_tree = init_fun(random.PRNGKey(0))
print("Ex 1")
print(param_tree)
print(apply_fun(param_tree, x))
print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))

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
  W = random.normal(random.fold_in(key, "W"), (num_features_out, num_features_in))
  b = random.normal(random.fold_in(key, "b"), (num_features_out,))
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

def batch_norm(key, x):
  scale = initializers.ones(random.fold_in(key, 'scale'), x.shape)
  offset = initializers.zeros(random.fold_in(key, 'offset'), x.shape)
  normalize, _ = jax.api._papply(partial(nn.normalize, axis=0), 'batch')
  return normalize(x) * scale + offset

def model(key, x):
  return jax.soft_pmap(partial(batch_norm, key), 'batch')(x)

x = random.normal(random.PRNGKey(0), (4, 3))
init_fun, apply_fun = unzip(model, x)
param_tree = init_fun(random.PRNGKey(0))
print(param_tree)
print(apply_fun(param_tree, x))
print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))
