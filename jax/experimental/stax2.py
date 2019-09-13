from __future__ import print_function

import numpy as onp
import operator as op

import jax
import jax.numpy as np
from jax import core
from jax import random
from jax.util import curry, partial


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

uniform_p = random._uniform.prim
normal_p = random._normal.prim
fold_in_p = random._fold_in.prim

class KeyTrace(core.Trace):
  def pure(self, val):
    return KeyTracer(self, val)

  def lift(self, val):
    return KeyTracer(self, val)

  def sublift(self, val):
    raise NotImplementedError

  def process_primitive(self, primitive, tracers, params):
    if primitive is fold_in_p:
      assert len(tracers) == 1
      key_tracer = tracers[0]
      key, path, tree = key_tracer.val, key_tracer.path, key_tracer.tree
      if key is None:
        out = None
      else:
        out, = primitive.bind(key, **params)
      return [KeyTracer(self, out, tree, path + [params['data']])]
    elif primitive in [uniform_p, normal_p]:
      key_tracer, args_tracers = tracers[0], tracers[1:]
      key, path, tree = key_tracer.val, key_tracer.path, key_tracer.tree
      assert all(len(tracer.path) == 0 for tracer in args_tracers[1:])
      args = [tracer.val for tracer in args_tracers]
      subtree = tree
      for path_key in path[:-1]:
        if path_key not in subtree:
          subtree[path_key] = dict()
        subtree = subtree[path_key]
      if path[-1] in subtree:
        out = subtree[path[-1]]
        print("using provided sample", out)
      else:
        assert key is not None
        out, = primitive.bind(key, *args, **params)
        subtree[path[-1]] = out
      return [KeyTracer(self, out, tree)]
    else:
      assert all(len(tracer.path) == 0 for tracer in tracers)
      tree = tracers[0].tree
      for tracer in tracers[1:]:
        if tree == dict():
          tree = tracer.tree
        elif tracer.tree == dict():
          continue
        elif tree != tracer.tree:
          raise AssertionError
      args = [tracer.val for tracer in tracers]
      out = primitive.bind(*args, **params)
      if primitive.multiple_results:
        return [KeyTracer(self, val, tree) for val in out]
      else:
        return KeyTracer(self, out, tree)

  def process_call(self, call_primitive, f, tracers, params):
    # let's inline it and see what happens
    return f.call_wrapped(*tracers)

  def process_map(self, map_primitive, f, tracers, params):
    raise NotImplementedError

  def post_process_call(self, call_primitive, out_tracers, params):
    raise NotImplementedError


class KeyTracer(core.Tracer):
  __slots__ = ['trace', 'val', 'tree', 'path']

  def __init__(self, trace, val, tree=None, path=None):
    self.trace = trace
    self.val = val
    self.tree = dict() if tree is None else tree
    self.path = [] if path is None else path

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

random._uniform.overrides.add(KeyTrace)
random._normal.overrides.add(KeyTrace)
random._fold_in.overrides.add(KeyTrace)

def unzip(f, key, *args):
  def apply(tree, *args):
    with core.new_master(KeyTrace) as master:
      trace = KeyTrace(master, core.cur_sublevel())
      args_tracers = [KeyTracer(trace, arg) for arg in args]
      key_tracer = KeyTracer(trace, None, tree)
      out_tracer = f(key_tracer, *args_tracers)
      return out_tracer.val
  with core.new_master(KeyTrace) as master:
    trace = KeyTrace(master, core.cur_sublevel())
    args_tracers = [KeyTracer(trace, arg) for arg in args]
    key_tracer = KeyTracer(trace, key)
    out_tracer = f(key_tracer, *args_tracers)
    out_tree = out_tracer.tree
  return out_tree, apply

### Example 0: Sanity check

def e(key, x):
  k = random.fold_in(key, "x")
  return random.uniform(k, (3,)) + x

x = np.ones(3)
key = random.PRNGKey(0)

param_tree, apply_fun = unzip(e, key, x)
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
key = random.PRNGKey(0)
# print(jax.make_jaxpr(f)(key, x))

param_tree, apply_fun = unzip(f, key, x)
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
key = random.PRNGKey(0)
param_tree, apply_fun = unzip(f, key, x)
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
key = random.PRNGKey(0)
param_tree, apply_fun = unzip(f, key, x)
print("Ex 3")
print(param_tree)
print(apply_fun(param_tree, x))
print(jax.make_jaxpr(partial(apply_fun, param_tree))(x))

# TODOs
#   - figure out batch norm
#   - recurse into subjaxprs, especially initial-style (scan!), not clear how to
#     represent splits under a scan so maybe it's an error for now
