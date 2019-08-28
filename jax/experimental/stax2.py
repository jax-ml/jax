from __future__ import print_function

import itertools as it
import numpy as onp
import operator as op

import jax.numpy as np
from jax import core
from jax import random
from jax.api_util import flatten_fun
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax.util import curry
from jax.interpreters import partial_eval as pe
from jax.abstract_arrays import ShapedArray
from jax import linear_util as lu


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


# First we set up prng splitting and sampling as new primitives so that we don't
# just inline our prng hash function. We also use a variant of splitting based
# on hashable name strings.

def randn(key, shape):
  return randn_p.bind(key, shape=shape)
randn_p = core.Primitive('randn')
randn_p.def_abstract_eval(lambda key, shape: ShapedArray(shape, onp.float32))

def named_subkey(key, name):
  return named_subkey_p.bind(key, name=name)
named_subkey_p = core.Primitive('named_subkey')
named_subkey_p.def_abstract_eval(lambda key, name: key)



# Now we can write the unzip function, starting with unzip_jaxpr which applies
# to a jaxpr before adding the boilerplate to handle Python callables and
# pytrees. The unzip_jaxpr function looks like a jaxpr interpreter.

def unzip_jaxpr(jaxpr):
  key_invar, data_invars = jaxpr.invars[0], jaxpr.invars[1:]
  params = {key_invar : Node()}
  canonical_paramvars = {}
  apply_eqns = []

  for eqn in jaxpr.eqns:
    if eqn.primitive is named_subkey_p:
      parent_var, = eqn.invars
      child_var, = eqn.outvars
      name = eqn.params['name']
      if name in params[parent_var].children:
        params[child_var] = params[parent_var].children[name]
      else:
        params[child_var] = Node()
        params[parent_var].children[name] = params[child_var]
    elif eqn.primitive is randn_p:
      parent_var, = eqn.invars
      param_var, = eqn.outvars
      shape = eqn.params['shape']
      if hasattr(params[parent_var], "leaf"):
        canonical_paramvars[param_var] = params[parent_var].leaf[1]
      else:
        params[parent_var].leaf = OpaqueTuple((onp.zeros(shape), param_var))
    else:
      canonical_invars = [canonical_paramvars.get(v, v) for v in eqn.invars]
      apply_eqns.append(core.new_jaxpr_eqn(
          canonical_invars, eqn.outvars, eqn.primitive, eqn.bound_subjaxprs,
          eqn.params))

  param_invar_tree = _strip_nodes(params[key_invar])
  param_tree = tree_map(op.itemgetter(0), param_invar_tree)
  invar_tree = tree_map(op.itemgetter(1), param_invar_tree)
  param_invars, treedef = tree_flatten(invar_tree)

  apply_jaxpr = core.Jaxpr(jaxpr.constvars, (),
                           param_invars + data_invars,
                           jaxpr.outvars, apply_eqns)
  core.check_jaxpr(apply_jaxpr)
  return param_tree, treedef, apply_jaxpr

def unzip(fun, *example_args, **example_kwargs):
  def abstractify(x):
    return ShapedArray(onp.shape(x), onp.result_type(x))
  args_flat, in_tree = tree_flatten((example_args, example_kwargs))
  fun, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
  pvals = [pe.PartialVal((abstractify(x), core.unit)) for x in args_flat]
  jaxpr, _, consts = pe.trace_to_jaxpr(fun, pvals, instantiate=True)
  param_tree, param_treedef, apply_jaxpr = unzip_jaxpr(jaxpr)

  def apply_fun(params, *args, **kwargs):
    args_flat, in_tree2 = tree_flatten(((core.unit,) + args, kwargs))
    assert in_tree == in_tree2
    params_flat, param_treedef2 = tree_flatten(params)
    assert param_treedef == param_treedef2
    out_flat = core.eval_jaxpr(apply_jaxpr, consts, (),
                               *it.chain(params_flat, args_flat[1:]))
    return tree_unflatten(out_tree(), out_flat)

  return param_tree, apply_fun


# These are just utility functions.

def _strip_nodes(params):
  assert type(params) is Node
  if hasattr(params, "leaf"):
    return params.leaf
  else:
    return {k : _strip_nodes(child) for k, child in params.children.items()}

class Node(object):
  __slots__ = ["leaf", "children"]  # only one!
  def __init__(self):
    self.children = {}
  def __repr__(self):
    if hasattr(self, "leaf"):
      return repr(self.leaf)
    else:
      return repr(self.children)

class OpaqueTuple(tuple): pass


###

### Example 1: The basics

# f :: Key -> a -> b
def f(key, x):
  k1 = named_subkey(key, "Layer 1")
  W = randn(named_subkey(k1, "W"), (4, 3))
  b = randn(named_subkey(k1, "b"), (4,))
  x = np.dot(W, x) + b

  k2 = named_subkey(key, "Layer 2")
  W = randn(named_subkey(k2, "W"), (2, 4))
  b = randn(named_subkey(k2, "b"), (2,))
  x = np.dot(W, x) + b

  return x

x = np.ones(3)
key = random.PRNGKey(0)
param_tree, apply_fun = unzip(f, key, x)
print("Ex 1")
print(param_tree)
print(apply_fun(param_tree, x))


# ### Example 2: writing point-free combinator libraries

# type Layer = Key -> Array -> Array
# Dense :: Int -> Layer
# Serial :: [Layer] -> Layer

@curry
def Serial(layers, key, x):
  for i, layer in enumerate(layers):
    x = layer(named_subkey(key, i), x)
  return x

@curry
def Dense(num_features_out, key, x):
  num_features_in, = x.shape
  W = randn(named_subkey(key, "W"), (num_features_out, num_features_in))
  b = randn(named_subkey(key, "b"), (num_features_out,))
  return np.dot(W, x) + b

f = Serial([Dense(4), Dense(2)])

x = np.ones(3)
key = random.PRNGKey(0)
param_tree, apply_fun = unzip(f, key, x)
print("Ex 2")
print(param_tree)
print(apply_fun(param_tree, x))


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
  layer_1 = Dense(2, named_subkey(key, 0))
  layer_2 = Dense(x.shape[0], named_subkey(key, 1))
  net = Serial2([layer_1, layer_2, layer_1])
  return net(x)

x = np.ones(3)
key = random.PRNGKey(0)
param_tree, apply_fun = unzip(f, key, x)
print("Ex 3")
print(param_tree)
print(apply_fun(param_tree, x))


# TODOs
#   - actually have prng primitives, merge this with real prng splitting
#   - pytree flattening stuff on data inputs/outputs
#   - split with integers
#   - figure out batch norm
#   - we should only be capturing prng splits that derive from the input
#     argument
#   - recurse into subjaxprs, especially initial-style (scan!), not clear how to
#     represent splits under a scan so maybe it's an error for now
