import jax
from jax import grad, lax, numpy as np
from jax.random import PRNGKey
from jax.lax import tag, push_tag
from jax.tree_util import tree_map, tree_multimap
from jax.experimental.stax2 import collect, inject, inject_add

def foo(scope, x):
  y = tag(x ** 2, scope, "y")
  z = y + 1
  return z

scope = PRNGKey(0)

assert foo(scope, 2.) == 5.
assert collect(foo)(scope, 2.) == {"y": 4.}
assert inject(foo, {"y": 3.})(scope, 2.) == 4.

def higher_foo(scope, x):
  y = foo(push_tag(scope, "foo"), x) # or scope + ["foo"] or scope["foo"]?
  z = tag(y * 2, scope, "z")
  return z

assert higher_foo(scope, 2.) == 10.
assert collect(higher_foo)(scope, 2.) == {"foo": {"y": 4.}, "z": 10.}
assert collect(higher_foo, path_filter=lambda path: "foo" in path)(
  scope, 2.) == {"foo": {"y": 4.}}

def grad_intermediates(f, path_filter=lambda path: True):
  def grad_fun(*args, **kwargs):
    orig_tree = collect(f, path_filter=path_filter)(*args, **kwargs)
    def fun_for_grad(tree):
      return inject_add(f, tree)(*args, **kwargs)
    zeros_tree = tree_map(np.zeros_like, orig_tree)
    return grad(fun_for_grad)(zeros_tree)
  return grad_fun

assert grad_intermediates(foo)(scope, 2.) == {"y": 1.}
assert grad_intermediates(higher_foo)(scope, 2.) == {"z": 1., "foo": {"y": 2.}}
assert grad_intermediates(higher_foo, path_filter=lambda path: "foo" in path)(
  scope, 2.) == {"foo": {"y": 2.}}