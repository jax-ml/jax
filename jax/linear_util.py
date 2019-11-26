# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for defining linear functions composed with transformations.

"Linear" here is meant in the sense of linear types; that is, a linear function
may be called at most once.

For example:

from jax import linear_util as lu

# A transformation that scales its argument down and its result up.
@lu.transformation
def scale_transformer(scale, x):
  ans = yield (x / scale,)
  yield x * scale

def f(x):
  return x + 1

g = lu.wrap_init(f)  # Wraps `f` as a `WrappedFun`.
g = scale_transformer(g, 2.0)  # Scale inputs/outputs by 2.0
g = scale_transformer(g, 0.7)  # Scale inputs/outputs further by 0.7.
print(g.call_wrapped(3.))  # Call the transformed function.


A `WrappedFun` object represents a function `f`, together with a
sequence of nested transformations that are to be applied to the positional
arguments at call time and function return values at return time.
`WrappedFun` objects explicitly represent the set of transformations so that
they can be used as dictionary keys for memoization. `WrappedFun` objects
compare as equal only if they compute the same function.

Transformations are implemented as generators to save call stack frames.
A transformation's generator takes arguments `gen args + args`, and yields
a tuple of transformed arguments that should be passed to the wrapped
function. The result of the wrapped function is passed back to the generator
using `gen.send()`, and the generator yields the transformed results to pass
back to the caller.

Transformations can also return auxiliary data using the `transform_with_aux`
decorator. For example:

@lu.transformation_with_aux
def scale_transformer_aux(scale, x):
  ans = yield (x / scale,)
  yield (x * scale, "Auxiliary data: {}".format(x))

g = lu.wrap_init(f)  # Wraps `f` as a `WrappedFun`.
g, aux_thunk = scale_transformer_aux(g, 2.0)  # Scale inputs/outputs by 2.0
print(g.call_wrapped(3.))  # Call the transformed function.
print(aux_thunk()) # Retrieves the auxiliary data computed during evaluation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import weakref

from .util import curry, partial


def thunk(f):
  store = Store()
  def f_memoized():
    if not store:
      # TODO(dougalm): save/restore relevant environment state too
      store.store(f())
    return store.val

  return f_memoized

class StoreException(Exception): pass


class EmptyStoreValue(object): pass
_EMPTY_STORE_VALUE = EmptyStoreValue()

class Store(object):
  __slots__ = ("_val",)

  def __init__(self):
    self._val = _EMPTY_STORE_VALUE

  def store(self, val):
    assert self._val is _EMPTY_STORE_VALUE, "Store occupied"
    self._val = val

  @property
  def val(self):
    if not self:
      raise StoreException("Store empty")
    return self._val

  def __nonzero__(self):
    return self._val is not _EMPTY_STORE_VALUE

  __bool__ = __nonzero__


class WrappedFun(object):
  """Represents a function `f` to which `transforms` are to be applied.

  Arguments:
    f: the function to be transformed.
    transforms: a list of `(gen, gen_args, out_store)` tuples representing
      transformations to apply to `f.`
    params: extra parameters to pass as keyword arguments to `f`.
  """
  __slots__ = ("f", "transforms", "stores", "params")

  def __init__(self, f, transforms, stores, params):
    self.f = f
    self.transforms = transforms
    self.stores = stores
    self.params = params

  @property
  def __name__(self):
    return getattr(self.f, '__name__', '<unnamed wrapped function>')

  def wrap(self, gen, gen_args, out_store):
    return WrappedFun(self.f, ((gen, gen_args),) + self.transforms,
                      (out_store,) + self.stores, self.params)

  def populate_stores(self, stores):
    for self_store, other_store in zip(self.stores, stores):
      if self_store is not None:
        self_store.store(other_store.val)

  def call_wrapped(self, *args, **kwargs):
    stack = []
    for (gen, gen_args), out_store in zip(self.transforms, self.stores):
      gen = gen(*(gen_args + tuple(args)), **kwargs)
      args, kwargs = next(gen)
      stack.append((gen, out_store))

    del gen
    ans = self.f(*args, **dict(self.params, **kwargs))
    del args
    while stack:
      gen, out_store = stack.pop()
      ans = gen.send(ans)
      if out_store is not None:
        ans, side = ans
        out_store.store(side)

    return ans

  def __repr__(self):
    def transform_to_str(x):
      i, (gen, args) = x
      return "{}   : {}   {}".format(i, fun_name(gen), fun_name(args))
    transformation_stack = map(transform_to_str, enumerate(self.transforms))
    return "Wrapped function:\n" + '\n'.join(transformation_stack) + '\nCore: ' + fun_name(self.f) + '\n'

  def __hash__(self):
    return hash((self.f, self.transforms, self.params))

  def __eq__(self, other):
    return (self.f == other.f and self.transforms == other.transforms and
            self.params == other.params)

@curry
def transformation(gen, fun, *transformation_args):
  return fun.wrap(gen, transformation_args, None)

@curry
def transformation_with_aux(gen, fun, *transformation_args):
  out_store = Store()
  out_thunk = lambda: out_store.val
  return fun.wrap(gen, transformation_args, out_store), out_thunk

def fun_name(f):
  try:
    return f.__name__
  except:
    return str(f)

def wrap_init(f, params={}):
  """Wraps function `f` as a `WrappedFun`, suitable for transformation."""
  return WrappedFun(f, (), (), tuple(sorted(params.items())))


def cache(call):
  fun_caches = weakref.WeakKeyDictionary()
  def memoized_fun(fun, *args):
    cache = fun_caches.setdefault(fun.f, {})
    key = (fun.transforms, fun.params, args)
    result = cache.get(key, None)
    if result is not None:
      ans, stores = result
      fun.populate_stores(stores)
    else:
      ans = call(fun, *args)
      cache[key] = (ans, fun.stores)
    return ans
  return memoized_fun

@transformation
def hashable_partial(x, *args):
  ans = yield (x,) + args, {}
  yield ans
