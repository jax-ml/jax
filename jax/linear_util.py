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
Utilities for defining functions composed with transformations.

For example,

   from jax import linear_util as lu

   wf = lu.wrap_init(f)  # Produce a WrappedFun for applying transformations on `f`

A `WrappedFun` object represents a function `f`, together with a sequence of
nested transformations that are to be applied to the positional and keyword
arguments at call time and function return values at return time.
A transformation can take some static positional arguments that are given
at the wrapping time, and may also return some auxiliary output:

    wf, aux_out_thunk = trans1(wf, static_arg)

We can call the transformed function. First, the transformation is applied
to the dynamic args and keyword args to produce new dynamic and keyword args.
Then the underlying function is called and the transformation is applied to
the results.
If there are multiple transformations, they form a stack. The arguments are
transformed first with the last applied transformation; the results are
transformed first with the first applied transformation.

    res = wf.call_wrapped(dynamic_args, kwargs)
    # Now `aux_out_thunk()` is the auxiliary output.

A transformation is written as a generator function that takes zero or more
static positional arguments (given when the transformation is instantiated),
along with positional and keyword arguments to be transformed.
The generator will yield twice:

    @lu.transformation_with_aux
    def trans1(static_arg, *dynamic_args, **kwargs):
      ...
      # First yield: pair of transformed (args, kwargs). Get back the results.
      results = yield (new_dynamic_args, new_kwargs)
      ...
      # Second yield: pair of (transformed results, and auxiliary output)
      yield new_results, auxiliary_output


`WrappedFun` objects explicitly represent the set of transformations so that
they can be used as dictionary keys for memoization. `WrappedFun` objects
compare as equal only if they compute the same function. The static and the
dynamic positional arguments for the generators, and also the auxiliary output
data must be immutable, because it will be stored in function memoization tables.
"""

from typing import Any, Tuple
import weakref

from .util import curry

class StoreException(Exception): pass


class EmptyStoreValue(object): pass
_EMPTY_STORE_VALUE = EmptyStoreValue()

class Store(object):
  """Storage for a value, with checks for overwriting or reading empty store."""
  __slots__ = ("_val",)

  def __init__(self):
    self._val = _EMPTY_STORE_VALUE

  def store(self, val):
    if self._val is not _EMPTY_STORE_VALUE:
      raise StoreException("Store occupied")
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
    transforms: a list of `(gen, gen_static_args)` tuples representing
      transformations to apply to `f.` Here `gen` is a generator function
      and `gen_static_args` is a tuple of static arguments for the generator. See
      description at the start of this module for the expected behavior of the
      generator.
    stores: a list of out_store for the auxiliary output of the `transforms`.
    params: extra parameters to pass as keyword arguments to `f`, along with the
      transformed keyword arguments.
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

  def wrap(self, gen, gen_static_args, out_store) -> 'WrappedFun':
    """Add another transform and its store."""
    return WrappedFun(self.f, ((gen, gen_static_args),) + self.transforms,
                      (out_store,) + self.stores, self.params)

  def populate_stores(self, stores):
    """Copy the values from the `stores` into `self.stores`."""
    for self_store, other_store in zip(self.stores, stores):
      if self_store is not None:
        self_store.store(other_store.val)

  def call_wrapped(self, *args, **kwargs):
    """Calls the underlying function, applying the transforms.

    The positional `args` and keyword `kwargs` are passed to the first
    transformation generator.
    """
    stack = []
    for (gen, gen_static_args), out_store in zip(self.transforms, self.stores):
      gen = gen(*(gen_static_args + tuple(args)), **kwargs)
      args, kwargs = next(gen)
      stack.append((gen, out_store))
    gen = None

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
def transformation(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  """Adds one more transformation to a WrappedFun.
  Args:
    gen: the transformation generator function
    fun: a WrappedFun on which to apply the transformation
    gen_static_args: static args for the generator function
  """
  return fun.wrap(gen, gen_static_args, None)

@curry
def transformation_with_aux(gen, fun: WrappedFun, *gen_static_args) -> Tuple[WrappedFun, Any]:
  """Adds one more transformation with auxiliary output to a WrappedFun."""
  out_store = Store()
  out_thunk = lambda: out_store.val
  return fun.wrap(gen, gen_static_args, out_store), out_thunk

def fun_name(f):
  try:
    return f.__name__
  except:
    return str(f)

def wrap_init(f, params={}) -> WrappedFun:
  """Wraps function `f` as a `WrappedFun`, suitable for transformation."""
  return WrappedFun(f, (), (), tuple(sorted(params.items())))


def cache(call):
  """Cache decorator for WrappedFun calls.
  Args:
    call: a function that takes a WrappedFun as a first argument

  Returns:
     the memoized `call` function.
  """
  fun_caches = weakref.WeakKeyDictionary()

  def memoized_fun(fun: WrappedFun, *args):
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

  memoized_fun.cache_clear = fun_caches.clear
  return memoized_fun

@transformation
def hashable_partial(x, *args):
  ans = yield (x,) + args, {}
  yield ans
