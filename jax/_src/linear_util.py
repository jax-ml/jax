# Copyright 2018 The JAX Authors.
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

   from jax._src import linear_util as lu

   # Produce a WrappedFun for applying transformations on `f`
   wf = lu.wrap_init(f, debug_info=api_util.debug_info("test", f, (), {}))

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
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
import re
import time
from typing import Any, Hashable, NamedTuple
import warnings
import weakref

from jax._src import config
from jax._src import core
from jax._src import traceback_util
from jax._src.tree_util import KeyPath, generate_key_paths, keystr
from jax._src.util import HashableFunction, cache_clearing_funs, curry, fun_name


traceback_util.register_exclusion(__file__)


class StoreException(Exception): pass


class EmptyStoreValue: pass
_EMPTY_STORE_VALUE = EmptyStoreValue()

class Store:
  """Storage for a value, with checks for overwriting or reading empty store."""
  __slots__ = ("_val",)

  def __init__(self):
    self._val = _EMPTY_STORE_VALUE

  def store(self, val):
    if self._val is not _EMPTY_STORE_VALUE:
      raise StoreException("Store occupied")
    self._val = val

  def reset(self):
    # This should only be called in exceptional circumstances (e.g. debugging).
    self._val = _EMPTY_STORE_VALUE

  @property
  def val(self):
    if not self:
      raise StoreException("Store empty")
    return self._val

  def __nonzero__(self):
    return self._val is not _EMPTY_STORE_VALUE

  __bool__ = __nonzero__

class EqualStore:
  __slots__ = ('_store',)

  def __init__(self):
    self._store = Store()

  @property
  def val(self):
    return self._store.val

  def store(self, val):
    try:
      self._store.store(val)
    except StoreException as e:
      try:
        okay = bool(self._store._val == val)
      except:
        raise e from None
      else:
        if not okay:
          raise StoreException("Store occupied with not-equal value") from None

  def reset(self):
    self._store.reset()


class WrappedFun:
  """Represents a function `f` to which `transforms` are to be applied.

  Args:
    f: the function to be transformed.
    f_transformed: transformed function.
    transforms: a tuple of `(gen, gen_static_args)` tuples representing
      transformations to apply to `f.` Here `gen` is a generator function and
      `gen_static_args` is a tuple of static arguments for the generator. See
      description at the start of this module for the expected behavior of the
      generator.
    stores: a list of out_store for the auxiliary output of the `transforms`.
    params: a tuple of `(name, param)` tuples representing extra parameters to
      pass as keyword arguments to `f`, along with the transformed keyword
      arguments.
    in_type: optional input type
    debug_info: debugging info about the function being wrapped.
  """
  __slots__ = ("f", "f_transformed", "transforms", "stores", "params", "in_type", "debug_info")

  f: Callable
  f_transformed: Callable
  transforms: tuple[tuple[Callable, tuple[Hashable, ...]], ...]
  stores: tuple[Store | EqualStore | None, ...]
  params: tuple[tuple[str, Any], ...]
  in_type: core.InputType | None
  debug_info: DebugInfo

  def __init__(self, f: Callable,
               f_transformed: Callable,
               transforms: tuple[tuple[Callable, tuple[Hashable, ...]], ...],
               stores: tuple[Store | EqualStore | None, ...],
               params: tuple[tuple[str, Hashable], ...],
               in_type: core.InputType | None,
               debug_info: DebugInfo):
    self.f = f
    self.f_transformed = f_transformed
    self.transforms = transforms
    self.stores = stores
    self.params = params
    self.in_type = in_type
    self.debug_info = debug_info

  @property
  def __name__(self):
    return fun_name(self.f, "<unnamed wrapped function>")

  def wrap(self, gen, gen_static_args,
           out_store: Store | EqualStore | None) -> WrappedFun:
    """Add another transform and its store."""
    if out_store is None:
      return WrappedFun(self.f, partial(gen, self.f_transformed, *gen_static_args),
                        ((gen, gen_static_args),) + self.transforms,
                        (out_store,) + self.stores, self.params, None, self.debug_info)
    else:
      return WrappedFun(self.f, partial(gen, self.f_transformed, out_store, *gen_static_args),
                        ((gen, gen_static_args),) + self.transforms,
                        (out_store,) + self.stores, self.params, None, self.debug_info)

  def populate_stores(self, stores):
    """Copy the values from the `stores` into `self.stores`."""
    for self_store, other_store in zip(self.stores, stores):
      if self_store is not None:
        self_store.store(other_store.val)

  def call_wrapped(self, *args, **kwargs):
    """Calls the transformed function"""
    return self.f_transformed(*args, **kwargs)

  def __repr__(self):
    def transform_to_str(x):
      i, (gen, args) = x
      return f"{i}   : {fun_name(gen)}   {fun_name(args)}"
    transformation_stack = map(transform_to_str, enumerate(self.transforms))
    return "Wrapped function:\n" + '\n'.join(transformation_stack) + '\nCore: ' + fun_name(self.f) + '\n'

  def __hash__(self):
    return hash((self.f, self.transforms, self.params, self.in_type,
                 self.debug_info))

  def __eq__(self, other):
    return (self.f == other.f and self.transforms == other.transforms and
            self.params == other.params and self.in_type == other.in_type and
            self.debug_info == other.debug_info)

@curry
def transformation2(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  """Adds one more transformation to a WrappedFun.

  Args:
    gen: the transformation generator function
    fun: a WrappedFun on which to apply the transformation
    gen_static_args: static args for the generator function
  """
  return fun.wrap(gen, gen_static_args, None)

# Backwards compat only. TODO: deprecate
@curry
def transformation(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  def gen2(f, *args, **kwargs):
    gen_inst = gen(*args, **kwargs)
    args_, kwargs_ = next(gen_inst)
    return gen_inst.send(f(*args_, **kwargs_))
  return transformation2(gen2, fun, *gen_static_args)()

# Backwards compat only. TODO: deprecate
@curry
def transformation_with_aux(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  def gen2(f, store, *args, **kwargs):
    gen_inst = gen(*args, **kwargs)
    args_, kwargs_ = next(gen_inst)
    ans, aux = gen_inst.send(f(*args_, **kwargs_))
    store.store(aux)
    return ans
  return transformation_with_aux2(gen2, fun, *gen_static_args)()

@curry
def transformation_with_aux2(
    gen, fun: WrappedFun, *gen_static_args, use_eq_store: bool = False
) -> tuple[WrappedFun, Callable[[], Any]]:
  """Adds one more transformation with auxiliary output to a WrappedFun."""
  out_store = Store() if not use_eq_store else EqualStore()
  out_thunk = lambda: out_store.val
  return fun.wrap(gen, gen_static_args, out_store), out_thunk


class DebugInfo(NamedTuple):
  """Debugging info about a func, its arguments, and results."""
  traced_for: str             # e.g. 'jit', 'scan', etc

  func_src_info: str
  """e.g. f'{fun.__name__} at {filename}:{lineno}' or {fun.__name__} if we have
  no source location information. The first word is always the function name,
  which may be '<unknown>'.
  """

  arg_names: tuple[str, ...]
  """The paths of the flattened non-static argnames,
  e.g. `('x', 'dict_arg["a"]', ... )`.
  Uses the empty string for the args that do not correspond to
  user-named arguments, e.g., tangent args in `jax.jvp`, or for arguments that
  we are not yet tracking properly.
  At the moment, `arg_names` accuracy is best-effort.
  Use `safe_arg_names` to detect and handle an unexpected
  number of elements in `arg_names`.
  """

  result_paths: tuple[str, ...] | Callable[[], tuple[str, ...]] | None
  """The paths to the flattened results, e.g., `('result[0]', result[1])` for a
  function that returns a tuple of arrays, or `(result,)` for a function that
  returns a single array.
  The result paths are not available while we are tracing the function,
  instead we keep a thunk. It is possible for the result paths to be `None`
  only when we first create a `DebugInfo`, before we put it in `lu.WrappedFun`
  and before we start tracing.
  Inside a `lu.WrappedFun` it can be only a thunk or a tuple of strings.
  Once we are done tracing, we use
  `self.resolve_result_paths()` to execute the thunk and replace the
  actual result paths.
  At the moment, `result_paths` accuracy is best-effort.
  Use `safe_result_paths` to detect and handle an unexpected
  number of elements in `result_paths`.
  """

  def resolve_result_paths(self) -> DebugInfo:
    """Return a debug info with resolved result paths."""
    assert self.result_paths is not None
    if callable(self.result_paths):
      return self._replace(result_paths=tuple(self.result_paths()))
    return self

  @property
  def func_name(self) -> str:
    return self.func_src_info.split(" ")[0]

  def replace_func_name(self, name: str) -> DebugInfo:
    func_src_comps = self.func_src_info.split(" ")
    func_src_comps[0] = name
    return self._replace(func_src_info=" ".join(func_src_comps))

  @property
  def func_filename(self) -> str | None:
    m = _re_func_src_info.match(self.func_src_info)
    if not m: return None
    return m.group(3)

  @property
  def func_lineno(self) -> int | None:
    m = _re_func_src_info.match(self.func_src_info)
    if not m or m.group(4) is None: return None
    return int(m.group(4))

  def safe_arg_names(self, expected: int) -> tuple[str, ...]:
    """Get the arg_names with a safety check."""
    if len(self.arg_names) == expected:
      return self.arg_names
    else:
      # TODO(necula): this should not happen
      return ("",) * expected

  def filter_arg_names(self, keep: Sequence[bool]) -> tuple[str, ...]:
    """Keep only the arg_names for which `keep` is True."""
    return tuple(v for v, b in zip(self.safe_arg_names(len(keep)), keep) if b)

  def safe_result_paths(self, expected: int) -> tuple[str, ...]:
    """Get the result paths with a safety check."""
    assert self.result_paths is not None and not callable(self.result_paths), self
    if self.result_paths is not None and len(self.result_paths) == expected:
      return self.result_paths
    else:
      # TODO(necula): this should not happen
      return ("",) * expected

  def filter_result_paths(self, keep: Sequence[bool]) -> tuple[str, ...]:
    """Keep only the result_paths for which `keep` is True."""
    assert self.result_paths is not None and not callable(self.result_paths), self
    return tuple(v for v, b in zip(self.safe_result_paths(len(keep)), keep) if b)

_re_func_src_info = re.compile(r"([^ ]+)( at (.+):(\d+))?$")

def _missing_debug_info(for_what: str) -> DebugInfo:
  warnings.warn(
      f"{for_what} is missing a DebugInfo object. "
      "This behavior is deprecated, use api_util.debug_info() to "
      "construct a proper DebugInfo object and propagate it to this function. "
      "See https://github.com/jax-ml/jax/issues/26480 for more details.",
      DeprecationWarning, stacklevel=2)
  return DebugInfo("missing_debug_info", "<missing_debug_info>", (), ())

def wrap_init(f: Callable, params=None, *,
              debug_info: DebugInfo) -> WrappedFun:
  """Wraps function `f` as a `WrappedFun`, suitable for transformation."""
  params_dict = {} if params is None else params
  params = () if params is None else tuple(sorted(params.items()))
  fun = WrappedFun(f, partial(f, **params_dict), (), (), params, None, debug_info)
  if debug_info.result_paths is None:
    fun, result_paths_thunk = _get_result_paths_thunk(fun)
    debug_info = debug_info._replace(
        result_paths=HashableFunction(result_paths_thunk, closure=()))
  fun = WrappedFun(fun.f, fun.f_transformed, fun.transforms, fun.stores,
                   fun.params, fun.in_type, debug_info)
  return fun


# We replace <flat index 0> with 0
_re_clean_keystr_arg_names = re.compile(r"<flat index ([^>]+)>")
def _clean_keystr_arg_names(k: KeyPath) -> str:
  res = keystr(k)
  return _re_clean_keystr_arg_names.sub(r"\1", res)

@transformation_with_aux2
def _get_result_paths_thunk(_fun: Callable, _store: Store, *args, **kwargs):
  ans = _fun(*args, **kwargs)
  result_paths = tuple(f"result{_clean_keystr_arg_names(path)}" for path, _ in generate_key_paths(ans))
  if _store:
    # In some instances a lu.WrappedFun is called multiple times, e.g.,
    # the bwd function in a custom_vjp
    assert _store.val == result_paths, (_store, result_paths)
  else:
    _store.store(result_paths)
  return ans

def annotate(f: WrappedFun, in_type: core.InputType | None) -> WrappedFun:
  assert f.in_type is None
  if in_type is None:
    return f
  _check_input_type(in_type)
  return WrappedFun(f.f, f.f_transformed, f.transforms, f.stores, f.params, in_type, f.debug_info)

def _check_input_type(in_type: core.InputType) -> None:
  # Check that in_type is syntactically well-formed
  assert type(in_type) is tuple and all(type(e) is tuple for e in in_type)
  assert all(isinstance(a, core.AbstractValue) and type(b) is bool
             for a, b in in_type)

  def valid_size(d) -> bool:
    if isinstance(d, core.DBIdx) and type(d.val) is int and d.val >= 0:
      return True
    return (isinstance(d, (int, core.DBIdx, core.DArray)) and
            (not isinstance(d, core.DArray) or type(d) is core.bint and not d.shape))
  assert all(valid_size(d) for a, _ in in_type if type(a) is core.DShapedArray
             for d in a.shape)

  # Check that all DBIdx point to positions to the left of the input on which
  # they appear.
  assert all(d.val < i for i, (aval, _) in enumerate(in_type)
             if isinstance(aval, core.DShapedArray) for d in aval.shape
             if isinstance(d, core.DBIdx))

  # Check that all implicit arguments have at least one DBIdx pointing to them.
  provided = [e for _, e in in_type]
  for aval, _ in in_type:
    if type(aval) is core.DShapedArray:
      for d in aval.shape:
        if isinstance(d, core.DBIdx):
          provided[d.val] = True
  assert all(provided)


def cache(call: Callable, *,
          explain: Callable[[WrappedFun, bool, dict, tuple, float], None] | None = None):
  """Memoization decorator for functions taking a WrappedFun as first argument.

  Args:
    call: a Python callable that takes a WrappedFun as its first argument. The
      underlying transforms and params on the WrappedFun are used as part of the
      memoization cache key.

    explain: a function that is invoked upon cache misses to log an explanation
      of the miss.
      Invoked with `(fun, is_cache_first_use, cache, key, elapsed_sec)`.

  Returns:
     A memoized version of ``call``.
  """
  fun_caches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

  def memoized_fun(fun: WrappedFun, *args):
    cache = fun_caches.setdefault(fun.f, new_cache := {})  # type: ignore
    key = (fun.transforms, fun.params, fun.in_type, args, config.trace_context())
    result = cache.get(key, None)
    if result is not None:
      ans, stores = result
      fun.populate_stores(stores)
    else:
      if do_explain := explain and config.explain_cache_misses.value:
        start = time.time()
      ans = call(fun, *args)
      if do_explain:
        explain(fun, cache is new_cache, cache, key, time.time() - start)  # type: ignore
      cache[key] = (ans, fun.stores)

    return ans

  def _evict_function(f):
    fun_caches.pop(f, None)

  memoized_fun.cache_clear = fun_caches.clear  # type: ignore
  memoized_fun.evict_function = _evict_function  # type: ignore
  cache_clearing_funs.add(memoized_fun.cache_clear)
  return memoized_fun

@transformation2
def hashable_partial(f, *args):
  return f(*args)


def merge_linear_aux(aux1, aux2):
  try:
    out1 = aux1()
  except StoreException:
    # store 1 was not occupied, so store 2 better be
    try:
      out2 = aux2()
    except StoreException:
      raise StoreException("neither store occupied") from None
    else:
      return False, out2
  else:
    # store 1 was occupied, so let's check store 2 is not occupied
    try:
      out2 = aux2()
    except StoreException:
      return True, out1
    else:
      raise StoreException("both stores occupied")
