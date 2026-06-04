# Copyright 2025 The JAX Authors.
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
"""Highly experimental and drafty code to extract repros.

This file contains the main data structure and the tracking code, which
attempts to intercept enough calls from user code to JAX and vice-versa, and
to build a call tree data structure (see `Call` and `Func`).

See https://docs.jax.dev/en/latest/debugging/repro.html

"""
import contextlib
import dataclasses
from functools import partial
import inspect
import itertools
import logging
import pathlib
import re
import threading
from typing import Any, Callable, Type, Union
import traceback
import weakref

from jax._src import config
from jax._src import tree_util
from jax._src.util import weakref_lru_cache

from jax._src.repro.trampolines import api_trampolines

Primitive = Any  # really, core.Primitive but don't want circular imports
Args = tuple[Any, ...]  # positional args
KWArgs = dict[str, Any]


class ReproError(Exception):
  pass


@dataclasses.dataclass
class ReproFlags:
  # Can be used to disable the repro tracker locally.
  # See https://docs.jax.dev/en/latest/debugging/repro.html#configuration-options.

  enable: bool = True

  log_calls: int = 0  # 0=no logging, 1=log all but primitives, 2=log all calls
  # The id of the Calls to log more details for
  log_calls_details: frozenset[int] = frozenset()

  # See documentation
  ERROR_MODE_IGNORE = "ignore"
  ERROR_MODE_LOG = "log"
  ERROR_MODE_DEFER = "defer"
  ERROR_MODE_RAISE_TRACKING = "raise_tracking"
  ERROR_MODE_RAISE = "raise"
  error_mode: str = ERROR_MODE_RAISE_TRACKING
  # How many stack frames to log when not "raise" and for log_calls_details
  log_traceback_frames = 40

  # Replace arrays with `np.ones` if larger than this threshold
  fake_array_threshold = 128

  # Inline the runtime files in the emitted repro
  inline_runtime = False

  # Check that the repro can be emitted for all top-level functions,
  # even in implicit mode when there are no errors.
  save_repro_on_success = False
  # This enables the saving of a repro on an uncaught exception from the
  # top-level call.
  save_repro_on_uncaught_exception = True

  def __init__(self):
    if config.repro_flags.value:
      for f in config.repro_flags.value.split(","):
        f, *fvalue = f.strip().split("=")
        def fvalue_to_int(default: int):
          return int(fvalue[0].strip()) if fvalue else default
        def fvalue_to_bool():
          return bool(fvalue_to_int(1))
        def fvalue_to_str():
          if len(fvalue) != 1: raise ValueError(f"Invalid value for flag `{f}`: {fvalue}")
          return fvalue[0].strip()

        if f == "enable":
          self.enable = fvalue_to_bool()
        elif f == "error_mode":
          val = fvalue_to_str()
          if val not in [ReproFlags.ERROR_MODE_DEFER,
                         ReproFlags.ERROR_MODE_LOG,
                         ReproFlags.ERROR_MODE_IGNORE,
                         ReproFlags.ERROR_MODE_RAISE,
                         ReproFlags.ERROR_MODE_RAISE_TRACKING]:
            raise ValueError(f"Invalid value for flag `{f}: {val}")
          self.error_mode = val
        elif f == "log_traceback_frames":
          self.log_traceback_frames = fvalue_to_int(30)
        elif f == "log_calls":
          self.log_calls = fvalue_to_int(1)
        elif f == "log_calls_details":
          call_ids = {int(anid.strip()) for anid in fvalue[0].split("+")}
          self.log_calls_details = self.log_calls_details.union(call_ids)

        elif f == "fake_array_threshold":
          self.fake_array_threshold = fvalue_to_int(128)
        elif f == "save_repro_on_uncaught_exception":
          self.save_repro_on_uncaught_exception = fvalue_to_bool()
        elif f == "save_repro_on_success":
          self.save_repro_on_success = fvalue_to_bool()
        elif f == "inline_runtime":
          self.inline_runtime = fvalue_to_bool()
        else:
          raise NotImplementedError(f"--jax_repro_flags: {f}")

  def log_calls_to(self, f: Union["Func", "Primitive"]) -> bool:
    log_calls = self.log_calls
    return (log_calls > 1 or log_calls == 1 and isinstance(f, Func))


@contextlib.contextmanager
def flags_override(**flag_values):
  """Usage: flags_override(log_calls=True, log_traceback_frames=40)
  """
  prev = {}
  for flag_name, flag_value in flag_values.items():
    prev[flag_name] = getattr(_thread_local_state.flags, flag_name)
    setattr(_thread_local_state.flags, flag_name, flag_value)
  try:
    yield
  finally:
    for flag_name, prev in prev.items():
      setattr(_thread_local_state.flags, flag_name, prev)


# Initialized lazily
save_statement_handler: Callable[..., None] = None  # type: ignore


# Same signature as pe.trace_to_jaxpr.
def _cached_trace_to_jaxpr(fun: Callable, rest_args, **params):
  run = _thread_local_state.trace_to_jaxpr_cache_miss_fun
  _thread_local_state.trace_to_jaxpr_cache_miss_fun = None  # No leaks
  assert run is not None
  return run()


@dataclasses.dataclass
class _CollectionPoint:
  """A request to collect the Statment upon completion of a given `level`."""
  stmt: Union["Statement", None]  # The collected statement
  stmt_level: int  # The level of the statement to collect
  deferred_error: Exception | None = None


class _ThreadLocalState(threading.local):
  def __init__(self):
    # Current call stack, [-1] is the top of the stack.
    # Bottom of the stack is always a JAX call.
    self.call_stack: list["Call"] = []

    # To simplify debugging, we assign deterministic ids to Func and Call
    # objects.
    self.func_index = itertools.count()  # Used only for non api_name funcs
    self.call_index = itertools.count()

    self.flags = ReproFlags()
    self.collection_points: list[_CollectionPoint] = []

    # The path and source for the last repro saved
    self.last_saved_repro: tuple[pathlib.Path, str] | None = None
    # For use with error_mode="defer"
    self.deferred_error: Exception | None = None
    # Map "fun at filename:lineno" to actual source location
    self.source_info_mapping: dict[str, str] = {}

    # Out of band value to communicate to the _cache_trace_to_jaxpr. We keep
    # this cache thread-local so that we can clear it when we pop the last
    # call from the stack.
    self.trace_to_jaxpr_cache_miss_fun: Callable[[], Any] | None = None
    self.trace_to_jaxpr_weakref_lru_cache = weakref_lru_cache(_cached_trace_to_jaxpr)

  def reset_counters(self):
    # Use at top-level when you want to start with fresh id counters
    assert not self.call_stack, self.call_stack
    self.func_index = itertools.count()
    self.call_index = itertools.count()

  def in_user_context(self):
    return len(self.call_stack) > 0 and self.call_stack[-1].is_function_def

  def warn_or_error(self, msg, traceback=None,
                    warning_only=False,
                    during_tracking=False):
    if self.flags.error_mode == ReproFlags.ERROR_MODE_IGNORE:
      return
    dolog = logging.warning if warning_only else logging.error
    dolog("Repro %s: %s", ("warning" if warning_only else "error"), msg)
    if not warning_only:
      if (self.flags.error_mode == ReproFlags.ERROR_MODE_RAISE or
          (self.flags.error_mode == ReproFlags.ERROR_MODE_RAISE_TRACKING and
           during_tracking)):
        raise ReproError(msg)
      if (self.flags.error_mode == ReproFlags.ERROR_MODE_DEFER or
          (self.flags.error_mode == ReproFlags.ERROR_MODE_RAISE_TRACKING and
           not during_tracking)):
        self.deferred_error = ReproError(
          "There were errors during repro "
          f"{'tracking' if during_tracking else 'emitting'}. "
          "See the logs for details.")

    if self.flags.log_traceback_frames > 0:
      if traceback:
        tb, where = traceback, "for call site "
      else:
        tb, where = current_traceback(), ""
      self.log_traceback(dolog, where, tb)

  def log_traceback(self, log_fn: Callable, where, tb, indent: str=""):
    log_fn("%s Traceback %s", indent, where)
    for lidx, l in enumerate(str(tb).splitlines()):
      if lidx >= self.flags.log_traceback_frames:
        log_fn("%s    ... reached %d limit", indent, self.flags.log_traceback_frames)
        break
      log_fn("%s     %s", indent, l)
    log_fn("%s    ... reached end of traceback, or a jit ...", indent)


_thread_local_state = _ThreadLocalState()


def last_saved_repro() -> tuple[pathlib.Path, str] | None:
  """The last repro (path and source) that was saved on the current thread."""
  return _thread_local_state.last_saved_repro


def reset_last_saved_repro():
  """Reset the last repro saved on the current thread."""
  _thread_local_state.last_saved_repro = None


def _monkey_patch_jax():
  from jax._src import core  # type: ignore
  old_bind = core.Primitive.bind
  if "Primitive.bind" in str(old_bind):
    core.Primitive.bind = bind_wrapper(old_bind)
  # Hook into pe.trace_to_jaxpr
  from jax._src.interpreters import partial_eval as pe  # type: ignore
  old_trace_to_jaxpr = pe.trace_to_jaxpr
  if "trace_to_jaxpr_wrapper" not in str(old_trace_to_jaxpr):
    pe.trace_to_jaxpr = trace_to_jaxpr_wrapper(old_trace_to_jaxpr)
  # and into pe.trace_to_jaxpr_dynamic
  old_trace_to_jaxpr_dynamic = pe.trace_to_jaxpr_dynamic
  if "trace_to_jaxpr_dynamic_wrapper" not in str(old_trace_to_jaxpr_dynamic):
    pe.trace_to_jaxpr_dynamic = trace_to_jaxpr_dynamic_wrapper(old_trace_to_jaxpr_dynamic)
  # Check that we have the expected wrappers in place.
  from jax._src.numpy import ufuncs  # type: ignore
  if not hasattr(ufuncs.remainder, "_is_repro_trampoline"):
    raise RuntimeError(
        "ufuncs.remainder does not seem to be the result of the repro "
        "jax_jit_trampoline. This suggests that the initialization order of "
        "the repro module and the rest of JAX is wrong. "
        "This may be because you are passing the --jax_repro_dir "
        "as a command line argument. It should be passed as an environment "
        "variable.")


# To avoid circular imports, we initialize some stuff lazily. Register
# here nullary initializers that will be called on first boundary call.
# Set to None once we have initialized
lazy_initializers: list[Callable[[], None]] | None = [_monkey_patch_jax]
_lazy_init_lock = threading.Lock()


def lazy_init():
  global lazy_initializers
  if lazy_initializers is None:
    return
  with _lazy_init_lock:
    if lazy_initializers is None:
      return
    while lazy_initializers:
      init_f = lazy_initializers.pop(0)
      init_f()
    lazy_initializers = None


class Func:
  def __init__(self, fun: Callable, *, is_user: bool,
               api_name: str | None = None,
               map_user_func_args: Callable | None = None):
    self.fun = fun
    self.real_api_fun = fun
    self.fun_info = fun_sourceinfo(fun)  # type: ignore
    self.fun_name = self.fun_info.split(" ")[0]
    self.is_user = is_user
    assert not (api_name and is_user), (fun, api_name)
    self.api_name = api_name
    if api_name is None:
      self.id = next(_thread_local_state.func_index)
    else:
      # Don't use up func_index for api_name because those happen at top-level
      # only before we start tracing; this allows us to reset the tracking state
      self.id = -1

    if is_user:
      assert map_user_func_args is None
      self.map_user_func_args = None
    elif map_user_func_args is None:  # See docstring for repro.boundary
      self.map_user_func_args = (
          lambda to_apply, *args, **kwargs: ((to_apply(args[0]), *args[1:]), kwargs))
    else:
      self.map_user_func_args = map_user_func_args
    # We can detect through caching that this function is a duplicate of another
    self.duplicate_of: Union["Func", None] = None

    self.__wrapped__ = fun  # Enable fun_sourceinfo(self)
    if hasattr(fun, "__name__"):
      setattr(self, "__name__", fun.__name__)
    if hasattr(fun, "__qualname__"):
      setattr(self, "__qualname__", fun.__name__)

    # For calls to USER functions, the invocation from which we emit a body
    self.function_def: Union["FunctionDef", None] = None
    # For calls to USER functions, we store a pre-processor for (args, kwargs)
    self.preprocess_args: Callable[[Args, KWArgs], tuple[Args, KWArgs]] = \
      lambda args, kwargs: (args, kwargs)

    if _thread_local_state.flags.log_calls_to(self.fun):
      logging.info(f"Created {self} for {self.fun_info}")

  def __call__(self, *args, **kwargs):
    return call_boundary(self, self.fun, args, kwargs)

  def __get__(self, instance, owner_class):
    # This is needed for bound methods, when we setattr(obj, "method", jit(f))
    if instance is None:
      # If accessed as Func.method, return self (the descriptor)
      return self
    # When accessed as func.method, return a callable that
    # binds 'instance' and calls the __call__ method of Func.
    return (lambda *args, **kwargs: self.__call__(instance, *args, **kwargs))

  python_name_re = re.compile(r'^fun(_.+)?_\d+$')
  def python_name(self):
    if self.api_name:
      return self.api_name
    if self.is_user and self.duplicate_of is not None:
      return self.duplicate_of.python_name()

    # If fun_name is likely the result of `f.python_name`, return the
    # original_fun_name
    m = Func.python_name_re.match(self.fun_name)
    if m is None:
      original_name = self.fun_name
    elif not m.group(1):
      original_name = UNKNOWN_FUN_NAME
    else:
      original_name = m.group(1)[1:]

    if original_name != UNKNOWN_FUN_NAME:
      return f"fun_{original_name}_{self.id}"
    else:
      return f"fun_{self.id}"

  def __repr__(self):  # Printing in debug logs
    return f"{'USER' if self.is_user else 'JAX'}[{self.python_name()}]"


@tree_util.register_pytree_node_class
class FuncPartial(Func):
  """A Func wrapper for a Partial, returned by some JAX APIs"""
  def __init__(self, func: tree_util.Partial, is_user: bool):
    super().__init__(func, is_user=is_user)
    self.args = func.args  # Some tests are looking for Partial attributes
    self.func = func.func
    self.keywords = func.keywords

  def tree_flatten(self):
    return ((self.fun,), self)

  @classmethod
  def tree_unflatten(cls, aux_data: "FuncPartial", children):
    return cls(children[0], is_user=aux_data.is_user)


@tree_util.register_pytree_node_class
class FuncVJP(Func):
  def __init__(self, func, is_user: bool):
    assert not is_user
    super().__init__(func, is_user=is_user)

  def tree_flatten(self):
    return ((self.fun,), self)

  @classmethod
  def tree_unflatten(cls, aux_data: "FuncVJP", children):
    return cls(children[0], is_user=aux_data.is_user)


def is_primitive(v: Union["Primitive", Func]) -> bool:
  from jax._src import core  # type: ignore
  return isinstance(v, core.Primitive)


def is_user_func(f: Union["Primitive", Func]) -> bool:
  return isinstance(f, Func) and f.is_user


def func_api_name(f: Union["Primitive", Func]) -> str:
  return f.api_name if isinstance(f, Func) else ""  # type: ignore


class WeakUnhashableKeyDictionary:
  """Dictionary keyed by id of unhashable keys.

  The entries are collected when the keys die.
  """
  class Id:
    def __init__(self, key):
      self._id = id(key)
    def __hash__(self):
      return self._id
    def __eq__(self, other):
      return self._id == other._id

  def __init__(self, *_, **__):
    self.keys = weakref.WeakValueDictionary()
    self.values = weakref.WeakKeyDictionary()

  def __getitem__(self, key):
    return self.values.__getitem__(WeakUnhashableKeyDictionary.Id(key))

  def __setitem__(self, key, value):
    # When key dies, its entry is deleted from self.keys, which means
    # _id dies, and its entry is deleted from self.values.
    _id = WeakUnhashableKeyDictionary.Id(key)
    self.keys.__setitem__(_id, key)
    self.values.__setitem__(_id, value)

  def __delitem__(self, key):
    _id = WeakUnhashableKeyDictionary.Id(key)
    self.keys.__delitem__(_id)
    self.values.__delitem__(_id)

  def get(self, key, default=None):
    _id = WeakUnhashableKeyDictionary.Id(key)
    if _id in self.keys:
      return self.values.__getitem__(_id)
    return default


class NormalizerContext:
  """We normalize values or arguments and results.

  See https://docs.jax.dev/en/latest/debugging/repro.html#handling-pytrees.

  During repro emitting, values are looked-up by id, so we use a
  memo table for normalization.
  """
  def __init__(self):
    self.memo_table: WeakUnhashableKeyDictionary = WeakUnhashableKeyDictionary()  # mapped by id(v)

  def normalize_value(self, v: Any, is_use: bool):
    from jax._src.repro import emitter  # TODO

    if isinstance(v, tuple) and not hasattr(v, "_fields"):  # skip NamedTuple
      return tuple(self.normalize_value(e, is_use) for e in v)
    if isinstance(v, list):
      return [self.normalize_value(e, is_use) for e in v]
    if isinstance(v, dict):
      return {k: self.normalize_value(e, is_use) for k, e in v.items()}

    # Emit first the data types for which we have special rules, because they
    # may also be pytrees and we don't want to loose their type.
    if v is None or isinstance(v, (int, float, bool, str, complex, Func)):
      return v

    if emitter.get_emitter(v) is not None:
      # If there is a special emitter we don't need to flatten this value.
      return v

    # Perhaps this is a user-defined pytree node, we flatten it but only
    # one level. We need to keep structure underneath, for APIs like
    # vmap in_axes that use tree prefix matching.
    try:
      vs = tree_util.flatten_one_level(v)[0]
    except ValueError:
      pass  # v is not an actual tree
    else:
      if emitter.maybe_singleton(vs) is not v:  # type: ignore
        return self.normalize_value(tuple(vs), is_use)  # type: ignore

    return v

_value_normalization_by_type: dict[Type, Callable[[NormalizerContext, Any], Any]] = {}


def register_normalizer_by_type(typ, normalizer: Callable[[NormalizerContext, Any], Any]) -> Callable[[NormalizerContext, Any], Any]:
  """Registers `normalizer` to use to emit operands of type `typ`"""
  _value_normalization_by_type[typ] = normalizer
  return normalizer


@dataclasses.dataclass(frozen=True)
class StateContext:
  default_lowering_platform: str | None
  trace_context: tuple


class Call:
  """Element of the stack calls, superclass for Statement and FunctionDef.

  The bottom of the stack is always a Statement. A FunctionDef is
  never called from a FunctionDef.
  """
  def __init__(self, parent: Union["Call", None],
               func: Union[Func, "Primitive"]):
    self.id = next(_thread_local_state.call_index)  # For debugging
    self.level: int = 0 if parent is None else parent.level + 1  # For debugging
    self.parent: Union["Call", None] = parent
    self.func = func
    self.is_function_def = is_user_func(func)
    # Collect the traceback before we are making the call
    self.raw_traceback = current_traceback()  # type: ignore

    # We save copies of args and kwargs, normalized. Initially None.
    self.args: Args | None = None
    self.kwargs: KWArgs | None = None
    self.result: Any | None = None
    self.uncaught_exception: Exception | None = None  # If not-None result is None
    self.uncaught_exception_traceback: str | None = None  # if level == 0 and uncaught_exception

    from jax._src.repro import repro_api  # type: ignore
    self.state_context = StateContext(
      default_lowering_platform=repro_api._local_state.default_lowering_platform,
      trace_context=config.trace_context())

    if self.level == 0:
      assert parent is None, ({self}, parent)
      assert not self.is_function_def, {self}
    else:
      assert parent is not None and parent.level == self.level - 1, {self}
    if self.is_function_def:
      assert parent
      if parent.is_function_def:
        _thread_local_state.warn_or_error(
            f"USER call {self} made from USER parent {parent}. "
            f"Perhaps the former was returned by a JAX function that does not wrap "
            "its results?",
            traceback=self.raw_traceback,
            during_tracking=True)

    user_parent: Call | None = parent
    while user_parent and not user_parent.func.is_user:
      user_parent = user_parent.parent
    self.parent_func_def: FunctionDef = user_parent  # type: ignore

    if self.level == 0:
      self.normalizer_ctx = NormalizerContext()
    else:
      assert self.parent is not None
      self.normalizer_ctx = self.parent.normalizer_ctx

  @staticmethod
  def start_call(func: Union["Func", "Primitive"], args, kwargs) -> tuple[Args, KWArgs]:
    """Start a call, and push a Call onto the call_stack."""
    if func_api_name(func):
      def do_wrap(f):
        return wrap_callable(f, is_user=True) if callable(f) else f
      wrapped_args, wrapped_kwargs = func.map_user_func_args(do_wrap, *args, **kwargs)  # pyrefly: ignore [not-callable]
    else:
      wrapped_args, wrapped_kwargs = args, kwargs

    is_function_def = is_user_func(func)
    parent = _thread_local_state.call_stack[-1] if _thread_local_state.call_stack else None
    if parent is None:
      _thread_local_state.deferred_error = None
    call = FunctionDef(parent, func) if is_function_def else Statement(parent, func)
    _thread_local_state.call_stack.append(call)
    call.log_start_call(wrapped_args, wrapped_kwargs)
    call.set_args(wrapped_args, wrapped_kwargs)

    return wrapped_args, wrapped_kwargs

  @staticmethod
  def end_call(*, result, exc: Exception | None):
    try:
      assert _thread_local_state.call_stack
      call = _thread_local_state.call_stack[-1]
      call.uncaught_exception = exc
      assert exc is None or result is None

      call.result = call.normalizer_ctx.normalize_value(result,
                                                        call.is_function_def)
      call.log_end_call(result, exc)

      _thread_local_state.call_stack.pop()
      if (_thread_local_state.collection_points and
          _thread_local_state.collection_points[-1].stmt_level == call.level):
        assert isinstance(call, Statement), call
        if exc is not None:
          call.uncaught_exception_traceback = traceback.format_exc()
        collection_point = _thread_local_state.collection_points.pop()
        collection_point.stmt = call
        collection_point.deferred_error = _thread_local_state.deferred_error
      else:
        collection_point = None  # No collection here

      if call.level == 0:  # We are called at the top-level
        assert isinstance(call, Statement), call
        # We don't carry the tracing cache to the next call tree.
        _thread_local_state.trace_to_jaxpr_weakref_lru_cache.cache_clear()

        if collection_point is None:
          if (exc is not None and
              _thread_local_state.flags.save_repro_on_uncaught_exception):
            call.uncaught_exception_traceback = traceback.format_exc()
            save_statement_handler(call, repro_name_prefix="jax_error_repro")
          elif (exc is None and
                _thread_local_state.flags.save_repro_on_success):
            save_statement_handler(
                call, extra_comment="save_repro_on_success",
                repro_name_prefix="jax_save_repro_on_success")

          if _thread_local_state.deferred_error is not None:
            raise _thread_local_state.deferred_error

    except Exception as e:
      # Exceptions here are bad, because they break the call_stack invariants
      logging.error(f"Exception caught in the end_call handler: {type(e)}: {e}\n{traceback.format_exc()}")
      raise

  def __repr__(self):
    return f"[{self.level}.{self.id}] {self.func}"

  def log_start_call(self, wrapped_args, kwargs):
    if _thread_local_state.flags.log_calls_to(self.func):
      # Log the Func and callable passed as args
      args_to_print = [a if isinstance(a, Func) or callable(a) else "" for a in wrapped_args]
      args_str = ", ".join(str(a) for a in args_to_print)
      args_str = re.sub(r'(, )+', ", ...", args_str) or "..."
      logging.info(f"{'  ' * self.level} start {self}({args_str})")
    if self.id in _thread_local_state.flags.log_calls_details:
      self.log_calls_arg_details(wrapped_args, kwargs)

  @staticmethod
  def log_value(v):
    if isinstance(v, Func):
      return f"{v} for {v.fun_info}"
    else:
      return f"{v}: {type(v)}"

  def log_calls_arg_details(self, args, kwargs):
    indent = "  " * self.level
    logging.info("%s calling: %s", indent,
                 self.func.fun_info if isinstance(self.func, Func) else self.func)
    _thread_local_state.log_traceback(logging.info, "at call site",
                                      self.raw_traceback, indent=indent)
    logging.info("%s detailed args", indent)
    for p, a in tree_util.tree_flatten_with_path(args)[0]:
      logging.info("%s args %s: %s", indent, tree_util.keystr(p),
                   Call.log_value(a))
    for p, a in tree_util.tree_flatten_with_path(kwargs)[0]:
      logging.info("%s kwargs %s: %s", indent, tree_util.keystr(p),
                   Call.log_value(a))

  def log_end_call(self, result, exc):
    if _thread_local_state.flags.log_calls_to(self.func):
      if exc is None:
        # Log the Func and callable results
        result_is_seq = isinstance(result, (tuple, list))
        result_tuple = result if result_is_seq else (result,)
        res_to_print = [r if isinstance(r, Func) or callable(r) else "" for r in result_tuple]
        res_str = ", ".join(str(r) for r in res_to_print)
        res_str = re.sub(r'(, )+', ", ...", res_str) or "..."
        res_str = f"res={res_str}"
      else:
        exc_str = str(exc)
        if self.level > 1:
          exc_str = exc_str[:1024] + "\n...."
        res_str = f"exc={exc_str}"
      logging.info(f"{'  ' * self.level} end {self}: {res_str}")
    if self.id in _thread_local_state.flags.log_calls_details:
      self.log_calls_res_details(result)

  def log_calls_res_details(self, result):
    logging.info(f"{'  ' * self.level} detailed result {self}")
    for p, a in tree_util.tree_flatten_with_path(result)[0]:
      logging.info(f"{'  ' * self.level} result {tree_util.keystr(p)}: {Call.log_value(a)}")

  @property
  def traceback(self):
    """A traceback inside the function being called, if available, otherwise
    the traceback where the Call was created (where the function is being called
    from)."""
    if self.body:  # type: ignore
      return self.body[0].raw_traceback  # type: ignore
    else:
      return self.raw_traceback


class Statement(Call):
  """A call to a JAX function inside a FunctionDef body"""
  def __init__(self, parent: Union["Call", None],
               func: Func):
    assert not is_user_func(func)
    super().__init__(parent, func)

  def set_args(self, args, kwargs):  # Statement
    # TODO: move these elsewhere, can we put them in repro_api?
    if isinstance(self.func, Func):
      if self.func.api_name in ("jit_call", "jit_aot_trace_call",
                                "jit_aot_lower_call",
                                "pjit_call", "pjit_aot_trace_call",
                                "pjit_aot_lower_call"):
        jitted_fun, jit_ctx_mesh, jit_kwargs, *rest_args = args
        jitted_fun.preprocess_args = (
          lambda args, kwargs: jit_call_filter_statics(
            jitted_fun, jit_kwargs, args, kwargs)[0:2])
        dyn_args, dyn_kwargs, new_jit_call_kwargs = jit_call_filter_statics(
            jitted_fun, jit_kwargs, tuple(rest_args), kwargs)
        args = (jitted_fun, jit_ctx_mesh, new_jit_call_kwargs, *dyn_args)
        kwargs = dyn_kwargs

      elif self.func.api_name == "custom_vjp_call":
        fun, fwd, bwd, custom_vjp_kwargs, *rest_args = args
        if custom_vjp_kwargs["nondiff_argnums"]:
          fun.preprocess_args = (
            lambda args, kwargs: filter_statics(args, kwargs,
                                                static_argnums=custom_vjp_kwargs["nondiff_argnums"]))
          fwd.preprocess_args = (
            lambda args, kwargs: filter_statics(args, kwargs,
                                                static_argnums=custom_vjp_kwargs["nondiff_argnums"]))
          bwd.preprocess_args = (
            lambda args, kwargs: filter_statics(args, kwargs,
                                                static_argnums=tuple(range(len(custom_vjp_kwargs["nondiff_argnums"])))))
          dyn_args, _ = filter_statics(rest_args, {},
                                       static_argnums=custom_vjp_kwargs["nondiff_argnums"],
                                       static_argnames=())
          new_custom_vjp_kwargs = dict(custom_vjp_kwargs, nondiff_argnums=())
          args = (fun, fwd, bwd, new_custom_vjp_kwargs, *dyn_args)

      elif self.func.api_name == "custom_jvp_call":
        fun, cjvp_kwargs, defjvp_kwargs, *fun_jvps_and_args = args
        fun_jvps, rest_args = fun_jvps_and_args[:kwargs["jvps_count"]], fun_jvps_and_args[kwargs["jvps_count"]:]
        if cjvp_kwargs["nondiff_argnums"]:
          assert not kwargs["uses_defjvps"]
          assert kwargs["jvps_count"] == 1, kwargs["jvps_count"]
          fun.preprocess_args = lambda args, kwargs: filter_statics(args, kwargs,
                                                                    static_argnums=cjvp_kwargs["nondiff_argnums"])
          for fun_jvp in fun_jvps:
            fun_jvp.preprocess_args = lambda args, kwargs: filter_statics(args, kwargs,
                                                                          static_argnums=tuple(range(len(cjvp_kwargs["nondiff_argnums"]))))
          dyn_args, _ = filter_statics(rest_args, {},
                                       static_argnums=cjvp_kwargs["nondiff_argnums"],
                                       static_argnames=())
          new_cjvp_kwargs = dict(cjvp_kwargs, nondiff_argnums=())
          args = (fun, new_cjvp_kwargs, defjvp_kwargs, *fun_jvps, *dyn_args)

      elif self.func.api_name == "checkpoint_call":
        from jax.ad_checkpoint import checkpoint_policies  # type: ignore
        f, t_args, t_kwargs, *rest_args = args
        new_t_kwargs = dict(t_kwargs)
        if (policy := new_t_kwargs.get("policy")) is not None:
          matching = [(policy_name, getattr(checkpoint_policies, policy_name))
                      for policy_name in dir(checkpoint_policies)
                      if policy is getattr(checkpoint_policies, policy_name)]
          if not matching:
            _thread_local_state.warn_or_error(
              f"Unrecognized jax.checkpoint policy: {policy}. Using 'dots_saveable'",
              traceback=self.raw_traceback,
              during_tracking=True)
            matching = [("dots_saveable", checkpoint_policies.dots_saveable)]
          new_t_kwargs["policy"] = EmitLiterally(f"jax.checkpoint_policies.{matching[0][0]}")
          args = (f, t_args, new_t_kwargs, *rest_args)

      elif self.func.api_name == "jax_repro_collect":
        fun, *rest_args = args
        static_argnums = kwargs.get("collect_static_argnums", ())
        static_argnames = kwargs.get("collect_static_argnames", {})
        new_kwargs = dict(kwargs)
        if "collect_static_argnums" in new_kwargs: del new_kwargs["collect_static_argnums"]
        if "collect_static_argnames" in new_kwargs: del new_kwargs["collect_static_argnames"]
        fun.preprocess_args = (
            lambda args, kwargs: filter_statics(
                args, kwargs,
                static_argnums=static_argnums, static_argnames=static_argnames))
        dyn_args, dyn_kwargs = filter_statics(rest_args, new_kwargs,
            static_argnums=static_argnums, static_argnames=static_argnames)
        args = (fun, *dyn_args)
        kwargs = dyn_kwargs

      elif self.func.api_name == "jax_vjphiprimitive_call":
        # The "prim" is treated as static, and not present in the repros
        kwargs = dict(kwargs)
        if "prim" in kwargs:
          del kwargs["prim"]

    self.args = self.normalizer_ctx.normalize_value(args, True)  # type: ignore
    self.kwargs = self.normalizer_ctx.normalize_value(kwargs, True)  # type: ignore
    if self.parent:
      assert self.parent.is_function_def, self.parent
      self.parent.body.append(self)  # pyrefly: ignore[missing-attribute]


class FunctionDef(Call):
  """A call to a USER function, collecting the body."""
  def __init__(self, parent: Union["Call", None],
               func: Func):
    assert is_user_func(func)
    super().__init__(parent, func)
    self.body : list[Statement] = []  # Only if `for_body`

  def set_args(self, args, kwargs):  # FunctionDef
    args, kwargs = self.func.preprocess_args(args, kwargs)
    self.args: Args = self.normalizer_ctx.normalize_value(args, False)  # type: ignore
    self.kwargs: KWArgs = self.normalizer_ctx.normalize_value(kwargs, False)  # type: ignore

    from jax._src import core  # type: ignore
    def has_tracers(call: FunctionDef) -> bool:
      leaves = tree_util.tree_leaves((call.args, call.kwargs))
      return any(isinstance(l, core.Tracer) for l in leaves)

    if self.func.function_def:
      # In Pallas the index_map are sometimes invoked with constants and then
      # only later are traced. We keep the invocation with tracers.
      prev_def = self.func.function_def
      if not has_tracers(prev_def) and has_tracers(self):
        _thread_local_state.warn_or_error(
            f"Replacing previous invocation {prev_def} (no tracers) "
            f"with new invocation {self} (contains tracers).",
            warning_only=True, traceback=self.raw_traceback,
            during_tracking=True)
        self.func.function_def = self
      else:
        # TODO: if both invocations have tracers, we ought to check that
        # they result in identical source
        self_tracers = "has tracers" if has_tracers(self) else "no tracers"
        prev_tracers = "has tracers" if has_tracers(prev_def) else "no tracers"
        _thread_local_state.warn_or_error(
            f"Ignoring additional invocation {self} ({self_tracers}). "
            f"Previous invocation was {prev_def} ({prev_tracers})",
            warning_only=True, traceback=self.raw_traceback,
            during_tracking=True)
    else:
      self.func.function_def = self


def bind_wrapper(actual_bind: Callable) -> Callable:
  def bind(*prim_and_args, **params):
    # We replace the core.Primitive._true_bind with this wrapper
    if not is_enabled():
      return actual_bind(*prim_and_args, **params)
    if not _thread_local_state.in_user_context():
      return actual_bind(*prim_and_args, **params)
    # We should not be seeing higher-order primitives in USER functions
    prim, *args = prim_and_args
    if (func_src_info := is_higher_order_primitive(prim, args, params)):
      _thread_local_state.warn_or_error(
          f"Binding primitive {prim} containing Jaxprs or functions, "
          f"for a function {func_src_info}. This means "
          "that there is a higher-order JAX API that is not "
          f"annotated with traceback_util.api_boundary(repro_api_name=...)",
          during_tracking=True)
    def do_bind(*_, **__):
      with enable(False):
        return actual_bind(*prim_and_args, **params)
    return call_boundary(prim, do_bind, tuple(args), params)

  bind.real_api_fun = actual_bind  # pyrefly: ignore[missing-attribute]
  return bind


def check_call_to_tracing():
  if is_enabled():
    # TODO(necula): we should also trigger the error if we are in implicit
    # mode at top-level (call_stack empty). But this would also trigger
    # under apply_primitive.
    if _thread_local_state.in_user_context():
      _thread_local_state.warn_or_error(
          "USER function calls directly into tracing. This means "
          "that there is a higher-order JAX API that is not "
          "annotated with traceback_util.api_boundary(repro_api_name=...)",
          during_tracking=True)
      # TODO(necula): check that fun is a USER function, or /jax/_src/ or
      # /flax/core/.
      # if False:
      #   msg = (
      #       "Expected that the call from `trace_to_jaxpr` (frame "
      #       f"index {i}) is to a USER function, and found instead call to "
      #       " an unwrapped function `{frames[i - 1]}`. This typically means a "
      #       "forgotten `core.repro_boundary` decorator.")
      #   _thread_local_state.warn_or_error(msg)


def trace_to_jaxpr_wrapper(actual_trace_to_jaxpr: Callable) -> Callable:
  def trace_to_jaxpr(fun, *rest_args, **params):
    if not is_enabled():
      return actual_trace_to_jaxpr(fun, *rest_args, **params)
    check_call_to_tracing()
    if isinstance(fun, Func):
      miss_fun = lambda: (actual_trace_to_jaxpr(fun, *rest_args, **params), fun)
      _thread_local_state.trace_to_jaxpr_cache_miss_fun = miss_fun
      # Use the real user function in the cache key
      try:
        res, hit_fun = _thread_local_state.trace_to_jaxpr_weakref_lru_cache(
          fun.fun, rest_args, **params)
      except TypeError as e:
        if "cannot create weak reference" in str(e):
          # TODO: explore further why test_jit_signature_fail fails here
          res, hit_fun = miss_fun()
        else:
          raise
      finally:
        _thread_local_state.trace_to_jaxpr_cache_miss_fun = None  # no leaks

      if hit_fun is not fun:
        # we had a cache hit
        fun.duplicate_of = hit_fun
      return res
    else:
      return actual_trace_to_jaxpr(fun, *rest_args, **params)

  trace_to_jaxpr.real_api_fun = actual_trace_to_jaxpr  # pyrefly: ignore[missing-attribute]
  return trace_to_jaxpr


def trace_to_jaxpr_dynamic_wrapper(actual_trace_to_jaxpr_dynamic: Callable) -> Callable:
  # TODO: implement caching for this one too
  def trace_to_jaxpr_dynamic(fun, *rest_args, **params):
    check_call_to_tracing()
    return actual_trace_to_jaxpr_dynamic(fun, *rest_args, **params)

  trace_to_jaxpr_dynamic.real_api_fun = actual_trace_to_jaxpr_dynamic  # pyrefly: ignore[missing-attribute]
  return trace_to_jaxpr_dynamic


class EmitLiterally:
  # These are used as replacements for arguments to be emitted literally.
  def __init__(self, literal: str):
    self.literal = literal


UNKNOWN_FUN_NAME = "unknown"
_fun_name_re = re.compile(r"(?:<built-in function (\S+)>)")
# TODO(necula): reuse this function from api_util.
def fun_sourceinfo(fun: Callable) -> str:
  # See DebugInfo.fun_src_info
  while isinstance(fun, partial):
    fun = fun.func
  fun = inspect.unwrap(fun)
  def sanitize_name(n: str) -> str:
    # TODO: Regexp
    if ("built-in" in n or "jnp.ufunc" in n or "<lambda>" in n or "<unnamed" in n):
      return UNKNOWN_FUN_NAME
    else:
      return n
  try:
    filename = fun.__code__.co_filename
    lineno = fun.__code__.co_firstlineno
    res = f"{sanitize_name(fun.__name__)} at {filename}:{lineno}"
    if (mapped := _thread_local_state.source_info_mapping.get(res)) is not None:
      return mapped
    return res
  except AttributeError:
    if isinstance(fun, Func):
      return repr(fun)
    try:
      fun_str = str(fun)
    except:
      return UNKNOWN_FUN_NAME
    # By contract, the function name has no spaces; also, we want to avoid
    # fun_sourceinfo of the form "<object Foo at 0x1234>", because it makes
    # lowering non-deterministic.
    if m := _fun_name_re.match(fun_str):
      return sanitize_name(m.group(1))
    return UNKNOWN_FUN_NAME


def current_traceback():
  from jax._src.lib import xla_client  # type: ignore
  # Use directly xla_client, otherwise we may get cached tracebacks
  tb = xla_client.Traceback.get_traceback()
  return tb.get_traceback() if tb is not None else None


def is_higher_order_primitive(prim, args, params) -> str | None:
  from jax._src import core  # type: ignore
  for arg in tuple(args) + tuple(params.values()):
    # Unpack tuples, and not more
    for a in (arg if isinstance(arg, tuple) else (arg,)):
      if isinstance(a, core.Jaxpr):
        func_src_info = a._debug_info.func_src_info
      elif isinstance(a, core.ClosedJaxpr):
        func_src_info = a.jaxpr._debug_info.func_src_info
      elif isinstance(a, core.lu.WrappedFun):
        func_src_info = a.debug_info.func_src_info
      else:
        continue
      return func_src_info or "<unknown>"
  return None


def filter_statics(args, kwargs,
                   static_argnums: tuple[int, ...]=(),
                   static_argnames: tuple[str, ...]=(),
                  ) -> tuple[Args, KWArgs]:
  dyn_args = []
  dyn_kwargs = {}
  # TODO: we should try to get this directly from JAX, there is a slight
  # chance we'll do something different otherwise. This is in C++ in JAX.
  static_argnums_set = {i if i >= 0 else len(args) - i for i in static_argnums}
  for i, a in enumerate(args):
    if i not in static_argnums_set:
      dyn_args.append(a)
  for k, a in sorted(kwargs.items()):
    if k not in static_argnames:
      dyn_kwargs[k] = a
  return tuple(dyn_args), dyn_kwargs


def jit_call_filter_statics(jitted_fun: Func,
                            jit_kwargs: KWArgs,
                            args: Args,
                            kwargs: KWArgs,
                            ) -> tuple[Args,
                                       KWArgs,
                                       KWArgs]:
  """Dynamic args and kwargs, and also new jit_kwargs, given a jit_call."""
  from jax._src import api_util  # type: ignore
  # Drop the static_argnums and static_argnames from the jit_kwargs
  new_jit_kwargs = dict(jit_kwargs)
  if "static_argnums" in jit_kwargs: del new_jit_kwargs["static_argnums"]
  if "static_argnames" in jit_kwargs: del new_jit_kwargs["static_argnames"]

  jitted_fun_sig = api_util.fun_signature(jitted_fun.fun)
  if jitted_fun_sig is None: # E.g., for built-in functions, or np.dot
    static_argnums: tuple[int, ...] = ()
    static_argnames: tuple[str, ...] = ()
  else:
    static_argnums, static_argnames = api_util.infer_argnums_and_argnames(
        jitted_fun_sig, jit_kwargs.get("static_argnums"),
        jit_kwargs.get("static_argnames"))
  dyn_args, dyn_kwargs = filter_statics(
      args, kwargs,
      static_argnums=static_argnums, static_argnames=static_argnames)
  if (static_argnums and
      (donate_argnums := new_jit_kwargs.get("donate_argnums")) is not None):
    new_donate_argnums = api_util.rebase_donate_argnums(donate_argnums, static_argnums)
    new_jit_kwargs["donate_argnums"] = new_donate_argnums

  return dyn_args, dyn_kwargs, new_jit_kwargs


def is_enabled() -> bool:
  return bool(config.repro_dir.value) and _thread_local_state.flags.enable


@contextlib.contextmanager
def enable(value: bool):
  """
  WARNING: This is part of the highly experimental repro feature.
  Subject to changes and removal.
  """
  prev = _thread_local_state.flags.enable
  _thread_local_state.flags.enable = value
  try:
    yield
  finally:
    _thread_local_state.flags.enable = prev


def boundary(boundary_fun: Callable, *,
             is_user: bool = False,
             repro_api_name: str | None = None,
             map_user_func_args: Callable | None = None,
             ) -> Callable:
  """
  Wraps a callable to track transitions from JAX to USER and back.

  Args:
    boundary_fun: the function that is called at the boundary of the JAX/USER
      region.
    is_user: whether the callee should be treated as a USER function.
    repro_api_name: the name to be emitted for the callee in repros. Must be
      present iff `not is_user`.
      See https://docs.jax.dev/en/latest/debugging/repro.html#trampolines for
      naming conventions.
    map_user_func_args: a function to return a replacement for the tuple of
      arguments, invoked
      `map_user_funcs(wrap_user, *args, **kwargs) -> (wrapped_args, wrapped_kwargs)`,
      where `args` and `kwargs` are the positional and keyword arguments,
      and `wrap_user` is a function to wrap function
      arguments as USER functions.
      `wrap_user` is the identity function when applied to a non-callable.
      If absent, then it maps only on the first positional argument.
  """
  # We assume that config.repro_dir does not change during execution and the
  # `boundary` when used as a function decorator does not change behavior. It
  # would be unsafe to use is_enabled() which may change during execution.
  if not bool(config.repro_dir.value): return boundary_fun
  if repro_api_name is not None and is_user:
    raise ValueError("api_name and is_user cannot both be present")
  if repro_api_name is not None and getattr(boundary_fun, "__module__", "") == "jax._src.repro.repro_api":
    pass
  elif (repro_api_name is not None and
        (trampoline := api_trampolines.get(repro_api_name)) is not None):
    return trampoline(boundary_fun)

  if not repro_api_name:
    # It is Ok to create Func for `is_user` because these are never called
    # while JAX initializes.
    return wrap_callable(boundary_fun, is_user=is_user)

  func = Func(boundary_fun, is_user=is_user, api_name=repro_api_name,
              map_user_func_args=map_user_func_args)
  return func


def call_boundary(func: Union[Func, "Primitive"],
                  boundary_fun: Callable,
                  args: Args, kwargs: KWArgs):
  if not is_enabled():
    return boundary_fun(*args, **kwargs)
  lazy_init()

  if (func_api_name(func) and
      _thread_local_state.call_stack and
      not _thread_local_state.call_stack[-1].is_function_def):
      # call a JAX API from within a JAX function
    if _thread_local_state.flags.log_calls:
      if not _thread_local_state.call_stack:
        detail = " (at top level)"
        indent = 0
      else:
        detail = f" (called from {_thread_local_state.call_stack[-1]})"
        indent = _thread_local_state.call_stack[-1].level
      logging.info(f"{'  ' * indent} Ignoring call to {func.api_name} from within JAX source (){detail}")
    return boundary_fun(*args, **kwargs)

  call_stack_level = len(_thread_local_state.call_stack)  # In case we get an exception during start_call
  try:
    wrapped_args, wrapped_kwargs = Call.start_call(func, args, kwargs)

    def call_jax_fun():  # Distinctive name for traceback readability
      return boundary_fun(*wrapped_args, **wrapped_kwargs)  # USER -> JAX

    def call_user_fun():
      return boundary_fun(*wrapped_args, **wrapped_kwargs)  # JAX -> USER

    if is_user_func(func):
      result = call_user_fun()
    else:
      result = call_jax_fun()
    Call.end_call(result=result, exc=None)
    return result
  except Exception as e:
    if len(_thread_local_state.call_stack) > call_stack_level:
      Call.end_call(result=None, exc=e)
    raise


def bypass_wrapper(f: Callable) -> Callable:
  """Bypasses the repro wrappers.

  Usage: `bypass_repro_wrapper(lax.scan)(f)` in order to use the real `lax.scan`,
  i.e., the one without the repro wrapper.
  """
  if hasattr(f, "real_api_fun"):
    return getattr(f, "real_api_fun")
  # If f is a bound method, and the real_boundary_fun is not set, we retrieve
  # only the function part, ignoring the __self__, so that we can invoke it
  # as a function
  return getattr(f, "__func__", f)


def wrap_callable(f: Callable, *, is_user: bool):
  assert callable(f), f
  if isinstance(f, Func):
    # I have seen this when I pass a BlockSpec into a push_block_spec and
    # we wrap the index_map as USER. Then if we return the same exact function
    # we wrap it as JAX. There are also situations where we pass the same
    # index_map multiple times around and it gets wrapped as a user function.
    if f.is_user == is_user:
      return f  # downside is that we may end up seeing multiple calls

  if isinstance(f, tree_util.Partial):
    res = FuncPartial(f, is_user=is_user)
    return res
  # Cannot import api here, we need to wrap functions before importing api
  if "api.VJP" in str(type(f)):
    res = FuncVJP(f, is_user=is_user)
    return res

  res = Func(f, is_user=is_user)  # type: ignore
  return res


@lazy_initializers.append  # type: ignore
def _():
  from jax._src import hijax  # type: ignore
  class ReproHiType(hijax.HiType):
    def __init__(self, typ):
      self._typ = typ

    def to_tangent_aval(self):
      return self._typ

    def __hash__(self):
      return hash(self._typ)
    def __eq__(self, other):
      return self._typ == other._typ


####

"""
TODO:
* print for function definitions the static arguments, as comments
* maybe print also the avals
* actual devices use jax_get_device, we need a way to make it platform
  agnostic
* why do we need to resolve_kwargs for the custom_jvp and friends? Because
  there are cases when we call the jvp function with kwargs but we never
  call the primal function, so the latter is generated with `*args, **kwargs`
  and then the repro will not be able to resolve the kwargs.
* reuse identical user functions when emitting
* If one forgets to annotate a higher-order repro_boundary then we won't wrap
  USER functions and we call into user code without tracking it. We would likely
  still be in USER mode, and then we'd give an error seeing a higher-order
  JAX primitive. We also have an error now when we create a Call to check the
  stack trace.
* test the emit_function cache
* try to figure out why we don't get an error in the fusible_matmul, if we
  comment out the repro_boundary on push_block_spec. Actually it seems that
  one can debug this by looking at the stack trace when an undefined tracer
  is encountered, by seeing that you call tracing (trace_to_jaxpr_dynamic)
  while in USER mode (after a call_jax_to_user)
* scatter carries a Jaxpr
* scatter_p exits in SC also, with the same name, e.g., addupdate_scatter,
  store_scatter
* had to wrap the callable result of USER functions, as USER function. This
  was needed to handle jax.custom_gradient, because it return a user function.
* must document the call stack structure, for explicit and implicit collection
* there is a problem when enabling jax_repro_dir from the command line: the
  command line processing is late and by then when we call jit(ufuncs.remainder)
  the jit api_boundary sees that repros as not enabled and then does not
  wrap. The JAX_REPRO_DIR can be passed only as an environment variable!!!
  Similarly, we seem to process the config.repro_flags only during the
  program initialization, which may be before the command-line is read.
* when we implement fuser.fusion, we have to carry over some properties from
  Fusion to the wrapped object. Do we really need to wrap the Fusion as
  opposed to the function it carries? And if so why don't we copy all
  properties over?
* if some functions are wrapped outside collect and then used inside collect
  we get collision on ids because the collector reset the tracker state
* I want to compress the stack trace, there are now 4 frames added for each
  transition from USER to JAX.
* finish implementing the deduplication of user functions
* I discovered for test_pallas_emit_pipeline_tpu_interpret_True that the
  index_maps may be called multiple times. The first time it is called
  with the argument 0 (from get_dma_slice) and then it is called with actual
  tracers. We ignore the **2nd** call, and we generate a body that uses the
  value 0 for the argument!!!
  I have seen other instances where the index_maps are traced multiple times,
  e.g., in the function _uses_arguments in the pipeline
* For the emitter, add a way to register type-based definers for structured
  types that can appear as args for FunctionDef (see TransformedRef)
"""
