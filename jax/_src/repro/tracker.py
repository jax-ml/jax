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
import functools
import inspect
import itertools
import logging
import pathlib
import re
import threading
from typing import Any, Callable, Type, Union
from functools import partial
import traceback
import weakref

from jax._src import config
from jax._src import tree_util


Primitive = Any  # really, core.Primitive but don't want circular imports
Args = tuple[Any, ...]  # positional args
KWArgs = dict[str, Any]

class ReproError(Exception):
  pass


# To avoid circular imports, we initialize some stuff lazily. Register
# here nullary initializers that will be called on first boundary call.
lazy_initializers: list[Callable[[], None]] | None = []  # Set to None once we have initialized
def lazy_init():
  global lazy_initializers
  if lazy_initializers is None:
    return
  initializers = lazy_initializers
  lazy_initializers = None
  for init_f in initializers:
    init_f()


@dataclasses.dataclass
class ReproFlags:
  enable: bool = True

  log_calls: int = 0  # 0=no logging, 1=log all but primitives, 2=log all calls
  # The id of the Calls to log more details for
  log_calls_details: frozenset[int] = frozenset()

  ERROR_MODE_IGNORE = "ignore"  # errors are ignored
  ERROR_MODE_LOG = "log"  # errors are only logged
  ERROR_MODE_DEFER = "defer"  # errors are deferred, and ReproError is raised at end of collection
  ERROR_MODE_RAISE = "raise"  # errors are raised eagerly
  error_mode: str = ERROR_MODE_DEFER
  log_traceback_frames = 40  # How many stack frames to log when not "raise"

  # Replace arrays with `np.ones` if larger than this threshold
  fake_array_threshold = 128
  collect_last_top_call = False

  # Inline the runtime files in the emitted repro
  inline_runtime = False

  # Check that the repro can be emitted, even in implicit mode when there
  # are no errors.
  check_repro_emit = False
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
          self.error_mode = fvalue_to_str()
          if self.error_mode not in [ReproFlags.ERROR_MODE_DEFER,
                                     ReproFlags.ERROR_MODE_LOG,
                                     ReproFlags.ERROR_MODE_IGNORE,
                                     ReproFlags.ERROR_MODE_RAISE]:
            raise ValueError("Invalid value for flag `{f}: {self.error_mode}")
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
        elif f == "check_repro_emit":
          self.check_repro_emit = fvalue_to_bool()
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
uncaught_exception_handler: Callable[["Statement", str], None] = None  # type: ignore
check_repro_emit: Callable[["Statement"], None] = None  # type: ignore


class _ThreadLocalState(threading.local):
  def __init__(self):
    self.initialized = False

    # Current call stack, [-1] is the top of the stack.
    # Bottom of the stack is always a JAX call.
    self.call_stack: list["Call"] = []

    # To simplify debugging, we assign deterministic ids to Func and Call
    # objects.
    self.func_index = itertools.count()  # Used only for non api_name funcs
    self.call_index = itertools.count()

    self.flags = ReproFlags()
    self.last_top_call: Union["Statement", None] = None

    # The path and source for the last repro saved
    self.last_saved_repro: tuple[pathlib.Path, str] | None = None
    # It is useful to report repro generation errors after the repro is emitted
    self.had_deferred_errors = False
    # Map "fun at filename:lineno" to actual source location
    self.source_info_mapping: dict[str, str] = {}

  def initialize_state_if_needed(self):
    """Deferred per-thread initialization, only once we start using repros.
    This ensures that repros and core are fully loaded.
    """
    if self.initialized: return
    self.initialized = True
    # Hook repros into core.Primitive._true_bind
    from jax._src import core  # type: ignore
    old_true_bind = core.Primitive._true_bind
    if "Primitive._true_bind" in str(old_true_bind):
      core.Primitive._true_bind = true_bind_wrapper(old_true_bind)

  def in_user_context(self):
    return self.call_stack and self.call_stack[-1].is_function_def

  def warn_or_error(self, msg, traceback=None, warning_only=False):
    if self.flags.error_mode == ReproFlags.ERROR_MODE_IGNORE:
      return
    logging.error("Repro error: %s", msg)
    if not warning_only:
      if self.flags.error_mode == ReproFlags.ERROR_MODE_RAISE:
        raise ReproError(msg)
      if self.flags.error_mode == ReproFlags.ERROR_MODE_DEFER:
        self.had_deferred_errors = True

    if self.flags.log_traceback_frames > 0:
      if traceback:
        tb, where = traceback, "for call site "
      else:
        tb, where = current_traceback(), ""
      logging.error(f"Traceback {where}(top {self.flags.log_traceback_frames} frames):")
      for l in str(tb).splitlines()[:self.flags.log_traceback_frames]:
        logging.error("     %s", l)
      logging.error("    ....")


_thread_local_state = _ThreadLocalState()

@contextlib.contextmanager
def new_tracking_state():
  """Install a fresh tracking state"""
  global _thread_local_state
  prev = _thread_local_state
  try:
    if _thread_local_state.flags.log_calls:
      logging.info("Reset repro tracking state")
    _thread_local_state = _ThreadLocalState()

    # TODO: maybe we ought to separate the parts of the state that should
    # not be freshened up
    _thread_local_state.flags = prev.flags
    yield
  finally:
    # TODO: clean this up
    prev.last_saved_repro = _thread_local_state.last_saved_repro
    _thread_local_state = prev


def last_saved_repro() -> tuple[pathlib.Path, str] | None:
  """The last repro (path and source) that was saved."""
  return _thread_local_state.last_saved_repro


class Func:
  def __init__(self, fun: Callable, *, is_user: bool,
               api_name: str | None = None,
               map_user_func_args: Callable | None = None):
    self.fun = fun
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

    if map_user_func_args is None:  # See docstring for repro.boundary
      if (api_name is not None and
          (f := _api_name_to_map_user_func_args.get(api_name)) is not None):
        self.map_user_func_args = f
      else:
        self.map_user_func_args = (
            lambda to_apply, *args, **_: (to_apply(args[0]), *args[1:]))
    else:
      self.map_user_func_args = map_user_func_args
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


# TODO: for now, to allow exploration we keep the traceback_util.api_boundary
# API simple, and we hard-code here some map_user_funcs
# `map_user_funcs(toapply, *args, **kwargs) -> mapped_args`
# mapped_args contains `toapply(a)` for each `a` that is a traced function in `args`

def _lax_cond_map_user_func_args(toapply, pred, true_fun: Callable, false_fun: Callable, *operands):
  if not (callable(true_fun) and callable(false_fun)):
    # try falling back to the old, deprecated version of `cond`
    if callable(false_fun) and len(operands) == 2 and callable(operands[1]):
      x_true, f_true, x_false, f_false = true_fun, false_fun, *operands
      return (pred, x_true, toapply(f_true), x_false, toapply(f_false))
    else:
      raise NotImplementedError("Unrecognized old-style (?) lax.cond")
  else:
    return (pred, toapply(true_fun), toapply(false_fun), *operands)

_api_name_to_map_user_func_args: dict[str, Callable] = {
    "jax_cond": _lax_cond_map_user_func_args,
    "jax_switch": (
      lambda toapply, idx, branches, operands, **kwargs: (idx, tuple(map(toapply, branches)), operands)),
    "jax.lax.fori_loop": (
      lambda toapply, l, u, body, *args, **kwargs: (l, u, toapply(body), *args)),
    "jax.lax.while_loop": (
      lambda toapply, cond, body, *args, **kwargs: (toapply(cond), toapply(body), *args)),
}


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

  def __init__(self, *args, **kwargs):
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

  We save normalized arguments and results in the call tree.
  The goal is to simplify the values by eliminating user-defined PyTrees,
  which are not needed for repro generation; however, we keep some PyTrees for which we have
  special emitter rules, e.g., lax.GatherDimensionNumbers

  NOTE: This also supports normalizing some types, but this is currently
  turned off. We used to normalize core.Tracer by replacing it with
  `NormTracer`, so that we don't leak tracers. But we have a bunch of
  other user-defined types that are not normalized and can contain
  tracers (e.g., to avoid leaked tracers;
  .

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

    # for t in type(v).__mro__:
    #   if (normalizer := _value_normalization_by_type.get(t)) is not None:
    #     # If 'v' has a normalizer, it is a leaf.
    #     if (vn := self.memo_table.get(v)) is not None:
    #       return vn
    #     if is_use:
    #       msg = f"Normalizing undefined {v} of type {type(v)}"
    #       _thread_local_state.warn_or_error(msg)
    #     vn = self.memo_table[v] = normalizer(self, v)
    #     return vn

    if emitter.get_emitter(v) is not None:
      return v

    # Perhaps this is a user-defined pytree node, we flatten it but only
    # one level. We need to keep structure underneath, for APIs like
    # vmap in_axes that use tree prefix matching.
    try:
      vs = tree_util.flatten_one_level(v)[0]
    except ValueError:
      pass  # v is not an actual tree
    else:
      if (vss := emitter.maybe_singleton(vs)) is not v:  # type: ignore
        return self.normalize_value(tuple(vs), is_use)  # type: ignore

    return v

_value_normalization_by_type: dict[Type, Callable[[NormalizerContext, Any], Any]] = {}


def register_normalizer_by_type(typ, normalizer: Callable[[NormalizerContext, Any], Any]) -> Callable[[NormalizerContext, Any], Any]:
  """Registers `normalizer` to use to emit operands of type `typ`"""
  _value_normalization_by_type[typ] = normalizer
  return normalizer


# class NormTracer:
#   def __init__(self, shape, dtype):
#     self.shape = shape
#     self.dtype = dtype
#   def __repr__(self):
#     return f"NormTracer({self.shape}, {self.dtype}, id={id(self)})"


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
    self.had_deferred_errors = False  # There were errors during repro collection

    # We save copies of args and kwargs, normalized.
    self.args: Args | None = None
    self.kwargs: KWArgs | None = None
    self.result: Any | None = None

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
            traceback=self.raw_traceback)

    user_parent: Call | None = parent
    while user_parent and not user_parent.func.is_user:
      user_parent = user_parent.parent
    self.parent_func_def: FunctionDef = user_parent  # type: ignore

    if self.level == 0:
      self.normalizer_ctx = NormalizerContext()
    else:
      self.normalizer_ctx = parent.normalizer_ctx  # type: ignore

    not config.enable_checks.value or check_traceback(self.raw_traceback.frames)  # type: ignore


  @staticmethod
  def start_call(func: Union["Func", "Primitive"], args, kwargs) -> tuple[Args, KWArgs]:
    """Start a call, and push a Call onto the call_stack."""
    _thread_local_state.initialize_state_if_needed()
    if func_api_name(func):
      def do_wrap(f):
        return wrap_callable(f, is_user=True) if callable(f) else f
      wrapped_args = func.map_user_func_args(do_wrap, *args, **kwargs)
    else:
      wrapped_args = args

    is_function_def = is_user_func(func)
    parent = _thread_local_state.call_stack[-1] if _thread_local_state.call_stack else None
    call = FunctionDef(parent, func) if is_function_def else Statement(parent, func)
    call.set_args(wrapped_args, kwargs)
    _thread_local_state.call_stack.append(call)
    call.log_start_call(wrapped_args, kwargs)

    return wrapped_args, kwargs

  @staticmethod
  def end_call(*, res, exc: Exception | None):
    try:
      assert _thread_local_state.call_stack
      call = _thread_local_state.call_stack[-1]
      result = res if exc is None else None

      # Watch for USER functions returning functions that are not already
      # wrapped. For now, we handle only tuple results.
      if call.is_function_def and isinstance(res, tuple):
        wrapped_res = tuple(
          wrap_callable(f, is_user=True) if (
            callable(f) and not isinstance(f, Func)) else f
          for f in res)
      else:
        wrapped_res = res

      call.result = call.normalizer_ctx.normalize_value(wrapped_res,
                                                        call.is_function_def)
      call.log_end_call(result, exc)

      _thread_local_state.call_stack.pop()
      if call.level == 0:  # We are called at the top-level
        assert isinstance(call, Statement)
        if _thread_local_state.flags.collect_last_top_call:
          _thread_local_state.last_top_call = call  # type: ignore

        if (exc is not None and not isinstance(exc, ReproError) and
            _thread_local_state.flags.save_repro_on_uncaught_exception):
          uncaught_exception_handler(call, traceback.format_exc())
        elif (_thread_local_state.had_deferred_errors and
              _thread_local_state.flags.save_repro_on_uncaught_exception):
          # We got deferred errors during repro collection in implicit mode.
          _thread_local_state.had_deferred_errors = False
          call.had_deferred_errors = True

        if _thread_local_state.flags.check_repro_emit:
          check_repro_emit(call)

      return wrapped_res
    except Exception as e:
      # Exceptions here are bad, because they break the call_stack invariants
      logging.error(f"Exception caught in the exit handler: {type(e)}: {e}\n{traceback.format_exc()}")
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
      self.log_arg_details(wrapped_args, kwargs)

  @staticmethod
  def log_value(v):
    if isinstance(v, Func):
      return f"{v} for {v.fun_info}"
    else:
      return f"{v}: {type(v)}"

  def log_arg_details(self, args, kwargs):
    logging.info(f"{'  ' * self.level} detailed args {self}")
    for p, a in tree_util.tree_flatten_with_path(args)[0]:
      logging.info(f"{'  ' * self.level} args {tree_util.keystr(p)}: {Call.log_value(a)}")
    for p, a in tree_util.tree_flatten_with_path(kwargs)[0]:
      logging.info(f"{'  ' * self.level} kwargs {tree_util.keystr(p)}: {Call.log_value(a)}")

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
      self.log_res_details(result)

  def log_res_details(self, result):
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
    if isinstance(self.func, Func):
      if self.func.api_name in ("jax_jit_call", "jax_jit_aot_trace_call",
                                "jax_jit_aot_lower_call",
                                "jax_pjit_call", "jax_pjit_aot_trace_call",
                                "jax_pjit_aot_lower_call"):
        jitted_fun, jit_ctx_mesh, jit_kwargs, *rest_args = args
        jitted_fun.preprocess_args = (
          lambda args, kwargs: jax_jit_call_filter_statics(
            jitted_fun, jit_kwargs, args, kwargs)[0:2])
        dyn_args, dyn_kwargs, new_jit_call_kwargs = jax_jit_call_filter_statics(
            jitted_fun, jit_kwargs, tuple(rest_args), kwargs)
        args = (jitted_fun, jit_ctx_mesh, new_jit_call_kwargs, *dyn_args)
        kwargs = dyn_kwargs

      elif self.func.api_name == "jax_custom_vjp_call":
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

      elif self.func.api_name == "jax_custom_jvp_call":
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

      elif self.func.api_name == "jax_checkpoint_call":
        from jax.ad_checkpoint import checkpoint_policies  # type: ignore
        from jax._src.repro import emitter
        f, t_args, t_kwargs, *rest_args = args
        new_t_kwargs = dict(t_kwargs)
        if (policy := new_t_kwargs.get("policy")) is not None:
          matching = [(policy_name, getattr(checkpoint_policies, policy_name))
                      for policy_name in dir(checkpoint_policies)
                      if policy is getattr(checkpoint_policies, policy_name)]
          if not matching:
            _thread_local_state.warn_or_error(
              f"Unrecognized jax.checkpoint policy: {policy}. Using 'dots_saveable'",
              traceback=self.raw_traceback)
            matching = [("dots_saveable", checkpoint_policies.dots_saveable)]
          new_t_kwargs["policy"] = emitter.EmitLiterally(f"jax.checkpoint_policies.{matching[0][0]}")
          args = (f, t_args, new_t_kwargs, *rest_args)

    self.args = self.normalizer_ctx.normalize_value(args, True)  # type: ignore
    self.kwargs = self.normalizer_ctx.normalize_value(kwargs, True)  # type: ignore
    if self.parent:
      assert self.parent.is_function_def
      self.parent.body.append(self)


class FunctionDef(Call):
  """A call to a USER function, collecting the body."""
  def __init__(self, parent: Union["Call", None],
               func: Func):
    assert is_user_func(func)
    super().__init__(parent, func)
    self.body : list[Statement] = []  # Only if `for_body`
    if func.function_def:
      # We know of cases when this happens, e.g., higher-order custom_vjp,
      # fusions
      _thread_local_state.warn_or_error(
          f"Ignoring additional invocation {self}. "
          f"Previous invocation was {self.func.function_def}",
          warning_only=True, traceback=self.raw_traceback)
    else:
      func.function_def = self

  def set_args(self, args, kwargs):  # FunctionDef
    args, kwargs = self.func.preprocess_args(args, kwargs)
    self.args: Args = self.normalizer_ctx.normalize_value(args, False)  # type: ignore
    self.kwargs: KWArgs = self.normalizer_ctx.normalize_value(kwargs, False)  # type: ignore


def true_bind_wrapper(actual_true_bind: Callable) -> Callable:
  def true_bind(*prim_and_args, **params):
    # We replace the core.Primitive._true_bind with this wrapper
    if not is_enabled():
      return actual_true_bind(*prim_and_args, **params)
    if not _thread_local_state.in_user_context():
      return actual_true_bind(*prim_and_args, **params)
    # We should not be seeing higher-order primitives in USER functions
    prim, *args = prim_and_args

    if (config.enable_checks.value and
        is_higher_order_primitive(prim, args, params)):
      _thread_local_state.warn_or_error(
          f"Encountered primitive {prim} containing Jaxprs or functions. This means "
          "that there is a higher-order JAX API that is not "
          "annotated with traceback_util.api_boundary(repro_api_name=...)")
    def do_bind(*_, **__):
      with enable(False):
        return actual_true_bind(*prim_and_args, **params)
    return call_boundary(prim, do_bind, tuple(args), params)

  true_bind.real_boundary_fun = actual_true_bind
  return true_bind


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
  except AttributeError as e:
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
  return xla_client.Traceback.get_traceback().get_traceback()


def is_higher_order_primitive(prim, args, params) -> bool:
  from jax._src import core  # type: ignore
  for arg in tuple(args) + tuple(params.values()):
    # Unpack tuples, and not more
    for a in (arg if isinstance(arg, tuple) else (arg,)):
      if isinstance(a, (core.Jaxpr, core.ClosedJaxpr, core.lu.WrappedFun)):
        return True
  return False


def filter_statics(args, kwargs,
                   static_argnums: tuple[int, ...]=(),
                   static_argnames: tuple[str, ...]=(),
                  ) -> tuple[Args, KWArgs]:
  dyn_args = []
  dyn_kwargs = {}
  # TODO: we should try to get this directly from JAX, there is a slight
  # chance we'll do something different otherwise. This is in C++ in JAX.
  static_argnums = {i if i >= 0 else len(args) - i for i in static_argnums}
  for i, a in enumerate(args):
    if i not in static_argnums:
      dyn_args.append(a)
  for k, a in sorted(kwargs.items()):
    if k not in static_argnames:
      dyn_kwargs[k] = a
  return tuple(dyn_args), dyn_kwargs


def jax_jit_call_filter_statics(jitted_fun: Func,
                                jit_kwargs: KWArgs,
                                args: Args,
                                kwargs: KWArgs,
                                ) -> tuple[Args,
                                           KWArgs,
                                           KWArgs]:
  """Dynamic args and kwargs, and also new jit_kwargs, given a jax_jit_call."""
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


# See comments in repro_api.py
boundary_trampolines: dict[str, Callable[..., Any]] = {}


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
             api_name: str | None = None,
             map_user_func_args: Callable | None = None,
             ) -> Callable:
  """
  Wraps a callable to track transitions from JAX to USER and back.

  Args:
    boundary_fun: the function that is called at the boundary of the JAX/USER
      region.
    is_user: whether the callee should be treated as a USER function.
    api_name: the name to be emitted for the callee in repros. Must be
      present iff `not is_user`.
    map_user_func_args: a function to return a replacement for the tuple of
      positional arguments, invoked `map_user_funcs(wrap_func, *args, *kwargs)'
      where `args` is the tuple of positional arguments and `wrap_func` is a
      function to wrap function arguments as USER functions. `wrap_func` is
      the identity function when applied to a non-callable. The function
      should return the tuple of wrapped positional args. For now we assume
      that JAX APIs can have USER functions only among the positional arguments.
      If absent, then it maps only on the first positional argument.
  """
  if not is_enabled(): return boundary_fun
  if api_name is not None and is_user:
    raise ValueError("api_name and is_user cannot both be present")
  if api_name is not None and (trampoline := boundary_trampolines.get(api_name)) is not None:
    return trampoline(boundary_fun)

  if not api_name:
    # It is Ok to create Func for `is_user` because these are never called
    # while JAX initializes.
    return wrap_callable(boundary_fun, is_user=is_user)

  func = Func(boundary_fun, is_user=is_user, api_name=api_name,
              map_user_func_args=map_user_func_args)


  @functools.wraps(boundary_fun)
  def repro_boundary_wrapper(*args, **kwargs):
    return call_boundary(func, boundary_fun, args, kwargs)

  repro_boundary_wrapper.real_boundary_fun = boundary_fun
  return repro_boundary_wrapper


def call_boundary(func: Union[Func, "Primitive"],
                  boundary_fun: Callable,
                  args: Args, kwargs: KWArgs):
  if not is_enabled():
    return boundary_fun(*args, **kwargs)
  lazy_init()

  if (func_api_name(func) and
      _thread_local_state.call_stack and
      not _thread_local_state.call_stack[-1].is_function_def):
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
    result = Call.end_call(res=result, exc=None)
    return result
  except Exception as e:
    if len(_thread_local_state.call_stack) > call_stack_level:
      Call.end_call(res=None, exc=e)
    raise


def bypass_wrapper(f: Callable) -> Callable:
  """Bypasses the repro wrappers.

  Usage: `bypass_repro_wrapper(lax.scan)(f)` in order to use the real `lax.scan`,
  i.e., the one without the repro api_boundary wrapper.
  """
  if hasattr(f, "real_boundary_fun"):
    return getattr(f, "real_boundary_fun")
  # If f is a bound method, and the real_boundary_fun is not set, we retrieve
  # only the function part, ignoring the __self__, so that we can invoke it
  # as a function
  return getattr(f, "__func__", f)


def wrap_callable(f: Callable, *, is_user: bool):
  assert callable(f), f
  if isinstance(f, Func) and f.is_user == is_user: return f
  if isinstance(f, tree_util.Partial):
    res = FuncPartial(f, is_user=is_user)
    return res

  res = Func(f, is_user=is_user)  # type: ignore
  return res


def generic_trampoline(transform_name: str, real_transform: Callable) -> Callable:
  """
  Builds a generic trampoline, e.g., for a transformation "trans" (grad, vmap, ...):
    jax.trans(f, *trans_args, **trans_kwargs)(*args, **kwargs) ->
       jax_trans_call(f, trans_args, trans_kwargs, *args, *kwargs)

  This trampoline works when there is one transformed function, passed as the
  first positional argument, and the transformation returns the transformed
  function (which itself is first-order).

  The `jax_trans_call` are defined in repro_api.py.
  """
  jax_trans_call_name = f"jax_{transform_name}_call"
  from jax._src.repro import repro_api

  def trampoline(fun: Callable, *trans_args: tuple[Any], **trans_kwargs: KWArgs):
    def call_trampoline(*args, **kwargs):
      return getattr(repro_api, jax_trans_call_name)(fun, trans_args, trans_kwargs, *args, **kwargs)
    call_trampoline.__name__ = f"jax_{transform_name}_call_trampoline"
    call_trampoline.__qualname__ = call_trampoline.__name__
    return call_trampoline
  trampoline.__name__ = f"jax_{transform_name}_trampoline"
  trampoline.__qualname__ = trampoline.__name__
  trampoline.real_boundary_fun = real_transform
  return trampoline

boundary_trampolines["jax.shard_map"] = partial(generic_trampoline, "shard_map")
boundary_trampolines["jax.pmap"] = partial(generic_trampoline, "pmap")
boundary_trampolines["jax.vmap"] = partial(generic_trampoline, "vmap")
boundary_trampolines["jax.grad"] = partial(generic_trampoline, "grad")
boundary_trampolines["jax.linear_transpose"] = partial(generic_trampoline, "linear_transpose")
boundary_trampolines["jax.jacfwd"] = partial(generic_trampoline, "jacfwd")
boundary_trampolines["jax.jacrev"] = partial(generic_trampoline, "jacrev")
boundary_trampolines["jax.hessian"] = partial(generic_trampoline, "hessian")
boundary_trampolines["jax.value_and_grad"] = partial(generic_trampoline, "value_and_grad")
boundary_trampolines["jax.checkpoint"] = partial(generic_trampoline, "checkpoint")


def jit_trampoline(is_pjit: bool, real_jit: Callable) -> Callable:
  # A trampoline for both jit and pjit
  from jax._src import mesh as mesh_lib

  def get_ctx_mesh() -> mesh_lib.Mesh | None:
    ctx_mesh = mesh_lib.get_concrete_mesh()
    if ctx_mesh.empty and is_pjit:
      ctx_mesh = mesh_lib.thread_resources.env.physical_mesh
    return ctx_mesh if not ctx_mesh.empty else None

  def jax_jit_trampoline(fun: Callable | None = None, /, **jit_kwargs):
    from jax._src.repro.repro_api import jax_jit_call, jax_jit_aot_trace_call, jax_jit_aot_lower_call
    from jax._src.repro.repro_api import jax_pjit_call, jax_pjit_aot_trace_call, jax_pjit_aot_lower_call
    if fun is None:  # Starting with JAX v0.8.1, jax.jit(**kwargs) can be a decorator
      return lambda fun: jax_jit_trampoline(fun, **jit_kwargs)

    # Ignore calls from xla_primitive_callable which use jit over an internal
    # function that just binds the primitive
    if getattr(fun, "_apply_primitive", None):
      return real_jit(fun, **jit_kwargs)

    def jax_jit_call_trampoline(*args, **kwargs):
      to_call = jax_pjit_call if is_pjit else jax_jit_call
      return to_call(fun, get_ctx_mesh(), jit_kwargs, *args, **kwargs)

    def jax_jit_aot_trace_trampoline(*args, **kwargs):
      to_call = jax_pjit_aot_trace_call if is_pjit else jax_jit_aot_trace_call
      return to_call(fun, get_ctx_mesh(), jit_kwargs, *args, **kwargs)

    jax_jit_call_trampoline.trace = jax_jit_aot_trace_trampoline

    def jax_jit_aot_lower_trampoline(*args, **kwargs):
      to_call = jax_pjit_aot_lower_call if is_pjit else jax_jit_aot_lower_call
      return to_call(fun, get_ctx_mesh(), jit_kwargs, *args, **kwargs)

    if hasattr(fun, "__name__"):
      jax_jit_call_trampoline.__name__ = fun.__name__
    if hasattr(fun, "__qualname__"):
      jax_jit_call_trampoline.__qualname__ = fun.__qualname__
    jax_jit_call_trampoline.lower = jax_jit_aot_lower_trampoline
    jax_jit_call_trampoline.clear_cache = lambda: None  # Caches are foiled
    # Save some stuff for jax_export_trampoline
    jax_jit_call_trampoline.fun = fun
    jax_jit_call_trampoline.jit_kwargs = jit_kwargs
    jax_jit_call_trampoline.is_pjit = is_pjit
    return jax_jit_call_trampoline

  jax_jit_trampoline.real_boundary_fun = real_jit
  return jax_jit_trampoline

boundary_trampolines["jax.jit"] = partial(jit_trampoline, False)
boundary_trampolines["pjit.pjit"] = partial(jit_trampoline, True)

def jax_export_trampoline(real_export: Callable) -> Callable:
  def export_trampoline(fun_jit, **exp_kwargs):
    def exported_call(*args, **kwargs):
      from jax._src.repro import repro_api
      assert (hasattr(fun_jit, "fun") and hasattr(fun_jit, "jit_kwargs") and
              hasattr(fun_jit, "is_pjit"))
      return repro_api.jax_export_call(
          fun_jit.fun, fun_jit.jit_kwargs, exp_kwargs, *args, **kwargs)

    return exported_call

  export_trampoline.real_boundary_fun = real_export
  return export_trampoline

boundary_trampolines["jax.export.export"] = jax_export_trampoline

def custom_jvp_trampoline(real_boundary_fun: Callable):
  from jax._src import api_util  # type: ignore
  from jax._src.repro.repro_api import jax_custom_jvp_call

  def jax_custom_jvp_call_trampoline(*args, **kwargs):
    cjvp_orig, *rest_args = args
    resolved_args = api_util.resolve_kwargs(cjvp_orig.fun, rest_args, kwargs)
    if cjvp_orig.jvps is None:
      jvps_count = 1
      uses_defjvps = False
      new_args = (cjvp_orig.jvp, *resolved_args)
    else:
      jvps_count = len(cjvp_orig.jvps)
      uses_defjvps = True
      new_args = (*cjvp_orig.jvps, *resolved_args)

    return jax_custom_jvp_call(cjvp_orig.fun,
                               dict(nondiff_argnums=cjvp_orig.nondiff_argnums),
                               dict(symbolic_zeros=cjvp_orig.symbolic_zeros),
                               *new_args, uses_defjvps=uses_defjvps, jvps_count=jvps_count)

  jax_custom_jvp_call_trampoline.real_boundary_fun = real_boundary_fun
  return jax_custom_jvp_call_trampoline

boundary_trampolines["jax.custom_jvp.__call__"] = custom_jvp_trampoline

def custom_vjp_trampoline(real_boundary_fun: Callable):
  def jax_custom_vjp_call_trampoline(*args, **kwargs):
    from jax._src import api_util  # type: ignore
    from jax._src.repro.repro_api import jax_custom_vjp_call
    cvjp_orig, *rest_args = args
    resolved_args = api_util.resolve_kwargs(cvjp_orig.fun, rest_args, kwargs)
    return jax_custom_vjp_call(cvjp_orig.fun, cvjp_orig.fwd, cvjp_orig.bwd,
                               dict(nondiff_argnums=cvjp_orig.nondiff_argnums),
                               *resolved_args)
  jax_custom_vjp_call_trampoline.real_boundary_fun = real_boundary_fun
  return jax_custom_vjp_call_trampoline

boundary_trampolines["jax.custom_vjp.__call__"] = custom_vjp_trampoline

def named_call_trampoline(real_boundary_fun: Callable):
  # TODO: handle named_call. The problem that a named_call can wrap a jit
  # with statics and the statics are then not handle properly
  return (lambda fun, *args, **kwargs: fun)

boundary_trampolines["jax.named_call"] = named_call_trampoline

def pallas_call_trampoline(real_boundary_fun: Callable):
  from jax._src.repro.repro_api import jax_pallas_call

  def jax_pallas_call_trampoline(kernel, out_shape, **pl_call_kwargs):
    def jax_pallas_call_call_trampoline(*args):
      return jax_pallas_call(kernel, out_shape, pl_call_kwargs,*args)
    return jax_pallas_call_call_trampoline
  jax_pallas_call_trampoline.real_boundary_fun = real_boundary_fun
  return jax_pallas_call_trampoline

boundary_trampolines["jax.experimental.pallas.pallas_call"] = pallas_call_trampoline


def fuser_fuse_trampoline(real_boundary_fun: Callable) -> Callable:
  from jax._src.repro.repro_api import jax_fuser_fuse

  def jax_fuser_fuse_trampoline(fun: Callable | None = None, **trans_kwargs):
    def actual_decorator(fun: Callable):
      return partial(jax_fuser_fuse, fun, trans_kwargs)
    if fun is None:
      return actual_decorator
    return actual_decorator(fun)

  jax_fuser_fuse_trampoline.real_boundary_fun = real_boundary_fun
  return jax_fuser_fuse_trampoline

boundary_trampolines["fuser.fuse"] = fuser_fuse_trampoline


def fuser_fusible_trampoline(real_boundary_fun: Callable) -> Callable:
  from jax._src.repro.repro_api import jax_fuser_fusible

  def jax_fuser_fusible_trampoline(fun: Callable | None = None, **trans_kwargs):
    def actual_decorator(fun: Callable):
      return partial(jax_fuser_fusible, fun, fun, trans_kwargs)
    if fun is None:
      return actual_decorator
    return actual_decorator(fun)

  jax_fuser_fusible_trampoline.real_boundary_fun = real_boundary_fun
  return jax_fuser_fusible_trampoline

boundary_trampolines["fuser.fusible"] = fuser_fusible_trampoline


def pallas_custom_fusion_trampoline(real_boundary_fun: Callable):
  from jax._src import api_util  # type: ignore
  from jax._src.repro.repro_api import pallas_custom_fusion_call

  def custom_fusion_trampoline(*args, **kwargs):
    cfus, *rest_args = args
    resolved_args = api_util.resolve_kwargs(cfus.fun, rest_args, kwargs)
    return pallas_custom_fusion_call(
        cfus.fun, cfus.eval_rule, cfus.pull_block_spec_rule,
        cfus.push_block_spec_rule, cfus.pallas_impl, *resolved_args)

  custom_fusion_trampoline.real_boundary_fun = real_boundary_fun
  return custom_fusion_trampoline

boundary_trampolines["jax.pallas.custom_fusion.__call__"] = pallas_custom_fusion_trampoline

def jax_linearize_trampoline(real_boundary_fun: Callable):
  from jax._src.repro.repro_api import jax_linearize

  def linearize_trampoline(*args, **kwargs):
    return jax_linearize(*args, **kwargs)

  linearize_trampoline.real_boundary_fun = real_boundary_fun
  return linearize_trampoline

boundary_trampolines["jax.linearize"] = jax_linearize_trampoline

def jax_vjp_trampoline(real_boundary_fun: Callable):
  from jax._src.repro.repro_api import jax_vjp

  def vjp_trampoline(*args, **kwargs):
    return jax_vjp(*args, **kwargs)

  vjp_trampoline.real_boundary_fun = real_boundary_fun
  return vjp_trampoline

boundary_trampolines["jax.vjp"] = jax_vjp_trampoline


def jax_custom_gradient_trampoline(real_boundary_fun: Callable):
  from jax._src.repro.repro_api import jax_custom_gradient_call

  def custom_gradient_trampoline(f: Callable):
    def custom_gradient_call_trampoline(*args, **kwargs):
      return jax_custom_gradient_call(f, *args, **kwargs)
    return custom_gradient_call_trampoline

  custom_gradient_trampoline.real_boundary_fun = real_boundary_fun
  return custom_gradient_trampoline

boundary_trampolines["jax.custom_gradient"] = jax_custom_gradient_trampoline


def jax_saved_input_vjp_trampoline(real_boundary_fun: Callable):
  from jax._src.repro.repro_api import jax_saved_input_vjp

  def saved_input_vjp_trampoline(*args, **kwargs):
    return jax_saved_input_vjp(*args, **kwargs)

  saved_input_vjp_trampoline.real_boundary_fun = real_boundary_fun
  return saved_input_vjp_trampoline

boundary_trampolines["jax.experimental.saved_input_vjp"] = jax_saved_input_vjp_trampoline


# TODO(necula): we should not need to mention Flax here
boundary_trampolines["flax.core.axes_scan.scan"] = partial(generic_trampoline, "flax_axes_scan")


# boundary_trampolines["jax_repro_collect"] = jax_repro_collect_trampoline
####

def check_traceback(frames):
  """Try to detect when JAX calls into USER code that is not intercepted.
  This happens during tracing and thus it would be the first indication that
  something is going to go wrong during repro generation.
  """
  for i, f in enumerate(frames):
    # This is the last JAX frame from the WrappedFun.call_wrapped sequence
    if f.function_name in ["_get_result_paths_thunk", "trace_to_jaxpr"]:
      if check_frame_is_unwrapped_function(frames[i - 1]):
        msg = (
            "Expected that the call from `_get_result_paths_thunk` (frame "
            f"index {i}) is to a USER function, and found instead call to "
            " an unwrapped function `{frames[i - 1]}`. This typically means a "
            "forgotten `core.repro_boundary` decorator.")
        _thread_local_state.warn_or_error(msg)


def check_frame_is_unwrapped_function(frame) -> bool:
  return (frame.function_name != "Func.__call__" and
          "/jax/_src/" not in frame.file_name and
          # TODO: this is because flax.core.axes_scan uses internals
          "/flax/core/" not in frame.file_name)

"""
TODO:
* print for function definitions the static arguments, as comments
* maybe print also the avals
* actual devices use jax_get_device, we need a way to make it platform
  agnostic
* instead of just emitting None for a return from a function with an error,
  print some indication of the error in comments
* document the pytree flattening that we do
* why do we need to resolve_kwargs for the custom_jvp and friends? Because
  there are cases when we call the jvp function with kwargs but we never
  call the primal function, so the latter is generated with `*args, **kwargs`
  and then the repro will not be able to resolve the kwargs.
* reuse identical user functions when emitting
* revisit the behavior and tests for errors during execution, in implicit
  mode, in collect mode, and also errors during emitting source.
* in error_mode="defer" if we get an error during collection (not repro generation)
  should we raise ReproError? If we do, we may interrupt lots of code, but
  maybe that is the point? If we don't we miss out on the signal.

* If one forgets to annotate a higher-order repro_boundary then we won't wrap
  USER functions and we call into user code without tracking it. We would likely
  still be in USER mode, and then we'd give an error seeing a higher-order
  JAX primitive. We also have an error now when we create a Call to check the
  stack trace.
* emit code to set the configuration flags the same way as they were during
  the repro.
* test the emit_function cache
* try to figure out why we don't get an error in the fusible_matmul, if we
  comment out the repro_boundary on push_block_spec. Actually it seems that
  one can debug this by looking at the stack trace when an undefined tracer
  is encountered, by seeing that you call tracing (trace_to_jaxpr_dynamic)
  while in USER mode (after a call_jax_to_user)
* scatter carries a Jaxpr
* scatter_p exits in SC also, with the same name, e.g., addupdate_scatter,
  store_scatter
* it is tricky to know which functions to wrap, cannot just look for callable
  because users can pass, e.g., Flax modules, which are callables, and are also
  pytree. The solution is to declare the callables, by way of
  repro_map_func_args, repro_map_fun_res.
* had to wrap the callable result of USER functions, as USER function. This
  was needed to handle jax.custom_gradient, because it return a user function.
"""
