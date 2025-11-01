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

We try to interfere as little as possible with the execution, to maximize
the chance that we can produce a faithful repro. However, we must foil the
JAX tracing caches. For this we use the `boundary_trampolines` (see more
in `repro_api.py`.).
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
from typing import Any, Callable, Union
from functools import partial
import traceback

from jax._src import config
from jax._src import tree_util


_fun_name_re = re.compile(r"(?:<built-in function (\S+)>)")

Primitive = Any  # really, core.Primitive but don't want circular imports

class ReproError(Exception):
  pass


UNKNOWN_FUN_NAME = "unknown"
# TODO(necula): reuse this function from api_util.
def _fun_sourceinfo(fun: Callable) -> str:
  # See DebugInfo.fun_src_info
  while isinstance(fun, partial):
    fun = fun.func
  fun = inspect.unwrap(fun)
  def sanitize_name(n: str) -> str:
    # TODO: Regexp
    if ("built-in" in n or "jnp.ufunc" in n or "<lambda>" in n or "<unamed" in n):
      return UNKNOWN_FUN_NAME
    else:
      return n
  try:
    filename = fun.__code__.co_filename
    lineno = fun.__code__.co_firstlineno
    return f"{sanitize_name(fun.__name__)} at {filename}:{lineno}"
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


def _current_traceback():
  from jax._src.lib import xla_client  # type: ignore
  # Use directly xla_client, otherwise we may get cached tracebacks
  return xla_client.Traceback.get_traceback().get_traceback()


def _higher_order_primitive(prim, args, params) -> bool:
  from jax._src import core  # type: ignore
  for arg in tuple(args) + tuple(params.values()):
    # Unpack tuples, and not more
    for a in (arg if isinstance(arg, tuple) else (arg,)):
      if isinstance(a, (core.Jaxpr, core.ClosedJaxpr, core.lu.WrappedFun)):
        return True
  return False


@dataclasses.dataclass
class ReproFlags:
  enable: bool = True

  log_calls: bool = False
  log_call_details: frozenset[int] = frozenset()

  save_repro_on_uncaught_exception = True

  enable_checks = True
  enable_checks_with_tracebacks = True
  log_traceback_frames = 40
  enable_checks_as_errors = False

  # Replace arrays with `np.ones` if larger than this threshold
  fake_array_threshold = 128
  collect_repro_on_success = False

  def __init__(self):
    if config.repro_flags.value:
      for f in config.repro_flags.value.split(","):
        f, *fvalue = f.strip().split("=")
        def fvalue_to_int(default: int):
          return int(fvalue[0]) if fvalue else default
        def fvalue_to_bool():
          return bool(fvalue_to_int(1))

        if f == "enable":
          self.enable = fvalue_to_bool()
        elif f == "enable_checks":
          self.enable_checks = fvalue_to_bool()
        elif f == "enable_checks_as_errors":
          self.enable_checks = fvalue_to_bool()
          self.enable_checks_as_errors = fvalue_to_bool()
        elif f == "enable_checks_with_tracebacks":
          self.enable_checks = fvalue_to_bool()
          self.enable_checks_with_tracebacks = fvalue_to_bool()
        elif f == "log_calls":
          self.log_calls = fvalue_to_bool()
        elif f == "log_call_details":
          call_ids = {int(anid.strip()) for anid in fvalue[0].split("+")}
          self.log_call_details = self.log_call_details.union(call_ids)
        elif f == "log_traceback_frames":
          self.log_traceback_frames = fvalue_to_int(30)
        elif f == "fake_array_threshold":
          self.fake_array_threshold = fvalue_to_int(128)
        elif f == "save_repro_on_uncaught_exception":
          self.save_repro_on_uncaught_exception = fvalue_to_bool()
        else:
          raise NotImplementedError(f"--jax_repro_flags: {f}")


@contextlib.contextmanager
def flags_override(**flag_values):
  """Overrides ReproFlags.

  Usage: flags_override(enable_checks=True, log_traceback_frames=40)
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


class _ThreadLocalState(threading.local):
  def __init__(self):
    # Current call stack, [0] is the main function.
    self.call_stack: list["Call"] = []

    # To simplify debugging, we assign deterministic ids to Func and Call
    # objects.
    self.func_index = itertools.count()  # Used only for non api_name funcs
    self.call_index = itertools.count()

    self.flags = ReproFlags()

    # The path and source for the last repro saved
    self.last_saved_repro: tuple[pathlib.Path, str] | None = None
    # It is useful to report repro generation errors after the repro is emitted
    self.had_deferred_errors = False

  def initialize_state_if_needed(self):
    """Deferred per-thread initialization, only once we start using repros.
    This ensures that Func is available.
    """
    if self.call_stack: return
    def main_repro(): return
    main_func = Func(main_repro, is_user=True)
    main_func.is_main = True
    self.push_call(main_func, (), {})

    # Hook repros into core.Primitive._true_bind
    from jax._src import core
    old_true_bind = core.Primitive._true_bind
    if "Primitive._true_bind" in str(old_true_bind):
      core.Primitive._true_bind = true_bind_wrapper(old_true_bind)
    stop = 1


    self.uncaught_exception_handler = uncaught_exception_handler

  def push_call(self, func: "Func", args, kwargs) -> "Call":
    call = Call(self.call_stack[-1] if self.call_stack else None,
                func, args, kwargs)
    self.call_stack.append(call)
    return call

  def warn_or_error(self, msg,
                    traceback=None,
                    warning_only=False,
                    during_repro_emit=False):
    logging.error("Repro error: %s", msg)
    if self.flags.enable_checks_as_errors and not warning_only:
      if during_repro_emit:
        _thread_local_state.had_deferred_errors = True
      else:
        raise ReproError(msg)  # This will include the current trace
    elif not self.flags.enable_checks_with_tracebacks:
      return

    if traceback:
      tb, where = traceback, "for call site "
    else:
      tb, where = _current_traceback(), ""
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
    self.fun_info = _fun_sourceinfo(fun)
    self.fun_name = self.fun_info.split(" ")[0]
    self.is_user = is_user
    self.is_main = False  # Whether this is the top-level function
    assert not (api_name and is_user), (fun, api_name)
    self.api_name = api_name
    if api_name is None:
      self.id = next(_thread_local_state.func_index)
    else:
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
    self.invocation: Union["Call", None] = None

    if _thread_local_state.flags.log_calls:
      logging.info(f"Created {self} for {self.fun_info}")

  def __call__(self, *args, **kwargs):
    return call_boundary(self, args, kwargs)

  def __get__(self, instance, owner_class):
    # This is needed when we setattr(obj, "method", jit(f))
    # TODO: if I define this the debugger does not show these values
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
    if self.is_main:
      return "main_repro"
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
_api_name_to_map_user_func_args: dict[str, Callable] = {
    "jax_cond": (
      lambda toapply, p, tf, ff, *args, **kwargs: (p, toapply(tf), toapply(ff), *args)),
    "jax_cond_with_per_branch_args": (
      lambda toapply, p, t_op, tf, f_op, ff, *args, **kwargs: (p, t_op, toapply(tf), f_op, toapply(ff), *args)),
    "jax.lax.fori_loop": (
      lambda toapply, l, u, body, *args, **kwargs: (l, u, toapply(body), *args)),
    "jax.lax.while_loop": (
      lambda toapply, cond, body, *args, **kwargs: (toapply(cond), toapply(body), *args)),
}


def is_primitive(v: Union["Primitive", Func]) -> bool:
  from jax._src import core  # type: ignore
  return isinstance(v, core.Primitive)


class Call:

  def __init__(self, parent: Union["Call", None],
               func: Union[Func, "Primitive"],
               args: tuple[Any, ...], kwargs: dict[str, Any]):
    self.id = next(_thread_local_state.call_index)
    self.level: int = 0 if parent is None else parent.level + 1
    self.parent: Union["Call", None] = parent
    self.func: Union[Func, "Primitive"] = func
    self.args = args
    self.kwargs = kwargs
    # Collect the traceback before we are making the call
    self.raw_traceback = _current_traceback()
    self.body : list["Call"] = []
    self.result: Any | None = None  # None is used also for functions that result in error
    assert ((isinstance(func, Func) and func.is_main) == (parent is None)), (func, parent)

    not _thread_local_state.flags.enable_checks or check_traceback(self.raw_traceback.frames)  # type: ignore

  def __repr__(self):
    return f"[{self.level}.{self.id}] {self.func}"

  def is_prim(self) -> bool:
    return not isinstance(self.func, Func)

  def log_arg_details(self):
    logging.info(f"{'  ' * self.level} detailed args {self}")
    def log_value(v):
      if isinstance(v, Func):
        return f"{v} for {v.fun_info}"
      else:
        return f"{v}: {type(v)}"
    for p, a in tree_util.tree_flatten_with_path(self.args)[0]:
      logging.info(f"{'  ' * self.level} args {tree_util.keystr(p)}: {log_value(a)}")
    for p, a in tree_util.tree_flatten_with_path(self.kwargs)[0]:
      logging.info(f"{'  ' * self.level} kwargs {tree_util.keystr(p)}: {log_value(a)}")

  def log_res_details(self):
    logging.info(f"{'  ' * self.level} detailed result {self}")
    def log_value(v):
      if isinstance(v, Func):
        return f"{v} for {v.fun_info}"
      else:
        return f"{v}: {type(v)}"
    for p, a in tree_util.tree_flatten_with_path(self.result)[0]:
      logging.info(f"{'  ' * self.level} result {tree_util.keystr(p)}: {log_value(a)}")

  @property
  def traceback(self):
    """A traceback inside the function being called, if available, otherwise
    the traceback where the Call was created (where the function is being called
    from)."""
    if self.body:
      return self.body[0].raw_traceback
    else:
      return self.raw_traceback

  def add_call_to_body(self, c: "Call"):
    assert self.func.is_user
    self.body.append(c)

  @staticmethod
  def start_call(args, kwargs, *, func: "Func") -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Start a call, and push a Call onto the call_stack."""
    wrapped_args = args

    if func.api_name:
      def do_wrap(f):
        if callable(f):
          return wrap_callable(f, is_user=True)
        else:
          return f
      wrapped_args = func.map_user_func_args(do_wrap, *args, **kwargs)

    wrapped_args_copy, kwargs_copy = Call.copy_args_results((wrapped_args, kwargs))

    call = _thread_local_state.push_call(func, wrapped_args_copy, kwargs_copy)
    if func.is_user:
      if func.invocation:
        # We know of cases when this happens, e.g., higher-order custom_vjp,
        # fusions
        _thread_local_state.warn_or_error(
            f"Ignoring additional invocation {call}. "
            f"Previous invocation was {func.invocation}",
            warning_only=True, traceback=call.raw_traceback)
      else:
        func.invocation = call

    if _thread_local_state.flags.log_calls:
      # Log the Func passed as args
      args_to_print = [a if isinstance(a, Func) else "" for a in wrapped_args]
      args_str = ", ".join(str(a) for a in args_to_print)
      args_str = re.sub(r'(, )+', ", ...", args_str) or "..."
      logging.info(f"{'  ' * call.level} start {call}({args_str})")
    if call.id in _thread_local_state.flags.log_call_details:
      call.log_arg_details()
    return wrapped_args, kwargs

  @staticmethod
  def end_call(*, res, exc: Exception | None):
    call_stack = _thread_local_state.call_stack
    try:
      assert call_stack
      call = call_stack[-1]
      res_str = ""  # For debugging
      if exc is None:
        assert call.result is None, call
        call.result = Call.copy_args_results(res)
        if _thread_local_state.flags.log_calls:
          res_str = "res=Ok"
      else:
        call.result = None
        exc_str = str(exc)
        if call.level > 1:
          exc_str = exc_str[:1024] + "\n...."
        if _thread_local_state.flags.log_calls:
          res_str = f"exc={exc_str}"

      if _thread_local_state.flags.log_calls:
        logging.info(f"{'  ' * call.level} end {call}: {res_str}")
      if call.id in _thread_local_state.flags.log_call_details:
        call.log_res_details()
      call_stack.pop()

      caller: Call = call.parent  # type: ignore
      assert caller is not None
      if caller.func.is_user:
        caller.add_call_to_body(call)

      if caller.func.is_main:
        if (exc is not None and not isinstance(exc, ReproError)):
          if _thread_local_state.flags.save_repro_on_uncaught_exception:
            uncaught_exception_handler(caller, traceback.format_exc())

        # Drop the last call from the main body, after emitting the repro,
        # so that we don't keep growing the main body.
        # TODO: instead of looking for absence of FuncPartial, look for
        # presence of only values for which we have value emitters
        returned_leaves = tree_util.tree_leaves(call.result, is_leaf=lambda l: isinstance(l, FuncPartial))
        if (all(not isinstance(l, (FuncPartial, Func)) for l in returned_leaves) and
            not _thread_local_state.flags.collect_repro_on_success):
          caller.body.pop()

      return res
    except Exception as e:
      # Exceptions here are bad, because they break the call_stack invariants
      logging.error(f"Exception caught in the exit handler: {type(e)}: {e}\n{traceback.format_exc()}")
      raise e

  @staticmethod
  def copy_args_results(v):
    # Make copies of the arguments in case they contain mutable parts; we will
    # need the originals when we emit repros. Don't instantiate new FuncPartial
    l, t = tree_util.tree_flatten(v,
                                  is_leaf=lambda v: isinstance(v, FuncPartial))
    return t.unflatten(l)

def true_bind_wrapper(actual_true_bind: Callable) -> Callable:
  def true_bind(*prim_and_args, **params):
    # We replace the core.Primitive._true_bind with this wrapper
    if not is_enabled():
      return actual_true_bind(*prim_and_args, **params)
    if not _thread_local_state.call_stack:  # The first call on this thread
      _thread_local_state.initialize_state_if_needed()
    stack_entry = _thread_local_state.call_stack[-1]
    if not stack_entry.func.is_user:
      return actual_true_bind(*prim_and_args, **params)
    # We should not be seeing higher-order primitives in USER functions
    prim, *args = prim_and_args
    if (_thread_local_state.flags.enable_checks and
        _higher_order_primitive(prim, args, params)):
      _thread_local_state.warn_or_error(
          f"Encountered primitive {prim} containing Jaxprs or functions. This means "
          "that there is a higher-order JAX API that is not "
          "annotated with traceback_util.api_boundary(repro_api_name=...)")

    c = Call(stack_entry, prim, tuple(args), params)
    if c.id in _thread_local_state.flags.log_call_details:
      c.log_arg_details()
      c.log_res_details()
    stack_entry.add_call_to_body(c)
    with enable(False):
      res = actual_true_bind(*prim_and_args, **params)

    c.result = res  # On success
    return res
  return true_bind


def uncaught_exception_handler(main_invocation: Call, traceback_str: str):
  from jax._src.repro import emitter
  c = emitter.collector(lambda: None)
  c.main_invocation = main_invocation
  # Save the repro even if there are errors during repro generation
  c.to_source(extra_comment="uncaught exception " + traceback_str,
              repro_name_prefix="jax_error_repro")

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
  else:
    func = Func(boundary_fun, is_user=is_user, api_name=api_name,
                map_user_func_args=map_user_func_args)
  # TODO: create somehow also the JAX functions, we don't want to keep recreating
  # jax_jit_call for each call
  # if not is_user and not api_name:
  #   return wrap_callable(boundary_fun, is_user=is_user)

  @functools.wraps(boundary_fun)
  def repro_boundary_wrapper(*args, **kwargs):
    # Create Func late; otherwise, we'd create some while JAX initializes and
    # it makes for less deterministic repros. There is no harm in having
    # several copies of, e.g., `lax.cond`.
    # func = Func(boundary_fun, is_user=is_user, api_name=api_name,
    #             map_user_func_args=map_user_func_args)
    return call_boundary(func, args, kwargs)

  repro_boundary_wrapper.real_boundary_fun = boundary_fun
  return repro_boundary_wrapper


def call_boundary(func: Func, args: tuple[Any, ...], kwargs: dict[str, Any]):
  boundary_fun = func.fun
  if not is_enabled():
    return boundary_fun(*args, **kwargs)

  _thread_local_state.initialize_state_if_needed()
  if func.api_name and not _thread_local_state.call_stack[-1].func.is_user:
    if _thread_local_state.flags.log_calls:
      logging.info(f"Ignoring call to {func.api_name} from within JAX source")
    return boundary_fun(*args, **kwargs)

  call_stack_level = len(_thread_local_state.call_stack)  # In case we get an exception during start_call
  try:
    wrapped_args, wrapped_kwargs = Call.start_call(args, kwargs, func=func)

    def call_user_to_jax():  # Distinctive name for traceback readability
      return boundary_fun(*wrapped_args, **wrapped_kwargs)  # USER -> JAX

    def call_jax_to_user():
      return boundary_fun(*wrapped_args, **wrapped_kwargs)  # JAX -> USER

    if func.is_user:
      result = call_jax_to_user()
    else:
      result = call_user_to_jax()
    result = Call.end_call(res=result, exc=None)
    return result
  except Exception as e:
    if len(_thread_local_state.call_stack) > call_stack_level:
      Call.end_call(res=None, exc=e)
    raise


def bypass_wrapper(f: Callable) -> Callable:
  """Bypasses the repro wrappers.

  WARNING: This is part of the highly experimental repro feature. Subject to changes
  and removal.

  Usage: `bypass_repro_wrapper(jax.jit)(f)` in order to use the real `jax.jit`,
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

  The `jax_trans_call` are defined in repro_api.py.
  """
  jax_trans_call_name = f"jax_{transform_name}_call"
  from jax._src.repro import repro_api

  def trampoline(fun: Callable, *trans_args: tuple[Any], **trans_kwargs: dict[str, Any]):
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
    if is_pjit:
      ctx_mesh = mesh_lib.thread_resources.env.physical_mesh
    else:
      ctx_mesh = mesh_lib.get_concrete_mesh()
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

    jax_jit_call_trampoline.lower = jax_jit_aot_lower_trampoline
    return jax_jit_call_trampoline

  jax_jit_trampoline.real_boundary_fun = real_jit
  return jax_jit_trampoline

boundary_trampolines["jax.jit"] = partial(jit_trampoline, False)
boundary_trampolines["pjit.pjit"] = partial(jit_trampoline, True)

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

def jax_saved_input_vjp_trampoline(real_boundary_fun: Callable):
  from jax._src.repro.repro_api import jax_saved_input_vjp

  def saved_input_vjp_trampoline(*args, **kwargs):
    return jax_saved_input_vjp(*args, **kwargs)

  saved_input_vjp_trampoline.real_boundary_fun = real_boundary_fun
  return saved_input_vjp_trampoline

boundary_trampolines["jax.experimental.saved_input_vjp"] = jax_saved_input_vjp_trampoline

####

def check_traceback(frames):
  """Try to detect when JAX calls into USER code that is not intercepted.
  This happens during tracing and thus it would be the first indication that
  something is going to go wrong during repro generation.
  """
  for i, f in enumerate(frames):
    # This is the last JAX frame from the WrappedFun.call_wrapped sequence
    if f.function_name == "_get_result_paths_thunk":
      if (frames[i - 1].function_name != "Func.__call__" and
          "/jax/_src/" not in frames[i - 1].file_name and
          # TODO: this is because flax.core.axes_scan uses internals
          "/flax/core/" not in frames[i - 1].file_name):
        msg = (
            "Expected that the call from `_get_result_paths_thunk` (frame "
            f"index {i}) is to a USER function, and found instead call to "
            " an unwrapped function `{frames[i - 1]}`. This typically means a "
            "forgotten `core.repro_boundary` decorator.")
        _thread_local_state.warn_or_error(msg)

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
"""
