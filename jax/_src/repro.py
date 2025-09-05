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

"""

import collections
import enum
from collections.abc import Sequence
import dataclasses
import functools
import inspect
import itertools
import logging
import os.path
import re
import threading
from typing import Any, Callable, NamedTuple, Union
from functools import partial
import traceback

import numpy as np

from jax._src import config
from jax._src import dtypes
from jax._src import path
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

def current_traceback():
  from jax._src.lib import xla_client  # type: ignore
  # Use directly xla_client, otherwise we may get cached tracebacks
  return xla_client.Traceback.get_traceback().get_traceback()

def maybe_singleton(x: list[Any]) -> Any | tuple[Any, ...]:
  return x[0] if len(x) == 1 else tuple(x)

def val_is_leaf(v):
  leaves = tree_util.tree_leaves(v)
  return (len(leaves) == 1 and leaves[0] is v)

_flat_index_re = re.compile(r"<flat index (\d+)>")
def keystr_no_flat_index(p: tree_util.KeyPath) -> str:
  return _flat_index_re.sub("\\1", tree_util.keystr(p))

def higher_order_primitive(prim, args, params) -> bool:
  from jax._src import core
  for arg in tuple(args) + tuple(params.values()):
    # Unpack tuples, and not more
    for a in (arg if isinstance(arg, tuple) else (arg,)):
      if isinstance(a, (core.Jaxpr, core.ClosedJaxpr, core.lu.WrappedFun)):
        return True
  return False


@dataclasses.dataclass
class ReproFlags:
  # Run some invariant checks while tracing to detect early errors, also enable
  # logging of repro operations.
  enable_checks = True
  enable_checks_with_tracebacks = True
  log_traceback_frames = 30
  enable_checks_as_errors = False

  # Replace arrays with `np.ones` if larger than this threshold
  fake_array_threshold = 128
  # # Squash sequences of array primitives, preserving the dependencies
  # squash_primitives = False

  def __init__(self):
    if config.repro_flags.value:
      for f in config.repro_flags.value.split(","):
        f, *fvalue = f.split("=")
        def fvalue_to_int(default: int):
          return int(fvalue[0]) if fvalue else default
        def fvalue_to_bool():
          return bool(fvalue_to_int(1))

        if f == "enable_checks":
          self.enable_checks = fvalue_to_bool()
        elif f == "enable_checks_as_errors":
          self.enable_checks = fvalue_to_bool()
          self.enable_checks_as_errors = fvalue_to_bool()
        elif f == "enable_checks_with_tracebacks":
          self.enable_checks = fvalue_to_bool()
          self.enable_checks_with_tracebacks = fvalue_to_bool()
        elif f == "log_traceback_frames":
          self.log_traceback_frames = fvalue_to_int(30)
        elif f == "fake_array_threshold":
          self.fake_array_threshold = fvalue_to_int(128)
        else:
          raise NotImplementedError(f"--jax_repro_flags: {f}")


class _ThreadLocalState(threading.local):
  def __init__(self):
    self.stack_node_id = collections.defaultdict(itertools.count)
    self.call_stack: list["Call"] = []

    self.flags = ReproFlags()
    # The collection of repros is enabled if config.repro_dir is set and
    # the collect_repro_enables is True
    self.collect_repro_enabled = True
    self.collect_repro_on_success = False

    # List of path names and source codes
    self.emitted_repros: list[tuple[str, str]] = []

    # For simpler debugging, use our own small value ids when printing
    self.small_id_index = itertools.count()
    self.small_id_map: dict[int, int] = {}  # id(v) -> small id
    self.func_index = itertools.count()  # Each constructed Func has an index
    self.call_index = itertools.count()  # Each constructed Call has an index

  def initialize_call_stack(self):
    """Deferred per-thread initialization, only once we start using.
    This ensures that Func is available
    """
    if self.call_stack: return
    def main(): return
    main_func = Func(main, is_jax=False)
    main_func.is_main = True
    self.push_call(main_func, (), {})

  def snapshot_state(self):
    self.initialize_call_stack()
    repro_call_stack = self.call_stack[-1]
    assert repro_call_stack.func.is_main, repro_call_stack
    return {
        "main_body_length": len(repro_call_stack.body),
        "func_index_counter": next(self.func_index),
        "call_index_counter": next(self.call_index),
        "small_id_index_counter": next(self.small_id_index),
        "small_id_keys": set(self.small_id_map.keys()),
    }

  def restore_state(self, snapshot):
    repro_call_stack = self.call_stack[-1]
    assert repro_call_stack.func.is_main, repro_call_stack
    repro_call_stack.body = repro_call_stack.body[:snapshot["main_body_length"]]
    self.func_index = itertools.count(snapshot["func_index_counter"])
    self.call_index = itertools.count(snapshot["call_index_counter"])
    self.small_id_index = itertools.count(snapshot["small_id_index_counter"])
    self.small_id_map = {k:v for k, v in self.small_id_map.items()
                         if k in snapshot["small_id_keys"]}

  def small_id(self, v_id: int) -> int:
    res = self.small_id_map.get(v_id)
    if res is not None: return res
    res = next(self.small_id_index)
    self.small_id_map[v_id] = res
    return res

  def push_call(self, func: "Func", args, kwargs) -> "Call":
    if not self.call_stack and not func.is_main:  # The first call on this thread
      self.initialize_call_stack()

    call = Call(len(self.call_stack),
                self.call_stack[-1] if self.call_stack else None,
                func, args, kwargs)
    self.call_stack.append(call)
    return call

  def warn_or_error(self, msg,
                    traceback=None):
    if self.flags.enable_checks_as_errors:
      logging.error("Repro error: %s", msg)
      raise ValueError(msg)
    elif self.flags.enable_checks_with_tracebacks:
      if traceback:
        tb, where = traceback, "for call site "
      else:
        tb, where = current_traceback(), ""
      logging.error("Repro error: %s",msg)
      logging.error(f"Traceback {where}(top {self.flags.log_traceback_frames} frames):")
      for l in str(tb).splitlines()[:self.flags.log_traceback_frames]:
        logging.error("     %s", l)
      logging.error("    ....")
    else:
      logging.warning("Repro error: %s",msg)

_thread_local_state = _ThreadLocalState()


class Func:
  MISSING_API_NAME = "MISSING_API_NAME"

  def __init__(self, fun: Callable, *, is_jax: bool,
               api_name: str | None = None,
               func_argnums: tuple[int, ...] | Callable[..., tuple[int, ...]] = (0,),
               introduced_for: Union["Call", None] = None):
    self.fun = fun
    self.fun_info = _fun_sourceinfo(fun)
    self.fun_name = self.fun_info.split(" ")[0]
    self.fun_name = self.fun_name
    self.is_jax = is_jax
    self.id = next(_thread_local_state.func_index)
    self.is_main = False  # Whether this is the top-level function
    assert not (api_name and not is_jax), (fun, api_name)
    self.api_name = api_name
    self.func_argnums = func_argnums
    self.__wrapped__ = fun  # Enable fun_sourceinfo(self)
    if hasattr(fun, "__name__"):
      setattr(self, "__name__", fun.__name__)
    if hasattr(fun, "__qualname__"):
      setattr(self, "__qualname__", fun.__name__)

    # For calls to USER functions.
    self.invocation: Union["Call", None] = None
    # The USER functions are introduced as arguments for API calls, and JAX
    # functions are results of API calls. This value is None only for API functions.
    self.introduced_for: Call | None = introduced_for

    fun_info_with_id = self.fun_info.replace(self.fun_info.split(" ")[0],
                                             f"{self.fun_name}<{Call.val_id(fun)}>")
    if _thread_local_state.flags.enable_checks:
      logging.info(f"Created {self} for {fun_info_with_id}")

  def __call__(self, *args, **kwargs):
    return jax_boundary(self.fun, func=self)(*args, **kwargs)

  def __get__(self, instance, owner_class):
    # This is needed when we setattr(obj, "method", jit(f))
    # TODO: if I define this the debugger does not show these values
    if instance is None:
      # If accessed as Func.method, return self (the descriptor)
      return self
    # When accessed as func.method, return a callable that
    # binds 'instance' and calls the __call__ method of Func.
    return (lambda *args, **kwargs: self.__call__(instance, *args, **kwargs))

  def python_name(self):
    if self.api_name:
      if self.api_name == Func.MISSING_API_NAME:
        assert False, (self.fun, self.fun_info)
        logging.error(f"missing api_name: {self.fun}, {self.fun_info}")
      return self.api_name

    if self.fun_name != UNKNOWN_FUN_NAME:
      return f"fun_{self.fun_name}_{self.id}"
    else:
      return f"fun_{self.id}"

  def __repr__(self):  # Printing in debug logs
    return f"{'JAX' if self.is_jax else 'USER'}[{self.python_name()}]"

  @staticmethod
  def wrap_func_in_out(f, *, is_jax: bool,
                       introduced_for: Union["Call", None] = None,
                       accum: list["Func"] | None = None,
                       ):
    assert callable(f), f
    assert is_jax == (introduced_for is not None)
    if isinstance(f, Func) and f.is_jax == is_jax: return f
    if isinstance(f, tree_util.Partial):
      assert is_jax  # In returned values
      res = FuncPartial(f, is_jax=is_jax, introduced_for=introduced_for)
      if accum is not None: accum.append(res)  # type: ignore
      return res

    res = Func(f, is_jax=is_jax, introduced_for=introduced_for)  # type: ignore
    if accum is not None: accum.append(res)  # type: ignore
    return res


@tree_util.register_pytree_node_class
class FuncPartial(Func):
  """A Func wrapper for a Partial, returned by some JAX APIs"""
  def __init__(self, func: tree_util.Partial,
               is_jax: bool, introduced_for: Union["Call", None]):
    super().__init__(func, is_jax=is_jax, introduced_for=introduced_for)
    self.args = func.args  # Some tests are looking for Partial attributes
    self.func = func.func
    self.keywords = func.keywords

  def tree_flatten(self):
    return ((self.fun,), self)

  @classmethod
  def tree_unflatten(cls, aux_data: "FuncPartial", children):
    return cls(children[0], is_jax=aux_data.is_jax,
               introduced_for=aux_data.introduced_for)


class Call:

  def __init__(self, level: int, parent,
               func: Func,
               args: tuple[Any, ...], kwargs: dict[str, Any]):
    self.id = next(_thread_local_state.call_index)
    self.level: int = level  # 0 is the top of the stack
    self.parent = parent
    self.func: Func = func
    self.args = args
    self.kwargs = kwargs
    # Collect the traceback before we are making the call
    self.raw_traceback = current_traceback()
    self.body : list["Call"] = []
    self.result: Any | None = None

  @property
  def traceback(self):
    """A traceback inside the function being called, if available, otherwise
    the traceback where the Call was created (where the function is being called
    from)."""
    if self.body:
      return self.body[0].raw_traceback
    else:
      return self.raw_traceback

  @staticmethod
  def val_id(v):
    return _thread_local_state.small_id(id(v))

  def __repr__(self):
    return f"[{self.level}.{self.id}] {self.func}"

  @staticmethod
  def start_call(args, kwargs, *, func: "Func"):
    introduced_funcs: list[Func] = []
    wrapped_args = args
    wrapped_kwargs = kwargs

    if func.api_name:
      func_argnums = (
          func.func_argnums(*args, **kwargs) if callable(func.func_argnums) else
          func.func_argnums)
      wrapped_args = list(args)
      for i in func_argnums:
        wrapped_args[i] = Func.wrap_func_in_out(wrapped_args[i], is_jax=False,
                                                accum=introduced_funcs)
    # Make copies of the arguments in case they contain mutable parts; we will
    # use them when we emit repros. Don't instantiate new FuncPartial
    l, t = tree_util.tree_flatten((wrapped_args, wrapped_kwargs),
                                  is_leaf=lambda v: isinstance(v, FuncPartial))
    wrapped_args_copy, wrapped_kwargs_copy = t.unflatten(l)
    del l, t

    call = _thread_local_state.push_call(func, wrapped_args_copy, wrapped_kwargs_copy)
    for f in introduced_funcs:
      f.introduced_for = call
    if not func.is_jax:
      if func.invocation:
        _thread_local_state.warn_or_error(
            f"Ignoring additional invocation {call}. "
            f"Previous invocation was {func.invocation}")
      else:
        func.invocation = call

    if _thread_local_state.flags.enable_checks:
      # Log the Func passed as args
      args_to_print = [a if isinstance(a, Func) else "" for a in wrapped_args]
      args_str = ", ".join(str(a) for a in args_to_print)
      args_str = re.sub(r'(, )+', ", ...", args_str) or "..."
      logging.info(f"{'  ' * call.level} start {call}({args_str})")
    return wrapped_args, wrapped_kwargs, call

  def end_call(self, *, res, exc: Exception | None):
    call_stack = _thread_local_state.call_stack
    try:
      assert call_stack
      assert self is call_stack[-1], (self, call_stack[-1])
      res_str = ""  # For debugging
      if exc is None:
        assert self.result is None, self

        self.result = res
        if self.func.api_name:
          # Look through the result of JAX functions and wrap callables as
          # Func. If we are returning functions, they are either the sole
          # result or in a tuple
          results: list[Any] = list(res) if isinstance(res, tuple) else [res]
          for i, r in enumerate(results):
            if callable(r):
              results[i] = Func.wrap_func_in_out(r, is_jax=True,
                                                 introduced_for=self)
          # Make copies in case the user function returns mutable values
          self.result = tuple(results) if isinstance(res, tuple) else results[0]

        if _thread_local_state.flags.enable_checks:
          res_str = "res=Ok"
      else:
        self.result = None
        exc_str = str(exc)
        if self.level > 1:
          exc_str = exc_str[:1024] + "\n...."
        if _thread_local_state.flags.enable_checks:
          res_str = f"exc={exc_str}"

      if _thread_local_state.flags.enable_checks:
        logging.info(f"{'  ' * self.level} end {self}: {res_str}")
      call_stack.pop()

      caller = self.parent
      if not caller.func.is_jax:
        # We were called from a user function
        caller.body.append(self)

      if caller.func.is_main:
        if (exc is not None and not isinstance(exc, ReproError)):
          # We caught an exception and we are at top-level
          _emit_repro(caller, "uncaught exception " + traceback.format_exc())
        # Drop the last call from the main body, after emitting the repro,
        # so that we don't keep growing the main body.
        returned_leaves = tree_util.tree_leaves(self.result, is_leaf=lambda l: isinstance(l, FuncPartial))
        if (all(not isinstance(l, FuncPartial) for l in returned_leaves) and
            not _thread_local_state.collect_repro_on_success):
          if _thread_local_state.flags.enable_checks:
            logging.info(f"Popping the last call from main: {self}")
          caller.body.pop()

      return self.result
    except Exception as e:
      # Exceptions here are bad, because they break the call_stack invariants
      logging.error(f"Exception caught in the exit handler: {type(e)}: {e}\n{traceback.format_exc()}")
      raise e

  def bind_primitive(self, prim: Primitive, args, params, outs):
    c = Call(self.level + 1, self, prim, args, params)
    c.result = outs
    # TODO: We don't collect primitive_bind inside main
    self.body.append(c)

def _true_bind_primitive(prim: Primitive, args, params):
  # Replacement for Primitive._true_bind when using repros
  if (not config.repro_dir.value or
      not _thread_local_state.collect_repro_enabled):
    return prim._true_bind_internal(*args, **params)
  if not _thread_local_state.call_stack:  # The first call on this thread
    _thread_local_state.initialize_call_stack()
  stack_entry = _thread_local_state.call_stack[-1]
  if stack_entry.func.is_jax:
    return prim._true_bind_internal(*args, **params)
  # We should not be seeing higher-order primitives in USER functions
  if (_thread_local_state.flags.enable_checks and
      higher_order_primitive(prim, args, params)):
    _thread_local_state.warn_or_error(
        f"Encountered primitive {prim} containing Jaxprs or functions. This means "
        "that there is a higher-order JAX API that is not "
        "annotated with core.jax_boundary")
  try:
    prev = _thread_local_state.collect_repro_enabled
    try:
      _thread_local_state.collect_repro_enabled = False
      res = prim._true_bind_internal(*args, **params)
    finally:
      _thread_local_state.collect_repro_enabled = prev
    stack_entry.bind_primitive(prim, args, params, res)
    return res
  except Exception as e:
    stack_entry.bind_primitive(prim, args, params, [])
    raise


# The line number of the call from USER->JAX. For debugging
this_file_name: str | None = None

# We rewrite some higher-order JAX API calls to uncurry them.
# TODO: Explain why!
# E.g., for a transformation "trans" (e.g., "vmap", "grad", ...)
#   jax.trans(f, *trans_args, **trans_kwargs)(*args, **kwargs) ->
#        jax_trans_call(f, trans_args, trans_kwargs, *args, *kwargs)
# The trampoline is called with one argument representing the real API call
# and should return a curried transformation function like "jax.trans" above.
api_boundary_trampolines: dict[str, Callable[..., Any]] = {}


def jax_boundary(boundary_fun: Callable, *,
                 is_jax: bool = True,
                 func: Func | None = None,
                 api_name: str | None = None,
                 func_argnums: tuple[int, ...] | Callable[..., tuple[int, ...]] = (0,),
                 ):
  """
  Args:
    boundary_fun: the function that is called at the boundary of the JAX/USER
      region.
    is_jax: whether this should be treated as a JAX function.
    func: the boundary object that is being entered. It is None for JAX
      API functions. It is not None for user functions, and for JAX functions
      that are returned from a JAX API function.
  """
  if not config.repro_dir.value: return boundary_fun
  if api_name is not None and (trampoline := api_boundary_trampolines.get(api_name)) is not None:
    return trampoline(boundary_fun)

  if func is None:
    func = Func(boundary_fun, is_jax=is_jax, api_name=api_name,
                func_argnums=func_argnums)

  @functools.wraps(boundary_fun)
  def wrapper(*args, **kwargs):
    if (not config.repro_dir.value or
        not _thread_local_state.collect_repro_enabled):
      return boundary_fun(*args, **kwargs)
    # API functions that do not return JAX functions can be skipped if
    # called for JAX-internal functions.
    if api_name in ["jax.lax.scan", "jax.lax.while_loop", "jax.lax.fori_loop",
                    "jax_vmap_call"]:
      if all(("/jax/jax/_src/" in _fun_sourceinfo(args[i])
              or isinstance(args[i], tree_util.Partial))
             for i in func_argnums):
        if _thread_local_state.flags.enable_checks:
          logging.info(f"Ignoring call to {api_name} from within JAX source")
        return boundary_fun(*args, **kwargs)

    wrapped_args, wrapped_kwargs, call = Call.start_call(args, kwargs, func=func)
    call_func = boundary_fun
    try:
      global this_file_name
      if this_file_name is None:
        this_frame = current_traceback().frames[1]
        this_file_name = this_frame.file_name
      def call_user_to_jax():  # Distinctive name for traceback readability
        return call_func(*wrapped_args, **wrapped_kwargs)  # USER -> JAX
      def call_jax_to_user():
        return call_func(*wrapped_args, **wrapped_kwargs)  # JAX -> USER
      if func.is_jax:
        result = call_user_to_jax()
      else:
        result = call_jax_to_user()
      result = call.end_call(res=result, exc=None)
      return result
    except Exception as e:
      call.end_call(res=None, exc=e)
      raise

  return wrapper


def generic_trampoline(transform_name: str, real_transform: Callable) -> Callable:
  """
  Builds a generic trampoline, e.g.,
    jax.trans(f, *trans_args, **trans_kwargs)(*args, **kwargs) ->
       jax_trans_call(f, trans_args, trans_kwargs, *args, *kwargs)

  """
  jax_trans_call_name = f"jax_{transform_name}_call"
  @partial(jax_boundary, api_name=jax_trans_call_name, func_argnums=(0,))
  def jax_call(fun: Func, trans_args: tuple[Any], trans_kwargs: dict[str, Any],
               *args, **kwargs):
    return real_transform(fun, *trans_args, **trans_kwargs)(*args, **kwargs)
  jax_call.__name__ == jax_trans_call_name
  jax_call.__qualname__ == jax_trans_call_name

  def trampoline(fun: Callable, *trans_args: tuple[Any], **trans_kwargs: dict[str, Any]):
    def call_trampoline(*args, **kwargs):
      return jax_call(fun, trans_args, trans_kwargs, *args, **kwargs)
    call_trampoline.__name__ = f"jax_{transform_name}_call_trampoline"
    call_trampoline.__qualname__ = call_trampoline.__name__
    return call_trampoline
  trampoline.__name__ = f"jax_{transform_name}_trampoline"
  trampoline.__qualname__ = trampoline.__name__
  return trampoline

api_boundary_trampolines["jax.shard_map"] = partial(generic_trampoline, "shard_map")
api_boundary_trampolines["jax.pmap"] = partial(generic_trampoline, "pmap")
api_boundary_trampolines["jax.vmap"] = partial(generic_trampoline, "vmap")
api_boundary_trampolines["jax.grad"] = partial(generic_trampoline, "grad")
api_boundary_trampolines["jax.linear_transpose"] = partial(generic_trampoline, "linear_transpose")
api_boundary_trampolines["jax.jacfwd"] = partial(generic_trampoline, "jacfwd")
api_boundary_trampolines["jax.jacrev"] = partial(generic_trampoline, "jacrev")
api_boundary_trampolines["jax.value_and_grad"] = partial(generic_trampoline, "value_and_grad")
api_boundary_trampolines["jax.checkpoint"] = partial(generic_trampoline, "checkpoint")


def jit_trampoline(real_jit: Callable) -> Callable:
  # TODO: share these with the runtime somehow?
  @partial(jax_boundary, api_name="jax_jit_call",
           func_argnums=(0,))
  def jax_jit_call(fun: Func, jit_args: tuple[Any], jit_kwargs: dict[str, Any],
                   *args, **kwargs):
    jit_new = real_jit(fun, *jit_args, **jit_kwargs)
    return jit_new(*args, **kwargs)


  @partial(jax_boundary, api_name="jax_jit_aot_trace_call",
           func_argnums=(0,))
  def jax_jit_aot_trace_call(fun: Func, jit_args: tuple[Any], jit_kwargs: dict[str, Any],
                             *args, **kwargs):
    jit_new = real_jit(fun, *jit_args, **jit_kwargs)
    return jit_new.trace(*args, **kwargs)


  @partial(jax_boundary, api_name="jax_jit_aot_lower_call",
           func_argnums=(0,))
  def jax_jit_aot_lower_call(fun: Func, jit_args: tuple[Any], jit_kwargs: dict[str, Any],
                             *args, **kwargs):
    jit_new = real_jit(fun, *jit_args, **jit_kwargs)
    return jit_new.lower(*args, **kwargs)

  def jax_jit_trampoline(fun: Callable, *jit_args, **jit_kwargs):
    assert not jit_args, f"jit_args not implemented: {jit_args}"

    # Ignore calls from xla_primitive_callable which use jit over an internal
    # function that just binds the primitive
    if getattr(fun, "_apply_primitive", None):
      return real_jit(fun, *jit_args, **jit_kwargs)

    def jax_jit_call_trampoline(*args, **kwargs):
      return jax_jit_call(fun, jit_args, jit_kwargs, *args, **kwargs)

    def jax_jit_aot_trace_trampoline(*args, **kwargs):
      return jax_jit_aot_trace_call(fun, jit_args, jit_kwargs, *args, **kwargs)

    jax_jit_call_trampoline.trace = jax_jit_aot_trace_trampoline

    def jax_jit_aot_lower_trampoline(*args, **kwargs):
      return jax_jit_aot_lower_call(fun, jit_args, jit_kwargs, *args, **kwargs)

    jax_jit_call_trampoline.lower = jax_jit_aot_lower_trampoline
    return jax_jit_call_trampoline

  return jax_jit_trampoline

api_boundary_trampolines["jax.jit"] = jit_trampoline

def custom_jvp_trampoline(real_boundary_fun: Callable):

  def func_argnums_jax_custom_jvp_call(*args, jvps_count: int, **__):
    return (0,) + tuple(3 + i for i in range(jvps_count) if callable(args[3 + i]))

  @partial(jax_boundary, api_name="jax_custom_jvp_call",
           func_argnums=func_argnums_jax_custom_jvp_call)
  def jax_custom_jvp_call(fun: Func, cjvp_kwargs: dict[str, Any], defjvp_kwargs: dict[str, Any],
                          *fun_jvps_and_args, uses_defjvps: bool, jvps_count: int, **kwargs):
    from jax._src import custom_derivatives
    cjvp_new = custom_derivatives.custom_jvp(fun, **cjvp_kwargs)
    if uses_defjvps:
      cjvp_new.defjvps(*fun_jvps_and_args[:jvps_count])
    else:
      assert jvps_count == 1
      cjvp_new.defjvp(fun_jvps_and_args[0], **defjvp_kwargs)
    return real_boundary_fun(cjvp_new, *fun_jvps_and_args[jvps_count:], **kwargs)

  def jax_custom_jvp_call_trampoline(*args, **kwargs):
    cjvp_orig, *rest_args = args
    if cjvp_orig.jvps is None:
      jvps_count = 1
      uses_defjvps = False
      new_args = (cjvp_orig.jvp, *rest_args)
    else:
      jvps_count = len(cjvp_orig.jvps)
      uses_defjvps = True
      new_args = (*cjvp_orig.jvps, *rest_args)

    return jax_custom_jvp_call(cjvp_orig.fun,
                               dict(nondiff_argnums=cjvp_orig.nondiff_argnums),
                               dict(symbolic_zeros=cjvp_orig.symbolic_zeros),
                               *new_args, uses_defjvps=uses_defjvps, jvps_count=jvps_count,
                               **kwargs)


  return jax_custom_jvp_call_trampoline

api_boundary_trampolines["jax.custom_jvp.__call__"] = custom_jvp_trampoline

def custom_vjp_trampoline(real_boundary_fun: Callable):
  @partial(jax_boundary, api_name="jax_custom_vjp_call",
           func_argnums=(0, 1, 2))
  def jax_custom_vjp_call(fun: Func, fwd: Func, bwd: Func,
                          cvjp_kwargs, *args, **kwargs):
    from jax._src import custom_derivatives
    cvjp_new = custom_derivatives.custom_vjp(fun, **cvjp_kwargs)
    cvjp_new.defvjp(fwd, bwd)
    return real_boundary_fun(cvjp_new, *args, **kwargs)


  def jax_custom_vjp_call_trampoline(*args, **kwargs):
    cvjp_orig, *rest_args = args
    return jax_custom_vjp_call(cvjp_orig.fun, cvjp_orig.fwd, cvjp_orig.bwd,
                               dict(nondiff_argnums=cvjp_orig.nondiff_argnums),
                               *rest_args, **kwargs)

  return jax_custom_vjp_call_trampoline

api_boundary_trampolines["jax.custom_vjp.__call__"] = custom_vjp_trampoline

def  named_call_trampoline(real_boundary_fun: Callable):
  # TODO: handle named_call. The problem that a named_call can wrap a jit
  # with statics and the statics are then not handle properly
  return (lambda fun, *args, **kwargs: fun)

api_boundary_trampolines["jax.named_call"] = named_call_trampoline

###
### Emit repro source code
###

# Maps types to source code emitters
_operand_emitter: dict[Any, Callable[["EmitFuncContext", Any], str]] = {}
_operand_emitter_initialized = False  # For lazy initialization, for circular imports

def register_emitter(typ, emitter: Callable[["EmitFuncContext", Any], str]) -> None:
  """Registers `emitter` to use to emit operands of type `typ`"""
  _operand_emitter[typ] = emitter


class EmitLiterally:
  # These are used as replacements for arguments to be emitted literally.
  def __init__(self, literal: str):
    self.literal = literal

def initialize_operand_emitter():
  # We wrap this in a function that is called late so that we can import here
  # other JAX modules
  global _operand_emitter_initialized
  _operand_emitter_initialized = True

  def emit_enum(enum_name: str) -> Callable[["EmitFuncContext", enum.Enum], str]:
    # For classes that derive from enum.Enum
    def emitter(ctx: "EmitFuncContext", v: enum.Enum):
      return f"{enum_name}.{v.name}"
    return emitter

  def emit_namedtuple(named_tuple_name: str) -> Callable[["EmitFuncContext", NamedTuple], str]:
    def emitter(ctx: "EmitFuncContext", v: NamedTuple):
      return f"{named_tuple_name}({ctx.emit_operand_sequence(v)})"
    return emitter

  _operand_emitter[frozenset] = lambda ctx, v: repr(v)
  _operand_emitter[EmitLiterally] = lambda ctx, v: v.literal

  from jax._src import literals
  _operand_emitter[literals.TypedNdArray] = (
      lambda ctx, v: f"literals.TypedNdArray({ctx.emit_operand_atom(v.val)}, weak_type={v.weak_type})")

  from jax._src import core
  _operand_emitter[core.Primitive] = (
      lambda ctx, v: f"jax_primitive_bind(\"{v}\")")

  def emit_np_array(ctx: EmitFuncContext, v) -> str:
    if v.size <= _thread_local_state.flags.fake_array_threshold:
      return repr(v)
    return f"np.ones({v.shape}, dtype={ctx.emit_operand_atom(v.dtype)})"

  _operand_emitter[np.ndarray] = emit_np_array
  _operand_emitter[np.generic] = emit_np_array

  from jax._src.array import ArrayImpl
  @partial(register_emitter, ArrayImpl)
  def emit_jax_array(ctx: EmitFuncContext, v) -> str:
    # TODO: emit it as a real jax.Array
    return emit_np_array(ctx, np.array(v))

  from jax import sharding  # type: ignore

  @partial(register_emitter, sharding.PartitionSpec)
  def emit_PartitionSpec(ctx: "EmitFuncContext", v: sharding.PartitionSpec) -> str:
    partitions = ctx.emit_operand_sequence(v._partitions)
    if partitions:
      partitions += ", "
    return (f"sharding.PartitionSpec({partitions}unreduced={v.unreduced}, reduced={v.reduced})")

  from jax._src.lib import _jax
  _operand_emitter[_jax.UnconstrainedSingleton] = (
      lambda ctx, v: "sharding.PartitionSpec.UNCONSTRAINED")

  _operand_emitter[sharding.AxisType] = emit_enum("sharding.AxisType")

  @partial(register_emitter, sharding.AbstractMesh)
  def emit_AbstractMesh(ctx: "EmitFuncContext", v: sharding.AbstractMesh) -> str:
    return (f"sharding.AbstractMesh({v.axis_sizes}, {ctx.emit_operand(v.axis_names)}, "
            f"axis_types={ctx.emit_operand(v.axis_types)}, "
            f"abstract_device={ctx.emit_operand(v.abstract_device)})")

  from jax._src.lib import xla_client  # type: ignore
  @partial(register_emitter, xla_client.Device)
  def emit_Device(ctx: "EmitFuncContext", v: sharding.Mesh) -> str:
    return f"jax_get_device(\"{v.platform}\", {v.id})"

  from jax._src.lib import xla_client  # type: ignore
  @partial(register_emitter, sharding.AbstractDevice)
  def emit_AbstractDevice(ctx: "EmitFuncContext", v: sharding.AbstractDevice) -> str:
    return f"sharding.AbstractDevice(\"{v.device_kind}\", {v.num_cores})"

  @partial(register_emitter, sharding.Mesh)
  def emit_Mesh(ctx: "EmitFuncContext", v: sharding.Mesh) -> str:
    devices_list = v.devices.tolist()
    devices_str = ctx.emit_operand(devices_list)
    return (f"sharding.Mesh(np.array({devices_str}), "
            f"axis_names={ctx.emit_operand(v.axis_names)}, "
            f"axis_types={ctx.emit_operand(v.axis_types)})")

  @partial(register_emitter, sharding.NamedSharding)
  def emit_NamedSharding(ctx: "EmitFuncContext", v: sharding.NamedSharding) -> str:
    mesh = ctx.emit_operand_atom(v.mesh)
    spec = ctx.emit_operand_atom(v.spec)
    memory_kind = ctx.emit_operand_atom(v.memory_kind)
    return f"sharding.NamedSharding({mesh}, {spec}, memory_kind={memory_kind})"

  from jax import lax
  register_emitter(lax.AccuracyMode, emit_enum("lax.AccuracyMode"))
  _operand_emitter[lax.ConvDimensionNumbers] = emit_namedtuple("lax.ConvDimensionNumbers")
  _operand_emitter[lax.DotAlgorithm] = emit_namedtuple("lax.DotAlgorithm")
  _operand_emitter[lax.DotAlgorithmPreset] = emit_enum("lax.DotAlgorithmPreset")
  _operand_emitter[lax.FftType] = emit_namedtuple("lax.FftType")
  _operand_emitter[lax.GatherDimensionNumbers] = emit_namedtuple("lax.GatherDimensionNumbers")
  _operand_emitter[lax.GatherScatterMode] = emit_enum("lax.GatherScatterMode")
  _operand_emitter[lax.Precision] = emit_enum("lax.Precision")
  _operand_emitter[lax.RandomAlgorithm] = emit_enum("lax.RandomAlgorithm")
  _operand_emitter[lax.RoundingMethod] = emit_enum("lax.RoundingMethod")
  _operand_emitter[lax.ScatterDimensionNumbers] = emit_namedtuple("lax.ScatterDimensionNumbers")
  def emit_Tolerance(ctx: "EmitFuncContext", v: lax.Tolerance) -> str:
    return f"lax.Tolerance({v.atol}, {v.rtol}, {v.ulps})"
  _operand_emitter[lax.Tolerance] = emit_Tolerance

  from jax._src import random
  def emit_PRNGImpl(ctx: "EmitFuncContext", v: random.PRNGImpl) -> str:
    return f"resolve_prng_impl(\"{v.name}\")"
  _operand_emitter[random.PRNGImpl] = emit_PRNGImpl

  from jax._src.prng import PRNGKeyArray
  @partial(register_emitter, PRNGKeyArray)
  def emit_PRNGKeyArray(ctx: "EmitFuncContext", v: PRNGKeyArray) -> str:
    return f"prng.PRNGKeyArray({emit_PRNGImpl(ctx, v._impl)}, {ctx.emit_operand_atom(v._base_array)})"

  _operand_emitter[xla_client.ArrayCopySemantics] = emit_enum("xla_client.ArrayCopySemantics")


@dataclasses.dataclass
class EmittedFunction():
  lines: list[str]
  # The immediate externals are those referenced in the `lines`, using
  # global variable names
  immediate_externals: dict[int, Any]
  # All externals include the `immediate_externals` and also those referenced
  # in functions that are recursively referenced by the `immediate_externals`.
  all_externals: dict[int, Any]


class EmitGlobalContext:
  _var_for_val: dict[int, str] # id(val) -> var_name

  def __init__(self):
    self._var_for_val = {}
    # Initialize the emitter rules
    if not _operand_emitter_initialized:
      initialize_operand_emitter()  # type: ignore

  def var_for_val(self, for_v: Any, *, prefix="v") -> str:
    v_id = id(for_v)
    vn = self._var_for_val.get(v_id, None)
    if vn is None:
      if isinstance(for_v, Func):
        vn = for_v.python_name()
      else:
        vn = f"{prefix}_{_thread_local_state.small_id(v_id)}"
      self._var_for_val[v_id] = vn
    return vn

  # TODO: make this weakref
  @functools.lru_cache()
  def emit_function(self, invocation: Call,
                    parent_ctx: Union["EmitFuncContext", None]
                    ) -> EmittedFunction:
    return EmitFuncContext(self, invocation, parent_ctx).emit_body()


class EmitFuncContext:

  def __init__(self, global_ctx: EmitGlobalContext, invocation: Call,
               parent_ctx: Union["EmitFuncContext", None]):
    self.global_ctx: EmitGlobalContext = global_ctx
    self.parent_ctx = parent_ctx
    self.invocation = invocation  # The body we are emitting
    self.lines: list[str] = []
    self.local_index = itertools.count()
    # For all the values defined here, their local name (by id)
    self.definitions: dict[int, str] = {}

    self.externals: dict[int, Any] = {}  # direct externals, by id
    self.externals_from_nested: dict[int, Any] = {}  # externals from nested
    self.indent: int = 0
    self.current_traceback: Union["Traceback", None] = None  # type: ignore  # noqa: F821

  def new_local_name(self, *, prefix="v") -> str:
    return f"{prefix}_{next(self.local_index)}"

  def define_value(self, v: Any, vn: str) -> None:
    v_id = id(v)
    if v_id in self.definitions:
      _thread_local_state.warn_or_error(
          f"Encountered definition of already defined value {self.definitions[v_id]} = {v}",
          traceback=self.current_traceback)
    self.definitions[v_id] = vn

  def use_value(self, v: Any) -> str:
    v_id = id(v)
    frame = self
    while True:
      if (vn := frame.definitions.get(v_id)) is not None:
        if frame is self:
          return vn
        else:
          self.externals[v_id] = v
          return self.global_ctx.var_for_val(v, prefix="g")
      frame = frame.parent_ctx  # type: ignore
      if not frame: break

    _thread_local_state.warn_or_error(f"Using undefined value {v}",
                                      traceback=self.current_traceback)
    return self.global_ctx.var_for_val(v, prefix="g")

  def emit_line(self, l: str):
    self.lines.append(" " * self.indent + l)

  def emit_operand(self, a) -> str:
    # TODO: is there a better way to handle a NamedTuple
    if isinstance(a, tuple) and not hasattr(a, "_fields"):
      return f"({self.emit_operand_sequence(a)}{',' if a else ''})"
    if isinstance(a, list):
      return f"[{self.emit_operand_sequence(a)}]"
    if isinstance(a, dict):
      return f"dict({self.emit_operand_key_value_sequence(a)})"
    return str(self.emit_operand_atom(a))

  def emit_operand_sequence(self, a) -> str:
    return ", ".join(self.emit_operand(v) for v in a)

  def emit_operand_key_value_sequence(self, a) -> str:
    return ", ".join(f"{k}={self.emit_operand(v)}" for k, v in a.items())

  def emit_operand_atom(self, v: Any) -> str:
    if isinstance(v, (int, float, bool, str, complex)) or v is None:
      if not isinstance(v, enum.IntEnum):
        return repr(v)
    for t in type(v).__mro__:
      if (emitter := _operand_emitter.get(t)) is not None:
        if (res := emitter(self, v)) is not None:
          return res
    # I don't know a way to handle dtypes using the _operand_emitter mechanism
    if isinstance(v, dtypes.DType):
      return f"dtypes.dtype(\"{v.name}\")"
    v_type_str = str(type(v))
    if "scalar_types._ScalarMeta" in v_type_str:
      return f"jnp.{v.dtype.name}"  # TODO: jnp.int32
    if "flax.core.axes_scan._Broadcast" in v_type_str:  # TODO: flax
      return "flax.core.axes_scan.broadcast"

    from jax._src import core  # type: ignore
    if not isinstance(v, (core.Tracer, Func, tree_util.Partial)):
      _thread_local_state.warn_or_error(
          f"Found non-tracer without custom emitter: {v} of type {type(v)}",
          traceback=self.current_traceback)
    v_id = id(v)
    # already known external
    if v_id in self.externals:
      return self.global_ctx.var_for_val(v, prefix="g")

    if isinstance(v, Func):
      if v.api_name:
        return v.api_name
      if not v.is_jax:
        # USER functions
        self.emit_operand_user_func(v)
        return v.python_name()

    return self.use_value(v)

  def emit_operand_user_func(self, f: Func) -> None:
    if not f.invocation:
      # A function that was never invoked is emitted in the main function
      if self.invocation.func.is_main:
        self.emit_line(f"def {self.global_ctx.var_for_val(f, prefix='fun')}(*args, **kwargs):")
        self.emit_line("  pass")
        self.emit_line("")
      else:
        self.externals[id(f)] = f
      return
    f_emitted = self.global_ctx.emit_function(f.invocation, self)
    if (self.invocation.func.is_main or
        any(ae in self.definitions for ae in f_emitted.all_externals)):
      # Emit (or mark as externals) the external functions
      for e in f_emitted.immediate_externals.values():
        if isinstance(e, Func):
          self.emit_operand_atom(e)
        else:
          self.emit_line(f"{self.global_ctx.var_for_val(e)} = {self.definitions[id(e)]}")
      for bl in f_emitted.lines:
        self.emit_line(bl)
      return

    # We make this function all external
    self.externals[id(f)] = f
    for ae in f_emitted.all_externals:
      self.externals_from_nested[id(ae)] = ae

  @staticmethod
  def flatten_custom_pytree(t) -> Any:
    if isinstance(t, tuple):
      return tuple(EmitFuncContext.flatten_custom_pytree(e) for e in t)
    if isinstance(t, list):
      return [EmitFuncContext.flatten_custom_pytree(e) for e in t]
    if isinstance(t, dict):
      return {k: EmitFuncContext.flatten_custom_pytree(e) for k, e in t.items()}
    if isinstance(t, tree_util.Partial):
      return t
    if t is None:
      return t
    return maybe_singleton(tree_util.tree_leaves(t, is_leaf=lambda l: isinstance(l, Func)))

  @staticmethod
  def flatten_custom_pytree_top(args, kwargs,
                                static_argnums: tuple[int, ...]=(),
                                static_argnames: tuple[str, ...]=()
                                ) -> tuple[list[Any], dict[str, Any]]:
    dyn_args = []
    dyn_kwargs = {}
    # TODO: we should try to get this directly from JAX, there is a slight
    # chance we'll do something different otherwise. This is in C++ in JAX.
    static_argnums = {i if i >= 0 else len(args) - i
                      for i in static_argnums}
    for i, a in enumerate(args):
      if i not in static_argnums:
        dyn_args.append(EmitFuncContext.flatten_custom_pytree(a))
    for k, a in sorted(kwargs.items()):
      if k not in static_argnames:
        dyn_kwargs[k] = EmitFuncContext.flatten_custom_pytree(a)
    return dyn_args, dyn_kwargs

  @staticmethod
  def jax_jit_call_dyn_args_kwargs(jax_jit_call: Call,
                                   args: tuple[Any, ...],
                                   kwargs: dict[str, Any]
                                   ) -> tuple[tuple[Any, ...],
                                              dict[str, Any],
                                              dict[str, Any]]:
    """Dynamic args and kwargs, and also new jit_kwargs, given a jax_jit_call."""
    from jax._src import api_util
    assert jax_jit_call.func.api_name == "jax_jit_call"
    jitted_fun, jit_args, jit_kwargs, *_ = jax_jit_call.args
    assert isinstance(jitted_fun, Func), jitted_fun
    assert not jit_args
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
    dyn_args, dyn_kwargs = EmitFuncContext.flatten_custom_pytree_top(
        args, kwargs,
        static_argnums=static_argnums, static_argnames=static_argnames)
    return dyn_args, dyn_kwargs, new_jit_kwargs

  def preprocess_args_for_call(self, call: Call) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """We are emitting a function call to `call`."""
    if call.func.api_name == "jax_jit_call":
      jitted_fun, jit_args, jit_kwargs, *rest_args = call.args
      dyn_args, dyn_kwargs, new_jit_call_kwargs = EmitFuncContext.jax_jit_call_dyn_args_kwargs(
          call, rest_args, call.kwargs)
      all_args = (jitted_fun, jit_args,
                  EmitFuncContext.flatten_custom_pytree(new_jit_call_kwargs),
                  *dyn_args)
      return all_args, dyn_kwargs

    if call.func.api_name == "jax_checkpoint_call":
      from jax.ad_checkpoint import checkpoint_policies
      f, t_args, t_kwargs, *rest_args = call.args
      new_t_kwargs = dict(t_kwargs)
      if (policy := new_t_kwargs.get("policy")) is not None:
        matching = [(policy_name, getattr(checkpoint_policies, policy_name))
                     for policy_name in dir(checkpoint_policies)
                     if policy is getattr(checkpoint_policies, policy_name)]
        if not matching:
          _thread_local_state.warn_or_error(
            f"Unrecognized jax.checkpoint policy: {policy}. Using 'dots_saveable'",
            traceback=self.current_traceback)
          matching = [("dots_saveable", checkpoint_policies.dots_saveable)]
        new_t_kwargs["policy"] = EmitLiterally(f"jax.checkpoint_policies.{matching[0][0]}")
        new_args = (f, t_args, new_t_kwargs, *rest_args)
        return (EmitFuncContext.flatten_custom_pytree(new_args),
                EmitFuncContext.flatten_custom_pytree(call.kwargs))

    return (EmitFuncContext.flatten_custom_pytree(call.args),
            EmitFuncContext.flatten_custom_pytree(call.kwargs))

  @staticmethod
  def preprocess_args_for_body(call: Call) -> tuple[Sequence[Any], dict[str, Any]]:
    """We are emitting the body of `call`."""
    assert not call.func.is_jax, call
    if call.func.is_main: return call.args, call.kwargs
    assert call.func.introduced_for is not None, call

    if call.func.introduced_for.func.api_name == "jax_jit_call":
      dyn_args, dyn_kwargs, _ = EmitFuncContext.jax_jit_call_dyn_args_kwargs(
          call.func.introduced_for, call.args, call.kwargs
      )
      return dyn_args, dyn_kwargs

    return (EmitFuncContext.flatten_custom_pytree(call.args),
            EmitFuncContext.flatten_custom_pytree(call.kwargs))


  def emit_body(self) -> EmittedFunction:
    """Emits the body of the function.

    This is called starting with the main function.
    Returns:
      * the source lines of the emitted body
      * the externals referenced directly in the emitted body. These are
        referenced by their global name, and can be either USER Func used or
        other values closed over from enclosing functions.
      * the transitive closure of the externals in the emitted body and USER
        functions referenced from the emitted body.
    """
    invocation = self.invocation
    assert not invocation.func.is_jax

    self.current_traceback = invocation.traceback
    self.emit_line("")
    self.emit_line(f"# body from invocation {invocation}")
    inv_args, inv_kwargs = EmitFuncContext.preprocess_args_for_body(invocation)
    inv_result = EmitFuncContext.flatten_custom_pytree(invocation.result)
    if all(val_is_leaf(v) for v in (*inv_args, *inv_kwargs.values())):
      args_str = ""
      for aidx, a in enumerate(inv_args):
        an = self.new_local_name()
        self.define_value(a, an)
        if args_str: args_str += ", "
        args_str += an
      if inv_kwargs:
        args_str += ", *"
      for k, a in inv_kwargs.items():
        self.define_value(a, k)
        if args_str: args_str += ", "
        args_str += f"{k}"
      self.emit_line(f"def {self.global_ctx.var_for_val(invocation.func)}({args_str}):")
      self.indent += 2
    else:
      self.emit_line(f"def {self.global_ctx.var_for_val(invocation.func)}(*args, **kwargs):")
      self.indent += 2
      # unpack the pytrees in args and kwargs
      # TODO: clean
      for path, a in tree_util.tree_leaves_with_path((inv_args, inv_kwargs),
                                                     is_leaf=lambda n: isinstance(n, FuncPartial)):
        an = self.new_local_name()
        self.define_value(a, an)
        path_str = f"{'args' if path[0].idx == 0 else 'kwargs'}{keystr_no_flat_index(path[1:])}"
        self.emit_line(f"{an} = {path_str}")
    del inv_args, inv_kwargs

    result = None  # The last result
    for c in invocation.body:  # For each call statement in the body
      self.current_traceback = c.traceback
      result = self.emit_call(c)

    # The end of the function
    self.current_traceback = invocation.traceback
    if invocation.func.is_main:
      # Return the last result
      assert inv_result is None
      if result is None:
        last_result = "_"
      elif val_is_leaf(result):
        last_result = self.use_value(result)
      else:
        last_result = "result"
      self.emit_line(f"return {last_result}")
    else:
      self.emit_line(f"return {self.emit_operand(inv_result)}")
    all_externals = dict(self.externals)  # copy
    for e in self.externals_from_nested.values():
      all_externals[id(e)] = e
    return EmittedFunction(self.lines, self.externals, all_externals)

  def emit_call(self, c: Call) -> Any:
    if isinstance(c.func, Func):
      args, kwargs = self.preprocess_args_for_call(c)
      result = EmitFuncContext.flatten_custom_pytree(c.result)
    else:
      args, kwargs, result = c.args, c.kwargs, c.result  # primitives

    if result is not None:
      res_is_leaf = val_is_leaf(result)
      if res_is_leaf:
        res_str = self.new_local_name()
      else:
        res_str = "result"
    else:
      res_is_leaf = True
      res_str = "_"

    callee_name = self.emit_operand_atom(c.func)
    call_args_str = self.emit_operand_sequence(args)
    if kwargs:
      call_args_str += (", " if args else "") + self.emit_operand_key_value_sequence(kwargs)
    self.emit_line(f"{res_str} = {callee_name}({call_args_str})  # {c}")
    if result is not None:
      if res_is_leaf:
        self.define_value(c.result, res_str)
      else:
        # TODO: clean
        for path, r in tree_util.tree_leaves_with_path(
            result,
            is_leaf=lambda n: (isinstance(n, FuncPartial))):
          r_n = self.new_local_name()
          self.define_value(r, r_n)
          path_str = f"result{keystr_no_flat_index(path)}"
          self.emit_line(f"{r_n} = {path_str}")
    return result


def _emit_repro(top_entry: Call, extra_message: str):
  assert config.repro_dir.value
  dump_to = config.repro_dir.value
  if not (out_dir := path.make_jax_dump_dir(dump_to)):
    return None
  fresh_id = itertools.count()
  while True:
    repro_path = out_dir / f"{top_entry.func.python_name()}_{next(fresh_id)}_repro.py"
    if not os.path.exists(repro_path):
      break
  assert not top_entry.func.is_jax
  # Collect the relevant functions, functions earlier in the list may depend
  # on function later in the list. We will emit the code starting with the
  # last function.
  global_ctx = EmitGlobalContext()
  main_invocation = top_entry
  if not top_entry.func.is_main:
    main_invocation = top_entry.parent.parent
  assert main_invocation.parent is None
  main_emitted = global_ctx.emit_function(main_invocation, None)
  preamble = f"""
# This file was generated by JAX repro extractor.

import jax
from jax._src import config

from jax._src.repro_runtime import *

if config.enable_x64.value != {config.enable_x64.value}:
  raise ValueError("This repro was saved with JAX_ENABLE_X64={config.enable_x64.value}."
                   "You should run with with the same value of the flag.")

# Repro was saved due to:
"""
  preamble += "\n".join([f"  # {l}" for l in extra_message.split("\n")]) + "\n\n"
  postamble = f"""

_repro_result = {main_invocation.func.python_name()}()
"""
  repro_source = preamble + "\n".join(main_emitted.lines) + postamble
  logging.warning(f"Dumped JAX repro at {repro_path} due to {extra_message[:128]}...")
  repro_path.write_text(repro_source)
  if main_emitted.all_externals:
    msg = ("Got undefined symbols: " +
           ", ".join(f"{e} = {global_ctx.var_for_val(e)}"
                     for e in main_emitted.all_externals.values()) +
           f"\nThe repro has been saved to {repro_path}.")
    _thread_local_state.warn_or_error(msg)

  _thread_local_state.emitted_repros.append((str(repro_path), repro_source))
  return repro_path, repro_source


"""
TODO:
* print for function definitions the static arguments, as comments
* maybe print also the avals
* why does the trampoline make it into the value_and_grad test?
* share the trampolines with the runtime
* actual devices use jax_get_device, we need a way to make it platform
  agnostic
* print ndarray properly, when using repr(...) it only prints a summary.
* Look in api.py, we also have jax.fwd_and_bwd, jax.jacrev, jax.jaxfwd
* instead of just emitting None for a return from a function with an error,
  print some indication of the error
* must set the enable_x64 flag the same during the repro as during the
  collection

* If one forgets to annotate a higher-order jax_boundary then we won't wrap
  USER functions and we call into user code without tracking it. We would likely
  still be in USER mode, and then we'd give an error seeing a higher-order
  JAX primitive.

What are we missing:
* caching is changed (mostly defeated)
* named_call is not handled
* we emit jax.Array as np.ndarray (loosing sharding)
* with custom_vjp, when we do higher-order differentiation the custom fwd and
  bwd functions are being called more than once, and we ignore the subsequent
  calls. This is a problem if those invocations would result in a different
  Jaxpr.
* if we don't recognize the jax.checkpoint policy param, we print a warning
  and we use `dots_saveable".
"""
