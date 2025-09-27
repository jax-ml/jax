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
  enable_checks_as_errors = False
  enable_checks_with_tracebacks = False

  # Replace arrays with `np.ones`
  fake_arrays = False
  # Squash sequences of array primitives, preserving the dependencies
  squash_primitives = False

  def __init__(self):
    if config.repro_flags.value:
      for f in config.repro_flags.value.split(","):
        if f == "enable_checks":
          self.enable_checks = True
        elif f == "enable_checks_as_errors":
          self.enable_checks = True
          self.enable_checks_as_errors = True
        elif f == "enable_checks_with_tracebacks":
          self.enable_checks = True
          self.enable_checks_with_tracebacks = True
        else:
          raise NotImplementedError(f"--jax_repro_flags: {f}")


class _ThreadLocalState(threading.local):
  def __init__(self):
    self.stack_node_id = collections.defaultdict(itertools.count)
    self.call_stack: list["Call"] = []

    self.flags = ReproFlags()
    self.emit_repro_enabled = True
    # List of path names and source codes
    self.emitted_repros: list[tuple[str, str]] = []

    # Set to true when have intercepted a primitive bind, and are calling JAX's
    # true primitive binding; we will just include this primitive call in the
    # repro, we do not want to see all that happens while this primitive is
    # processed
    self.inside_bind_primitive = False

    # For simpler debugging, use our own small value ids when printing
    self.small_id_index = itertools.count()
    self.small_id_map: dict[int, int] = {}  # id(v) -> small id
    self.func_index = itertools.count()  # Each constructed Func has an index
    self.call_index = itertools.count()  # Each constructed Call has an index

  def initialize(self):
    def main(): return
    main_func = Func(main, is_jax=False)
    main_func.is_main = True
    self.push_call(main_func, (), {})

  def snapshot_state(self):
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
    call = Call(len(self.call_stack),
                self.call_stack[-1] if self.call_stack else None,
                func, args, kwargs)
    self.call_stack.append(call)
    return call

  def warn_or_error(self, msg):
    if self.flags.enable_checks_as_errors:
      logging.error(msg)
      raise ValueError(msg)
    elif self.flags.enable_checks_with_tracebacks:
      tb = current_traceback()
      logging.error(msg)
      logging.error("Traceback:")
      for l in str(tb).splitlines()[:20]:
        logging.error("     " + l)
      logging.error("    ....")
    else:
      logging.warning(msg)

_thread_local_state = _ThreadLocalState()


class Func:
  MISSING_API_NAME = "MISSING_API_NAME"

  def __init__(self, fun: Callable, *, is_jax: bool,
               api_name: str | None = None, api_constructor: bool = False,
               func_argnums: tuple[int, ...] | Callable[..., tuple[int, ...]] = (0,),
               introduced_for: Union["Call", None] = None):
    # TODO: we should not need the prefix, we have introduced_for.
    fun_name_prefix = ""
    if introduced_for is not None:
      for_api_name: str = introduced_for.func.api_name
      if for_api_name == "jax_jit_call":
        fun_name_prefix = "jit_"
      elif for_api_name == "jax.jvp":
        fun_name_prefix = "jvp_"
      elif for_api_name == "jax_grad_call":
        fun_name_prefix = "grad_"
      elif for_api_name == "jax.vjp":
        fun_name_prefix = "vjp_"
      elif for_api_name == "jax.checkpoint":
        fun_name_prefix = "checkpoint_"

    self.fun = fun
    self.fun_info = _fun_sourceinfo(fun)
    self.fun_name = self.fun_info.split(" ")[0]
    self.fun_name = fun_name_prefix + self.fun_name
    self.is_jax = is_jax
    self.id = next(_thread_local_state.func_index)
    self.is_main = False  # Whether this is the top-level function
    assert not (api_name and not is_jax), (fun, api_name)
    self.api_name = api_name
    self.func_argnums = func_argnums
    self.api_constructor = api_constructor
    assert not (api_constructor and not api_name), (fun, api_name)
    self.__wrapped__ = fun  # Enable fun_sourceinfo(self)
    if hasattr(fun, "__name__"):
      setattr(self, "__name__", fun.__name__)
    if hasattr(fun, "__qualname__"):
      setattr(self, "__qualname__", fun.__name__)
    # For PjitWrapper
    from jax._src.lib import xla_client as xc
    if self.is_jax and type(fun) is xc._xla.PjitFunction:
      self.lower = (
          lambda *args, **kwargs: aot_lower(self, *args, **kwargs))
      self.trace = (
          lambda *args, **kwargs: aot_trace(self, *args, **kwargs))
      self.eval_shape = (
          lambda *args, **kwargs: aot_eval_shape(self, *args, **kwargs))
      self.clear_cache = fun.clear_cache
      self._cache_size = fun._cache_size

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
      # TODO: do we need to do this special thing for Partial? Maybe it is
      # enough to just register Func as a tree node
      assert is_jax  # In returned values
      res = FuncPyTreeNode(f, is_jax=is_jax, introduced_for=introduced_for)
      if accum is not None: accum.append(res)  # type: ignore
      return res

    res = Func(f, is_jax=is_jax, introduced_for=introduced_for)  # type: ignore
    if accum is not None: accum.append(res)  # type: ignore
    return res


@tree_util.register_pytree_node_class
class FuncPyTreeNode(Func):

  def __init__(self, func: tree_util.Partial,
               is_jax: bool, introduced_for: Union["Call", None]):
    super().__init__(func, is_jax=is_jax, introduced_for=introduced_for)

  def tree_flatten(self):
    return ((self.fun,), self)

  @classmethod
  def tree_unflatten(cls, aux_data: "FuncPyTreeNode", children):
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
    self.body : list["Call"] = []
    self.result: Any | None = None
    self.source_info = current_traceback()
    # The values defined in this scope, indexed by id(v)
    self.definitions: dict[int, str] = {}

  @staticmethod
  def val_id(v):
    return _thread_local_state.small_id(id(v))

  def __repr__(self):
    return f"[{self.level}.{self.id}] {self.func}"

  @staticmethod
  def pytree_to_str(t, callback: Callable[[Any, str], None] | None):
    from jax._src.array import ArrayImpl
    from jax._src import literals
    def a_to_str(a):
      # I don't know a way to handle dtypes using the _operand_emitter mechanism
      if isinstance(a, dtypes.DType):
        return f"dtypes.dtype(\"{a.name}\")"
      if isinstance(a, (np.ndarray, np.generic, literals.TypedNdArray)):
        res = f"ndarr[{','.join(str(d) for d in a.shape)}]<{Call.val_id(a)}>"
        return res
      if isinstance(a, ArrayImpl):
        res = f"jax_arr[{','.join(str(d) for d in a.shape)}]<{Call.val_id(a)}>"
        return res
      if hasattr(a, "shape"):
        typ = "tracer" if hasattr(a, "_trace") else "other_arr_" + re.sub(r"[^a-zA-Z0-9_]", "_", str(type(a)))
        res = f"{typ}[{','.join(str(d) for d in a.shape)}]<{Call.val_id(a)}>"
        not callback or callback(a, res)
        return res
      if isinstance(a, (int, float, bool, str)):
        return str(a)  # Don't use callback for constants
      if isinstance(a, frozenset):
        return repr(a)
      if isinstance(a, Func):
        res = str(a)
        if a.is_jax and not a.api_name and callback:
          callback(a, res)
        return res

      res = f"{a}<{Call.val_id(a)}>"
      not callback or callback(a, res)
      return res
    return tree_util.tree_map(a_to_str, t)

  def define_value(self, v: Any, v_str: str):
    v_id = id(v)
    if v_id in self.definitions:
      _thread_local_state.warn_or_error(f"Encountered definition of already defined value {v_str}")
    self.definitions[v_id] = v_str

  def use_value(self, v: Any, v_str: str):
    v_id = id(v)
    frame = self
    while True:
      if v_id in frame.definitions: return
      # Look for a USER function up the stack
      frame = frame.parent
      while frame and frame.func.is_jax:
        frame = frame.parent
      if not frame: break
      continue

    _thread_local_state.warn_or_error(f"Using undefined value {v_str}")


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

    call = _thread_local_state.push_call(func,
                                         wrapped_args, wrapped_kwargs)
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
      def callback(a, a_str):
        # use it in the caller
        if not call.parent.func.is_jax:
          if not (call.func.api_constructor and
                  "custom_derivatives.custom_" in str(a)):
            # the first arg of a constructor is the custom_jvp object
            call.parent.use_value(a, a_str)
        if not call.func.is_jax:
          call.define_value(a, a_str)
      args_str = Call.pytree_to_str(wrapped_args,
                                    callback=callback)
      kwargs_str = Call.pytree_to_str(wrapped_kwargs,
                                      callback=callback)
      logging.info(f"{'  ' * call.level} start {call}(*{args_str}, **{kwargs_str})")
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
          if self.func.api_constructor:  # TODO: hack for class constructors
            self.result = self.args[0]
            self.args = self.args[1:]
          else:
            # Look through the result of JAX functions and wrap callables as
            # Func. If we are returning functions, they are either the sole
            # result of in a tuple
            if isinstance(self.result, tuple):
              multiple_results = True
              results = list(self.result)
            else:
              multiple_results = False
              results = [self.result]

            for i, r in enumerate(results):
              if callable(r):
                results[i] = Func.wrap_func_in_out(r, is_jax=True,
                                                   introduced_for=self)
            if multiple_results:
              self.result = tuple(results)
            else:
              self.result = results[0]

        if _thread_local_state.flags.enable_checks:
          def callback(a, a_str):
            # use it here
            if not self.func.is_jax:
              self.use_value(a, a_str)
            if not self.parent.func.is_jax:
              self.parent.define_value(a, a_str)
          res_str = Call.pytree_to_str(
              self.result, callback=callback)
          res_str = f"res={res_str}"
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

      if not call_stack[-1].func.is_jax:
        # We were called from a user function
        caller = call_stack[-1]
        caller.body.append(self)

      if call_stack[-1].func.is_main:
        if (exc is not None and not isinstance(exc, ReproError)):
          # We caught an exception and we are at top-level
          _emit_repro(call_stack[-1], "uncaught exception " + traceback.format_exc())

      if self.func.api_constructor:
        return None
      return self.result
    except Exception as e:
      # Exceptions here are bad, because they break the call_stack invariants
      logging.error(f"Exception caught in the exit handler: {type(e)}: {e}\n{traceback.format_exc()}")
      raise e

  def bind_primitive(self, prim: Primitive, args, params, outs):
    not _thread_local_state.flags.enable_checks or Call.pytree_to_str(args, callback=self.use_value)
    c = Call(self.level + 1, self, prim, args, params)
    c.result = outs
    not _thread_local_state.flags.enable_checks or Call.pytree_to_str(outs, callback=self.define_value)
    self.body.append(c)

def _true_bind_primitive(prim: Primitive, args, params):
  # Replacement for Primitive._true_bind when using repros
  if (not config.repro_dir.value or
      not _thread_local_state.emit_repro_enabled or
      _thread_local_state.inside_bind_primitive):
    return prim._true_bind_internal(*args, **params)
  not _thread_local_state.flags.enable_checks or check_traceback()
  stack_entry = _thread_local_state.call_stack[-1]
  if stack_entry.func.is_jax:
    return prim._true_bind_internal(*args, **params)
  # We should not be seeing higher-order primitives in USER functions
  if higher_order_primitive(prim, args, params):
    _thread_local_state.warn_or_error(
        f"Seen primitive {prim} containing Jaxprs or functions. This means "
        "that there is a higher-order JAX API that is not "
        "annotated with core.jax_boundary")
  try:
    prev = _thread_local_state.inside_bind_primitive
    try:
      _thread_local_state.inside_bind_primitive = True
      res = prim._true_bind_internal(*args, **params)
    finally:
      _thread_local_state.inside_bind_primitive = prev
    stack_entry.bind_primitive(prim, args, params, res)
    return res
  except Exception as e:
    stack_entry.bind_primitive(prim, args, params, [])
    raise


# The line number of the call from USER->JAX. For debugging
this_file_name: str | None = None
user_to_jax_line: int | None = None
jax_to_user_line: int | None = None

def jax_boundary(boundary_fun: Callable, *,
                 is_jax: bool = True,
                 func: Func | None = None,
                 api_name: str | None = None,
                 func_argnums: tuple[int, ...] | Callable[..., tuple[int, ...]] = (0,),
                 api_constructor: bool = False):
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
  if api_name == "jax.jit":
    real_jit = boundary_fun
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

  if api_name == "jax.shard_map":
    real_shard_map = boundary_fun
    @partial(jax_boundary, api_name="jax_shard_map_call",
             func_argnums=(0,))
    def jax_shard_map_call(fun: Func, shard_map_kwargs: dict[str, Any],
                           *args, **kwargs):
      shard_map_new = real_shard_map(fun, **shard_map_kwargs)
      return shard_map_new(*args, **kwargs)

    def jax_shard_map_trampoline(fun: Callable, *t_args, **shard_map_kwargs):
      assert not t_args, ("shard_map does not take more *args", t_args)
      def jax_shard_map_call_trampoline(*args, **kwargs):
        return jax_shard_map_call(fun, shard_map_kwargs, *args, **kwargs)
      return jax_shard_map_call_trampoline

    return jax_shard_map_trampoline

  if api_name == "jax.grad":
    real_grad = boundary_fun
    @partial(jax_boundary, api_name="jax_grad_call",
             func_argnums=(0,))
    def jax_grad_call(fun: Func, grad_args: tuple[Any], grad_kwargs: dict[str, Any],
                      *args, **kwargs):
      grad_new = real_grad(fun, *grad_args, **grad_kwargs)
      return grad_new(*args, **kwargs)

    def jax_grad_trampoline(fun: Callable, *grad_args, **grad_kwargs):
      def jax_grad_call_trampoline(*args, **kwargs):
        return jax_grad_call(fun, grad_args, grad_kwargs, *args, **kwargs)
      return jax_grad_call_trampoline

    return jax_grad_trampoline

  if api_name == "jax.value_and_grad":
    real_value_and_grad = boundary_fun
    @partial(jax_boundary, api_name="jax_value_and_grad_call",
             func_argnums=(0,))
    def jax_value_and_grad_call(fun: Func, value_and_grad_args: tuple[Any], value_and_grad_kwargs: dict[str, Any],
                      *args, **kwargs):
      value_and_grad_new = real_value_and_grad(fun, *value_and_grad_args, **value_and_grad_kwargs)
      return value_and_grad_new(*args, **kwargs)

    def jax_value_and_grad_trampoline(fun: Callable, *value_and_grad_args, **value_and_grad_kwargs):
      def jax_value_and_grad_call_trampoline(*args, **kwargs):
        return jax_value_and_grad_call(fun, value_and_grad_args, value_and_grad_kwargs, *args, **kwargs)
      return jax_value_and_grad_call_trampoline

    return jax_value_and_grad_trampoline

  if api_name == "jax.vmap":
    real_vmap = boundary_fun
    @partial(jax_boundary, api_name="jax_vmap_call",
             func_argnums=(0,))
    def jax_vmap_call(fun: Func, vmap_args: tuple[Any], vmap_kwargs: dict[str, Any],
                      *args, **kwargs):
      vmap_new = real_vmap(fun, *vmap_args, **vmap_kwargs)
      return vmap_new(*args, **kwargs)

    def jax_vmap_trampoline(fun: Callable, *vmap_args: tuple[Any], **vmap_kwargs: dict[str, Any]):
      def jax_vmap_call_trampoline(*args, **kwargs):
        return jax_vmap_call(fun, vmap_args, vmap_kwargs, *args, **kwargs)
      return jax_vmap_call_trampoline

    return jax_vmap_trampoline

  if api_name == "jax.custom_vjp.__call__":
    @partial(jax_boundary, api_name="jax_custom_vjp_call",
             func_argnums=(0, 1, 2))
    def jax_custom_vjp_call(fun: Func, fwd: Func, bwd: Func,
                            cvjp_kwargs, *args, **kwargs):
      from jax._src import custom_derivatives
      cvjp_new = custom_derivatives.custom_vjp(fun, **cvjp_kwargs)
      cvjp_new.defvjp(fwd, bwd)
      return boundary_fun(cvjp_new, *args, **kwargs)
    def jax_custom_vjp_call_trampoline(*args, **kwargs):
      cvjp_orig, *rest_args = args
      return jax_custom_vjp_call(cvjp_orig.fun, cvjp_orig.fwd, cvjp_orig.bwd,
                                 dict(nondiff_argnums=cvjp_orig.nondiff_argnums),
                                 *rest_args, **kwargs)
    return jax_custom_vjp_call_trampoline

  if api_name == "jax.custom_jvp.__call__":
    def func_argnums_jax_custom_jvp_call(*args, jvps_count:int, **__):
      return (0,) + tuple(3 + i for i in range(jvps_count) if callable(args[3 + i]))
    @partial(jax_boundary, api_name="jax_custom_jvp_call",
             func_argnums=func_argnums_jax_custom_jvp_call)
    def jax_custom_jvp_call(fun: Func, cjvp_kwargs: dict[str, Any], defjvp_kwargs: dict[str, Any],
                            *fun_jvps_and_args, uses_defjvps:bool, jvps_count: int, **kwargs):
      from jax._src import custom_derivatives
      cjvp_new = custom_derivatives.custom_jvp(fun, **cjvp_kwargs)
      if uses_defjvps:
        cjvp_new.defjvps(* fun_jvps_and_args[:jvps_count])
      else:
        assert jvps_count == 1
        cjvp_new.defjvp(fun_jvps_and_args[0], **defjvp_kwargs)
      return boundary_fun(cjvp_new, *fun_jvps_and_args[jvps_count:], **kwargs)
    def jax_custom_jvp_call_trampoline(*args, **kwargs):
      cjvp_orig, *rest_args = args
      if cjvp_orig.jvps is None:
        jvps_count = 1
        uses_defjvps = False
        new_args = (cjvp_orig.jvp, *rest_args)
      else:
        jvps_count = len(cjvp_orig.jvps)
        uses_defjvps = True
        new_args = (* cjvp_orig.jvps, *rest_args)

      return jax_custom_jvp_call(cjvp_orig.fun,
                                 dict(nondiff_argnums=cjvp_orig.nondiff_argnums),
                                 dict(symbolic_zeros=cjvp_orig.symbolic_zeros),
                                 *new_args, uses_defjvps=uses_defjvps, jvps_count=jvps_count,
                                 **kwargs)
    return jax_custom_jvp_call_trampoline

  if api_name == "jax.named_call":
    # TODO: handle named_call. The problem that a named_call can wrap a jit
    # with statics and the statics are then not handle properly
    return (lambda fun, *, name: fun)

  if func is None:
    func = Func(boundary_fun, is_jax=is_jax, api_name=api_name,
                api_constructor=api_constructor, func_argnums=func_argnums)

  @functools.wraps(boundary_fun)
  def wrapper(*args, **kwargs):
    if (not config.repro_dir.value or
        not _thread_local_state.emit_repro_enabled or
        _thread_local_state.inside_bind_primitive):
      return boundary_fun(*args, **kwargs)
    # API functions that do not return JAX functions can be skipped if
    # called for JAX-internal functions.
    if api_name in ["jax.lax.scan", "jax.lax.while_loop", "jax.lax.fori_loop",
                    "jax.vmap"]:
      if all("/jax/jax/_src/" in _fun_sourceinfo(args[i])
             for i in func_argnums):
        if _thread_local_state.flags.enable_checks:
          logging.info(f"Ignoring call to {api_name} from within JAX source")
        return boundary_fun(*args, **kwargs)

    wrapped_args, wrapped_kwargs, call = Call.start_call(args, kwargs, func=func)
    call_func = boundary_fun
    try:
      global this_file_name, user_to_jax_line, jax_to_user_line
      if user_to_jax_line is None:
        this_frame = current_traceback().frames[1]
        this_file_name = this_frame.file_name
        user_to_jax_line = 5 + this_frame.line_num
        jax_to_user_line = 2 + user_to_jax_line
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
      raise e

  return wrapper


# TODO: move these to a run time file
@partial(jax_boundary, api_name="jax_aot_trace")
def aot_trace(jitted_fun: Func, *args, **kwargs):
  fun = jitted_fun.fun  # Remove wrappers
  while isinstance(fun, Func):
    fun = fun.fun
  return fun.trace(*args, **kwargs)  # type: ignore

@partial(jax_boundary, api_name="jax_aot_lower")
def aot_lower(jitted_fun: Func, *args, **kwargs):
  fun = jitted_fun.fun  # Remove wrappers
  while isinstance(fun, Func):
    fun = fun.fun
  return fun.lower(*args, **kwargs)  # type: ignore

@partial(jax_boundary, api_name="jax_aot_eval_shape")
def aot_eval_shape(jitted_fun: Func, *args, **kwargs):
  fun = jitted_fun.fun  # Remove wrappers
  while isinstance(fun, Func):
    fun = fun.fun
  return fun.eval_shape(*args, **kwargs)  # type: ignore


def check_traceback():
  """Check the invariants of the current traceback and Func call stack.

  TODO

  """
  def is_transition_to_user(f):
    return this_file_name == f.file_name and f.line_num == jax_to_user_line
  def is_transition_to_jax(f):
    return this_file_name == f.file_name and f.line_num == user_to_jax_line

  def is_filename_in_test(file_name):
    # TODO: make a regex
    # These are considered USER even though they are in /jax/jax/
    if ("/jax/jax/_src/test_util.py" in file_name or
        "/jax/jax/_src/public_test_util.py" in file_name):
      return True
    return "/jax/tests/" in file_name

  # Pop current() and check_traceback()
  frames = current_traceback().frames[2:]
  call_stack = _thread_local_state.call_stack
  # Annotations for each frame
  annotations: list[list[str]] = [[] for _ in frames]

  prev_f = None
  # We scan the traceback from bottom and the call_stack in parallel
  call_stack_idx = 0  # The last call we have seen; at index 0 we have the
                      # fake "main"
  assert call_stack[call_stack_idx].func.is_main
  has_errors = False
  seen_test_frame = False
  in_jax = False  # We have seen a transition USER -> JAX

  for traceback_idx in reversed(range(len(frames))):
    f = frames[traceback_idx]
    if not seen_test_frame:
      if is_filename_in_test(f.file_name):
        seen_test_frame = True
        annotations[traceback_idx].append("Effective stack bottom (first test frame)")

      prev_f = f
      continue

    if is_filename_in_test(f.file_name):
      # A frame in test must follow another test frame, or a transition to USER
      if in_jax:
        has_errors = True
        annotations[traceback_idx].append("ERROR: current frame points to test, but in JAX mode")
      assert prev_f is not None
      if not is_filename_in_test(prev_f.file_name) and not is_transition_to_user(prev_f):
        has_errors = True
        annotations[traceback_idx].append("ERROR: Current frame is test and the previous one is not.")

    def annotate_transition(to_jax: bool):
      to_str = "JAX" if to_jax else "USER"
      nonlocal call_stack_idx, has_errors
      call_stack_idx += 1
      if call_stack_idx >= len(call_stack):
        has_errors = True
        annotations[traceback_idx].append(
            f"ERROR: Ran out of call_stack for call to {to_str}: {call_stack_idx} >= {len(call_stack)}")
      elif call_stack[call_stack_idx].func.is_jax != to_jax:
        has_errors = True
        annotations[traceback_idx].append(
            f"ERROR: Expected call to {to_str}, but {call_stack[call_stack_idx]}")
      else:
        annotations[traceback_idx].append(
            "Call " + str(call_stack[call_stack_idx]))

    if is_transition_to_user(f):
      annotate_transition(False)
      in_jax = False
    elif is_transition_to_jax(f):
      annotate_transition(True)
      in_jax = True

    prev_f = f

  if has_errors:
    def show_frames():
      acc = []
      for i, f in enumerate(frames):
        ann = annotations[i]
        acc.append(f"[{i:4d}] {f}")
        for a in ann:
          acc.append("   ^^^ " + a)
      return "\n".join(acc)

    frames_str = show_frames()
    call_stack_str = "\n".join([f"  {c}" for c in _thread_local_state.call_stack])
    logging.error(f"The following are the frames, starting with top of "
                  f"the stack:\n{frames_str}\nCall stack is:\n{call_stack_str}")
    assert False, "check_traceback errors, see error log"

###
### Emit repro source code
###

# Maps types to source code emitters
_operand_emitter: dict[Any, Callable[["EmitFuncContext", Any], str]] = {}
_operand_emitter_initialized = False  # For lazy initialization, circular imports

@dataclasses.dataclass()
class EmitLiteral:
  what: str
_operand_emitter[EmitLiteral] = lambda _, v: v.what

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

  from jax._src import literals
  _operand_emitter[literals.TypedNdArray] = (
      lambda ctx, v: f"literals.TypedNdArray({ctx.emit_operand_atom(v.val)}, weak_type={v.weak_type})")

  from jax._src import core
  _operand_emitter[core.Primitive] = (
      lambda ctx, v: f"jax_primitive_bind(\"{v}\")")

  def emit_np_array(ctx: EmitFuncContext, v) -> str:
    return repr(v)
  _operand_emitter[np.ndarray] = emit_np_array
  _operand_emitter[np.generic] = emit_np_array

  from jax._src.array import ArrayImpl
  def emit_jax_array(ctx: EmitFuncContext, v) -> str:
    # TODO: emit it as a real jax.Array
    return ctx.emit_operand(np.array(v))
  _operand_emitter[ArrayImpl] = emit_jax_array

  from jax import sharding  # type: ignore

  def emit_PartitionSpec(ctx: "EmitFuncContext", v: sharding.PartitionSpec) -> str:
    return (f"sharding.PartitionSpec({ctx.emit_operand_sequence(v._partitions)}, unreduced={v.unreduced}, reduced={v.reduced})")
  _operand_emitter[sharding.PartitionSpec] = emit_PartitionSpec

  _operand_emitter[sharding.AxisType] = emit_enum("sharding.AxisType")
  def emit_AbstractMesh(ctx: "EmitFuncContext", v: sharding.AbstractMesh) -> str:
    return (f"sharding.AbstractMesh({v.axis_sizes}, {ctx.emit_operand(v.axis_names)}, "
            f"axis_types={ctx.emit_operand(v.axis_types)}, "
            f"abstract_device={ctx.emit_operand(v.abstract_device)})")
  _operand_emitter[sharding.AbstractMesh] = emit_AbstractMesh

  from jax._src.lib import xla_client  # type: ignore
  def emit_Device(ctx: "EmitFuncContext", v: sharding.Mesh) -> str:
    return f"jax_get_device(\"{v.platform}\", {v.id})"
  _operand_emitter[xla_client.Device] = emit_Device

  def emit_Mesh(ctx: "EmitFuncContext", v: sharding.Mesh) -> str:
    devices_list = v.devices.tolist()
    devices_str = ctx.emit_operand(devices_list)
    return (f"sharding.Mesh(np.array({devices_str}), "
            f"axis_names={ctx.emit_operand(v.axis_names)}, "
            f"axis_types={ctx.emit_operand(v.axis_types)})")
  _operand_emitter[sharding.Mesh] = emit_Mesh

  def emit_NamedSharding(ctx: "EmitFuncContext", v: sharding.NamedSharding) -> str:
    mesh = ctx.emit_operand_atom(v.mesh)
    spec = ctx.emit_operand_atom(v.spec)
    memory_kind = ctx.emit_operand_atom(v.memory_kind)
    return f"sharding.NamedSharding({mesh}, {spec}, memory_kind={memory_kind})"
  _operand_emitter[sharding.NamedSharding] = emit_NamedSharding

  from jax import lax
  _operand_emitter[lax.AccuracyMode] = emit_enum("lax.AccuracyMode")
  _operand_emitter[lax.ConvDimensionNumbers] = emit_namedtuple("lax.ConvDimensionNumbers")
  _operand_emitter[lax.DotAlgorithm] = emit_namedtuple("lax.DotAlgorithm")
  _operand_emitter[lax.DotAlgorithmPreset] = emit_enum("lax.DotAlgorithmPreset")
  _operand_emitter[lax.FftType] = emit_namedtuple("lax.FftType")
  _operand_emitter[lax.GatherDimensionNumbers] = emit_namedtuple("lax.GatherDimensionNumbers")
  _operand_emitter[lax.GatherScatterMode] = emit_enum("lax.GatherScatterMode")
  _operand_emitter[lax.Precision] = emit_enum("lax.Precision")
  # _operand_emitter[lax.RaggedDotDimensionNumbers] =
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

class EmitGlobalContext:
  _var_for_val: dict[int, str] # id(val) -> var_name
  # by id(val) the values that appear on the left of an assignment
  _definitions: set[int]

  def __init__(self):
    self._var_for_val = {}
    self._definitions = set()
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

  def define_var_for_val(self, for_v: Any, *, prefix="v"):
    self._definitions.add(id(for_v))
    return self.var_for_val(for_v, prefix=prefix)

  def preprocess_emit_primitive(self, c: "Call") -> tuple[Any, Any, Any]:
    preproc = None  # _primitive_emitter_preprocessor.get(c.func.name, None)
    if preproc is None:
      return c.func, c.args, c.kwargs
    return preproc(self, c.func, * c.args, ** c.kwargs)


class EmitFuncContext:

  def __init__(self, global_ctx: EmitGlobalContext, call: Call):
    self.global_ctx: EmitGlobalContext = global_ctx
    self.call = call  # The body we are emitting
    self.lines: list[str] = []
    # For all the values defined here, their name (by id)
    self.definitions: dict[int, str] = {}
    self.definition_index = itertools.count()

    self.externals: dict[int, Any] = {}  # direct externals, by id
    self.all_externals: dict[int, Any] = {}  # direct and indirect externals
    self.indent: int = 0

  def emit_line(self, l: str):
    self.lines.append(" " * self.indent + l)

  def emit_operand_atom(self, v: Any) -> str | None:
    from jax._src import core  # type: ignore

    for t in type(v).__mro__:
      if (emitter := _operand_emitter.get(t)) is not None:
        if (res := emitter(self, v)) is not None:
          return res
    # I don't know a way to handle dtypes using the _operand_emitter mechanism
    if isinstance(v, dtypes.DType):
      return f"dtypes.dtype(\"{v.name}\")"
    # Constants and globals
    # if isinstance(v, (np.ndarray, np.generic)):
    #   return repr(v)
    if "scalar_types._ScalarMeta" in str(type(v)):
      return f"jnp.{v.dtype.name}"  # TODO: jnp.int32
    if isinstance(v, core.Primitive):
      return f"jax_primitive_bind(\"{v}\")"

    # Already defined here
    v_id = id(v)
    v_n = self.definitions.get(v_id)
    if v_n is not None:
      return v_n
    # already known externals
    if v_id in self.externals:
      return self.global_ctx.var_for_val(v)

    if isinstance(v, Func):
      if v.api_name:
        return v.api_name
      v_n = self.global_ctx.var_for_val(v, prefix="f")
      if v.is_jax:
        self.externals[v_id] = v
        return v_n
      # USER functions
      self.emit_operand_user_func(v)
      return v_n

    if v_id not in self.global_ctx._definitions:
      logging.warning(f"Found undefined value {v} while emitting")
      return repr(v)
    self.externals[v_id] = v
    return self.global_ctx.var_for_val(v)

  def emit_operand_sequence(self, a) -> str:
    return ", ".join(self.emit_operand(v) for v in a)

  def emit_operand_key_value_sequence(self, a) -> str:
    return ", ".join(f"{k}={self.emit_operand(v)}" for k, v in a.items())

  def emit_operand(self, a) -> str:
    if isinstance(a, (int, float, bool, str)) or a is None:
      return repr(a)
    # TODO: is there a better way to handle a NamedTuple
    if isinstance(a, tuple) and not hasattr(a, "_fields"):
      return f"({self.emit_operand_sequence(a)}{',' if a else ''})"
    if isinstance(a, list):
      return f"[{self.emit_operand_sequence(a)}]"
    if isinstance(a, dict):
      return f"dict({self.emit_operand_key_value_sequence(a)})"
    return str(self.emit_operand_atom(a))

  def emit_operand_user_func(self, f: Func) -> None:
    if not f.invocation:
      # A function that was never invoked, is emitted in the main function
      if self.call.func.is_main:
        self.emit_line(f"def {self.global_ctx.var_for_val(f, prefix='fun')}(*args, **kwargs):")
        self.emit_line("  pass")
        self.emit_line("")
      else:
        self.externals[id(f)] = f
      return
    body_lines, body_externals, body_all_externals = EmitFuncContext(self.global_ctx, f.invocation).emit_body()
    if (not self.call.func.is_main and
        not any(ae in self.definitions for ae in body_all_externals)):
      # We make this function all external
      self.externals[id(f)] = f
      for ae in body_all_externals:
        self.all_externals[id(ae)] = ae
      return
    # We can emit some external Funcs here
    for e in body_externals.values():
      if isinstance(e, Func):
        self.emit_operand_atom(e)
    for bl in body_lines:
      self.emit_line(bl)

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
                                static_argnums=(), static_argnames=()
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
  def preprocess_args_for_call(call: Call) -> tuple[Any, dict[str, Any], Any]:
    from jax._src import api_util
    """We are emitting a function call to `call`."""
    if call.func.api_name == "jax_jit_call":
      f, jit_args, jit_kwargs, *rest_args = call.args
      assert isinstance(f, Func), f
      # Drop the static_argnums and static_argnames from the jit_kwargs
      new_jit_kwargs = dict(jit_kwargs)
      if "static_argnums" in jit_kwargs: del new_jit_kwargs["static_argnums"]
      if "static_argnames" in jit_kwargs: del new_jit_kwargs["static_argnames"]

      jitted_fun_sig = api_util.fun_signature(f.fun)
      assert jitted_fun_sig, f
      static_argnums, static_argnames = api_util.infer_argnums_and_argnames(
          jitted_fun_sig, jit_kwargs.get("static_argnums"),
          jit_kwargs.get("static_argnames"))
      dyn_args, dyn_kwargs = EmitFuncContext.flatten_custom_pytree_top(
          rest_args, call.kwargs,
          static_argnums=static_argnums, static_argnames=static_argnames)
      return ((f, jit_args,EmitFuncContext.flatten_custom_pytree(new_jit_kwargs),
               *dyn_args), dyn_kwargs,
              EmitFuncContext.flatten_custom_pytree(call.result))

    return (EmitFuncContext.flatten_custom_pytree(call.args),
            EmitFuncContext.flatten_custom_pytree(call.kwargs),
            EmitFuncContext.flatten_custom_pytree(call.result))


  @staticmethod
  def preprocess_args_for_body(call: Call) -> tuple[Sequence[Any], dict[str, Any], Any]:
    """We are emitting the body of `call`."""
    from jax._src import api_util
    assert not call.func.is_jax, call
    if call.func.is_main: return call.args, call.kwargs, call.result
    assert call.func.introduced_for is not None, call

    if call.func.introduced_for.func.api_name == "jax_jit_call":
      jitted_fun, jit_args, jit_kwargs, *_ = call.func.introduced_for.args
      assert not jit_args
      jitted_fun_sig = api_util.fun_signature(jitted_fun.fun)
      assert jitted_fun_sig, jitted_fun
      static_argnums, static_argnames = api_util.infer_argnums_and_argnames(
          jitted_fun_sig, jit_kwargs.get("static_argnums"),
          jit_kwargs.get("static_argnames"))
      dyn_args, dyn_kwargs = EmitFuncContext.flatten_custom_pytree_top(
          call.args, call.kwargs,
          static_argnums=static_argnums, static_argnames=static_argnames)
      new_jit_kwargs = dict(jit_kwargs)
      if "static_argnums" in new_jit_kwargs: del new_jit_kwargs["static_argnums"]
      if "static_argnames" in new_jit_kwargs: del new_jit_kwargs["static_argnames"]
      return dyn_args, dyn_kwargs, EmitFuncContext.flatten_custom_pytree(call.result)

    return (EmitFuncContext.flatten_custom_pytree(call.args),
            EmitFuncContext.flatten_custom_pytree(call.kwargs),
            EmitFuncContext.flatten_custom_pytree(call.result))

  def emit_body(self) -> tuple[list[str], dict[int, Any], dict[int, Any]]:
    """Emits the body of the function.

    This is called starting with the main function.
    Returns:
      the source lines of the emitted body
      the externals referenced directly in the emitted body
      the transitive closure of the externals in the emitted body and functions
        referenced from the emitted body.
    """
    # Emits the body and returns the lines of the emitted body,
    # the immediate externals, and all the nested externals
    call = self.call
    assert not call.func.is_jax

    def val_is_leaf(v):
      leaves = tree_util.tree_leaves(v)
      return (len(leaves) == 1 and leaves[0] is v)

    self.emit_line("")
    self.emit_line(f"# body from invocation {call}")
    call_args, call_kwargs, call_result = EmitFuncContext.preprocess_args_for_body(call)
    if all(val_is_leaf(v) for v in list(call_args) + list(call_kwargs.values())):
      args_str = ""
      args_used_here: set[str] = set()
      for aidx, a in enumerate(call_args):
        an = self.global_ctx.var_for_val(a)
        if an in args_used_here:
          # In some rare cases, e.g., test_custom_jvp_defjvps_0 a user function is
          # called with two identical arguments. Ignore, duplicates in the definition.
          an = f"{an}_duplicate_{aidx}"
        else:
          self.definitions[id(a)] = an
        args_used_here.add(an)
        if args_str: args_str += ", "
        args_str += an
      if call_kwargs:
        args_str += ", *"
      for k, a in call_kwargs.items():
        an = self.global_ctx.var_for_val(a)
        self.definitions[id(a)] = k
        if args_str: args_str += ", "
        args_str += f"{k}"
      self.emit_line(f"def {self.global_ctx.define_var_for_val(call.func)}({args_str}):")
      self.indent += 2
    else:
      self.emit_line(f"def {self.global_ctx.define_var_for_val(call.func)}(*args, **kwargs):")
      self.indent += 2
      # unpack the pytrees in args and kwargs
      # TODO: clean
      for path, a in tree_util.tree_leaves_with_path((call_args, call_kwargs),
                                                     is_leaf=lambda n: (n is None or
                                                                        isinstance(n, tree_util.Partial) or
                                                                        isinstance(n, FuncPyTreeNode))):
        self.definitions[id(a)] = self.global_ctx.var_for_val(a)
        path_str = f"{'args' if path[0].idx == 0 else 'kwargs'}{tree_util.keystr(path[1:])}"
        self.emit_line(f"{self.global_ctx.define_var_for_val(a)} = {path_str}")
    del call_args, call_kwargs

    result = None  # The last result
    for c in call.body:
      func = c.func
      if isinstance(c.func, Func):
        args, kwargs, result = EmitFuncContext.preprocess_args_for_call(c)
      else:
        args, kwargs, result = c.args, c.kwargs, c.result
      if result is not None:
        res_is_leaf = val_is_leaf(result)
        if res_is_leaf:
          res_str = f"{self.global_ctx.define_var_for_val(result)}"
        else:
          res_str = "result"
      else:
        res_is_leaf = True
        res_str = "_"

      callee_name = self.emit_operand_atom(func)
      call_args_str = self.emit_operand_sequence(args)
      if kwargs:
        call_args_str += (", " if args else "") + self.emit_operand_key_value_sequence(kwargs)
      self.emit_line(f"{res_str} = {callee_name}({call_args_str})  # {c}")
      if result is not None:
        if res_is_leaf:
          self.definitions[id(c.result)] = res_str
        else:
          # TODO: clean
          for path, r in tree_util.tree_leaves_with_path(result,
                                                         is_leaf=lambda n: (n is None or
                                                                            isinstance(n, tree_util.Partial) or
                                                                            isinstance(n, FuncPyTreeNode))):
            r_n = self.global_ctx.define_var_for_val(r)
            self.definitions[id(r)] = r_n
            path_str = f"result{tree_util.keystr(path)}"
            self.emit_line(f"{r_n} = {path_str}")

    # The end of the function
    if call.func.is_main:
      assert call_result is None
      if result is None:  # Return the last result
        last_result = "_"
      elif val_is_leaf(result):
        last_result = self.global_ctx.var_for_val(result)
      else:
        last_result = "result"
      self.emit_line(f"return {last_result}")
    else:
      self.emit_line(f"return {self.emit_operand(call_result)}")
    for e in self.externals.values():
      self.all_externals[id(e)] = e
    return self.lines, self.externals, self.all_externals


def _emit_repro(top_entry: Call, extra_message: str):
  assert config.repro_dir.value
  dump_to = config.repro_dir.value
  # dump_to = os.getcwd()
  if not (out_dir := path.make_jax_dump_dir(dump_to)):
    return None
  fresh_id = itertools.count()
  while True:
    repro_path = out_dir / f"{top_entry.func.python_name()}_{next(fresh_id)}_repro.py"
    if not os.path.exists(repro_path):
      break
  assert not top_entry.func.is_jax
  # to_emit: list[Func] = _collect_repro_funcs(top_entry)
  # Collect the relevant functions, functions earlier in the list may depend
  # on function later in the list. We will emit the code starting with the
  # last function.
  global_ctx = EmitGlobalContext()
  main_invocation = top_entry
  if not top_entry.func.is_main:
    main_invocation = top_entry.parent.parent
  assert main_invocation.parent is None
  main_lines, main_externals, main_all_externals = EmitFuncContext(global_ctx, main_invocation).emit_body()
  preamble = """
# This file was generated by JAX repro extractor.

import jax

from jax._src.repro_runtime import *

# Repro was saved due to:
"""
  preamble += "\n".join([f"  # {l}" for l in extra_message.split("\n")]) + "\n\n"
  postamble = f"""

_repro_result = {main_invocation.func.python_name()}()
"""
  repro_source = preamble + "\n".join(main_lines) + postamble
  logging.warning(f"Dumped JAX repro at {repro_path} due to {extra_message[:128]}...")
  repro_path.write_text(repro_source)
  if main_all_externals:
    msg = ("Got undefined symbols: " +
           ", ".join(f"{e} = {global_ctx.var_for_val(e)}"
                     for e in main_all_externals.values()) +
           f"\nThe repro has been saved to {repro_path}.")
    _thread_local_state.warn_or_error(msg)

  _thread_local_state.emitted_repros.append((str(repro_path), repro_source))
  return repro_path, repro_source

_thread_local_state.initialize()

"""
TODO:
* print for function definitions the static arguments, as comments
* maybe print also the avals
* why does the trampoline make it into the value_and_grad test?
* share the trampolines with the runtime
* actual devices use jax_get_device, we need a way to make it platform
  agnostic
* print ndarray properly, when using repr(...) it only prints a summary.


What are we missing:
* caching is changed (mostly defeated)
* named_call is not handled
* we emit jax.Array as np.ndarray (loosing sharding)
"""
