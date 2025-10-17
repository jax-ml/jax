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

import collections
import contextlib
import dataclasses
import functools
import inspect
import itertools
import logging
import re
import threading
from typing import Any, Callable, Union
from functools import partial
import traceback

from jax._src import config
from jax._src.traceback_util import enable_repro, repro_enabled
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
  # Run some invariant checks while tracing to detect early errors, also enable
  # logging of repro operations.
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

@contextlib.contextmanager
def flags_override(*flag_values):
  """Overrides ReproFlags.

  Usage: flags_override(enable_checks=True, log_traceback_frames=40)
  """
  if len(flag_values) % 2:
    raise ValueError("flag_values must have an even number of elements")
  prev = {}
  for i in range(len(flag_values) // 2):
    flag_name, flag_value = flag_values[i * 2], flag_values[i * 2 + 1]
    prev[flag_name] = getattr(_thread_local_state.flags, flag_name)
    setattr(_thread_local_state.flags, flag_name, flag_value)
  try:
    yield
  finally:
    for flag_name, prev in prev.items():
      setattr(_thread_local_state.flags, flag_name, prev)


class _ThreadLocalState(threading.local):
  def __init__(self):
    self.stack_node_id = collections.defaultdict(itertools.count)
    self.call_stack: list["Call"] = []

    self.flags = ReproFlags()

    # We are in the process of emitting repros. We defer errors
    self.emitting_repro: bool = False
    self.emitting_repro_had_errors: bool = False

    # The path and source for the last repro saved
    self.last_repro: tuple[str, str] | None = None

    # For simpler debugging, use our own small value ids when printing
    self.small_id_index = itertools.count()
    self.small_id_map: dict[int, int] = {}  # id(v) -> small id
    self.func_index = itertools.count()  # Each constructed Func has an index
    self.call_index = itertools.count()  # Each constructed Call has an index

    self.emit_call_preprocessor: Callable = lambda *args: args[1:]
    self.undefined_value_handler: Callable[..., str] | None = None

  def initialize_state(self):
    """Deferred per-thread initialization, only once we start using.
    This ensures that Func is available
    """
    if self.call_stack: return
    def main(): return
    main_func = Func(main, is_jax=False)
    main_func.is_main = True
    self.push_call(main_func, (), {})

  def snapshot_state(self):
    self.initialize_state()
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
      self.initialize_state()

    call = Call(len(self.call_stack),
                self.call_stack[-1] if self.call_stack else None,
                func, args, kwargs)
    self.call_stack.append(call)
    return call

  def warn_or_error(self, msg,
                    traceback=None,
                    warning_only=False):
    logging.error("Repro error: %s", msg)
    if self.flags.enable_checks_as_errors and not warning_only:
      if not _thread_local_state.emitting_repro:
        raise ReproError(msg)  # This will include the current trace

      _thread_local_state.emitting_repro_had_errors = True
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


class Func:
  MISSING_API_NAME = "MISSING_API_NAME"

  def __init__(self, fun: Callable, *, is_jax: bool,
               api_name: str | None = None,
               map_user_funcs: Callable | None = None):
    self.fun = fun
    self.fun_info = _fun_sourceinfo(fun)
    self.fun_name = self.fun_info.split(" ")[0]
    self.is_jax = is_jax
    self.id = next(_thread_local_state.func_index)
    self.is_main = False  # Whether this is the top-level function
    assert not (api_name and not is_jax), (fun, api_name)
    self.api_name = api_name
    if map_user_funcs is None:  # See docstring for repro_boundary
      self.map_user_funcs = (
          lambda to_apply, *args, **_:
            tuple(to_apply(a) if callable(a) and not isinstance(a, FuncPartial) else a
                  for a in args))
    else:
      self.map_user_funcs = map_user_funcs
    self.__wrapped__ = fun  # Enable fun_sourceinfo(self)
    if hasattr(fun, "__name__"):
      setattr(self, "__name__", fun.__name__)
    if hasattr(fun, "__qualname__"):
      setattr(self, "__qualname__", fun.__name__)

    # For calls to USER functions.
    self.invocation: Union["Call", None] = None

    fun_info_with_id = self.fun_info.replace(
        self.fun_info.split(" ")[0],
        f"{self.fun_name}<{Call.val_id(fun)}>")
    if _thread_local_state.flags.enable_checks:
      logging.info(f"Created {self} for {fun_info_with_id}")

  def __call__(self, *args, **kwargs):
    return repro_boundary(self.fun, func=self)(*args, **kwargs)

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
      if self.api_name == Func.MISSING_API_NAME:
        assert False, (self.fun, self.fun_info)
        logging.error(f"missing api_name: {self.fun}, {self.fun_info}")
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
    return f"{'JAX' if self.is_jax else 'USER'}[{self.python_name()}]"

# TODO: solve this reference
_emit_repro = None


@tree_util.register_pytree_node_class
class FuncPartial(Func):
  """A Func wrapper for a Partial, returned by some JAX APIs"""
  def __init__(self, func: tree_util.Partial, is_jax: bool):
    super().__init__(func, is_jax=is_jax)
    self.args = func.args  # Some tests are looking for Partial attributes
    self.func = func.func
    self.keywords = func.keywords

  def tree_flatten(self):
    return ((self.fun,), self)

  @classmethod
  def tree_unflatten(cls, aux_data: "FuncPartial", children):
    return cls(children[0], is_jax=aux_data.is_jax)


class Call:

  def __init__(self, level: int, parent: Union["Call", None],
               func: Func,
               args: tuple[Any, ...], kwargs: dict[str, Any]):
    self.id = next(_thread_local_state.call_index)
    self.level: int = level  # 0 is the bottom of the stack (first call)
    self.parent: Union["Call", None] = parent
    self.func: Func = func
    self.args = args
    self.kwargs = kwargs
    # Collect the traceback before we are making the call
    self.raw_traceback = _current_traceback()
    self.body : list["Call"] = []
    self.result: Any | None = None  # None is used also for functions that result in error

    # Cache here the emitted function body, for user functions
    self.emitted_function: Union["EmittedFunction", None] = None  # type: ignore # noqa: F821

    not _thread_local_state.flags.enable_checks or check_traceback(self.raw_traceback.frames)  # type: ignore

  def __repr__(self):
    return f"[{self.level}.{self.id}] {self.func}"

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

  def add_call_to_body(self, c: "Call"):
    assert not self.func.is_jax
    self.emitted_function = None
    self.body.append(c)

  @staticmethod
  def start_call(args, kwargs, *, func: "Func"):
    """Start a call, and push a Call onto the call_stack."""
    wrapped_args = args

    if func.api_name:
      def do_wrap(f):
        if callable(f):
          return wrap_callable(f, is_jax=False)
        else:
          return f
      wrapped_args = func.map_user_funcs(do_wrap, *args, **kwargs)

    # Make copies of the arguments in case they contain mutable parts; we will
    # need the originals when we emit repros. Don't instantiate new FuncPartial
    l, t = tree_util.tree_flatten((wrapped_args, kwargs),
                                  is_leaf=lambda v: isinstance(v, FuncPartial))
    wrapped_args_copy, kwargs_copy = t.unflatten(l)
    del l, t

    call = _thread_local_state.push_call(func, wrapped_args_copy, kwargs_copy)
    if not func.is_jax:
      if func.invocation:
        # We know of cases when this happens, e.g., higher-order custom_vjp,
        # fusions
        _thread_local_state.warn_or_error(
            f"Ignoring additional invocation {call}. "
            f"Previous invocation was {func.invocation}",
            warning_only=True, traceback=call.raw_traceback)
      else:
        func.invocation = call

    if _thread_local_state.flags.enable_checks:
      # Log the Func passed as args
      args_to_print = [a if isinstance(a, Func) else "" for a in wrapped_args]
      args_str = ", ".join(str(a) for a in args_to_print)
      args_str = re.sub(r'(, )+', ", ...", args_str) or "..."
      logging.info(f"{'  ' * call.level} start {call}({args_str})")
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

        call.result = res
        if call.func.is_jax:  # TODO: ideally we want only api_name function to return functions
          # Look through the result of JAX functions and wrap callables as
          # Func. If we are returning functions, they are either the sole
          # result or in a tuple
          results: list[Any] = list(res) if isinstance(res, tuple) else [res]
          for i, r in enumerate(results):
            if callable(r):
              results[i] = wrap_callable(r, is_jax=True)
          # Make copies in case the user function returns mutable values
          call.result = tuple(results) if isinstance(res, tuple) else results[0]

        if _thread_local_state.flags.enable_checks:
          res_str = "res=Ok"
      else:
        call.result = None
        exc_str = str(exc)
        if call.level > 1:
          exc_str = exc_str[:1024] + "\n...."
        if _thread_local_state.flags.enable_checks:
          res_str = f"exc={exc_str}"

      if _thread_local_state.flags.enable_checks:
        logging.info(f"{'  ' * call.level} end {call}: {res_str}")
      call_stack.pop()

      caller: Call = call.parent  # type: ignore
      assert caller is not None
      if not caller.func.is_jax:
        caller.add_call_to_body(call)

      if caller.func.is_main:
        if (exc is not None and not isinstance(exc, ReproError)):
          # We caught an exception and we are at top-level
          from jax._src.repro.emitter import emit_repro
          emit_repro(caller, "uncaught exception " + traceback.format_exc(),
                     repro_name_prefix="jax_error_repro")
        # Drop the last call from the main body, after emitting the repro,
        # so that we don't keep growing the main body.
        returned_leaves = tree_util.tree_leaves(call.result, is_leaf=lambda l: isinstance(l, FuncPartial))
        if (all(not isinstance(l, (FuncPartial, Func)) for l in returned_leaves) and
            not _thread_local_state.flags.collect_repro_on_success):
          if _thread_local_state.flags.enable_checks:
            logging.info(f"{'  ' * call.level} popping from main {call}")
          caller.body.pop()

      return call.result
    except Exception as e:
      # Exceptions here are bad, because they break the call_stack invariants
      logging.error(f"Exception caught in the exit handler: {type(e)}: {e}\n{traceback.format_exc()}")
      raise e


def true_bind_primitive(prim: Primitive, args, params):
  # Replacement for Primitive._true_bind when using repros
  if not repro_enabled():
    return prim._true_bind_internal(*args, **params)
  if not _thread_local_state.call_stack:  # The first call on this thread
    _thread_local_state.initialize_state()
  stack_entry = _thread_local_state.call_stack[-1]
  if stack_entry.func.is_jax:
    return prim._true_bind_internal(*args, **params)
  # We should not be seeing higher-order primitives in USER functions
  if (_thread_local_state.flags.enable_checks and
      _higher_order_primitive(prim, args, params)):
    _thread_local_state.warn_or_error(
        f"Encountered primitive {prim} containing Jaxprs or functions. This means "
        "that there is a higher-order JAX API that is not "
        "annotated with traceback_util.api_boundary(repro_api_name=...)")

  c = Call(stack_entry.level + 1, stack_entry, prim, args, params)
  stack_entry.add_call_to_body(c)
  with enable_repro(False):
    res = prim._true_bind_internal(*args, **params)

  c.result = res  # On success
  # TODO: We shouldn't collect primitive_bind inside main
  return res


# See comments in repro_api.py
boundary_trampolines: dict[str, Callable[..., Any]] = {}


def repro_boundary(boundary_fun: Callable, *,
                   is_jax: bool = True,
                   func: Func | None = None,
                   api_name: str | None = None,
                   map_user_funcs: Callable | None = None,
                   ):
  """
  Args:
    boundary_fun: the function that is called at the boundary of the JAX/USER
      region.
    is_jax: whether this should be treated as a JAX function.
    func: the boundary object that is being entered. It is None for JAX
      API functions. It is not None for user functions, and for JAX functions
      that are returned from a JAX API function.
    api_name: the name to be emitted for this function in repros.
    map_user_funcs: a function to return a replacement for the tuple of
      positional arguments, invoked `map_user_funcs(wrap_func, *args, *kwargs)'
      where `args` is the tuple of positional arguments and `wrap_func` is a
      function to wrap function arguments as USER functions. For now we assume
      that JAX APIs can have USER functions only among the positional arguments.
      If absent, then it looks for all callables among `args`.
  """
  if not repro_enabled(): return boundary_fun
  if api_name is not None and (trampoline := boundary_trampolines.get(api_name)) is not None:
    return trampoline(boundary_fun)

  if func is None:
    func = Func(boundary_fun, is_jax=is_jax, api_name=api_name,
                map_user_funcs=map_user_funcs)

  @functools.wraps(boundary_fun)
  def wrapper(*args, **kwargs):
    if not repro_enabled():
      return boundary_fun(*args, **kwargs)
    # API functions that do not return JAX functions can be skipped if
    # called for JAX-internal functions.
    if api_name in ["jax.lax.scan", "jax.lax.while_loop", "jax.lax.fori_loop",
                    "jax_vmap_call"]:
      has_user_func = False
      def find_user_func(a):
        nonlocal has_user_func
        has_user_func |= ("/jax/_src/" not in _fun_sourceinfo(a) and
                          not isinstance(a, tree_util.Partial))
      func.map_user_funcs(find_user_func, *args, **kwargs)
      if not has_user_func:
        if _thread_local_state.flags.enable_checks:
          logging.info(f"Ignoring call to {api_name} from within JAX source")
        return boundary_fun(*args, **kwargs)
    call_func = boundary_fun

    call_stack_level = len(_thread_local_state.call_stack)  # In case we get an exception during start_call
    try:
      wrapped_args, wrapped_kwargs = Call.start_call(args, kwargs, func=func)
      def call_user_to_jax():  # Distinctive name for traceback readability
        return call_func(*wrapped_args, **wrapped_kwargs)  # USER -> JAX
      def call_jax_to_user():
        return call_func(*wrapped_args, **wrapped_kwargs)  # JAX -> USER
      if func.is_jax:
        result = call_user_to_jax()
      else:
        result = call_jax_to_user()
      result = Call.end_call(res=result, exc=None)
      return result
    except Exception as e:
      if len(_thread_local_state.call_stack) > call_stack_level:
        Call.end_call(res=None, exc=e)

      raise

  wrapper.real_boundary_fun = boundary_fun
  return wrapper


def wrap_callable(f: Callable, *, is_jax: bool):
  assert callable(f), f
  if not repro_enabled():
    return f
  if isinstance(f, Func) and f.is_jax == is_jax: return f
  if isinstance(f, tree_util.Partial):
    assert is_jax  # In returned values
    res = FuncPartial(f, is_jax=is_jax)
    return res

  res = Func(f, is_jax=is_jax)  # type: ignore
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
  from jax._src.repro.repro_api import jax_jit_call, jax_jit_aot_trace_call, jax_jit_aot_lower_call
  from jax._src.repro.repro_api import jax_pjit_call, jax_pjit_aot_trace_call, jax_pjit_aot_lower_call
  from jax._src import mesh as mesh_lib

  def get_ctx_mesh() -> mesh_lib.Mesh | None:
    if is_pjit:
      ctx_mesh = mesh_lib.thread_resources.env.physical_mesh
    else:
      ctx_mesh = mesh_lib.get_concrete_mesh()
    return ctx_mesh if not ctx_mesh.empty else None

  def jax_jit_trampoline(fun: Callable, *jit_args, **jit_kwargs):
    assert not jit_args, f"jit_args not implemented: {jit_args}"

    # Ignore calls from xla_primitive_callable which use jit over an internal
    # function that just binds the primitive
    if getattr(fun, "_apply_primitive", None):
      return real_jit(fun, *jit_args, **jit_kwargs)

    def jax_jit_call_trampoline(*args, **kwargs):
      to_call = jax_pjit_call if is_pjit else jax_jit_call
      return to_call(fun, get_ctx_mesh(), jit_args, jit_kwargs, *args, **kwargs)

    def jax_jit_aot_trace_trampoline(*args, **kwargs):
      to_call = jax_pjit_aot_trace_call if is_pjit else jax_jit_aot_trace_call
      return to_call(fun, get_ctx_mesh(), jit_args, jit_kwargs, *args, **kwargs)

    jax_jit_call_trampoline.trace = jax_jit_aot_trace_trampoline

    def jax_jit_aot_lower_trampoline(*args, **kwargs):
      to_call = jax_pjit_aot_lower_call if is_pjit else jax_jit_aot_lower_call
      return to_call(fun, get_ctx_mesh(), jit_args, jit_kwargs, *args, **kwargs)

    jax_jit_call_trampoline.lower = jax_jit_aot_lower_trampoline
    return jax_jit_call_trampoline

  jax_jit_trampoline.real_boundary_fun = real_jit
  return jax_jit_trampoline

boundary_trampolines["jax.jit"] = partial(jit_trampoline, False)
boundary_trampolines["pjit.pjit"] = partial(jit_trampoline, True)

def custom_jvp_trampoline(real_boundary_fun: Callable):
  from jax._src.repro.repro_api import jax_custom_jvp_call

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

  jax_custom_jvp_call_trampoline.real_boundary_fun = real_boundary_fun
  return jax_custom_jvp_call_trampoline

boundary_trampolines["jax.custom_jvp.__call__"] = custom_jvp_trampoline

def custom_vjp_trampoline(real_boundary_fun: Callable):
  from jax._src.repro.repro_api import jax_custom_vjp_call
  def jax_custom_vjp_call_trampoline(*args, **kwargs):
    cvjp_orig, *rest_args = args
    return jax_custom_vjp_call(cvjp_orig.fun, cvjp_orig.fwd, cvjp_orig.bwd,
                               dict(nondiff_argnums=cvjp_orig.nondiff_argnums),
                               *rest_args, **kwargs)
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
  from jax._src.repro.repro_api import pallas_custom_fusion_call

  def custom_fusion_trampoline(*args, **kwargs):
    cfus, *rest_args = args
    return pallas_custom_fusion_call(
        cfus.fun, cfus.eval_rule, cfus.pull_block_spec_rule,
        cfus.push_block_spec_rule, cfus.pallas_impl, *rest_args, **kwargs)

  custom_fusion_trampoline.real_boundary_fun = real_boundary_fun
  return custom_fusion_trampoline

boundary_trampolines["jax.pallas.custom_fusion.__call__"] = pallas_custom_fusion_trampoline

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
* scatter carries a Jaxprx
* scatter_p exits in SC also, with the same name, e.g., addupdate_scatter,
  store_scatter
* the test_scatter_add looses the ScatterDimensionNumbers, because the
  emit_custom_pytree already takes the leaves. Maybe we should first try
  the emitters.
"""
