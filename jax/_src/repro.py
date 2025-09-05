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
import dataclasses
import functools
import inspect
import itertools
import logging
import os.path
import re
import threading
from typing import Any, Callable, Union
from functools import partial
import traceback

import numpy as np

from jax._src import config
from jax._src import dtypes
from jax._src import path

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
  from jax._src.lib import xla_client
  # Use directly xla_client, otherwise we may get cached tracebacks
  return xla_client.Traceback.get_traceback().get_traceback()

def maybe_singleton(x: list[Any]) -> Any | tuple[Any, ...]:
  return x[0] if len(x) == 1 else tuple(x)

@dataclasses.dataclass
class ReproFlags:
  # Run some invariant checks while tracing to detect early errors
  enable_checks = False
  # Replace arrays with `np.ones`
  fake_arrays = False
  # Squash sequences of array primitives, preserving the dependencies
  squash_primitives = False

  def __init__(self):
    if config.repro_flags.value:
      for f in config.repro_flags.value.split(","):
        if f == "enable_checks":
          self.enable_checks = True
        else:
          raise NotImplementedError(f"--jax_repro_flags: {f}")


class _ThreadLocalState(threading.local):
  def __init__(self):
    self.stack_node_id = collections.defaultdict(itertools.count)
    self.call_stack: list["Call"] = []

    self.flags = ReproFlags()
    self.emit_repro_enabled = True
    self.emit_repro_on_success = False
    # List of path names and source codes
    self.emitted_repros: list[tuple[str, str]] = []

    # For simpler debugging, use our own small value ids when printing
    self.small_id_index = itertools.count()
    self.small_id_map: dict[int, int] = {}  # id(v) -> small id
    self.func_index = itertools.count()  # Each constructed Func has an index
    self.call_index = itertools.count()  # Each constructed Call has an index

  def initialize(self):
    def main(): return
    main_func = Func(main, is_jax=False)
    main_func.is_main = True
    self.push_call(main_func, (), {}, (), {})

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

  def push_call(self, func: "Func", args, kwargs,
                orig_args, orig_kwargs) -> "Call":
    call = Call(len(self.call_stack),
                self.call_stack[-1] if self.call_stack else None,
                func, args, kwargs, orig_args, orig_kwargs)
    self.call_stack.append(call)
    return call


_thread_local_state = _ThreadLocalState()

class Func:
  MISSING_API_NAME = "MISSING_API_NAME"

  def __init__(self, fun: Callable, *, is_jax: bool,
               api_name: str | None = None, api_constructor: bool = False,
               func_argnums: tuple[int, ...] = (0,),
               fun_name_prefix = "",
               introduced_for: Union["Call"] | None = None):

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
      # def func_lower(*args, **kwargs):
      #   return aot_lower(self, *args, **kwargs)
      # self.lower = func_lower
      # def func_trace(*args, **kwargs):
      #   return aot_trace(self, *args, **kwargs)
      # self.trace = func_trace
      # self.eval_shape = (
      #     lambda *args, **kwargs: aot_eval_shape(self, *args, **kwargs))
      self.clear_cache = fun.clear_cache
      self._cache_size = fun._cache_size

    # For calls to USER functions.
    self.invocation: Call | None = None
    # The USER functions are introduced as arguments for API calls, and JAX
    # functions are results of API calls. This value is None only for API functions.
    self.introduced_for: Call | None = introduced_for

    fun_info_with_id = self.fun_info.replace(self.fun_info.split(" ")[0],
                                             f"{self.fun_name}<{Call.val_id(fun)}>")
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
                       fun_name_prefix: str = "",
                       ):
    from jax._src import tree_util
    assert callable(f), f
    if isinstance(f, Func) and f.is_jax == is_jax: return f
    if isinstance(f, tree_util.Partial):
      assert is_jax  # In returned values
      res = tree_util.Partial(
          Func.wrap_func_in_out(f.func, is_jax=True,
                                fun_name_prefix=fun_name_prefix,
                                introduced_for=introduced_for,
                                accum=accum),
          * f.args, ** f.keywords)
      if isinstance(f.func, tree_util._HashableCallableShim):
        # There are some api_test.py that look in here
        res.func.args = f.func.args
    res = Func(f, is_jax=is_jax, fun_name_prefix=fun_name_prefix,
               introduced_for=introduced_for)
    if accum is not None: accum.append(res)
    return res

def clone_func(func: Func):
  assert func.introduced_for is not None
  assert func.introduced_for.func.api_name
  introduced_for = func.introduced_for
  # Call again the producer of this JAX function
  logging.info(f"{'  ' * len(_thread_local_state.call_stack)} create a new version of {func}")
  new_func = introduced_for.func(*introduced_for.orig_args, **introduced_for.orig_kwargs)
  # TODO: fix this
  if isinstance(new_func, Func):
    assert introduced_for.result is func
    func = new_func
    override_fun = new_func.fun
  elif isinstance(new_func, tuple):
    assert isinstance(introduced_for.result, tuple) and len(new_func) == len(introduced_for.result)
    if introduced_for.result[0] is func:
      func = new_func[0]
      override_fun = new_func[0].fun
    elif introduced_for.result[1] is func:
      func = new_func[1]
      override_fun = new_func[1].fun
    else:
      assert False, (introduced_for.result, new_func)
  else:
    assert False, (introduced_for.result, new_func)

  return func, override_fun


class Call:

  def __init__(self, level: int, parent,
               func: Func,
               args: tuple[Any, ...], kwargs: dict[str, Any],
               orig_args: tuple[Any, ...], orig_kwargs: dict[str, Any],):
    self.id = next(_thread_local_state.call_index)
    self.level: int = level  # 0 is the top of the stack
    self.parent = parent
    self.func: Func = func
    self.args = args
    self.kwargs = kwargs
    self.orig_args = orig_args
    self.orig_kwargs = orig_kwargs
    self.body : list["Call"] = []
    self.result: Any | None = None
    self.source_info = current_traceback()
    self.computed_values = []

  @staticmethod
  def val_id(v):
    return _thread_local_state.small_id(id(v))

  def __repr__(self):
    return f"[{self.level}.{self.id}] {self.func}"

  @staticmethod
  def pytree_to_str(t):
    from jax import tree_util  # type: ignore
    def a_to_str(a):
      if hasattr(a, "shape"):
        typ = "tracer" if hasattr(a, "_trace") else "arr"
        return f"{typ}[{','.join(str(d) for d in a.shape)}]<{Call.val_id(a)}>"
      if isinstance(a, Func):
        return str(a)
      if isinstance(a, (int, float, bool, str)):
        return str(a)
      return f"{a}<{Call.val_id(a)}>"
    return tree_util.tree_map(a_to_str, t)

  @staticmethod
  def start_call(args, kwargs, *, func: "Func"):
    introduced_funcs: list[Func] = []
    override_fun = None
    wrapped_args = args
    if func.api_name:
      if func.api_name == "jax_defjvps":
        func_argnums = []
        for i, a in enumerate(args):
          if i == 0: continue
          if callable(a):
            func_argnums.append(i)
      else:
        func_argnums = func.func_argnums
      wrapped_args = list(args)
      for i in func_argnums:
        wrapped_args[i] = Func.wrap_func_in_out(wrapped_args[i], is_jax=False,
                                                accum=introduced_funcs)
    elif func.is_jax:
      func, override_fun = clone_func(func)

    wrapped_kwargs = kwargs

    args_str = Call.pytree_to_str(wrapped_args)
    kwargs_str = Call.pytree_to_str(wrapped_kwargs)
    call = _thread_local_state.push_call(func,
                                         wrapped_args, wrapped_kwargs,
                                         # TODO: wrapped_kwargs == kwargs
                                         args, kwargs)
    for f in introduced_funcs:
      f.introduced_for = call
    if not func.is_jax:
      if func.invocation:
        logging.warning(f"Adding additional invocation to {func}")
      func.invocation = call

    logging.info(f"{'  ' * call.level} start {call}(*{args_str}, **{kwargs_str})")
    return override_fun, wrapped_args, wrapped_kwargs, call

  def end_call(self, *, res, exc: Exception | None):
    call_stack = _thread_local_state.call_stack
    try:
      assert call_stack
      assert self is call_stack[-1], (self, call_stack[-1])
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

            # TODO: we should not need the prefix, we have introduced_for.
            fun_name_prefix = ""
            if self.func.api_name == "jax.jit":
              fun_name_prefix = "jit_"
            elif self.func.api_name == "jax.jvp":
              fun_name_prefix = "jvp_"
            elif self.func.api_name == "jax.grad":
              fun_name_prefix = "grad_"
            elif self.func.api_name == "jax.vjp":
              fun_name_prefix = "vjp_"
            elif self.func.api_name == "jax.checkpoint":
              fun_name_prefix = "checkpoint_"

            for i, r in enumerate(results):
              if callable(r):
                results[i] = Func.wrap_func_in_out(r, is_jax=True,
                                                   fun_name_prefix=fun_name_prefix,
                                                   introduced_for=self)
            if multiple_results:
              self.result = tuple(results)
            else:
              self.result = results[0]

        res_str = f"res={Call.pytree_to_str(self.result)}"
      else:
        self.result = None
        exc_str = str(exc)
        if self.level > 1:
          exc_str = exc_str[:1024] + "\n...."
        res_str = f"exc={exc_str}"
        res_wrapped = None
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
    c = Call(self.level + 1, self, prim, args, params, args, params)
    c.result = outs
    self.body.append(c)

def _true_bind_primitive(prim: Primitive, args, params):
  # Replacement for Primitive._true_bind when using repros
  if not config.repro_dir.value or not _thread_local_state.emit_repro_enabled:
    return prim._true_bind(*args, **params)
  not _thread_local_state.flags.enable_checks or check_traceback()
  stack_entry = _thread_local_state.call_stack[-1]
  if stack_entry.func.is_jax:
    return prim._true_bind(*args, **params)
  try:
    res = prim._true_bind(*args, **params)
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
                 func_argnums: tuple[int, ...] = (0,),
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
  if func is None:
    func = Func(boundary_fun, is_jax=is_jax, api_name=api_name,
                api_constructor=api_constructor, func_argnums=func_argnums)

  @functools.wraps(boundary_fun)
  def wrapper(*args, **kwargs):
    if not config.repro_dir.value or not _thread_local_state.emit_repro_enabled:
      return boundary_fun(*args, **kwargs)
    override_fun, wrapped_args, wrapped_kwargs, call = Call.start_call(args, kwargs, func=func)
    fun = override_fun or boundary_fun
    try:
      global this_file_name, user_to_jax_line, jax_to_user_line
      if user_to_jax_line is None:
        this_frame = current_traceback().frames[1]
        this_file_name = this_frame.file_name
        user_to_jax_line = 5 + this_frame.line_num
        jax_to_user_line = 2 + user_to_jax_line
      if func.is_jax:
        result = fun(*wrapped_args, **wrapped_kwargs)  # USER -> JAX
      else:
        result = fun(*wrapped_args, **wrapped_kwargs)  # JAX -> USER
      result = call.end_call(res=result, exc=None)
      return result
    except Exception as e:
      call.end_call(res=None, exc=e)
      raise e

  return wrapper


# TODO: move these to a run time file
@partial(jax_boundary, api_name="jax_aot_trace")
def aot_trace(jitted_fun: Func, *args, **kwargs):
  new_func, _ = clone_func(jitted_fun)
  return new_func.fun.trace(*args, **kwargs)

@partial(jax_boundary, api_name="jax_aot_lower")
def aot_lower(jitted_fun: Func, *args, **kwargs):
  return jitted_fun.fun.lower(*args, **kwargs)

@partial(jax_boundary, api_name="jax_aot_eval_shape")
def aot_eval_shape(jitted_fun: Func, *args, **kwargs):
  return jitted_fun.fun.eval_shape(*args, **kwargs)


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

class EmitGlobalContext:
  _var_for_val: dict[int, str] # id(val) -> var_name
  # by id(val) the values that appear on the left of an assignment
  _definitions: set[int]

  def __init__(self):
    self._var_for_val = {}
    self._definitions = set()
    # Initialize the emitter rules
    if not _operand_emitter_initialized:
      initialize_operand_emitter()

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
    preproc = _primitive_emitter_preprocessor.get(c.func.name, None)
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
    self.externals: dict[int, Any] = {}  # direct externals, by id
    self.all_externals: dict[int, Any] = {}  # direct and indirect externals
    self.indent: int = 0

  def emit_line(self, l: str):
    self.lines.append(" " * self.indent + l)

  def emit_operand(self, v: Any) -> str | None:
    from jax._src import core
    # from jax._src import custom_derivatives
    # from jax._src import tree_util

    for t in type(v).__mro__:
      if (emitter := _operand_emitter.get(t)) is not None:
        if (res := emitter(self, v)) is not None:
          return res
    # I don't know a way to handle dtypes using tyhe _operand_emitter mechanism
    if isinstance(v, dtypes.DType):
      return f"dtypes.dtype(\"{v.name}\")"
    # Constants and globals
    if isinstance(v, (np.ndarray, np.generic)):
      return repr(v)
      return self.global_ctx.var_for_val(v, prefix="ndarray")
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
      logging.warning(f"Found undefined value {v}")
      return repr(v)
    self.externals[v_id] = v
    return self.global_ctx.var_for_val(v)

  def emit_operand_list_str(self, a) -> str:
    return ", ".join(self.emit_operand_pytree_str(v) for v in a)

  def emit_operand_dict_str(self, a) -> str:
    return ", ".join(f"{k}={self.emit_operand_pytree_str(v)}" for k, v in a.items())

  def emit_operand_pytree_str(self, a) -> str:
    if isinstance(a, tuple):
      return f"({self.emit_operand_list_str(a)}{',' if a else ''})"
    if isinstance(a, list):
      return f"[{self.emit_operand_list_str(a)}]"
    if isinstance(a, dict):
      return f"dict({self.emit_operand_dict_str(a)})"
    return str(self.emit_operand(a))

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
        self.emit_operand(e)
    for bl in body_lines:
      self.emit_line(bl)

  @staticmethod
  def flatten_custom_pytree(t) -> list[Any]:
    from jax._src import tree_util
    if isinstance(t, tuple):
      return tuple(EmitFuncContext.flatten_custom_pytree(e) for e in t)
    if isinstance(t, list):
      return list(EmitFuncContext.flatten_custom_pytree(e) for e in t)
    if isinstance(t, dict):
      return {k: EmitFuncContext.flatten_custom_pytree(e) for k, e in t.items()}
    if isinstance(t, tree_util.Partial):
      return t
    if t is None:
      return t
    return maybe_singleton(tree_util.tree_leaves(t))

  @staticmethod
  def flatten_custom_pytree_top(args, kwargs,
                                static_argnums=(), static_argnames={}) -> list[Any]:
    dyn_args = []
    dyn_kwargs = {}
    # TODO: we should try to get this directly from JAX, there is a slight
    # chance we'll do something different otherwise
    static_argnums = set(i if i >= 0 else len(args) - i
                         for i in static_argnums)
    for i, a in enumerate(args):
      if i not in static_argnums:
        dyn_args.append(EmitFuncContext.flatten_custom_pytree(a))
    for k, a in sorted(kwargs.items()):
      if k not in static_argnames:
        dyn_kwargs[k] = EmitFuncContext.flatten_custom_pytree(a)
    return dyn_args, dyn_kwargs

  @staticmethod
  def preprocess_args_for_call(call: Call) -> tuple[tuple[Any,...], dict[str, Any], Any]:
    """We are emitting a function call to `call`."""
    if call.func.api_name == "jax.jit":
      f, = call.args
      assert isinstance(f, Func), f
      # Drop the static_argnums and static_argnames
      kwargs = dict(call.kwargs)
      if "static_argnums" in kwargs: del kwargs["static_argnums"]
      if "static_argnames" in kwargs: del kwargs["static_argnames"]
      return (call.args, kwargs, EmitFuncContext.flatten_custom_pytree(call.result))

    if call.func.is_jax and call.func.introduced_for is not None and call.func.introduced_for.func.api_name == "jax.jit":
      jit_call = call.func.introduced_for
      dyn_args, dyn_kwargs = EmitFuncContext.flatten_custom_pytree_top(
          call.args, call.kwargs,
          static_argnums=jit_call.kwargs.get("static_argnums", ()),
          static_argnames=jit_call.kwargs.get("static_argnames", ()))
      return dyn_args, dyn_kwargs, EmitFuncContext.flatten_custom_pytree(call.result)

    return (EmitFuncContext.flatten_custom_pytree(call.args),
            call.kwargs,
            EmitFuncContext.flatten_custom_pytree(call.result))

  @staticmethod
  def preprocess_args_for_body(call: Call) -> tuple[tuple[Any,...], dict[str, Any], Any]:
    """We are emitting the body of `call`."""
    assert not call.func.is_jax, call
    if call.func.is_main: return call.args, call.kwargs, call.result
    assert call.func.introduced_for is not None, call

    if call.func.introduced_for.func.api_name == "jax.jit":
      jit_call = call.func.introduced_for
      dyn_args, dyn_kwargs = EmitFuncContext.flatten_custom_pytree_top(
          call.args, call.kwargs,
          static_argnums=jit_call.kwargs.get("static_argnums", ()),
          static_argnames=jit_call.kwargs.get("static_argnames", ()))
      return dyn_args, dyn_kwargs, EmitFuncContext.flatten_custom_pytree(call.result)

    return (EmitFuncContext.flatten_custom_pytree(call.args),
            call.kwargs,
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
    from jax._src import tree_util
    from jax._src import core
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
      for a in call_args:
        an = self.global_ctx.var_for_val(a)
        self.definitions[id(a)] = an
        if args_str: args_str += ", "
        args_str += an
      for k, a in call_kwargs.items():
        an = self.global_ctx.var_for_val(a)
        self.definitions[id(a)] = an
        if args_str: args_str += ", "
        args_str += f"{k}={an}"
      self.emit_line(f"def {self.global_ctx.define_var_for_val(call.func)}({args_str}):")
      self.indent += 2
    else:
      self.emit_line(f"def {self.global_ctx.define_var_for_val(call.func)}(*args, **kwargs):")
      self.indent += 2
      # unpack the pytrees in args and kwargs
      for path, a in tree_util.tree_leaves_with_path((call_args, call_kwargs),
                                                     is_leaf=lambda n: n is None or isinstance(n, tree_util.Partial)):
        self.definitions[id(a)] = self.global_ctx.var_for_val(a)
        path_str = f"{'args' if path[0].idx == 0 else 'kwargs'}{tree_util.keystr(path[1:])}"
        self.emit_line(f"{self.global_ctx.define_var_for_val(a)} = {path_str}")

    result = None  # The last result
    for c in call.body:
      # TODO: clean this up
      # if isinstance(c.func, Func) and c.func.jax_jit_kwargs:
      #   from jax._src.lib import jax_jit
      #   signature, dynargs = jax_jit.parse_arguments(
      #       c.args, tuple(c.kwargs.values()), tuple(c.kwargs.keys()),
      #       c.func.jax_jit_kwargs.get("static_argnums", ()),
      #       c.func.jax_jit_kwargs.get("static_argnames", ()),
      #       tree_util.default_registry)

      if isinstance(c.func, core.Primitive):
        func, args, kwargs = self.global_ctx.preprocess_emit_primitive(c)
        from jax._src import core
        if next(core.jaxprs_in_params(kwargs), None) is not None:
          raise ReproError(f"Seen primitive {c.func} containing Jaxprs. This means "
                           "that there is a higher-order JAX API that is not "
                           "annotated with core.jax_boundary")
        result = c.result
      else:
        assert c.func.is_jax
        args, kwargs, result = EmitFuncContext.preprocess_args_for_call(c)
        func = c.func
      if result is not None:
        res_is_leaf = val_is_leaf(result)
        if res_is_leaf:
          res_str = f"{self.global_ctx.define_var_for_val(result)}"
        else:
          res_str = "result"
      else:
        res_str = "_"

      callee_name = self.emit_operand(func)
      call_args_str = self.emit_operand_list_str(args)
      if kwargs:
        call_args_str += (", " if args else "") + self.emit_operand_dict_str(kwargs)
      self.emit_line(f"{res_str} = {callee_name}({call_args_str})  # {c}")
      if result is not None:
        if res_is_leaf:
          self.definitions[id(c.result)] = res_str
        else:
          for path, r in tree_util.tree_leaves_with_path(result,
                                                         is_leaf=lambda n: n is None or isinstance(n, tree_util.Partial)):
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
      self.emit_line(f"return {self.emit_operand_pytree_str(call_result)}")
    for e in self.externals.values():
      self.all_externals[id(e)] = e
    return self.lines, self.externals, self.all_externals


# Maps types to source code emitters
_operand_emitter: dict[Any, Callable[[EmitFuncContext, Any], str]] = {}
_operand_emitter_initialized = False  # For lazy initialization, circular imports

@dataclasses.dataclass()
class EmitLiteral:
  what: str
_operand_emitter[EmitLiteral] = lambda _, v: v.what

def initialize_operand_emitter():
  global _operand_emitter_initialized
  _operand_emitter_initialized = True

  for ty in [bool, int, float, str, type(None)]:
    _operand_emitter[ty] = lambda ctx, v: repr(v)

  from jax._src import literals
  _operand_emitter[literals.TypedNdArray] = (
      lambda ctx, v: f"literals.TypedNdArray({ctx.emit_operand(v.val)}, weak_type={v.weak_type})")

  def emit_np_array(ctx: EmitFuncContext, v) -> str:
    flags = _thread_local_state.flags
    if flags.fake_arrays:
      return "np.ones("
    return repr(v)
  _operand_emitter[np.ndarray] = emit_np_array

  from jax import sharding

  def emit_PartitionSpec(ctx: EmitFuncContext, v: sharding.PartitionSpec) -> str:
    return (f"sharding.PartitionSpec({v._partitions}, unreduced={v.unreduced}, reduced={v.reduced})")
  _operand_emitter[sharding.PartitionSpec] = emit_PartitionSpec
  def emit_AbstractMesh(ctx: EmitFuncContext, v: sharding.AbstractMesh) -> str:
    return (f"sharding.AbstractMesh({v.axis_sizes}, {v.axis_names}, "
            f"axis_types={v.axis_types}, abstract_device={v.abstract_device})")
  _operand_emitter[sharding.AbstractMesh] = emit_AbstractMesh

  def emit_NamedSharding(ctx: EmitFuncContext, v: sharding.NamedSharding) -> str:
    mesh = _operand_emitter[type(v.mesh)](ctx, v.mesh)
    spec = _operand_emitter[type(v.spec)](ctx, v.spec)
    return f"sharding.NamedSharding({mesh}, {spec}, memory_kind={v.memory_kind})"
  _operand_emitter[sharding.NamedSharding] = emit_NamedSharding


_primitive_emitter_preprocessor: dict[str, Callable] = {}
def random_seed_preproc(ctx: EmitFuncContext, prim, *args, impl):
  return prim, args, dict(impl=EmitLiteral(f"jax_random_prngs(\"{impl.name}\")"))

_primitive_emitter_preprocessor["random_seed"] = random_seed_preproc

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
    pass # assert False, msg
  _thread_local_state.emitted_repros.append((repro_path, repro_source))
  return repro_path, repro_source

_thread_local_state.initialize()
