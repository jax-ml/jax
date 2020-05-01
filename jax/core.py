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


import operator
from operator import attrgetter
from contextlib import contextmanager
from collections import namedtuple
from functools import total_ordering
import itertools as it
from weakref import ref
import threading
from typing import Dict, Generator, Iterator, Sequence, Type
import types
from typing import Any, Callable, ClassVar, Dict, List, Optional, Set

import numpy as onp

from . import dtypes
from .config import FLAGS
from . import linear_util as lu

from .util import safe_zip, safe_map, partial, curry, prod, partialmethod
from .pprint_util import pp, vcat, hcat, pp_kv_pairs, PrettyPrint

# TODO(dougalm): the trace cache breaks the leak detector. Consisder solving.
check_leaks = False

"""Disables internal invariant checks."""
skip_checks = not FLAGS.jax_enable_checks  # not __debug__  # google doesn't use -O

@contextmanager
def skipping_checks():
  """Context manager for temporarily disabling checks."""
  global skip_checks
  old_value, skip_checks = skip_checks, True
  try:
    yield
  finally:
    skip_checks = old_value

zip = safe_zip
map = safe_map


# -------------------- jaxprs --------------------

class Jaxpr(object):
  def __init__(self, constvars, invars, outvars, eqns):
    """
    Params:
      constvars: list of variables introduced for constants (either literals
        in the Python program, or the result of constant folding during the
        generation of the Jaxpr). Array constants are replaced with such variables
        while scalar constants are kept inline.
      invars: list of input variables. Together, `constvars` and `invars` are
        the inputs to the Jaxpr.
      outvars: list of output variables.
      eqns: list of equations."""
    self.constvars = list(constvars)
    self.invars = list(invars)
    self.outvars = list(outvars)
    self.eqns = list(eqns)

  def __str__(self):
    return str(pp_jaxpr(self))
  __repr__ = __str__


def subjaxprs(jaxpr: Jaxpr) -> Iterator[Jaxpr]:
  """Generator for all subjaxprs found in the params of jaxpr.eqns.
  Does not descend recursively into the found subjaxprs.
  """
  for eqn in jaxpr.eqns:
    for param in eqn.params.values():
      if type(param) is Jaxpr:
        yield param
      elif type(param) is TypedJaxpr:
        yield param.jaxpr


class TypedJaxpr(object):
  def __init__(self, jaxpr: Jaxpr, literals: Sequence,
               in_avals: Sequence['AbstractValue'], out_avals: Sequence['AbstractValue']):
    assert len(literals) == len(jaxpr.constvars)
    assert len(in_avals) == len(jaxpr.invars)

    if not skip_checks:
      in_avals_raised = [raise_to_shaped(v) for v in in_avals]
      out_avals_raised = [raise_to_shaped(v) for v in out_avals]
      exp_in_avals = [v.aval for v in jaxpr.invars]
      exp_out_avals = [v.aval for v in jaxpr.outvars]
      assert in_avals_raised == exp_in_avals, "expected: {}, got: {}".format(exp_in_avals, in_avals_raised)
      assert out_avals_raised == exp_out_avals, "expected: {}, got: {}".format(exp_out_avals, out_avals_raised)

    self.jaxpr = jaxpr
    self.literals = list(literals)
    self.in_avals = list(in_avals)
    self.out_avals = list(out_avals)

  def __iter__(self):
    return iter((self.jaxpr, self.literals, self.in_avals, self.out_avals))

  def __str__(self):
    # TODO(mattjj): improve this with type annotations?
    return str(pp_jaxpr(self.jaxpr))
  __repr__ = __str__

@curry
def jaxpr_as_fun(typed_jaxpr: TypedJaxpr, *args):
  return eval_jaxpr(typed_jaxpr.jaxpr, typed_jaxpr.literals, *args)



class JaxprEqn(namedtuple('JaxprEqn',
                          ['invars', 'outvars', 'primitive', 'params'])):
  def __repr__(self): return str(pp_eqn(self)).rstrip()

new_jaxpr_eqn = JaxprEqn


@total_ordering
class Var(object):
  # TODO(frostig,mattjj): We don't override __eq__ or __hash__, so comparison is
  # by object id, but pretty printing might collide.

  def __init__(self, count, suffix, aval):
    self.count = count
    self.suffix = suffix
    self.aval = raise_to_shaped(aval)

  def __lt__(self, other):
    if not isinstance(other, Var):
      return NotImplemented
    else:
      return (self.count, self.suffix) < (other.count, other.suffix)

  def __repr__(self):
    rem = self.count
    s = ''
    while True:
      rem, i = rem // 26, rem % 26
      s = chr(97 + i % 26) + s
      if not rem:
        break
    return s + self.suffix

def gensym(suffix):
  counter = it.count()
  return lambda aval: Var(next(counter), suffix, aval)

class Literal(object):
  __slots__ = ["val", "hash"]

  def __init__(self, val):
    self.val = val
    try:
      self.hash = hash(val)
    except TypeError:
      if type(val) in literalable_types:
        try:
          self.hash = hash((val.item(), val.dtype))
        except (TypeError, AttributeError):
          self.hash = None

  @property
  def aval(self):
    return raise_to_shaped(get_aval(self.val))

  def __hash__(self):
    assert False

  def __eq__(self, other):
    assert False

  def __repr__(self):
    if self.hash is None:
      return 'Literal(val={})'.format(self.val)
    else:
      return '{}'.format(self.val)

literalable_types: Set[type] = set()

class Primitive(object):
  multiple_results = False  # set for multi-output primitives
  call_primitive = False    # set for call primitives processed in final style
  map_primitive = False     # set for map primitives processed in final style

  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return '{}'.format(self.name)

  def bind(self, *args, **kwargs):
    assert skip_checks or all(isinstance(arg, Tracer)
                              or valid_jaxtype(arg) for arg in args), args
    top_trace = find_top_trace(args)
    if top_trace is None:
      return self.impl(*args, **kwargs)

    tracers = map(top_trace.full_raise, args)
    out_tracer = top_trace.process_primitive(self, tracers, kwargs)
    if self.multiple_results:
      return map(full_lower, out_tracer)
    else:
      return full_lower(out_tracer)

  def def_impl(self, impl):
    self.impl = impl
    return impl

  def def_abstract_eval(self, abstract_eval):
    self.abstract_eval = abstract_eval
    return abstract_eval

  def def_custom_bind(self, bind):
    self.bind = bind
    return bind

  def impl(self, *args, **kwargs):
    raise NotImplementedError("Evaluation rule for '{}' not implemented"
                              .format(self.name))

  def abstract_eval(self, *args, **kwargs):
    raise NotImplementedError("Abstract evaluation for '{}' not implemented"
                              .format(self.name))


# -------------------- lifting --------------------

# TODO(necula): this belongs next to pe.new_eqn_recipe, but is needed in
# core.py. Plan to move all these utilities to jaxpr.py.
def extract_call_jaxpr(primitive, params):
  """Extract the call primitive subjaxpr from the params.

  Returns the subjaxpr and the params without the "call_jaxpr" value. If this is
  not a call primitive then returns (None, params).
  """
  if not (primitive.call_primitive or primitive.map_primitive):
    return (None, params)
  else:
    assert "call_jaxpr" in params
    new_params = dict(params)
    del new_params["call_jaxpr"]
    return (params["call_jaxpr"], new_params)


def eval_jaxpr(jaxpr, consts, *args):
  def read(v):
    if type(v) is Literal:
      return v.val
    else:
      return env[v]

  def write(v, val):
    env[v] = val

  env = {}
  write(unitvar, unit)
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    call_jaxpr, params = extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr:
      subfuns = [lu.wrap_init(partial(eval_jaxpr, call_jaxpr, ()))]
    else:
      subfuns = []
    ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  return map(read, jaxpr.outvars)


# -------------------- tracing --------------------


class Trace:
  master: 'MasterTrace'
  level: int
  sublevel: 'Sublevel'

  def __init__(self, master: 'MasterTrace', sublevel: 'Sublevel') -> None:
    self.master = master
    self.level = master.level
    self.sublevel = sublevel

  def full_raise(self, val) -> 'Tracer':
    if not isinstance(val, Tracer):
      return self.pure(val)
    level = self.level
    sublevel = self.sublevel
    if val._trace.master is self.master:
      if val._trace.sublevel == sublevel:
        return val
      elif val._trace.sublevel < sublevel:
        return self.sublift(val)
      else:
        raise escaped_tracer_error("Can't lift sublevels {} to {}"
                                   .format(val._trace.sublevel, sublevel))
    elif val._trace.level < level:
      if val._trace.sublevel > sublevel:
        raise escaped_tracer_error("Incompatible sublevel: {}, {}"
                                   .format(val._trace, (level, sublevel)))
      return self.lift(val)
    elif val._trace.level > level:
      raise escaped_tracer_error("Can't lift level {} to {}"
                                 .format(val, self))
    else:  # val._trace.level == self.level:
      raise escaped_tracer_error("Different traces at same level: {}, {}"
                                 .format(val, self))

  def pure(self, val):
    raise NotImplementedError("must override")

  def lift(self, tracer):
    raise NotImplementedError("must override")

  def sublift(self, tracer):
    raise NotImplementedError("must override")

  def process_primitive(self, primitive, tracers, params):
    raise NotImplementedError("must override")

  def __repr__(self):
    return '{}(level={}/{})'.format(
        self.__class__.__name__, self.level, self.sublevel)

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError("must override to handle call-like primitives")

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers):
    # As a default implementation, drop the custom differentiation rule. This
    # behavior is desirable when staging out of the JAX system, but not when
    # there are further differentiation transformations to be applied. Override
    # this method to allow differentiation to be performed downstream.
    del primitive, jvp  # Unused.
    return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
    # See comment in the above process_custom_jvp_call method.
    del primitive, fwd, bwd, out_trees  # Unused.
    return fun.call_wrapped(*tracers)

def escaped_tracer_error(detail):
  msg = ("Encountered an unexpected tracer. Perhaps this tracer escaped "
         "through global state from a previously traced function.\n"
         "The functions being transformed should not save traced values to "
         "global state.\nDetails: {}.")
  return UnexpectedTracerError(msg.format(detail))

class UnexpectedTracerError(Exception): pass


class Tracer(object):
  __array_priority__ = 1000
  __slots__ = ['_trace', '__weakref__']

  def __array__(self, *args, **kw):
    raise Exception("Tracer can't be used with raw numpy functions. "
                    "You might have\n"
                    "  import numpy as np\n"
                    "instead of\n"
                    "  import jax.numpy as np")

  def __init__(self, trace):
    self._trace = trace

  def __iter__(self):
    return iter(self.aval._iter(self))

  def __len__(self):
    return self.aval._len(self)

  @property
  def aval(self):
    raise NotImplementedError("must override")

  def __neg__(self): return self.aval._neg(self)
  def __pos__(self): return self.aval._pos(self)
  def __eq__(self, other): return self.aval._eq(self, other)
  def __ne__(self, other): return self.aval._ne(self, other)
  def __lt__(self, other): return self.aval._lt(self, other)
  def __le__(self, other): return self.aval._le(self, other)
  def __gt__(self, other): return self.aval._gt(self, other)
  def __ge__(self, other): return self.aval._ge(self, other)
  def __abs__(self): return self.aval._abs(self)
  def __add__(self, other): return self.aval._add(self, other)
  def __radd__(self, other): return self.aval._radd(self, other)
  def __sub__(self, other): return self.aval._sub(self, other)
  def __rsub__(self, other): return self.aval._rsub(self, other)
  def __mul__(self, other): return self.aval._mul(self, other)
  def __rmul__(self, other): return self.aval._rmul(self, other)
  def __div__(self, other): return self.aval._div(self, other)
  def __rdiv__(self, other): return self.aval._rdiv(self, other)
  def __truediv__(self, other): return self.aval._truediv(self, other)
  def __rtruediv__(self, other): return self.aval._rtruediv(self, other)
  def __floordiv__(self, other): return self.aval._floordiv(self, other)
  def __rfloordiv__(self, other): return self.aval._rfloordiv(self, other)
  def __divmod__(self, other): return self.aval._divmod(self, other)
  def __rdivmod__(self, other): return self.aval._rdivmod(self, other)
  def __mod__(self, other): return self.aval._mod(self, other)
  def __rmod__(self, other): return self.aval._rmod(self, other)
  def __pow__(self, other): return self.aval._pow(self, other)
  def __rpow__(self, other): return self.aval._rpow(self, other)
  def __matmul__(self, other): return self.aval._matmul(self, other)
  def __rmatmul__(self, other): return self.aval._rmatmul(self, other)
  def __and__(self, other): return self.aval._and(self, other)
  def __rand__(self, other): return self.aval._rand(self, other)
  def __or__(self, other): return self.aval._or(self, other)
  def __ror__(self, other): return self.aval._ror(self, other)
  def __xor__(self, other): return self.aval._xor(self, other)
  def __rxor__(self, other): return self.aval._rxor(self, other)
  def __invert__(self): return self.aval._invert(self)
  def __lshift__(self, other): return self.aval._lshift(self, other)
  def __rshift__(self, other): return self.aval._rshift(self, other)
  def __getitem__(self, idx): return self.aval._getitem(self, idx)
  def __nonzero__(self): return self.aval._nonzero(self)
  def __bool__(self): return self.aval._bool(self)
  def __int__(self): return self.aval._int(self)
  def __long__(self): return self.aval._long(self)
  def __hex__(self): return self.aval._hex(self)
  def __oct__(self): return self.aval._oct(self)

  def __float__(self):
    raise TypeError("JAX Tracer object cannot be interpreted as a float. "
                    "Try using `x.astype(float)` instead.")

  def __complex__(self):
    raise TypeError("JAX Tracer object cannot be interpreted as a complex. "
                    "Try using `x.astype(complex)` instead.")

  def __setitem__(self, idx, val):
    raise TypeError("JAX 'Tracer' objects do not support item assignment")

  def __getattr__(self, name):
    # if the aval property raises an AttributeError, gets caught here
    assert skip_checks or name != "aval"

    try:
      attr = getattr(self.aval, name)
    except KeyError as err:
      raise AttributeError(
          "{} has no attribute {}".format(self.__class__.__name__, name)
      ) from err
    else:
      t = type(attr)
      if t is aval_property:
        return attr.fget(self)
      elif t is aval_method:
        return types.MethodType(attr.fun, self)
      else:
        return attr

  def __repr__(self):
    base = pp('Traced<{}>with<{}>'.format(self.aval, self._trace))
    contents = self._contents()
    if contents:
      base += pp('  with ') >> vcat(pp('{} = '.format(name)) >> pp_payload
                                    for name, pp_payload in contents)
    return str(base)

  def _contents(self):
    try:
      return [(name, pp(repr(getattr(self, name)))) for name in self.__slots__]
    except AttributeError:
      return ()

  def __copy__(self):
    return self

  def __deepcopy__(self, unused_memo):
    return self

# these can be used to set up forwarding of properties and instance methods from
# Tracer instances to the underlying avals
aval_property = namedtuple("aval_property", ["fget"])
aval_method = namedtuple("aval_method", ["fun"])


class MasterTrace:
  level: int
  trace_type: Type[Trace]

  def __init__(self, level, trace_type) -> None:
    self.level = level
    self.trace_type = trace_type

  def __repr__(self) -> str:
    return "MasterTrace({},{})".format(self.level, self.trace_type.__name__)

  def __hash__(self) -> int:
    return hash((self.level, self.trace_type))

  def __eq__(self, other: object) -> bool:
    return (isinstance(other, MasterTrace) and
            self.level == other.level and self.trace_type == other.trace_type)

class TraceStack:
  upward: List[MasterTrace]
  downward: List[MasterTrace]

  def __init__(self):
    self.upward = []
    self.downward = []

  def next_level(self, bottom: bool) -> int:
    if bottom:
      return - (len(self.downward) + 1)
    else:
      return len(self.upward)

  def push(self, master_trace: MasterTrace, bottom: bool) -> None:
    if bottom:
      self.downward.append(master_trace)
    else:
      self.upward.append(master_trace)

  def pop(self, bottom: bool) -> None:
    if bottom:
      self.downward.pop()
    else:
      self.upward.pop()

  def __repr__(self) -> str:
    return  'Trace stack\n{} ---\n{}'.format(
      map('  {}\n'.format, self.upward[::-1]),
      map('  {}\n'.format, self.downward))

  def copy(self):
    new = TraceStack()
    new.upward = self.upward[:]
    new.downward = self.downward[:]
    return new

class Sublevel(int): pass


# The global state of the tracer is accessed by a thread-local object.
# This allows concurrent tracing in separate threads; passing traced objects
# between threads is forbidden.
class TraceState(threading.local):
  trace_stack: TraceStack
  substack: List[Sublevel]
  initial_style: bool

  def __init__(self) -> None:
    self.trace_stack = TraceStack()
    self.substack = [Sublevel(0)]
    self.initial_style = False

  def copy(self):
    new = TraceState()
    new.trace_stack = self.trace_stack.copy()
    new.substack = self.substack[:]
    new.initial_style = self.initial_style
    return new
trace_state = TraceState()

def reset_trace_state() -> bool:
  "Reset the global trace state and return True if it was already clean."
  if (trace_state.substack != [Sublevel(0)] or
      trace_state.trace_stack.downward or
      trace_state.trace_stack.upward):
    trace_state.__init__()  # type: ignore
    return False
  else:
    return True

def cur_sublevel() -> Sublevel:
  return trace_state.substack[-1]

@contextmanager
def new_master(trace_type: Type[Trace], bottom=False) -> Generator[MasterTrace, None, None]:
  level = trace_state.trace_stack.next_level(bottom)
  master = MasterTrace(level, trace_type)
  trace_state.trace_stack.push(master, bottom)

  try:
    yield master
  finally:
    trace_state.trace_stack.pop(bottom)

  if check_leaks:
    t = ref(master)
    del master
    if t() is not None:
      print(trace_state.trace_stack)
      raise Exception('Leaked trace {}'.format(t()))

@contextmanager
def new_sublevel() -> Generator[None, None, None]:
  sublevel = Sublevel(len(trace_state.substack))
  trace_state.substack.append(sublevel)
  try:
    yield
  finally:
    trace_state.substack.pop()

  if check_leaks:
    t = ref(sublevel)
    del sublevel
    if t() is not None:
      raise Exception('Leaked sublevel {}'.format(t()))

def full_lower(val):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val

def find_top_trace(xs):
 try:
   top_trace = max((x._trace for x in xs if isinstance(x, Tracer)),
                   key=attrgetter('level'))
 except ValueError:
   return None
 else:
   return type(top_trace)(top_trace.master, cur_sublevel())

@contextmanager
def initial_style_staging():
  prev, trace_state.initial_style = trace_state.initial_style, True
  try:
    yield
  finally:
    trace_state.initial_style = prev


# -------------------- abstract values --------------------


class AbstractValue(object):
  __slots__: List[str] = []

  def at_least_vspace(self):
    assert False

  def __repr__(self):
    try:
      kv_pairs = ('{}={}'.format(k, v) for k, v in self.__dict__.items())
      return '{}({})'.format(self.__class__.__name__, ','.join(kv_pairs))
    except AttributeError:
      return self.__class__.__name__

  def strip_weak_type(self):
    return self

class Bot(AbstractValue): pass

bot = Bot()

class AbstractUnit(AbstractValue):
  def join(self, other):
    if not skip_checks:
      assert other is abstract_unit, other
    return self
  def _eq(self, self_traced, other): return get_aval(other) is self

abstract_unit = AbstractUnit()

def lattice_join(x, y):
  if x is None:
    return y
  elif y is None:
    return x
  elif isinstance(x, type(y)):
    return y.join(x)
  elif isinstance(y, type(x)):
    return x.join(y)
  else:
    raise TypeError((x, y))

# For use in typing annotations to denote either a Tracer or a `valid_jaxtype`.
Value = Any

def valid_jaxtype(x):
  try:
    concrete_aval(x)
  except TypeError:
    return False
  else:
    return True


def concrete_aval(x):
  try:
    return pytype_aval_mappings[type(x)](x)
  except KeyError as err:
    raise TypeError("{} is not a valid Jax type".format(type(x))) from err


def get_aval(x):
  if isinstance(x, Tracer):
    return x.aval
  else:
    return concrete_aval(x)


pytype_aval_mappings: Dict[type, Callable[[Any], AbstractValue]] = {}


class Unit(object):
  def __repr__(self): return '*'
unit = Unit()
literalable_types.add(Unit)

class UnitVar(object):
  @property
  def aval(self): return abstract_unit
  def __repr__(self): return '*'
unitvar = UnitVar()

pytype_aval_mappings[Unit] = lambda _: abstract_unit

identity_p = Primitive('id')
identity_p.def_impl(lambda x: x)
identity_p.def_custom_bind(lambda x: x)

class ConcretizationTypeError(TypeError): pass

def raise_concretization_error(val, context=""):
  msg = (f"Abstract tracer value encountered where concrete value is expected ({context}).\n"
          "Use transformation parameters such as `static_argnums` for `jit` "
          "to avoid tracing input values.\n"
          "See `https://jax.readthedocs.io/en/latest/faq.html#abstract-tracer-value-encountered-where-concrete-value-is-expected-error`.\n"
          f"Encountered value: {val}")
  raise ConcretizationTypeError(msg)


def concretization_function_error(fun, context=""):
  fname = getattr(fun, "__name__", fun)
  fname_context = f"in `{fname}`"
  if context:
    fname_context += f" {context}"
  def error(self, arg):
    raise_concretization_error(arg, fname_context)
  return error


def concrete_or_error(typ: Type, val: Any, context=""):
  """Like typ(val), but gives the context in the error message.
  Use with typ either `int`, or `bool`.
  """
  if isinstance(val, Tracer):
    if isinstance(val.aval, ConcreteArray):
      return typ(val.aval.val)
    else:
      raise_concretization_error(val, context)
  else:
    return typ(val)

class UnshapedArray(AbstractValue):
  __slots__ = ['dtype', 'weak_type']
  array_abstraction_level = 2

  def __init__(self, dtype, weak_type=False):
    self.dtype = onp.dtype(dtypes.canonicalize_dtype(dtype))
    self.weak_type = weak_type

  def __eq__(self, other):
    return (type(self) is type(other) and self.dtype == other.dtype and
            self.weak_type == other.weak_type)

  def __ne__(self, other):
    return not self == other

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `onp.zeros(3).dtype is onp.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.dtype, self.weak_type))

  def __repr__(self):
    return '{}({}{})'.format(self.__class__.__name__, self.str_short(),
                             ", weak_type=True" if self.weak_type else "")

  _bool = _nonzero = concretization_function_error(bool)
  _float   = concretization_function_error(
      float, "Try using `x.astype(float)` instead.")
  _int     = concretization_function_error(
      int, "Try using `x.astype(int)` instead.")
  _complex = concretization_function_error(
      complex, "Try using `x.astype(complex)` instead.")
  _hex     = concretization_function_error(hex)
  _oct     = concretization_function_error(oct)

  def at_least_vspace(self):
    return self

  def join(self, other):
    if self.dtype == other.dtype:
      if self.weak_type == other.weak_type:
        return self
      else:
        return UnshapedArray(self.dtype, weak_type=False)
    else:
      raise TypeError(self, other)

  def str_short(self):
    return self.dtype.name

  def strip_weak_type(self):
    """Returns a copy of the aval with weak_type=False."""
    return UnshapedArray(self.dtype) if self.weak_type else self

  @property
  def shape(self):
    msg = ("UnshapedArray has no shape. Please open an issue at "
           "https://github.com/google/jax/issues because it's unexpected for "
           "UnshapedArray instances to ever be produced.")
    raise TypeError(msg)

class ShapedArray(UnshapedArray):
  __slots__ = ['shape']
  array_abstraction_level = 1

  def __init__(self, shape, dtype, weak_type=False):
    super(ShapedArray, self).__init__(dtype, weak_type=weak_type)
    self.shape = canonicalize_shape(shape)

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: prod(self.shape))

  broadcast: ClassVar[Optional[aval_method]] = None
  transpose: ClassVar[Optional[aval_method]] = None
  reshape: ClassVar[Optional[aval_method]] = None
  _iter: ClassVar[Optional[staticmethod]] = None

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape
            and self.weak_type == other.weak_type)

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `onp.zeros(3).dtype is onp.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.shape, self.dtype, self.weak_type))

  def at_least_vspace(self):
    return self

  def join(self, other):
    if self.shape == other.shape and self.dtype == other.dtype:
      if self.weak_type == other.weak_type:
        return self
      else:
        return ShapedArray(self.shape, self.dtype, weak_type=False)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(self, other)

  def str_short(self):
    shapestr = ','.join(map(str, self.shape))
    return '{}[{}]'.format(self.dtype.name, shapestr)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError:
      raise TypeError("len() of unsized object")  # same as numpy error

  def _len(self, ignored_tracer):
    return len(self)

  def strip_weak_type(self):
    return ShapedArray(self.shape, self.dtype) if self.weak_type else self


def _forward_to_value(self, fun, ignored_tracer, *args):
  return fun(self.val, *args)

class ConcreteArray(ShapedArray):
  __slots__ = ['val']
  array_abstraction_level = 0

  def __init__(self, val, weak_type=False):
    super(ConcreteArray, self).__init__(onp.shape(val), onp.result_type(val),
                                        weak_type=weak_type)
    # Note: canonicalized self.dtype doesn't necessarily match self.val
    self.val = val
    assert self.dtype != onp.dtype('O')

  def __eq__(self, other):
    return (type(self) is type(other) and self.dtype == other.dtype
            and self.shape == other.shape and self.weak_type == other.weak_type
            and onp.all(self.val == other.val))

  def __hash__(self):
    return id(self.val)

  def at_least_vspace(self):
    return ShapedArray(self.shape, self.dtype, weak_type=self.weak_type)

  def join(self, other):
    if self == other:
      return self
    elif self.shape == other.shape and self.dtype == other.dtype:
      return ShapedArray(self.shape, self.dtype,
                         weak_type=self.weak_type and other.weak_type)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype,
                           weak_type=self.weak_type and other.weak_type)
    else:
      raise TypeError(self, other)

  def str_short(self):
    return str(self.val)

  def strip_weak_type(self):
    return ConcreteArray(self.val) if self.weak_type else self

  _bool = _nonzero = partialmethod(_forward_to_value, bool)
  _int     = partialmethod(_forward_to_value, int)
  _hex     = partialmethod(_forward_to_value, hex)
  _oct     = partialmethod(_forward_to_value, oct)


class AbstractToken(AbstractValue): pass

abstract_token = AbstractToken()


def raise_to_shaped(aval, weak_type=False):
  if isinstance(aval, ShapedArray):
    return ShapedArray(aval.shape, aval.dtype, weak_type=weak_type)
  elif aval is abstract_unit:
    return abstract_unit
  elif aval is abstract_token:
    return abstract_token
  else:
    raise TypeError(type(aval))

# Registry for valid dimension types. This is used by masking.Poly.
_DIMENSION_TYPES: Set[type] = {int}

def _canonicalize_dimension(dim):
  if type(dim) in _DIMENSION_TYPES:
    return dim
  else:
    return operator.index(dim)

def canonicalize_shape(shape):
  """Canonicalizes and checks for errors in a user-provided shape value.

  Args:
    shape: a Python value that represents a shape.

  Returns:
    A tuple of integers.
  """
  try:
    return tuple(map(_canonicalize_dimension, shape))
  except TypeError:
    pass
  msg = ("Shapes must be 1D sequences of concrete values of integer type, "
         "got {}.")
  if any(isinstance(x, Tracer) and isinstance(get_aval(x), ShapedArray)
         and not isinstance(get_aval(x), ConcreteArray) for x in shape):
    msg += ("\nIf using `jit`, try using `static_argnums` or applying `jit` to "
            "smaller subfunctions.")
  raise TypeError(msg.format(shape))


# ------------------- Call and map -------------------

def apply_todos(todos, outs):
  todos_list = list(todos)
  while todos_list:
    outs = map(full_lower, todos_list.pop()(outs))
  return outs

@lu.transformation_with_aux
def process_env_traces(post_processor: str, primitive: Primitive,
                           level: int, params_tuple: tuple, *args):
  outs = yield args, {}
  params = dict(params_tuple)
  todo = []
  while True:
    tracers = [x for x in outs if isinstance(x, Tracer) and x._trace.level > level]
    if tracers:
      ans = max(tracers, key=lambda x: x._trace.level)
    else:
      break
    trace = type(ans._trace)(ans._trace.master, cur_sublevel())
    outs = map(trace.full_raise, outs)
    post_process = getattr(trace, post_processor)
    outs, cur_todo = post_process(primitive, outs, params)
    todo.append(cur_todo)
  yield outs, tuple(todo)  # Ensure the aux output is immutable

def _call_bind(processor: str, post_processor: str, primitive: Primitive,
               f: lu.WrappedFun, *args, **params):
  top_trace = find_top_trace(args)
  level = trace_state.trace_stack.next_level(True) if top_trace is None else top_trace.level
  params_tuple = tuple(params.items())
  f, env_trace_todo = process_env_traces(f, post_processor, primitive, level, params_tuple)
  if top_trace is None:
    with new_sublevel():
      outs = primitive.impl(f, *args, **params)
  else:
    tracers = map(top_trace.full_raise, args)
    process = getattr(top_trace, processor)
    outs = map(full_lower, process(primitive, f, tracers, params))
  return apply_todos(env_trace_todo(), outs)

call_bind = partial(_call_bind, 'process_call', 'post_process_call')
map_bind = partial(_call_bind, 'process_map', 'post_process_map')


def call_impl(f: lu.WrappedFun, *args, **params):
  del params  # params parameterize the call primitive, not the function
  return f.call_wrapped(*args)

call_p = Primitive('call')
call_p.multiple_results = True
call_p.call_primitive = True
call = partial(call_bind, call_p)
call_p.def_custom_bind(call)
call_p.def_impl(call_impl)


# ------------------- Jaxpr printed representation -------------------

def check_jaxpr(jaxpr: Jaxpr):
  """Checks well-formedness of a jaxpr.

  Specifically it checks that all variabled used are previously defined.
  """
  def context():
    return "\njaxpr:\n{}\n".format(jaxpr)

  def read_env(env: Set[Var], v: Var):
    if type(v) is not Literal and v not in env:
      raise Exception("Variable '{}' not defined".format(v) + context())

  def write_env(env: Set[Var], v: Var):
    if v in env:
      raise Exception("Variable {} already bound".format(v) + context())
    env.add(v)

  env: Set[Var] = set()
  read = partial(read_env, env)
  write = partial(write_env, env)

  write(unitvar)
  map(write, jaxpr.constvars)
  map(write, jaxpr.invars)
  for eqn in jaxpr.eqns:
    if eqn.primitive.call_primitive or eqn.primitive.map_primitive:
      if "call_jaxpr" not in eqn.params:
        raise Exception("Call primitive {} should have a 'call_jaxpr' parameter"
                        .format(eqn.primitive))
    map(read, eqn.invars)
    map(write, eqn.outvars)

  for subjaxpr in subjaxprs(jaxpr):
    check_jaxpr(subjaxpr)

  map(read, jaxpr.outvars)


def pp_vars(vs) -> str:
    return ' '.join(map(str, vs))

def pp_eqn_compact(primitive_name: str, params: Dict) -> PrettyPrint:
  filtered_params = {k: v for k, v in params.items()
                     if not isinstance(v, (Jaxpr, TypedJaxpr))}
  return pp(primitive_name) >> pp_kv_pairs(sorted(filtered_params.items()))

def pp_eqn(eqn: JaxprEqn) -> PrettyPrint:
  lhs = pp_vars(eqn.outvars)
  pp_subexpr = pp('')
  return (pp('{} = '.format(lhs)) >>
          pp(eqn.primitive.name) >> pp_kv_pairs(sorted(eqn.params.items()))
          >> pp(' ') >> pp(pp_vars(eqn.invars))) + pp_subexpr


def pp_jaxpr(jaxpr) -> PrettyPrint:
  pp_outvars = str(tuple(jaxpr.outvars))
  return (pp('{{ lambda {} ; {}.'.format(pp_vars(jaxpr.constvars),
                                         pp_vars(jaxpr.invars))) +
          ((pp('let ') >>
            vcat(map(pp_eqn, jaxpr.eqns))) +
           pp('in {} }}'.format(pp_outvars))).indent(2))
