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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import attrgetter
from contextlib import contextmanager
from collections import namedtuple, Counter, defaultdict
from weakref import ref
import six
import types

from . import linear_util as lu
from .util import unzip2, safe_zip, safe_map, partial
from .pprint_util import pp, vcat, hcat, pp_kv_pairs

# TODO(dougalm): the trace cache breaks the leak detector. Consisder solving.
# TODO(mattjj): put each of these behind flags
check_leaks = False
skip_checks = True  # not __debug__  # google doesn't use -O

zip = safe_zip
map = safe_map


# -------------------- jaxprs --------------------

class Jaxpr(object):
  def __init__(self, constvars, freevars, invars, outvar, eqns):
    self.constvars = constvars
    self.freevars = freevars
    self.invars = invars
    self.outvar = outvar
    self.eqns = eqns

  def __str__(self):
    return str(pp_jaxpr(self))

  def __repr__(self):
    return self.__str__()


JaxprEqn = namedtuple('JaxprEqn', ['invars', 'outvars', 'primitive',
                                   'bound_subjaxprs', 'destructure', 'params'])

class Primitive(object):
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
    return full_lower(out_tracer)

  def def_impl(self, impl):
    self.impl = impl
    return impl

  def def_custom_bind(self, bind):
    self.bind = bind
    return bind

  def impl(self, *args, **kwargs):
    raise NotImplementedError("Evaluation rule for '{}' not implemented"
                              .format(self.name))


# -------------------- lifting --------------------


def eval_jaxpr(jaxpr, consts, freevar_vals, *args):
  def read(v):
    return env[v]

  def write(v, val):
    env[v] = val

  env = {}
  write(unitvar, unit)
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  map(write, jaxpr.freevars, freevar_vals)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    subfuns = [partial(eval_jaxpr, subjaxpr, map(read, const_bindings),
                                             map(read, freevar_bindings))
               for subjaxpr, const_bindings, freevar_bindings
               in eqn.bound_subjaxprs]
    subfuns = map(lu.wrap_init, subfuns)
    ans = eqn.primitive.bind(*(subfuns + in_vals), **eqn.params)
    outvals = list(ans) if eqn.destructure else [ans]
    map(write, eqn.outvars, outvals)
  return read(jaxpr.outvar)


def full_lower(val):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val


def find_top_trace(xs):
 try:
   top_trace = max((x.trace for x in xs if isinstance(x, Tracer)),
                   key=attrgetter('level'))
 except ValueError:
   return None
 else:
   return type(top_trace)(top_trace.master, cur_sublevel())


# -------------------- tracing --------------------


class Trace(object):
  def __init__(self, master, sublevel):
    self.master = master
    self.level = master.level
    self.sublevel = sublevel

  def full_raise(self, val):
    if not isinstance(val, Tracer):
      return self.pure(val)
    level = self.level
    sublevel = self.sublevel
    if val.trace.master is self.master:
      if val.trace.sublevel == sublevel:
        return val
      elif val.trace.sublevel < sublevel:
        return self.sublift(val)
      else:
        raise Exception("Can't lift sublevels {} to {}"
                        .format(val.trace.sublevel, sublevel))
    elif val.trace.level < level:
      if val.trace.sublevel > sublevel:
        raise Exception("Incompatible sublevel: {}, {}"
                        .format(val.trace, (level, sublevel)))
      return self.lift(val)
    elif val.trace.level > level:
      print_trace_stack()
      raise Exception("Can't lift {} to {}".format(val, self))
    elif val.trace.level == self.level:
      raise Exception("Different traces at same level: {}, {}".format(val, self))
    else:
      raise Exception("Can't lift {} to {}".format(val, self))


  def pure(self, val):
    assert False

  def lift(self, tracer):
    assert False

  def sublift(self, tracer):
    assert False

  def __repr__(self):
    return '{}(level={}/{})'.format(
        self.__class__.__name__, self.level, self.sublevel)


class Tracer(object):
  __array_priority__ = 1000
  __slots__ = ['trace']

  def __array__(self):
    raise Exception("Tracer can't be used with raw numpy functions. "
                    "You might have\n  import numpy as np\ninstead of\n  import jax.numpy as np")

  def __init__(self, trace):
    self.trace = trace

  def __iter__(self):
    return iter(self.aval._iter(self))

  def __len__(self):
    # TODO(dougalm,mattjj): replace with something more efficient
    return len(list(iter(self)))

  @property
  def aval(self):
    assert False

  def __neg__(self): return self.aval._neg(self)
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
  def __invert__(self, other): return self.aval._invert(self, other)
  def __lshift__(self, other): return self.aval._lshift(self, other)
  def __rshift__(self, other): return self.aval._rshift(self, other)
  def __getitem__(self, idx): return self.aval._getitem(self, idx)
  def __nonzero__(self): return self.aval._nonzero(self)
  def __bool__(self): return self.aval._bool(self)
  def __float__(self): return self.aval._float(self)
  def __int__(self): return self.aval._int(self)
  def __long__(self): return self.aval._long(self)
  def __complex__(self): return self.aval._complex(self)
  def __hex__(self): return self.aval._hex(self)
  def __oct__(self): return self.aval._oct(self)


  def __getattr__(self, name):
    # if the aval property raises an AttributeError, gets caught here
    assert skip_checks or name != "aval"

    try:
      attr = getattr(self.aval, name)
    except KeyError:
      raise AttributeError(
          "{} has no attribute {}".format(self.__class__.__name__, name))
    else:
      t = type(attr)
      if t is aval_property:
        return attr.fget(self)
      elif t is aval_method:
        if six.PY3:
          return types.MethodType(attr.fun, self)
        else:
          return types.MethodType(attr.fun, self, None)
      else:
        return attr

  def __repr__(self):
    return 'Traced<{}>with<{}>'.format(self.aval, self.trace)


# these can be used to set up forwarding of properties and instance methods from
# Tracer instances to the underlying avals
aval_property = namedtuple("aval_property", ["fget"])
aval_method = namedtuple("aval_method", ["fun"])


class MasterTrace(object):
  def __init__(self, level, trace_type):
    self.level = level
    self.trace_type = trace_type

  def __repr__(self):
    return "MasterTrace({},{})".format(self.level, self.trace_type.__name__)

  def __hash__(self):
    return hash((self.level, self.trace_type))

  def __eq__(self, other):
    return self.level == other.level and self.trace_type == other.trace_type


class TraceStack(object):
  def __init__(self):
    self.upward = []
    self.downward = []

  def next_level(self, bottom):
    if bottom:
      return - (len(self.downward) + 1)
    else:
      return len(self.upward)

  def push(self, val, bottom):
    if bottom:
      self.downward.append(val)
    else:
      self.upward.append(val)

  def pop(self, bottom):
    if bottom:
      self.downward.pop()
    else:
      self.upward.pop()

  def __repr__(self):
    return  'Trace stack\n{} ---\n{}'.format(
      map('  {}\n'.format, self.upward[::-1]),
      map('  {}\n'.format, self.downward))


trace_stack = TraceStack()

class Sublevel(int): pass
substack = [Sublevel(0)]

def cur_sublevel():
  return substack[-1]


@contextmanager
def new_master(trace_type, bottom=False):
  level = trace_stack.next_level(bottom)
  master = MasterTrace(level, trace_type)
  trace_stack.push(master, bottom)

  try:
    yield master
  finally:
    trace_stack.pop(bottom)

  if check_leaks:
    t = ref(master)
    del master
    if t() is not None:
      print(trace_stack)
      raise Exception('Leaked trace {}'.format(t()))


@contextmanager
def new_sublevel():
  sublevel = Sublevel(len(substack))
  substack.append(sublevel)
  try:
    yield
  finally:
    substack.pop()

  if check_leaks:
    t = ref(sublevel)
    del sublevel
    if t() is not None:
      print_trace_stack()
      raise Exception('Leaked sublevel {}'.format(t()))

# -------------------- abstract values --------------------


class AbstractValue(object):
  __slots__ = []

  def at_least_vspace(self):
    assert False

  def __repr__(self):
    try:
      kv_pairs = ('{}={}'.format(k, v) for k, v in self.__dict__.items())
      return '{}({})'.format(self.__class__.__name__, ','.join(kv_pairs))
    except AttributeError:
      return self.__class__.__name__


class Bot(AbstractValue): pass

bot = Bot()


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


def valid_jaxtype(x):
  try:
    concrete_aval(x)
  except TypeError:
    return False
  return True


def concrete_aval(x):
  try:
    return pytype_aval_mappings[type(x)](x)
  except KeyError:
    raise TypeError("{} is not a valid Jax type".format(type(x)))


def get_aval(x):
  if isinstance(x, Tracer):
    return x.aval
  else:
    return concrete_aval(x)


pytype_aval_mappings = {}


# ------------------- Products -------------------

class JaxTuple(tuple):
  def __new__(cls, xs):
    assert skip_checks or all(map(valid_jaxtype, xs)), xs
    return tuple.__new__(cls, xs)

  def __repr__(self):
    if self is unit:
      return unitvar
    else:
      return 'JaxTuple({})'.format(','.join(map(repr, self)))


class AbstractTuple(AbstractValue, tuple):
  @staticmethod
  def _iter(tracer):
    return tracer.unpack()

  def at_least_vspace(self):
    return AbstractTuple(x.at_least_vspace() for x in self)

  def join(self, other):
    return AbstractTuple(map(lattice_join, self, other))

  def __repr__(self):
    return '({})'.format(','.join(map(repr, self)))


unit = JaxTuple(())
unitvar = '*'

def tuple_to_jaxtuple(x):
  if type(x) is tuple:
    return JaxTuple(map(tuple_to_jaxtuple, x))
  else:
    return x

def pack(args):
  return pack_p.bind(*args)

def concrete_jaxtuple(xs):
  return AbstractTuple(map(concrete_aval, xs))

pytype_aval_mappings[JaxTuple] = concrete_jaxtuple

identity_p = Primitive('id')
identity_p.def_impl(lambda x: x)
identity_p.def_custom_bind(lambda x: x)

pack_p = Primitive('pack')
pack_p.def_impl(lambda *xs: JaxTuple(xs))

@pack_p.def_custom_bind
def pack_p_bind(*args):
  top_trace = find_top_trace(args)
  if top_trace is None:
    return JaxTuple(args)
  else:
    tracers = map(top_trace.full_raise, args)
    return top_trace.pack(tracers)


# ------------------- Call -------------------


def apply_todos(todos, x):
  while todos:
    x = full_lower(todos.pop()(x))
    assert skip_checks or isinstance(x, Tracer) or valid_jaxtype(x), x
  return x

@lu.transformation_with_aux
def process_env_traces(primitive, level, *args):
  ans = yield args
  todo = []
  while isinstance(ans, Tracer) and ans.trace.level > level:
    t = ans.trace
    sublevel = cur_sublevel()
    trace = type(t)(t.master, sublevel)
    ans = trace.full_raise(ans)
    ans, cur_todo = ans.trace.post_process_call(primitive, ans)
    todo.append(cur_todo)
  yield ans, todo

def call_bind(primitive, f, *args, **kwargs):
  top_trace = find_top_trace(args)
  level = trace_stack.next_level(True) if top_trace is None else top_trace.level
  f, env_trace_todo = process_env_traces(f, primitive, level)
  if top_trace is None:
    with new_sublevel():
      ans = primitive.impl(f, *args, **kwargs)
  else:
    tracers = map(top_trace.full_raise, args)
    ans = full_lower(top_trace.process_call(primitive, f, tracers, kwargs))
  return apply_todos(env_trace_todo(), ans)


def call_impl(f, *args, **kwargs):
  return f(*args, **kwargs)


call_p = Primitive('call')
call = partial(call_bind, call_p)
call_p.def_custom_bind(call)
call_p.def_impl(call_impl)


# ------------------- Jaxpr printed representation -------------------

def check_jaxpr(jaxpr):
  def context():
    return "\njaxpr:\n{}\n".format(jaxpr)

  def read_env(env, v):
    if v not in env:
      raise Exception("Variable '{}' not defined".format(v) + context())

  def write_env(env, v):
    if v in env:
      raise Exception("Variable {} already bound".format(v) + context())
    env.add(v)

  env = set()
  read = partial(read_env, env)
  write = partial(write_env, env)

  const_env = set()
  read_const= partial(read_env, const_env)
  write_const= partial(write_env, const_env)

  map(write_const, jaxpr.constvars)
  write(unitvar)
  map(write, jaxpr.constvars)
  map(write, jaxpr.freevars)
  map(write, jaxpr.invars)
  for eqn in jaxpr.eqns:
    map(read, eqn.invars)
    for subjaxpr, constvars, freevars in eqn.bound_subjaxprs:
      map(read, freevars)
      map(read_const, constvars)
      check_jaxpr(subjaxpr)
    map(write, eqn.outvars)
  read(jaxpr.outvar)


def pp_jaxpr(jaxpr):
  def print_vars(vs):
    return ' '.join(map(str, vs))

  def pp_eqn(eqn):
    if eqn.destructure:
      lhs = '(' + print_vars(eqn.outvars) + ')'
    else:
      lhs = eqn.outvars[0]

    pp_subexpr = pp('')
    if eqn.bound_subjaxprs:
      for subjaxpr, const_vars, bound_vars in eqn.bound_subjaxprs:
        pp_subexpr = pp_subexpr + (
            pp_jaxpr(subjaxpr).indent(2)
            >> pp(' [ {} ; {} ]'.format(print_vars(const_vars),
                                        print_vars(bound_vars))))
    return (pp('{} = '.format(lhs)) >>
            pp(eqn.primitive.name) >> pp_kv_pairs(eqn.params.items())
            >> pp(' ') >> pp(print_vars(eqn.invars))) + pp_subexpr

  return (pp('{{ lambda {} ; {} ; {}.'.format(print_vars(jaxpr.constvars),
                                              print_vars(jaxpr.freevars),
                                              print_vars(jaxpr.invars))) +
          ((pp('let ') >>
            vcat(map(pp_eqn, jaxpr.eqns))) +
           pp('in {} }}'.format(jaxpr.outvar))).indent(2))
