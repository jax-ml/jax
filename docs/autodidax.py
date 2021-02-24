# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md:myst,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# # Autodidax: JAX core from scratch
#
# Ever want to learn how JAX works, but the implementation seemed too
# impenetrable? Well, you're in luck! By reading this tutorial, you'll learn
# every big idea in JAX's core system. You'll even get clued into our weird
# jargon!

# ## Part 1: Transformations as interpreters: standard evaluation, `jvp`, and `vmap`
#
# We want to transform functions that look like this:
#
# ```python
# def f(x):
#   y = sin(x) * 2
#   z = - y + x
#   return z
# ```
#
# Think of functions like `sin` and the arithmetic operations underlying the
# infix operators (`mul`, `add`, and `neg`) as primitive operations, meaning
# atomic units of processing rather than compositions.
#
# "Transform" means "interpret differently." Instead of standard interpretation
# where we apply primitive functions to numerical inputs to produce numerical
# outputs, we want to override primitive application and let different values
# flow through our program. For example, we might want to replace the
# application of every primitive with type `a -> b` with an application of its
# JVP rule with type `(a, T a) -> (b, T b)`, and let primal-tangent pairs flow
# through our program. Moreover, we want to apply a composition of multiple
# transformations, leading to stacks of interpreters.

# ### JAX core machinery
#
# We can implement stacks of interpreters and even have them all discharge on
# the fly as we execute the Python function to be transformed. To start, let's
# define these primitives so that we can intercept their application:


# +
from typing import NamedTuple

class Primitive(NamedTuple):
  name: str

add_p = Primitive('add')
mul_p = Primitive('mul')
neg_p = Primitive("neg")
sin_p = Primitive("sin")
cos_p = Primitive("cos")
reduce_sum_p = Primitive("reduce_sum")
greater_p = Primitive("greater")

def add(x, y): return bind(add_p, x, y)
def mul(x, y): return bind(mul_p, x, y)
def neg(x): return bind(neg_p, x)
def sin(x): return bind(sin_p, x)
def cos(x): return bind(cos_p, x)
def reduce_sum(x, axis=None): return bind(reduce_sum_p, x, axis=axis)
def greater(x, y): return bind(greater_p, x, y)


# -

# We'll set up array data types and infix operator methods in a moment.
#
# A `Primitive` is just an object with a name, to which we attach our
# interpretation rules (one for each transformation). The `bind` function is our
# interception point: it'll figure out which transformation rule to apply, based
# on how the arguments are boxed in tracers and what interpreters are active.
#
# The functions that user code calls, like `add` and `sin`, are just wrappers
# around calls to `bind`. These wrappers let us control how arguments are passed
# to `bind`, and in particular we follow a handy internal convention: when we
# call `bind`, we pass values representing array data as positional arguments,
# and we pass metadata like the `axis` argument to `sum_p` via keyword. This
# calling convention simplifies some core logic (since e.g. instances of the
# `Tracer` class to be defined below can only occurr in positional arguments to
# `bind`). The wrappers can also provide docstrings!
#
# We represent active interpreters as a stack. The stack is just a simple
# `list`, and each element is a container with an integer level (corresponding
# to the element's height in the stack), an interpreter type (which we'll call a
# `trace_type`), and an optional field for any global data the interpreter
# needs. We call each element a `MainTrace`, though maybe "Interpreter" would be
# more descriptive.

# +
from contextlib import contextmanager
from typing import Type, List, Optional, Any

class MainTrace(NamedTuple):
  level: int
  trace_type: Type['Trace']
  global_data: Optional[Any]

trace_stack: List[MainTrace] = []

@contextmanager
def new_main(trace_type: Type['Trace'], global_data=None):
  level = len(trace_stack)
  main = MainTrace(level, trace_type, global_data)
  trace_stack.append(main)

  try:
    yield main
  finally:
    trace_stack.pop()


# -

# When we're about to apply a transformed function, we'll push another
# interpreter onto the stack using `new_main`. Then, as we apply primitives in
# the function, we can think of the `bind` first being interprted by the trace
# at the top of the stack (i.e. with the highest level). If that first
# interpreter itself binds other primitives in its interpretation rule for the
# primitive, like how the JVP rule of `sin_p` might bind `cos_p` and `mul_p`,
# then those `bind` calls will be handled by the interpreter at the next level
# down.
#
# What goes at the bottom of the interpreter stack? At the bottom, we know all
# the transformation interpreters are finished, and we just want to do standard
# evaluation. So at the bottom we'll put an evaluation interpreter.
#
# Let's sketch out the interface for interpreters, which is based on the `Trace`
# and `Tracer` base classes. A `Tracer` represents a boxed-up value, perhaps
# carrying some extra context data used by the interpreter. A `Trace` handles
# boxing up vales into `Tracers` and also handles primitive application.

class Trace:
  main: MainTrace

  def __init__(self, main: MainTrace) -> None:
    self.main = main

  def pure(self, val): assert False  # must override
  def lift(self, val): assert False  # must override

  def process_primitive(self, primitive, tracers, params):
    assert False  # must override


# The first two methods are about boxing up values in `Tracer`s, which are the
# objects that flow through the Python programs we transform. The last method is
# the callback we'll use to interpret primitive application.
#
# The `Trace` itself doesn't contain any data, other than a reference to its
# corresponding `MainTrace` instance. In fact, multiple instances of a `Trace`
# might be created and discarded during an application of a transformation,
# whereas only a single `MainTrace` instance is created per application of a
# transformation.
#
# As for `Tracer`s themselves, each one carries an abstract value (and forwards
# infix operators to it), and the rest is up to the transformation. (The
# relationship between `Tracer`s and `AbstractValue`s is that there's one
# `Tracer` per transformation, and at least one `AbstractValue` per base type,
# like arrays.)

# +
import numpy as np
from typing import Tuple

class Tracer:
  _trace: Trace

  __array_priority__ = 1000

  @property
  def aval(self):
    assert False  # must override

  def full_lower(self):
    return self  # default implementation

  def __neg__(self): return self.aval._neg(self)
  def __add__(self, other): return self.aval._add(self, other)
  def __radd__(self, other): return self.aval._radd(self, other)
  def __mul__(self, other): return self.aval._mul(self, other)
  def __rmul__(self, other): return self.aval._rmul(self, other)
  def __gt__(self, other): return self.aval._gt(self, other)
  def __bool__(self): return self.aval._bool(self)
  def __nonzero__(self): return self.aval._nonzero(self)

  def __getattr__(self, name):
    try:
      return getattr(self.aval, name)
    except AttributeError:
      raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

class ShapedArray:
  array_abstraction_level = 1
  shape: Tuple[int]
  dtype: np.dtype

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  @property
  def ndim(self):
    return len(self.shape)

  _neg = staticmethod(neg)
  _add = staticmethod(add)
  _radd = staticmethod(add)
  _mul = staticmethod(mul)
  _rmul = staticmethod(mul)
  _gt = staticmethod(greater)

  @staticmethod
  def _bool(tracer):
    raise Exception("ShapedArray can't be unambiguously converted to bool")

  @staticmethod
  def _nonzero(tracer):
    raise Exception("ShapedArray can't be unambiguously converted to bool")

  def str_short(self):
    return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'

class ConcreteArray(ShapedArray):
  array_abstraction_level = 2
  val: np.ndarray

  def __init__(self, val):
    self.val = val
    self.shape = val.shape
    self.dtype = val.dtype

  @staticmethod
  def _bool(tracer):
    return bool(tracer.aval.val)

  @staticmethod
  def _nonzero(tracer):
    return bool(tracer.aval.val)

def get_aval(x):
  if isinstance(x, Tracer):
    return x.aval
  else:
    return ConcreteArray(np.asarray(x))


# -

# Notice that we actually have two `AbstractValue`s for arrays, representing
# different levels of abstraction. A `ShapedArray` represents the set of all
# possible arrays with a given shape and dtype. A `ConcreteArray` represents a
# singleton set consisting of a single array value.
#
# Now that we've set up the trace stack, the Trace/Tracer API for interpreters,
# and abstract values, we can come back to implement `bind`:

def bind(prim, *args, **params):
  top_trace = find_top_trace(args)
  tracers = [full_raise(top_trace, arg) for arg in args]
  out = top_trace.process_primitive(prim, tracers, params)
  return full_lower(out)


# The main action is that we call `find_top_trace` to figure out which
# interpreter should handle this primitive application as a function of the
# arguments and the active traces on the trace stack. We then call that top
# trace's `process_primitive` so that the trace can apply its interpretation
# rule. The calls to `full_raise` just ensure that the inputs are boxed in the
# top trace's `Tracer` instances, and the call to `full_lower` is an optional
# optimization so that we unbox values out of `Tracer`s as much as possible.

# +
from operator import attrgetter

def find_top_trace(xs) -> Trace:
  top_main = max((x._trace.main for x in xs if isinstance(x, Tracer)),
                 default=trace_stack[0], key=attrgetter('level'))
  return top_main.trace_type(top_main)


# -

# In words, `find_top_trace` returns the highest-level interpreter associated
# with the `Tracer`s on its inputs, and otherwise returns the interpreter at the
# bottom of the stack (which is always an evaluation trace, at least for now).
# This corresponds to JAX transformations mostly working by data dependence
# _except_ for the special bottom-of-the-stack interpreter, which interprets
# everything.

# +
def full_lower(val):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val

def full_raise(trace, val) -> Tracer:
  if not isinstance(val, Tracer):
    return trace.pure(val)
  level = trace.main.level
  if val._trace.main is trace.main:
    return val
  elif val._trace.main.level < level:
    return trace.lift(val)
  elif val._trace.main.level > level:
    raise Exception(f"Can't lift level {val._trace.main.level} to {level}.")
  else:  # val._trace.level == level
    raise Exception(f"Different traces at same level: {val._trace}, {trace}.")


# -

# The logic in `full_raise` serves to box values into `Tracer`s for a particular
# `Trace`, calling different methods on the `Trace` based on context:
# `Trace.pure` is called on non-`Tracer` constants, and `Trace.lift` is called
# for values that are already `Tracer`s from a lower-level interpreter. These
# two methods could share the same implementation, but by distinguishing them in
# the core logic we can provide more information to the `Trace` subclass.
#
# That's it for the JAX core! Now we can start adding interpreters.

# ### Evaluation interpreter
#
# We'll start with the simplest interpreter: the evaluation interpreter that
# will sit at the bottom of the interpreter stack.

# +
class EvalTrace(Trace):
  pure = lift = lambda self, x: x  # no boxing in Tracers needed

  def process_primitive(self, primitive, tracers, params):
    return impl_rules[primitive](*tracers, **params)

trace_stack.append(MainTrace(0, EvalTrace, None))  # special bottom of the stack

impl_rules = {}
impl_rules[add_p] = np.add
impl_rules[mul_p] = np.multiply
impl_rules[neg_p] = np.negative
impl_rules[sin_p] = np.sin
impl_rules[cos_p] = np.cos
impl_rules[reduce_sum_p] = np.sum
impl_rules[greater_p] = np.greater


# -

# With this interpreter, we can evaluate user functions:

# +
def f(x):
  y = sin(x) * 2
  z = - y + x
  return z

print(f(3.0))


# -

# Woo! Like going around in a big circle. But the point of this indirection is
# that now we can add some real transformations.

# ### Forward-mode autodiff with `jvp`
#
# First, a couple helper functions:

# +
def zeros_like(val):
  return np.zeros_like(val)

def unzip2(pairs):
  lst1, lst2 = [], []
  for x1, x2 in pairs:
    lst1.append(x1)
    lst2.append(x2)
  return lst1, lst2


# -

# The `Tracer` for forward-mode autodiff carries a primal-tangent pair. The
# `Trace` applies JVP rules.

# +
class JVPTracer(Tracer):
  def __init__(self, trace, primal, tangent):
    self._trace = trace
    self.primal = primal
    self.tangent = tangent

  @property
  def aval(self):
    return get_aval(self.primal)

class JVPTrace(Trace):
  pure = lift = lambda self, val: JVPTracer(self, val, zeros_like(val))

  def process_primitive(self, primitive, tracers, params):
    primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
    jvp_rule = jvp_rules[primitive]
    primal_out, tangent_out = jvp_rule(primals_in, tangents_in, **params)
    return JVPTracer(self, primal_out, tangent_out)

jvp_rules = {}


# -

# Notice both `lift` and `sublift` package a value into a `JVPTracer` with the
# minimal amount of context, which is a zero tangent value.

# Let's add some JVP rules for primitives:

# +
def add_jvp(primals, tangents):
  (x, y), (x_dot, y_dot) = primals, tangents
  return x + y, x_dot + y_dot
jvp_rules[add_p] = add_jvp

def mul_jvp(primals, tangents):
  (x, y), (x_dot, y_dot) = primals, tangents
  return x * y, x_dot * y + x * y_dot
jvp_rules[mul_p] = mul_jvp

def sin_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  return sin(x), cos(x) * x_dot
jvp_rules[sin_p] = sin_jvp

def cos_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  return cos(x), -sin(x) * x_dot
jvp_rules[cos_p] = cos_jvp

def neg_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  return neg(x), neg(x_dot)
jvp_rules[neg_p] = neg_jvp

def reduce_sum_jvp(primals, tangents, *, axis):
  (x,), (x_dot,) = primals, tangents
  return reduce_sum(x, axis), reduce_sum(x_dot, axis)
jvp_rules[reduce_sum_p] = reduce_sum_jvp

def greater_jvp(primals, tangents):
  (x, y), _ = primals, tangents
  out_primal = greater(x, y)
  return out_primal, zeros_like(out_primal)
jvp_rules[greater_p] = greater_jvp


# -

# Finally, we add a transformation API to kick off the trace:

def jvp(f, primals, tangents):
  with new_main(JVPTrace) as main:
    trace = JVPTrace(main)
    tracers_in = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
    out = f(*tracers_in)
    tracer_out = full_raise(trace, out)
    primal_out, tangent_out = tracer_out.primal, tracer_out.tangent
  return primal_out, tangent_out


# And with that, we can differentiate!

x = 3.0
y, sin_deriv_at_3 = jvp(sin, (x,), (1.0,))
print(sin_deriv_at_3)
print(cos(3.0))


# +
def f(x):
  y = sin(x) * 2
  z = - y + x
  return z

x, xdot = 3., 1.
y, ydot = jvp(f, (x,), (xdot,))
print(y)
print(ydot)


# +
def deriv(f):
  return lambda x: jvp(f, (x,), (1.,))[1]

print(deriv(sin)(3.))
print(deriv(deriv(sin))(3.))
print(deriv(deriv(deriv(sin)))(3.))
print(deriv(deriv(deriv(deriv(sin))))(3.))


# +
def f(x):
  if x > 0.:  # Python control flow
    return 2. * x
  else:
    return x

print(deriv(f)(3.))
print(deriv(f)(-3.))


# -

# ### Vectorized batching with `vmap`
#
# First, a couple helper functions, one for producing mapped abstract values
# from unmapped ones (by removing an axis), and one for moving batch dimensions
# around:

# +
def mapped_aval(batch_dim, aval):
  shape = list(aval.shape)
  del shape[batch_dim]
  return ShapedArray(tuple(shape), aval.dtype)

def move_batch_axis(axis_size, src, dst, x):
  if src is not_mapped:
    target_shape = list(np.shape(x))
    target_shape.insert(dst, axis_size)
    return np.broadcast_to(np.expand_dims(x, dst), target_shape)
  else:
    return np.moveaxis(x, src, dst)


# -

# The `Tracer` for vectorized batching carries a batched value and an optional
# integer indicating which axis (if any) is the batch axis.

# +
from typing import Union

class NotMapped: pass
not_mapped = NotMapped()

class BatchTracer(Tracer):
  def __init__(self, trace, val, batch_dim: Union[NotMapped, int]):
    self._trace = trace
    self.val = val
    self.batch_dim = batch_dim

  @property
  def aval(self):
    if self.batch_dim is not_mapped:
      return get_aval(self.val)
    else:
      return mapped_aval(self.batch_dim, get_aval(self.val))

  def full_lower(self):
    if self.batch_dim is not_mapped:
      return full_lower(self.val)
    else:
      return self

class BatchTrace(Trace):
  pure = lift = lambda self, val: BatchTracer(self, val, not_mapped)

  def process_primitive(self, primitive, tracers, params):
    vals_in, bdims_in = unzip2((t.val, t.batch_dim) for t in tracers)
    vmap_rule = vmap_rules[primitive]
    val_out, bdim_out = vmap_rule(self.axis_size, vals_in, bdims_in, **params)
    return BatchTracer(self, val_out, bdim_out)

  @property
  def axis_size(self):
    return self.main.global_data

vmap_rules = {}
# -

# Here we've implemented the optional `Tracer.full_lower` method, which lets
# peel off a batching tracer if it's not needed because it doesn't represent a
# batched value.
#
# For `BatchTrace`, analogous to `JVPTrace`, the methods `pure` and `lift` just
# box a value in a `BatchTracer` with the minimal amount of context, which in
# this case is a `batch_dim` taking the sentinel value `not_mapped`. Notice we
# use the `MainTrace`'s interpreter-global data field to store the batch axis
# size.
#
# Next we can define batching interpreter rules for each primitive:

# +
from functools import partial

def broadcasting_binop_batching_rule(op, axis_size, vals_in, dims_in):
  (x, y), (x_bdim, y_bdim) = vals_in, dims_in
  if x_bdim != y_bdim:
    y = move_batch_axis(axis_size, y_bdim, x_bdim, y)
  return op(x, y), x_bdim
vmap_rules[add_p] = partial(broadcasting_binop_batching_rule, add)
vmap_rules[mul_p] = partial(broadcasting_binop_batching_rule, mul)

def vectorized_unop_batching_rule(op, axis_size, vals_in, dims_in):
  (x,), (x_bdim,) = vals_in, dims_in
  return op(x), x_bdim
vmap_rules[sin_p] = partial(vectorized_unop_batching_rule, sin)
vmap_rules[cos_p] = partial(vectorized_unop_batching_rule, cos)
vmap_rules[neg_p] = partial(vectorized_unop_batching_rule, neg)

def reduce_sum_batching_rule(axis_size, vals_in, dims_in, *, axis):
  (x,), (x_bdim,) = vals_in, dims_in
  new_axis = axis + (x_bdim <= axis)
  out_bdim = x_bdim - (new_axis < x_bdim)
  return reduce_sum(x, new_axis), out_bdim
vmap_rules[reduce_sum_p] = reduce_sum_batching_rule


# -

# Finally, we add a transformation API to kick off the trace:

def vmap(f, in_axes, out_axis):
  def batched_f(*args):
    axis_size, = {x.shape[ax] for x, ax in zip(args, in_axes)
                  if ax is not None}
    with new_main(BatchTrace, axis_size) as main:
      trace = BatchTrace(main)
      tracers_in = [BatchTracer(trace, x, ax) if ax is not None else x
                    for x, ax in zip(args, in_axes)]
      out = f(*tracers_in)
      tracer_out = full_raise(trace, out)
      val_out, batch_dim_out = tracer_out.val, tracer_out.batch_dim
    return move_batch_axis(axis_size, batch_dim_out, out_axis, val_out)
  return batched_f


# +
def add_one_to_a_scalar(scalar):
  assert np.ndim(scalar) == 0
  return 1 + scalar

vector_in = np.arange(3.)
vector_out = vmap(add_one_to_a_scalar, (0,), 0)(vector_in)

print(vector_in)
print(vector_out)


# +
def jacfwd(f, x):
  pushfwd = lambda v: jvp(f, (x,), (v,))[1]
  vecs_in = np.eye(np.size(x)).reshape(np.shape(x) * 2)
  return vmap(pushfwd, (0,), 0)(vecs_in)

def f(x):
  return sin(x)

jacfwd(f, np.arange(3.))
# -

# That's it for `jvp` and `vmap`! Before moving on, let's highlight a few
# simplifications in what we've seen so far compared to the full JAX
# implementation:
# 1. **Fewer, simpler primitives.** More primitives means more interpretation
# rules, and for more complex primitives (like for convolution or advanced
# indexing) each rule is harder to write. But the overarching design is no
# different.
# 1. **Transformations expect arrays in, single array out.**
# 2. **No symbolic zeros in autodiff.**
# 3. **No special call primitives yet.** The core machinery needs to be
#     generalized to handle the most flexible kind of higher-order primitive,
#     used by `jax.custom_jvp` and `jax.custom_vjp`.

# ## Part 2: Jaxprs, for `jit` and `vjp`
#
# The next transformations are the horizon are `jit` for just-in-time
# compilation and `vjp` for reverse-mode autodiff.  (`grad` is just a small
# wrapper around `vjp`.) For `jvp` and `vmap` we only needed each `Tracer` to
# carry a little    bit of extra context, but for both `jit` and `vjp` we need
# much richer context: we need to represent _programs_. That is, we need jaxprs!
#
# We need a program representation for `jit` because the purpose of `jit` is to
# stage computation out of Python. For    any computation we want to stage out,
# we need to be able to represent it as data, and build it up as we trace a
# Python function. Similarly, `vjp` needs a way to represent the computation for
# the backward pass of reverse-mode      autodiff. We use the same jaxpr program
# representation for both needs.
#
# (Building a program representation is the most
# [free](https://en.wikipedia.org/wiki/Free_object) kind of trace-
# transformation, and so except for issues around handling native Python control
# flow, any transformation could be      implemented by first tracing to a jaxpr
# and then interpreting the jaxpr.)
#
# The jaxpr term syntax is roughly:
#
# ```
# jaxpr ::=
#   { lambda <binder> , ... .
#     let <eqn>
#         ...
#     in <atom> }
#
# binder ::= <var>:<array_type>
# var ::= a | b | c | ...
# atom ::= <var> | <literal>
# literal ::= <int32> | <float32>
#
# eqn ::= <binder> = <primitive> [ <params> ] <atom> , ...
# ```
#
# The syntax of types is:
#
# ```
# jaxpr_type ::= [<array_type>, ...] -> [<array_type>, ...]
# array_type ::= <dtype>[<shape>]
# dtype ::= f32 | f64 | i32 | i64
# shape ::= <int> , ...
# ```
#
# How do we represent these as Python data structures? We reuse ShapedArrays to
# represent types, and we can represent the term syntax with a few Python
# structs:

# +
from typing import Dict, Set

class Var:
  aval: ShapedArray
  def __init__(self, aval): self.aval = aval

class Lit:
  val: Any
  aval: ShapedArray

  def __init__(self, val):
    self.val = val
    self.aval = raise_to_shaped(get_aval(self.val))

Atom = Union[Var, Lit]

class JaxprEqn(NamedTuple):
  primitive: Primitive
  inputs: List[Atom]
  params: Dict[str, Any]
  out_binder: Var

class Jaxpr(NamedTuple):
  in_binders: List[Var]
  eqns: List[JaxprEqn]
  out: Atom


def raise_to_shaped(aval):
  return ShapedArray(aval.shape, aval.dtype)


# +
class JaxprType:
  in_types: List[ShapedArray]
  out_type: ShapedArray

  def __init__(self, in_types, out_type):
    self.in_types = in_types
    self.out_type = out_type

  def __repr__(self):
    in_types = ', '.join(aval.str_short() for aval in self.in_types)
    out_type = self.out_type.str_short()
    return f'({in_types}) -> {out_type}'


def typecheck_jaxpr(jaxpr: Jaxpr) -> JaxprType:
  env: Set[Var] = set()

  for v in jaxpr.in_binders:
    env.add(v)

  for eqn in jaxpr.eqns:
    in_types = [typecheck_atom(env, x) for x in eqn.inputs]
    out_type = abstract_eval_rules[eqn.primitive](*in_types, **eqn.params)
    if not types_equal(out_type, eqn.out_binder.aval): raise TypeError
    env.add(eqn.out_binder)

  out_type = typecheck_atom(env, jaxpr.out)
  return JaxprType([v.aval for v in jaxpr.in_binders], out_type)

def typecheck_atom(env: Set[Var], x: Atom) -> ShapedArray:
  if isinstance(x, Var):
    if x not in env: raise TypeError("unbound variable")
    return x.aval
  elif isinstance(x, Lit):
    return raise_to_shaped(get_aval(x.val))
  else:
    assert False

def types_equal(a: ShapedArray, b: ShapedArray) -> bool:
  return a.shape == b.shape and a.dtype == b.dtype


# -

# Now that we have jaxprs as a data structure, we need ways to produce these
# from tracing Python code. In general there are two variants of how we trace to
# a jaxpr; `jit` uses one and `vjp` uses the other. We'll start with the one
# used   by `jit`, which is also used by control flow primitives like
# `lax.cond`, `lax.while_loop`, and `lax.scan`.

# +
# NB: the analogous class in JAX is called 'DynamicJaxprTracer'
class JaxprTracer(Tracer):
  __slots__ = ['aval']
  aval: ShapedArray

  def __init__(self, trace, aval):
    self._trace = trace
    self.aval = aval

# NB: the analogous class in JAX is called 'DynamicJaxprTrace'
class JaxprTrace(Trace):
  def new_arg(self, aval: ShapedArray) -> JaxprTracer:
    aval = raise_to_shaped(aval)
    tracer = JaxprTracer(self, aval)
    self.builder.tracer_to_var[id(tracer)] = Var(aval)
    return tracer

  def get_or_make_const_tracer(self, val: Any) -> JaxprTracer:
    tracer = self.builder.const_tracers.get(id(val))
    if tracer is None:
      tracer = JaxprTracer(self, raise_to_shaped(get_aval(val)))
      self.builder.add_const(tracer, val)
    return tracer
  pure = lift = get_or_make_const_tracer

  def process_primitive(self, primitive, tracers, params):
    avals_in = [t.aval for t in tracers]
    aval_out = abstract_eval_rules[primitive](*avals_in, **params)
    out_tracer = JaxprTracer(self, aval_out)
    inputs = [self.builder.getvar(t) for t in tracers]
    outvar = self.builder.add_var(out_tracer)
    self.builder.add_eqn(JaxprEqn(primitive, inputs, params, outvar))
    return out_tracer

  @property
  def builder(self):
    return self.main.global_data

# NB: in JAX, instead of a dict we attach impl rules to the Primitive instance
abstract_eval_rules = {}


# -

# Notice that we keep as interpreter-global data a builder object, which keeps
# track of variables, constants, and eqns  as we build up the jaxpr.

class JaxprBuilder:
  eqns: List[JaxprEqn]
  tracer_to_var: Dict[int, Var]
  const_tracers: Dict[int, JaxprTracer]
  constvals: Dict[Var, Any]

  def __init__(self):
    self.eqns = []
    self.tracer_to_var = {}
    self.const_tracers = {}
    self.constvals = {}

  def add_eqn(self, eqn: JaxprEqn) -> None:
    self.eqns.append(eqn)

  def add_var(self, tracer: JaxprTracer) -> Var:
    var = self.tracer_to_var.get(id(tracer))
    assert var is None
    var = self.tracer_to_var[id(tracer)] = Var(tracer.aval)
    return var

  def getvar(self, tracer: JaxprTracer) -> Var:
    var = self.tracer_to_var.get(id(tracer))
    assert var is not None
    return var

  def add_const(self, tracer: JaxprTracer, val: Any) -> Var:
    var = self.add_var(tracer)
    self.const_tracers[id(val)] = tracer
    self.constvals[var] = val
    return var

  def build(self, in_tracers: List[JaxprTracer], out_tracer: JaxprTracer
            ) -> Tuple[Jaxpr, List[Any]]:
    constvars, constvals = unzip2(self.constvals.items())
    t2v = lambda t: self.tracer_to_var[id(t)]
    in_binders = constvars + [t2v(t) for t in in_tracers]
    jaxpr = Jaxpr(in_binders, self.eqns, t2v(out_tracer))
    typecheck_jaxpr(jaxpr)
    return jaxpr, constvals


# The rules we need for `JaxprTrace.process_primitive` are essentially typing
# rules for primitive applications: given   the primitive, its parameters, and
# types for the inputs, the rule must produce a type for the output, which is
# then   packaged with the output `JaxprTracer`. We can use abstract evaluation
# rules for this same purpose, even though they  can be more general (since
# abstract evaluation rules need to work on ConcreteArray inputs as well). We'll
# reuse these abstract evaluation rules for the other jaxpr-producing trace
# machinery, where the potential extra generality is useful.

def broadcast_shapes(*shapes):
  assert len(shapes) > 1
  for sizes in zip(*shapes):
    sizes = [d for d in sizes if d != 1]
    if sizes[:-1] != sizes[1:]:
      raise Exception
  return tuple(next((d for d in sizes if d != 1), 1) for sizes in zip(*shapes))


# +
def broadcasting_binop_abstract_eval_rule(*avals_in):
  out_dtype = np.result_type(*map(np.result_type, avals_in))
  out_shape = broadcast_shapes(*map(np.shape, avals_in))
  return ShapedArray(out_shape, out_dtype)

abstract_eval_rules[add_p] = broadcasting_binop_abstract_eval_rule
abstract_eval_rules[mul_p] = broadcasting_binop_abstract_eval_rule

def vectorized_unop_abstract_eval_rule(aval_in):
  return ShapedArray(np.shape(aval_in), np.result_type(aval_in))

abstract_eval_rules[sin_p] = vectorized_unop_abstract_eval_rule
abstract_eval_rules[cos_p] = vectorized_unop_abstract_eval_rule
abstract_eval_rules[neg_p] = vectorized_unop_abstract_eval_rule

def reduce_sum_abstract_eval_rule(aval_in, *, axis):
  new_shape = [d for i, d in enumerate(aval_in.shape) if i != axis]
  return ShapedArray(tuple(new_shape), aval_in.dtype)
abstract_eval_rules[reduce_sum_p] = reduce_sum_abstract_eval_rule


# -

# To check our implementation, we can add a `make_jaxpr` transformation and
# first pretty-printer:

def make_jaxpr(f, avals_in):
  builder = JaxprBuilder()
  with new_main(JaxprTrace, builder) as main:
    trace = JaxprTrace(main)
    tracers_in = [trace.new_arg(aval) for aval in avals_in]
    out = f(*tracers_in)
    tracer_out = full_raise(trace, out)
    return builder.build(tracers_in, tracer_out)

# +
from collections import defaultdict
import itertools as it
import string

class PPrint:
  lines: List[Tuple[int, str]]

  def __init__(self, lines):
    self.lines = lines

  def indent(self, indent: int) -> 'PPrint':
    return PPrint([(indent + orig_indent, s) for orig_indent, s in self.lines])

  def __add__(self, rhs: 'PPrint') -> 'PPrint':
    return PPrint(self.lines + rhs.lines)

  def __rshift__(self, rhs: 'PPrint') -> 'PPrint':
    if not rhs.lines: return self
    if not self.lines: return rhs
    indent, s = self.lines[-1]
    indented_block = rhs.indent(indent + len(s))
    common_line = s + ' ' * rhs.lines[0][0] + rhs.lines[0][1]
    return PPrint(self.lines[:-1]
                  + [(indent, common_line)]
                  + indented_block.lines[1:])

  def __str__(self) -> str:
    return '\n'.join(' ' * indent + s for indent, s in self.lines)

def pp(s: Any) -> PPrint:
  return PPrint([(0, line) for line in str(s).splitlines()])

def vcat(ps: List[PPrint]) -> PPrint:
  return sum(ps, pp(''))

def pp_jaxpr(jaxpr: Jaxpr):
  namegen = (''.join(s) for r in it.count(1)
             for s in it.permutations(string.ascii_lowercase, r))
  names = defaultdict(lambda: next(namegen))
  in_binders = ', '.join(var_str(names, x) for x in jaxpr.in_binders)
  eqns = vcat([pp_eqn(names, e) for e in jaxpr.eqns])
  out = names[jaxpr.out] if isinstance(jaxpr.out, Var) else str(jaxpr.out.val)
  return (pp(f'{{ lambda {in_binders} .') +
          ((pp('let ') >> eqns) + pp(f'in {out} }}')).indent(2))

def var_str(names: Dict[Var, str], v: Var) -> str:
  return f'{names[v]}:{v.aval.str_short()}'

def pp_eqn(names: Dict[Var, str], eqn: JaxprEqn) -> PPrint:
  lhs = pp(var_str(names, eqn.out_binder))
  rhs = (pp(eqn.primitive.name) >> pp_params(eqn.params) >>
         pp(' '.join(names[x] if isinstance(x, Var) else str(x.val)
                     for x in eqn.inputs)))
  return lhs >> pp(' = ') >> rhs

def pp_params(params: Dict[str, Any]) -> PPrint:
  items = sorted(params.items())
  if items:
    return pp(' [ ') >> vcat([pp(f'{k}={v}') for k, v in items]) >> pp(' ] ')
  else:
    return pp(' ')


# -

jaxpr, consts = make_jaxpr(lambda x: 2. * x, [raise_to_shaped(get_aval(3.))])
print(pp_jaxpr(jaxpr))
print(typecheck_jaxpr(jaxpr))
