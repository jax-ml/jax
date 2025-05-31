# ---
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
#
# jupyter:
#   jupytext:
#     formats: ipynb,md:myst,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Autodidax2, part 1: JAX from scratch, again

# If you want to understand how JAX works you could trying reading the code. But
# the code is complicated, often for no good reason. This notebook presents a
# stripped-back version without the cruft. It's a minimal version of JAX from
# first principles. Enjoy!

# ## Main idea: context-sensitive interpretation

# JAX is two things:
#   1. a set of primitive operations (roughly the NumPy API)
#   2. a set of interpreters over those primitives (compilation, AD, etc.)
#
# In this minimal version of JAX we'll start with just two primitive operations,
# addition and multiplication, and we'll add interpreters one by one. Suppose we
# have a user-defined function like this:

def foo(x):
  return mul(x, add(x, 3.0))


# We want to be able to interpret `foo` in different ways without changing its
# implementation: we want to evaluate it on concrete values, differentiate it,
# stage it out to an IR, compile it and so on.

# Here's how we'll do it. For each of these interpretations we'll define an
# `Interpreter` object with a rule for handling each primitive operation. We'll
# keep track of the *current* interpreter using a global context variable. The
# user-facing functions `add` and `mul` will dispatch to the current
# interpreter. At the beginning of the program the current interpreter will be
# the "evaluating" interpreter which just evaluates the operations on ordinary
# concrete data. Here's what this all looks like so far.

# +
from enum import Enum, auto
from contextlib import contextmanager
from typing import Any

# The full (closed) set of primitive operations
class Op(Enum):
  add = auto()  # addition on floats
  mul = auto()  # multiplication on floats

# Interpreters have rules for handling each primitive operation.
class Interpreter:
  def interpret_op(self, op: Op, args: tuple[Any, ...]):
    assert False, "subclass should implement this"

# Our first interpreter is the "evaluating interpreter" which performs ordinary
# concrete evaluation.
class EvalInterpreter:
  def interpret_op(self, op, args):
    assert all(isinstance(arg, float) for arg in args)
    match op:
      case Op.add:
        x, y = args
        return x + y
      case Op.mul:
        x, y = args
        return x * y
      case _:
        raise ValueError(f"Unrecognized primitive op: {op}")

# The current interpreter is initially the evaluating interpreter.
current_interpreter = EvalInterpreter()

# A context manager for temporarily changing the current interpreter
@contextmanager
def set_interpreter(new_interpreter):
  global current_interpreter
  prev_interpreter = current_interpreter
  try:
    current_interpreter = new_interpreter
    yield
  finally:
    current_interpreter = prev_interpreter

# The user-facing functions `mul` and `add` dispatch to the current interpreter.
def add(x, y): return current_interpreter.interpret_op(Op.add, (x, y))
def mul(x, y): return current_interpreter.interpret_op(Op.mul, (x, y))


# -

# At this point we can call `foo` with ordinary concrete inputs and see the
# results:

print(foo(2.0))

# ## Aside: forward-mode automatic differentiation

# For our second interpreter we're going to try forward-mode automatic
# differentiation (AD). Here's a quick introduction to forward-mode AD in case
# this is the first time you've come across it. Otherwise skip ahead to the
# "JVPInterprer" section.

# Suppose we're interested in the derivative of `foo(x)` evaluated at `x=2.0`.
# We could approximate it with finite differences:

print((foo(2.00001) - foo(2.0)) / 0.00001)

# The answer is close to 7.0 as expected. But computing it this way required two
# evaluations of the function (not to mention the roundoff error and truncation
# error). Here's a funny thing though. We can almost get the answer with a
# single evaluation:

print(foo(2.00001))

# The answer we're looking for, 7.0, is right there in the insignificant digits!

# Here's one way to think about what's happening. The initial argument to `foo`,
# `2.00001`, carries two pieces of data: a "primal" value, 2.0, and a "tangent"
# value, `1.0`. The representation of this primal-tangent pair, `2.00001`, is
# the sum of the two, with the tangent scaled by a small fixed epsilon, `1e-5`.
# Ordinary evaluation of `foo(2.00001)` propagates this primal-tangent pair,
# producing `10.0000700001` as the result. The primal and tangent components are
# well separated in scale so we can visually interpret the result as the
# primal-tangent pair (10.0, 7.0), ignoring the the ~1e-10 truncation error at
# the end.

# The idea with forward-mode differentiation is to do the same thing but exactly
# and explicitly (eyeballing floats doesn't really scale). We'll represent the
# primal-tangent pair as an actual pair instead of folding them both into a
# single floating point number. For each primitive operation we'll have a rule
# that describes how to propagate these primal tangent pairs. Let's work out the
# rules for our two primitives.

# Addition is easy. Consider `x + y` where `x = xp + xt * eps` and `y = yp + yt * eps`
# ("p" for "primal", "t" for "tangent"):
#
#      x + y = (xp + xt * eps) + (yp + yt * eps)
#            =   (xp + yp)             # primal component
#              + (xt + yt) * eps       # tangent component
#
# The result is a first-order polynomial in `eps` and we can read off the
# primal-tangent pair as (xp + yp, xt + yt).

# Multiplication is more interesting:
#
#      x * y = (xp + xt * eps) * (yp + yt * eps)
#            =    (xp * yp)                        # primal component
#               + (xp * yt + xt * yp) * eps        # tangent component
#               + (xt * yt)           * eps * eps  # quadratic component, vanishes in the eps->0 limit
#
# Now we have a second order polynomial. But as epsilon goes to zero the
# quadratic term vanishes and our primal-tangent pair
# is just `(xp * yp, xp * yt + xt * yp)`
# (In our earlier example with finite `eps` this term not vanishing is
# why we had the 1e-10 "truncation error").

# Putting this into code, we can write down the forward-AD rules for addition
# and multiplication and express `foo` in terms of these:

# +
from dataclasses import dataclass

# A primal-tangent pair is conventionally called a "dual number"
@dataclass
class DualNumber:
  primal  : float
  tangent : float

def add_dual(x : DualNumber, y: DualNumber) -> DualNumber:
  return DualNumber(x.primal + y.primal, x.tangent + y.tangent)

def mul_dual(x : DualNumber, y: DualNumber) -> DualNumber:
  return DualNumber(x.primal * y.primal, x.primal * y.tangent + x.tangent * y.primal)

def foo_dual(x : DualNumber) -> DualNumber:
  return mul_dual(x, add_dual(x, DualNumber(3.0, 0.0)))

print (foo_dual(DualNumber(2.0, 1.0)))


# -

# That works! But rewriting `foo` to use the `_dual` versions of addition and
# multiplication was a bit tedious. Let's get back to the main program and use
# our interpretation machinery to do the rewrite automatically.

# ## JVP Interpreter

# We'll set up a new interpreter called `JVPInterpreter` ("JVP" for
# "Jacobian-vector product") which propagates these dual numbers instead of
# ordinary values. The `JVPInterpreter` has methods 'add' and 'mul' that operate
# on dual number. They cast constant arguments to dual numbers as needed by
# calling `JVPInterpreter.lift`. In our manually rewritten version above we did
# that by replacing the literal `3.0` with `DualNumber(3.0, 0.0)`.

# +
# This is like DualNumber above except that is also has a pointer to the
# interpreter it belongs to, which is needed to avoid "perturbation confusion"
# in higher order differentiation.
@dataclass
class TaggedDualNumber:
  interpreter : Interpreter
  primal  : float
  tangent : float

class JVPInterpreter(Interpreter):
  def __init__(self, prev_interpreter: Interpreter):
    # We keep a pointer to the interpreter that was current when this
    # interpreter was first invoked. That's the context in which our
    # rules should run.
    self.prev_interpreter = prev_interpreter

  def interpret_op(self, op, args):
    args = tuple(self.lift(arg) for arg in args)
    with set_interpreter(self.prev_interpreter):
      match op:
        case Op.add:
          # Notice that we use `add` and `mul` here, which are the
          # interpreter-dispatching functions defined earlier.
          x, y = args
          return self.dual_number(
              add(x.primal, y.primal),
              add(x.tangent, y.tangent))

        case Op.mul:
          x, y = args
          x = self.lift(x)
          y = self.lift(y)
          return self.dual_number(
              mul(x.primal, y.primal),
              add(mul(x.primal, y.tangent), mul(x.tangent, y.primal)))

  def dual_number(self, primal, tangent):
    return TaggedDualNumber(self, primal, tangent)

  # Lift a constant value (constant with respect to this interpreter) to
  # a TaggedDualNumber.
  def lift(self, x):
    if isinstance(x, TaggedDualNumber) and x.interpreter is self:
      return x
    else:
      return self.dual_number(x, 0.0)

def jvp(f, primal, tangent):
  jvp_interpreter = JVPInterpreter(current_interpreter)
  dual_number_in = jvp_interpreter.dual_number(primal, tangent)
  with set_interpreter(jvp_interpreter):
    result = f(dual_number_in)
  dual_number_out = jvp_interpreter.lift(result)
  return dual_number_out.primal, dual_number_out.tangent

# Let's try it out:
print(jvp(foo, 2.0, 1.0))

# Because we were careful to consider nesting interpreters, higher-order AD
# works out of the box:

def derivative(f, x):
  _, tangent = jvp(f, x, 1.0)
  return tangent

def nth_order_derivative(n, f, x):
  if n == 0:
    return f(x)
  else:
    return derivative(lambda x: nth_order_derivative(n-1, f, x), x)


# -

print(nth_order_derivative(0, foo, 2.0))

print(nth_order_derivative(1, foo, 2.0))

print(nth_order_derivative(2, foo, 2.0))

# The rest are zero because `foo` is only a second-order polymonial
print(nth_order_derivative(3, foo, 2.0))

print(nth_order_derivative(4, foo, 2.0))


# There are some subtleties worth discussing. First, how do you tell if
# something is constant with respect to differentiation? It's tempting to say
# "it's a constant if and only if it's not a dual number". But actually dual
# numbers created by a *different* JVPInterpreter also need to be considered
# constants with respect to the JVPInterpreter we're currently handling. That's
# why we need the `x.interpreter is self` check in `JVPInterpreter.lift`. This
# comes up in higher order differentiation when there are multiple JVPInterprers
# in scope. The sort of bug where you accidentally interpret a dual number from
# a different interpreter as non-constant is sometimes called "perturbation
# confusion" in the literature. Here's an example program that would have given
# the wrong answer if we hadn't had the `and x.interpreter is self` check in
# `JVPInterpreter.lift`.

# +
def f(x):
  # g is constant in its (ignored) argument `y`. Its derivative should be zero
  # but our AD will mess it up if we don't distinguish perturbations from
  # different interpreters.
  def g(y):
    return x
  should_be_zero = derivative(g, 0.0)
  return mul(x, should_be_zero)

print(derivative(f, 0.0))
# -

# Another subtlety: `JVPInterpreter.add` and `JVPInterpreter.mul` describe
# addition and multiplication on dual numbers in terms of addition and
# multiplication on the primal and tangent components. But we don't use ordinary
# `+` and `*` for this. Instead we use our own `add` and `mul` functions which
# dispatch to the current interpreter. Before calling them we set the current
# interpreter to be the *previous* interpreter, i.e. the interpreter that was
# current when `JVPInterpreter` was first invoked. If we didn't do this we'd
# have an infinite recursion, with `add` and `mul` dispatching to
# `JVPInterpreter` endlessly. The advantage of using own `add` and `mul` instead
# of ordinary `+` and `*` is that it means we can nest these interpreters and do
# higher-order AD.

# At this point you might be wondering: have we just reinvented operator
# overloading? Python overloads the infix ops `+` and `*` to dispatch to the
# argument's `__add__` and `__mul__`. Could we have just used that mechanism
# instead of this whole interpreter business? Yes, actually. Indeed, the earlier
# automatic differentiation (AD) literature uses the term "operator overloading"
# to describe this style of AD implementation. One detail is that we can't rely
# exclusively on Python built-in overloading because that only lets us overload
# a handful of built-in infix ops whereas we eventually want to overload
# numpy-level operations like `sin` and `cos`. So we need our own mechanism.

# But there's a more important difference: our dispatch is based on *context*
# whereas traditional Python-style overloading is based on *data*. This is
# actually a recent development for JAX. The earliest versions of JAX looked
# more like traditional data-based overloading. An interpreter (a "trace" in JAX
# jargon) for an operation would be chosen based on data attached to the
# arguments to that operation. We've gradually made the interpreter-dispatch
# decision rely more and more on context rather than data (omnistaging [link],
# stackless [link]). The reason to prefer context-based interpretation over
# data-based interpretation is that it makes the implementation much simpler.

# All that said, we do *also* want to take advantage of Python's built-in
# overloading mechanism. That way we get the syntactic convenience of using
# infix operators `+` and `*` instead of writing out `add(..)` and `mul(..)`.
# But we'll put that aside for now.

# # 3. Staging to an untyped IR

# The two program transformations we've seen so far -- evaluation and JVP --
# both traverse the input program from top to bottom. They visit the operations
# one by one in the same order as ordinary evaluation. A convenient thing about
# top-to-bottom transformations is that they can be implemented eagerly, or
# "online", meaning that we can evaluate the program from top to bottom and
# perform the necessary transformations as we go. We never look at the entire
# program at once.

# But not all transformations work this way. For example, dead-code elimination
# requires traversing from bottom to top, collecting usage statistics on the way
# up and eliminating pure operations whose results have no uses. Another
# bottom-to-top transformation is AD transposition, which we use to implement
# reverse-mode AD. For these we need to first "stage" the program into an IR
# (internal representation), a data structure representing the program, which we
# can then traverse in any order we like. Building this IR from a Python program
# will be the goal of our third and final interpreter.

# First, let's define the IR. We'll do an untypes ANF IR to start. A function
# (we call IR functions "jaxprs" in JAX) will have a list of formal parameters,
# a list of operations, and a return value. Each argument to an operation must
# be an "atom", which is either a variable or a literal. The return value of the
# function is also an atom.

# +
Var = str           # Variables are just strings in this untyped IR
Atom = Var | float  # Atoms (arguments to operations) can be variables or (float) literals

# Equation - a single line in our IR like `z = mul(x, y)`
@dataclass
class Equation:
  var  : Var         # The variable name of the result
  op   : Op          # The primitive operation we're applying
  args : tuple[Atom] # The arguments we're applying the primitive operation to

# We call an IR function a "Jaxpr", for "JAX expression"
@dataclass
class Jaxpr:
  parameters : list[Var]      # The function's formal parameters (arguments)
  equations  : list[Equation] # The body of the function, a list of instructions/equations
  return_val : Atom           # The function's return value

  def __str__(self):
    lines = []
    lines.append(', '.join(b for b in self.parameters) + ' ->')
    for eqn in self.equations:
      args_str = ', '.join(str(arg) for arg in eqn.args)
      lines.append(f'  {eqn.var} = {eqn.op}({args_str})')
    lines.append(self.return_val)
    return '\n'.join(lines)


# -

# To build the IR from a Python function we define a `StagingInterpreter` that
# takes each operation and adds it to a growing list of all the operations we've
# seen so far:

# +
class StagingInterpreter(Interpreter):
  def __init__(self):
    self.equations = []         # A mutable list of all the ops we've seen so far
    self.name_counter = 0  # Counter for generating unique names

  def fresh_var(self):
    self.name_counter += 1
    return "v_" + str(self.name_counter)

  def interpret_op(self, op, args):
    binder = self.fresh_var()
    self.equations.append(Equation(binder, op, args))
    return binder

def build_jaxpr(f, num_args):
  interpreter = StagingInterpreter()
  parameters = tuple(interpreter.fresh_var() for _ in range(num_args))
  with set_interpreter(interpreter):
    result = f(*parameters)
  return Jaxpr(parameters, interpreter.equations, result)


# -

# Now we can construct an IR for a Python program and print it out:

print(build_jaxpr(foo, 1))


# We can also evaluate our IR by writing an explicit interpreter that traverses
# the operations one by one:

# +
def eval_jaxpr(jaxpr, args):
  # An environment mapping variables to values
  env = dict(zip(jaxpr.parameters, args))
  def eval_atom(x): return env[x] if isinstance(x, Var) else x
  for eqn in jaxpr.equations:
    args = tuple(eval_atom(x) for x in eqn.args)
    env[eqn.var] = current_interpreter.interpret_op(eqn.op, args)
  return eval_atom(jaxpr.return_val)

print(eval_jaxpr(build_jaxpr(foo, 1), (2.0,)))
# -

# We've written this interpreter in terms of `current_interpreter.interpret_op`
# which means we've done a full round-trip: interpretable Python program to IR
# to interpretable Python program. Since the result is "interpretable" we can
# differentiate it again, or stage it out or anything we like:

print(jvp(lambda x: eval_jaxpr(build_jaxpr(foo, 1), (x,)), 2.0, 1.0))

# ## Up next...

# That's it for part one of this tutorial. We've done two primitives, three
# interpreters and the tracing mechanism that weaves them together. In the next
# part we'll add types other than floats, error handling, compilation,
# reverse-mode AD and higher-order primitives. Note that the second part is
# structured differently. Rather than trying to have a top-to-bottom order that
# obeys both code dependencies (e.g. data structures need to be defined before
# they're used) and pedagogical dependencies (concepts need to be introduced
# before they're implemented) we're going with a single file that can be approached
# in any order.
