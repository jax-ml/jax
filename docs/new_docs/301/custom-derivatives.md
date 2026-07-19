---
jupytext:
  formats: md:myst
  notebook_metadata_filter: nosearch
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
nosearch: true
---

(jax-301-custom-derivatives)=
# Custom derivative rules with hijax primitives

<!--* freshness: { reviewed: '2026-07-19' } *-->

Sometimes you want to tell JAX how to differentiate something yourself.
Maybe the rule-by-rule derivative that autodiff computes is numerically
unstable, and you know a better formula for it as a whole. Maybe you're
calling out to code JAX can't trace into, like an external solver or
simulator. Or maybe the mathematically right derivative isn't the mechanical
one — as with iterative fixed-point solvers, where implicit differentiation
of the solution beats differentiating through the iterations. This page
works examples of each.

The recommended way to define custom derivatives is with a *hijax primitive*. A
hijax primitive is a unit of computation treated atomically by JAX's
higher-level transformations, namely autodiff and batching. Being atomic, a
primitive needs a rule defining how it behaves under each transformation.  But
instead of carrying rules for lower-level transformations, like evaluation or
translation to HLO or Mosaic, hijax primitives carry a Python implementation
written with ordinary JAX operations and using ordinary JAX types. After
higher-level transformations are finished, a hijax primitive is expanded into
that Python implementation.

By reading this page, you'll learn how to control JAX's autodiff and batching
transformations of your own operations. This page assumes familiarity with
`jax.jvp` and `jax.grad` and the mathematical meaning of JVPs and VJPs (see
{doc}`cookbook`).

Hijax primitives are still experimental -- expect imports from
`jax.experimental.hijax`, and expect the APIs to evolve. JAX's classic tools
for this job, the `jax.custom_jvp` and `jax.custom_vjp` decorators, remain
fully supported and can be more convenient for simple cases; they're covered
in {doc}`custom-jvp-vjp`. But hijax primitives allow for customizing both modes
at once, as well as controlling interactions with custom types and other
advanced autodiff features.

## TL;DR

Subclass `VJPHiPrimitive`, declare the input and output types, and define an
`expand` method giving the implementation. Then attach the derivative rules
you need: `vjp_fwd`/`vjp_bwd_retval` for reverse mode, `jvp` for forward
mode:

```{code-cell}
import jax
import jax.numpy as jnp
from jax.experimental.hijax import VJPHiPrimitive

class SinTimesY(VJPHiPrimitive):
  def __init__(self, x_aval, y_aval):
    self.in_avals = (x_aval, y_aval)  # input types
    self.out_aval = x_aval            # output type
    self.params = {}                  # static parameters (none here)
    super().__init__()

  # Implementation, used for evaluation and lowering (e.g. under jit).
  def expand(self, x, y):
    return jnp.sin(x) * y

  # Reverse-mode: forward pass returns (primal_out, residuals).
  def vjp_fwd(self, nzs_in, x, y):
    return self(x, y), (jnp.cos(x), jnp.sin(x), y)

  # Reverse-mode: backward pass maps (residuals, output cotangent) to a tuple
  # of input cotangents.
  def vjp_bwd_retval(self, res, g):
    cos_x, sin_x, y = res
    return (cos_x * g * y, sin_x * g)

  # Forward-mode rule (optional, only needed under e.g. jax.jvp).
  def jvp(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return self(x, y), jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot

def f(x, y):
  return SinTimesY(jax.typeof(x), jax.typeof(y))(x, y)
```

```{code-cell}
from jax import jvp, grad

print(f(2., 3.))
y, y_dot = jvp(f, (2., 3.), (1., 0.))
print(y)
print(y_dot)
print(grad(f)(2., 3.))
```

Note that a single hijax primitive can carry both forward- and reverse-mode
rules. (With the classic `jax.custom_jvp` and `jax.custom_vjp` decorators,
covered in {doc}`custom-jvp-vjp`, you must choose one mode per function.)

## Example problems

To get an idea of what problems hijax custom derivatives are meant to solve,
let's go over a few examples. A more thorough introduction to the
`VJPHiPrimitive` API is in the next section.

### Example: Numerical stability

One application of custom derivative rules is to improve the numerical
stability of differentiation.

Say we want to write a function called `log1pexp`, which computes
$x \mapsto \log ( 1 + e^x )$. We can write that using `jax.numpy`:

```{code-cell}
def log1pexp(x):
  return jnp.log(1. + jnp.exp(x))

log1pexp(3.)
```

Since it's written in terms of `jax.numpy`, it's JAX-transformable:

```{code-cell}
from jax import jit, grad, vmap

print(jit(log1pexp)(3.))
print(jit(grad(log1pexp))(3.))
print(vmap(jit(grad(log1pexp)))(jnp.arange(3.)))
```

But there's a numerical stability problem lurking here:

```{code-cell}
print(grad(log1pexp)(100.))
```

That doesn't seem right! After all, the derivative of $x \mapsto \log (1 +
e^x)$ is $x \mapsto \frac{e^x}{1 + e^x}$, and so for large values of $x$ we'd
expect the value to be about 1.

We can get a bit more insight into what's going on by looking at the jaxpr
for the gradient computation:

```{code-cell}
jit(grad(log1pexp)).trace(100.).jaxpr
```

Stepping through how the jaxpr would be evaluated, we can see that the last
line would involve multiplying values that floating point math will round to
0 and $\infty$, respectively, which is never a good idea. That is, we're
effectively evaluating `lambda x: (1 / (1 + jnp.exp(x))) * jnp.exp(x)` for
large `x`, which effectively turns into `0. * jnp.inf`.

Instead of generating such large and small values, hoping for a cancellation
that floats can't always provide, we'd rather just express the derivative
function as a more numerically stable program. In particular, we can write a
program that more closely evaluates the equal mathematical expression $1 -
\frac{1}{1 + e^x}$, with no cancellation in sight.

This problem is interesting because even though our definition of `log1pexp`
could already be JAX-differentiated (and transformed with `jit`, `vmap`,
...), we're not happy with the result of applying standard autodiff rules to
the primitives comprising `log1pexp` and composing the result. Instead, we'd
like to specify how the whole function `log1pexp` should be differentiated,
as a unit, and thus arrange those exponentials better.

Here's a solution using a hijax primitive:

```{code-cell}
class Log1pExp(VJPHiPrimitive):
  def __init__(self, x_aval):
    self.in_avals = (x_aval,)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    return jnp.log(1. + jnp.exp(x))

  def vjp_fwd(self, nzs_in, x):
    return self(x), x

  def vjp_bwd_retval(self, x, g):
    return ((1 - 1/(1 + jnp.exp(x))) * g,)

  def jvp(self, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return self(x), (1 - 1/(1 + jnp.exp(x))) * x_dot

  def batch_dim_rule(self, axis_data, in_dims):
    return in_dims[0]

def log1pexp(x):
  return Log1pExp(jax.typeof(x))(x)
```

```{code-cell}
print(grad(log1pexp)(100.))
```

```{code-cell}
print(jit(log1pexp)(3.))
print(jit(grad(log1pexp))(3.))
print(vmap(jit(grad(log1pexp)))(jnp.arange(3.)))
```

The `expand` method plays the role of the original Python function: it's
what runs when we evaluate `log1pexp` eagerly, and it's what gets traced when
we apply `jit`. The `vjp_fwd`/`vjp_bwd_retval` pair defines reverse-mode
differentiation, and `jvp` defines forward-mode. (The `batch_dim_rule` method
is a one-liner that tells `vmap` where the batch dimension of the output is;
more on that below.)

We can inspect the jaxpr of the gradient computation to confirm the stable
formula is what runs:

```{code-cell}
jit(grad(log1pexp)).trace(100.).jaxpr
```

### Example: Enforcing a differentiation convention

A related application is to enforce a differentiation convention, perhaps at
a boundary.

Consider the function $f : \mathbb{R}_+ \to \mathbb{R}_+$ with $f(x) =
\frac{x}{1 + \sqrt{x}}$, where we take $\mathbb{R}_+ = [0, \infty)$. We might
implement $f$ as a program like this:

```{code-cell}
def f(x):
  return x / (1 + jnp.sqrt(x))
```

As a mathematical function on $\mathbb{R}$ (the full real line), $f$ is not
differentiable at zero (because the limit defining the derivative doesn't
exist from the left). Correspondingly, autodiff produces a `nan` value:

```{code-cell}
print(grad(f)(0.))
```

But mathematically if we think of $f$ as a function on $\mathbb{R}_+$ then it
is differentiable at 0 [Rudin's Principles of Mathematical Analysis
Definition 5.1, or Tao's Analysis I 3rd ed. Definition 10.1.1 and Example
10.1.6]. Alternatively, we might say as a convention we want to consider the
directional derivative from the right. So there is a sensible value for the
Python function `grad(f)` to return at `0.0`, namely `1.0`. By default, JAX's
machinery for differentiation assumes all functions are defined over
$\mathbb{R}$ and thus doesn't produce `1.0` here.

We can use a custom derivative rule! In particular, we can define the rule in
terms of the derivative function $x \mapsto \frac{\sqrt{x} + 2}{2(\sqrt{x} +
1)^2}$ on $\mathbb{R}_+$:

```{code-cell}
class FOnRPlus(VJPHiPrimitive):
  def __init__(self, x_aval):
    self.in_avals = (x_aval,)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    return x / (1 + jnp.sqrt(x))

  def vjp_fwd(self, nzs_in, x):
    return self(x), x

  def vjp_bwd_retval(self, x, g):
    return (((jnp.sqrt(x) + 2) / (2 * (jnp.sqrt(x) + 1)**2)) * g,)

def f(x):
  return FOnRPlus(jax.typeof(x))(x)
```

```{code-cell}
print(grad(f)(0.))
```

### Example: Gradient clipping

While in some cases we want to express a mathematical differentiation
computation, in other cases we may even want to take a step away from
mathematics to adjust the computation autodiff performs. One canonical
example is reverse-mode gradient clipping.

For gradient clipping, we can use `jnp.clip` together with a
reverse-mode-only rule. The bounds `lo` and `hi` are ordinary traced inputs:
they aren't involved in differentiation, but they are runtime data (they
could even be `jit` tracers), so the forward rule saves them as residuals
and the backward rule returns `None` for their cotangents:

```{code-cell}
class ClipGradient(VJPHiPrimitive):
  def __init__(self, lo_aval, hi_aval, x_aval):
    self.in_avals = (lo_aval, hi_aval, x_aval)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, lo, hi, x):
    return x  # identity function

  def vjp_fwd(self, nzs_in, lo, hi, x):
    return self(lo, hi, x), (lo, hi)  # save bounds as residuals

  def vjp_bwd_retval(self, res, g):
    lo, hi = res
    return (None, None, jnp.clip(g, lo, hi))  # None: zero cotangents for lo, hi

  def batch_dim_rule(self, axis_data, in_dims):
    return in_dims[2]

def clip_gradient(lo, hi, x):
  return ClipGradient(jax.typeof(lo), jax.typeof(hi), jax.typeof(x))(lo, hi, x)
```

(Static parameters in `self.params` wouldn't work for the bounds: params
are baked into the primitive instance when it's constructed, so they can't
be traced values. Anything that might be dynamic data — like bounds passed
as arguments to a `jit`-compiled function — should be an ordinary input.
Save `self.params` for genuinely static data, like the Python function in
the `fixed_point` example below.)

```{code-cell}
import matplotlib.pyplot as plt

t = jnp.linspace(0, 10, 1000)

plt.plot(jnp.sin(t))
plt.plot(vmap(grad(jnp.sin))(t))
```

```{code-cell}
def clip_sin(x):
  x = clip_gradient(-0.75, 0.75, x)
  return jnp.sin(x)

plt.plot(clip_sin(t))
plt.plot(vmap(grad(clip_sin))(t))
```

### Example: Python debugging

Another application that is motivated by development workflow rather than
numerics is to set a `pdb` debugger trace in the backward pass of
reverse-mode autodiff.

When trying to track down the source of a `nan` runtime error, or just
examine carefully the cotangent (gradient) values being propagated, it can be
useful to insert a debugger at a point in the backward pass that corresponds
to a specific point in the primal computation:

```{code-cell}
import pdb

class Debug(VJPHiPrimitive):
  def __init__(self, x_aval):
    self.in_avals = (x_aval,)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    return x  # acts like identity

  def vjp_fwd(self, nzs_in, x):
    return self(x), x

  def vjp_bwd_retval(self, x, g):
    pdb.set_trace()
    return (g,)

def debug(x):
  return Debug(jax.typeof(x))(x)

def foo(x):
  y = x ** 2
  y = debug(y)  # insert pdb in corresponding backward pass step
  return jnp.sin(y)
```

```python
jax.grad(foo)(3.)

> <ipython-input-113-b19a2dc1abf7>(12)vjp_bwd_retval()
-> return (g,)
(Pdb) p x
Array(9., dtype=float32)
(Pdb) p g
Array(-0.91113025, dtype=float32)
(Pdb) q
```

### Example: Implicit function differentiation of iterative implementations

This example gets pretty deep in the mathematical weeds!

Another application for custom VJP rules is reverse-mode differentiation of
functions that are JAX-transformable (by `jit`, `vmap`, ...) but not
efficiently JAX-differentiable for some reason, perhaps because they involve
`lax.while_loop`. (It's not possible to produce an XLA HLO program that
efficiently computes the reverse-mode derivative of an XLA HLO While loop
because that would require a program with unbounded memory use, which isn't
possible to express in XLA HLO, at least without side-effecting interactions
through infeed/outfeed.)

For example, consider this `fixed_point` routine which computes a fixed
point by iteratively applying a function in a `while_loop`:

```{code-cell}
from jax.lax import while_loop

def fixed_point(f, a, x_guess):
  def cond_fun(carry):
    x_prev, x = carry
    return jnp.abs(x_prev - x) > 1e-6

  def body_fun(carry):
    _, x = carry
    return x, f(a, x)

  _, x_star = while_loop(cond_fun, body_fun, (x_guess, f(a, x_guess)))
  return x_star
```

This is an iterative procedure for numerically solving the equation $x =
f(a, x)$ for $x$, by iterating $x_{t+1} = f(a, x_t)$ until $x_{t+1}$ is
sufficiently close to $x_t$. The result $x^*$ depends on the parameters $a$,
and so we can think of there being a function $a \mapsto x^*(a)$ that is
implicitly defined by equation $x = f(a, x)$.

We can use `fixed_point` to run iterative procedures to convergence, for
example running Newton's method to calculate square roots while only
executing adds, multiplies, and divides:

```{code-cell}
def newton_sqrt(a):
  update = lambda a, x: 0.5 * (x + a / x)
  return fixed_point(update, a, a)
```

```{code-cell}
print(newton_sqrt(2.))
```

We can `vmap` or `jit` the function as well:

```{code-cell}
print(jit(vmap(newton_sqrt))(jnp.array([1., 2., 3., 4.])))
```

We can't apply reverse-mode automatic differentiation because of the
`while_loop`, but it turns out we wouldn't want to anyway: instead of
differentiating through the implementation of `fixed_point` and all its
iterations, we can exploit the mathematical structure to do something that is
much more memory-efficient (and FLOP-efficient in this case, too!). We can
instead use the implicit function theorem [Prop A.25 of Bertsekas's Nonlinear
Programming, 2nd ed.], which guarantees (under some conditions) the existence
of the mathematical objects we're about to use. In essence, we linearize at
the solution and solve those linear equations iteratively to compute the
derivatives we want.

Consider again the equation $x = f(a, x)$ and the function $x^*$. We want to
evaluate vector-Jacobian products like $v^\mathsf{T} \mapsto v^\mathsf{T}
\partial x^*(a_0)$.

At least in an open neighborhood around the point $a_0$ at which we want to
differentiate, let's assume that the equation $x^*(a) = f(a, x^*(a))$ holds
for all $a$. Since the two sides are equal as functions of $a$, their
derivatives must be equal as well, so let's differentiate both sides:

$\qquad \partial x^*(a) = \partial_0 f(a, x^*(a)) + \partial_1 f(a, x^*(a))
\partial x^*(a)$.

Setting $A = \partial_1 f(a_0, x^*(a_0))$ and $B = \partial_0 f(a_0,
x^*(a_0))$, we can write the quantity we're after more simply as

$\qquad \partial x^*(a_0) = B + A \partial x^*(a_0)$,

or, by rearranging,

$\qquad \partial x^*(a_0) = (I - A)^{-1} B$.

That means we can evaluate vector-Jacobian products like

$\qquad v^\mathsf{T} \partial x^*(a_0) = v^\mathsf{T} (I - A)^{-1} B =
w^\mathsf{T} B$,

where $w^\mathsf{T} = v^\mathsf{T} (I - A)^{-1}$, or equivalently
$w^\mathsf{T} = v^\mathsf{T} + w^\mathsf{T} A$, or equivalently
$w^\mathsf{T}$ is the fixed point of the map $u^\mathsf{T} \mapsto
v^\mathsf{T} + u^\mathsf{T} A$. That last characterization gives us a way to
write the VJP for `fixed_point` in terms of a call to `fixed_point`!
Moreover, after expanding $A$ and $B$ back out, we can see we need only to
evaluate VJPs of $f$ at $(a_0, x^*(a_0))$.

Here's the upshot. The function argument `f` isn't differentiated, so it goes
in `params` (functions are hashable), while `a` and `x_guess` are ordinary
traced inputs:

```{code-cell}
from functools import partial
from jax import vjp

class FixedPoint(VJPHiPrimitive):
  def __init__(self, a_aval, x_aval, *, f):
    self.in_avals = (a_aval, x_aval)
    self.out_aval = x_aval
    self.params = dict(f=f)
    super().__init__()

  def expand(self, a, x_guess):
    def cond_fun(carry):
      x_prev, x = carry
      return jnp.abs(x_prev - x) > 1e-6

    def body_fun(carry):
      _, x = carry
      return x, self.f(a, x)

    _, x_star = while_loop(cond_fun, body_fun, (x_guess, self.f(a, x_guess)))
    return x_star

  def vjp_fwd(self, nzs_in, a, x_guess):
    x_star = self(a, x_guess)
    return x_star, (a, x_star)

  def vjp_bwd_retval(self, res, x_star_bar):
    a, x_star = res
    _, vjp_a = vjp(lambda a: self.f(a, x_star), a)
    a_bar, = vjp_a(fixed_point(partial(rev_iter, self.f),
                               (a, x_star, x_star_bar),
                               x_star_bar))
    return a_bar, jnp.zeros_like(x_star)

def rev_iter(f, packed, u):
  a, x_star, x_star_bar = packed
  _, vjp_x = vjp(lambda x: f(a, x), x_star)
  return x_star_bar + vjp_x(u)[0]

def fixed_point(f, a, x_guess):
  a_aval = jax.tree.map(jax.typeof, a)
  x_aval = jax.tree.map(jax.typeof, x_guess)
  return FixedPoint(a_aval, x_aval, f=f)(a, x_guess)
```

```{code-cell}
print(newton_sqrt(2.))
```

```{code-cell}
print(grad(newton_sqrt)(2.))
print(grad(grad(newton_sqrt))(2.))
```

We can check our answers by differentiating `jnp.sqrt`, which uses a totally
different implementation:

```{code-cell}
print(grad(jnp.sqrt)(2.))
print(grad(grad(jnp.sqrt))(2.))
```

Notice that the backward rule calls `fixed_point` again, on the linear
problem, and that the parameter `a` passed there is a pytree of arrays: the
`in_avals` entries can themselves be pytrees of types, as discussed below.

A limitation to this approach is that the argument `f` can't close over any
values involved in differentiation, since it's a static parameter of the
primitive. That is, you might notice that we kept the parameter `a` explicit
in the argument list of `fixed_point`. For this use case, consider using the
low-level primitive `lax.custom_root`, which allows for derivatives in
closed-over variables with custom root-finding functions.

## Basic usage of `VJPHiPrimitive`

### Anatomy of a hijax primitive

A hijax primitive is a subclass of `VJPHiPrimitive`. Its `__init__` must set
three attributes and then call `super().__init__()`:

* `in_avals`, a tuple with one entry per positional argument, giving each
  argument's type (each entry can also be a pytree of types);
* `out_aval`, the output type (also possibly a pytree of types);
* `params`, a dict of hashable static parameters, which are made available as
  attributes on the instance (e.g. `self.f` for `params = dict(f=f)`).
  Params are static: they're baked into the primitive instance, so they
  can't be traced values, and dynamic data must instead be an input.

Since the types are fixed at construction time, the usual idiom is a wrapper
function that builds the primitive instance from the types of the arguments,
using `jax.typeof`, and immediately applies it:

```{code-cell}
class Square(VJPHiPrimitive):
  def __init__(self, x_aval):
    self.in_avals = (x_aval,)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    return x * x

def square(x):
  return Square(jax.typeof(x))(x)

print(square(3.))
```

The instance is callable, and calling it is what binds the primitive: in an
eager context it evaluates, and under a `jit` trace it records itself into
the jaxpr as a single equation. The types of the actual arguments are checked
against `in_avals`.

The only required method is `expand`, which gives the implementation as a
JAX-traceable Python function of the (non-static) arguments. Everything else
is optional, and is only needed if you use the corresponding transformation:

| method(s) | transformation |
|---|---|
| `expand` | evaluation and lowering (`jit`) |
| `vjp_fwd` and `vjp_bwd_retval` (or `vjp_bwd`) | reverse-mode autodiff (`grad`, `vjp`) |
| `jvp` | forward-mode autodiff (`jvp`) |
| `lin` and `linearized` | `jax.linearize` |
| `batch_dim_rule` (or `batch`) | `vmap` |
| `transpose` | transposition, for primitives linear in some inputs |

If you apply a transformation without having defined the corresponding
method, you get a `NotImplementedError` telling you what to implement.
Some rules can be defined generically in terms of others; see the helpers
below.

### Custom VJPs with `vjp_fwd` and `vjp_bwd_retval`

The pair `vjp_fwd`/`vjp_bwd_retval` works just like the `f_fwd`/`f_bwd` pair
of `jax.custom_vjp` ({doc}`custom-jvp-vjp`). In Haskell-like type
signatures:

```haskell
vjp_fwd :: (NonZeros, a) -> (b, c)
vjp_bwd_retval :: (c, CT b) -> CT a
```

The function `vjp_fwd` describes the forward pass: it takes the primal
inputs and returns a pair of the primal output and any "residual" data to be
stored for use by the backward pass. (Its extra first argument `nzs_in` is a
tuple of booleans indicating which inputs are being differentiated; you can
ignore it, or use it to avoid saving residuals that won't be needed.) The
primal output should usually be computed by calling `self(...)`, i.e. by
binding the primitive itself; that way the custom rules also apply under
higher-order differentiation.

The function `vjp_bwd_retval` describes the backward pass: it takes the
residuals and the cotangent of the output, and returns a tuple of cotangents
with one entry per primal input.

```{code-cell}
class Mul(VJPHiPrimitive):
  def __init__(self, x_aval, y_aval):
    self.in_avals = (x_aval, y_aval)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x, y):
    return x * y

  def vjp_fwd(self, nzs_in, x, y):
    return self(x, y), (x, y)

  def vjp_bwd_retval(self, res, g):
    x, y = res
    return (g * y, x * g)

def mul(x, y):
  return Mul(jax.typeof(x), jax.typeof(y))(x, y)

print(grad(mul)(2., 3.))
print(grad(mul, 1)(2., 3.))
```

(jax-301-vjp-bwd-accums)=

### The general backward API: `vjp_bwd` and gradient accumulators

`vjp_bwd_retval` is actually a convenience wrapper. The more general API is
`vjp_bwd`, with signature `vjp_bwd(self, res, outgrad, *arg_accums)`.
Instead of returning cotangent values, it receives one *gradient
accumulator* per primal input (matching the pytree structure of each entry
of `in_avals`) and pushes each cotangent contribution into the corresponding
accumulator; it returns `None` (or a dict of backward-pass logs, covered in
{ref}`jax-301-bwd-logging`). A subclass should override one of `vjp_bwd`
or `vjp_bwd_retval`, not both.

An accumulator is a `GradAccum` instance. It carries the expected cotangent
type as `acc.aval` — determined by the corresponding primal input's type,
since cotangent types are always a function of primal types — and accepts
contributions via `acc.accum(ct)`. There are three subclasses, importable
from `jax.experimental.hijax`, and which one your rule receives is decided
by the caller of autodiff:

* `ValAccum` holds a value and adds contributions to it functionally. This
  is what you get in ordinary `jax.grad` or `jax.vjp` usage, where the
  caller wants the gradient back as a value.
* `RefAccum` wraps a mutable array `Ref` (see {doc}`refs`) as `acc.ref`, and
  `accum` adds in place. You get one when the caller binds a gradient ref
  with `.with_refs(...)`. Your rule can also update `acc.ref` directly —
  including *indexed* updates that touch only part of the gradient buffer.
* `NullAccum` discards everything: the gradient for this input isn't needed
  (the input isn't being differentiated, or the caller passed
  `jax.ad.DontWant()`). Its `accum` is a no-op, so accumulating
  unconditionally is safe; your rule can also check for it and skip the work.

In fact, the base class's default `vjp_bwd` is just a few lines in terms of
`vjp_bwd_retval`:

```python
def vjp_bwd(self, res, outgrad, /, *arg_accums):
  args_grad = self.vjp_bwd_retval(res, outgrad)
  maybe_accum = lambda acc, v: isinstance(acc, GradAccum) and acc.accum(v)
  jax.tree.map(maybe_accum, arg_accums, args_grad)
```

So why override `vjp_bwd` directly? The main reason is in-place, sparse
gradient accumulation. If your operation reads only a small piece of a large
input, its cotangent is mostly zeros — and with `vjp_bwd_retval` you have no
choice but to materialize that dense array of zeros, every time. With
`vjp_bwd`, when you're handed a `RefAccum` you can instead add-update just
the entries you touched:

```{code-cell}
from jax.experimental.hijax import ShapedArray, ValAccum, RefAccum, NullAccum

class TakeElt(VJPHiPrimitive):
  def __init__(self, x_aval, i):
    self.in_avals = (x_aval,)
    self.out_aval = ShapedArray((), x_aval.dtype)
    self.params = dict(i=i)
    super().__init__()

  def expand(self, x):
    return x[self.i]

  def vjp_fwd(self, nzs_in, x):
    return self(x), None

  def vjp_bwd(self, res, g, x_acc):
    if isinstance(x_acc, NullAccum):
      return                              # gradient not wanted: skip the work
    elif isinstance(x_acc, RefAccum):
      x_acc.ref[self.i] += g              # sparse in-place update
    else:
      one_hot = jnp.zeros(x_acc.aval.shape, x_acc.aval.dtype).at[self.i].add(g)
      x_acc.accum(one_hot)                # dense functional fallback

def take_elt(x, i):
  return TakeElt(jax.typeof(x), i)(x)
```

Under plain `jax.grad`, the rule sees a `ValAccum` and takes the dense path:

```{code-cell}
x = jnp.arange(10.)
print(grad(lambda x: take_elt(x, 3))(x))
```

But when the caller binds a gradient ref, the rule sees a `RefAccum`, and
each backward pass writes a single element in place — here accumulating
gradients across several calls with no dense one-hot array in sight:

```{code-cell}
grad_ref = jax.new_ref(jnp.zeros(10))

for i in [3, 5, 3]:
  _, f_vjp = jax.vjp(lambda x: take_elt(x, i), x)
  f_vjp.with_refs(grad_ref)(1.0)

print(grad_ref)
```

```{code-cell}
def bwd_jaxpr():
  _, f_vjp = jax.vjp(lambda x: take_elt(x, 3), x)
  f_vjp.with_refs(grad_ref)(1.0)

print(jax.jit(bwd_jaxpr).trace().jaxpr)
```

The backward pass is a one-element read-add-write on the gradient ref.
JAX's own primitives play the same game: for example, `dynamic_slice`'s
transpose checks for a ref-backed accumulator and add-updates only the
sliced window, which is what makes the sparse-gradient examples in
{doc}`refs` work.

### Custom JVPs with `jvp`

The `jvp` method defines forward-mode differentiation. It takes a tuple of
primal inputs and a tuple of tangent inputs, and returns a pair of the primal
output and the tangent output. (The input tangents can be symbolic zeros in
some cases; see the symbolic zeros section below.)

```{code-cell}
class Sin(VJPHiPrimitive):
  def __init__(self, x_aval):
    self.in_avals = (x_aval,)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    return jnp.sin(x)

  def jvp(self, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return self(x), jnp.cos(x) * x_dot

def sin(x):
  return Sin(jax.typeof(x))(x)

y, y_dot = jvp(sin, (3.,), (1.,))
print(y)
print(y_dot)
```

One difference from `jax.custom_jvp`: by default, JAX does not automatically
derive reverse-mode differentiation from a hijax primitive's `jvp` rule, so
applying `grad` to `sin` as defined above raises a `NotImplementedError`
asking for `vjp_fwd`. You can define both sets of rules on the same primitive
(as in the TL;DR example above); unlike with `jax.custom_jvp` and
`jax.custom_vjp`, you never have to choose between them. (Or you can derive
the reverse-mode rules from the `jvp` rule — see
{ref}`jax-301-derived-rules` below.)

Both kinds of rules are also what make higher-order differentiation work:
`grad`-of-`grad` composes the VJP rules, while e.g. `jax.hessian`, which is
forward-over-reverse, needs the `jvp` rule as well.

### Custom linearization with `lin` and `linearized`

`jax.linearize` doesn't use the `jvp` or VJP rules; it has its own pair of
methods. The `lin` method is like `vjp_fwd`: it takes `nzs_in` and the primal
inputs, and returns the primal output paired with residuals. The `linearized`
method is the linear map itself: it takes the residuals and the input
tangents, and returns the output tangents:

```{code-cell}
class Sin(VJPHiPrimitive):
  def __init__(self, x_aval):
    self.in_avals = (x_aval,)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    return jnp.sin(x)

  def lin(self, nzs_in, x):
    return self(x), jnp.cos(x)

  def linearized(self, cos_x, x_dot):
    return cos_x * x_dot

def sin(x):
  return Sin(jax.typeof(x))(x)

y, f_lin = jax.linearize(sin, 3.)
print(y)
print(f_lin(1.))
```

(If you don't need to control the linearization itself, a primitive with a
`jvp` rule can derive these two methods instead — that's next.)

(jax-301-derived-rules)=

### Deriving rules from other rules

You don't have to write every rule by hand: helpers in
`jax.experimental.hijax` can derive some rules from others, giving
`jax.custom_jvp`-style behavior where everything follows from one
handwritten rule. Each helper name resolves to a pair that unpacks, right in
the class body, to the two methods it defines:

* `lin, linearized = linearize_from_jvp` derives linearization support by
  partially evaluating the `jvp` rule;
* `vjp_fwd, vjp_bwd_retval = vjp_from_lin` derives reverse mode from the
  `lin`/`linearized` rules (whether handwritten or themselves derived from
  `jvp`): it stores the linearization's residuals and transposes
  `linearized` on the backward pass;
* `vjp_fwd, vjp_bwd_retval = vjp_from_jvp` instead derives reverse mode
  directly from the `jvp` rule: it stores the primal inputs as residuals
  and, on the backward pass, linearizes and transposes the `jvp` rule —
  recomputing the rule's primal-dependent intermediates there, remat-style;
* `jvp = jvp_from_lin` goes the other way, deriving forward mode from
  handwritten `lin`/`linearized` rules. (It's a single function rather than
  a pair, since it defines a single method.)

Here's a primitive with a handwritten `jvp` rule and everything else
derived:

```{code-cell}
from jax.experimental.hijax import linearize_from_jvp, vjp_from_lin

class Sin(VJPHiPrimitive):
  def __init__(self, x_aval):
    self.in_avals = (x_aval,)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    return jnp.sin(x)

  def jvp(self, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return self(x), jnp.cos(x) * x_dot

  lin, linearized = linearize_from_jvp
  vjp_fwd, vjp_bwd_retval = vjp_from_lin

def sin(x):
  return Sin(jax.typeof(x))(x)

print(grad(sin)(3.))
y, sin_lin = jax.linearize(sin, 3.)
print(sin_lin(1.))
print(grad(grad(sin))(3.))
```

This combination — partially evaluate the `jvp` rule once in the forward
pass, save the linearization's residuals, and transpose the linear
remainder in the backward pass — is exactly how `jax.custom_jvp` implements
reverse mode. (`vjp_from_jvp` computes the same values with a different
memory/compute tradeoff, saving only the primal inputs and redoing the
linearization work in the backward pass.)

As with `jax.custom_jvp`, for the derived transposition to work, the JVP
rule's output tangents must be linear as a function of the input tangents.
And deriving in a circle, with `jvp = jvp_from_lin` and `lin, linearized =
linearize_from_jvp` on the same primitive, is an error.

### Hijax primitives in jaxprs

Because a hijax primitive is a real primitive, it appears as a single
equation in jaxprs:

```{code-cell}
jit(mul).trace(2., 3.).jaxpr
```

That's true under `jit` too. It's only at lowering time that `expand` is
traced and inlined. In an eager context, each call to the primitive calls
`expand` again (so, like any JAX function, it's best to keep `expand` free of
side effects, though harmless ones like `print` can be instructive):

```{code-cell}
class Noisy(VJPHiPrimitive):
  def __init__(self, x_aval):
    self.in_avals = (x_aval,)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    print('called expand!')
    return jnp.sin(x)

def noisy(x):
  return Noisy(jax.typeof(x))(x)

print(noisy(3.))
```

```{code-cell}
print(jit(noisy)(3.))
print(jit(noisy)(3.))  # tracing is cached: no more 'called expand!'
```

You can also use Python control flow in `expand` and in the derivative
rules, as long as the primitive is used eagerly (outside of `jit`), since the
rules then see concrete values:

```{code-cell}
class G(VJPHiPrimitive):
  def __init__(self, x_aval):
    self.in_avals = (x_aval,)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    if x > 0:
      return jnp.sin(x)
    else:
      return jnp.cos(x)

  def vjp_fwd(self, nzs_in, x):
    return self(x), x

  def vjp_bwd_retval(self, x, g):
    if x > 0:
      return (2 * g,)
    else:
      return (3 * g,)

def g(x):
  return G(jax.typeof(x))(x)

print(grad(g)(1.))
print(grad(g)(-1.))
```

### `vmap` with `batch_dim_rule` or `batch`

For `vmap` support, the easiest option is to define `batch_dim_rule`, which
takes axis metadata and the batch dimension of each argument (`None` for
unbatched arguments) and just returns the batch dimension of the output.
Given that data movement answer, JAX derives the batched computation
automatically by `vmap`-ing the primitive's other rules:

```{code-cell}
class MulV(Mul):
  def batch_dim_rule(self, axis_data, in_dims):
    x_dim, y_dim = in_dims
    return y_dim if x_dim is None else x_dim

def mul(x, y):
  return MulV(jax.typeof(x), jax.typeof(y))(x, y)

x = jnp.arange(3.)
y = jnp.arange(3.) + 1.
print(vmap(mul)(x, y))
print(vmap(mul, in_axes=(0, None))(x, 2.))
print(vmap(grad(mul))(x, y))
```

For full control over the batched computation itself, override the `batch`
method instead. It takes the axis metadata, the batched argument values, and
their batch dimensions (`None` for unbatched arguments), and returns the
batched output paired with its batch dimension — computed however you like,
in ordinary JAX operations. The classic reason is a kernel with a dedicated
batched variant: if `expand` calls a hand-written kernel (via Pallas,
`jax.ffi`, ...), `vmap`-ing it may be impossible or inefficient, and a
`batch` rule can instead dispatch to the batched kernel. But `batch` is the
right tool whenever you don't want the batched computation to be "`vmap`
the ops in `expand`" — it gives finer control over the batched program,
down to subtle details like where the `reduce_sum`s that autodiff
introduces for transposed broadcasts end up.

Here's the shape of the dedicated-batched-kernel case, with stand-in
"kernels" (and handling, for this example, only batching over the vector
argument):

```{code-cell}
def matvec_kernel(A, x):          # stand-in for e.g. a Pallas or ffi call
  return A @ x

def batched_matvec_kernel(A, X):  # the kernel's dedicated batched variant
  return X @ A.T

class MatVec(VJPHiPrimitive):
  def __init__(self, A_aval, x_aval):
    self.in_avals = (A_aval, x_aval)
    self.out_aval = ShapedArray((A_aval.shape[0],), A_aval.dtype)
    self.params = {}
    super().__init__()

  def expand(self, A, x):
    return matvec_kernel(A, x)

  def batch(self, axis_data, args, dims):
    A, X = args
    A_dim, x_dim = dims
    assert A_dim is None
    X = jnp.moveaxis(X, x_dim, 0)  # stack the batch of vectors along axis 0
    return batched_matvec_kernel(A, X), 0

def matvec(A, x):
  return MatVec(jax.typeof(A), jax.typeof(x))(A, x)

A = jnp.arange(6.).reshape(2, 3)
xs = jnp.arange(12.).reshape(4, 3)  # batch of 4 vectors
print(vmap(matvec, in_axes=(None, 0))(A, xs))
```

## More features and details

### Working with `list` / `tuple` / `dict` containers (and other pytrees)

The entries of `in_avals`, and `out_aval` itself, can be
pytrees ({ref}`jax-101-pytrees`) of types, so arguments
and outputs can be pytrees of arrays. Here's a contrived example:

```{code-cell}
from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])

from jax.experimental.hijax import Zero, instantiate_zeros

class FPt(VJPHiPrimitive):
  def __init__(self, pt_aval):
    self.in_avals = (pt_aval,)
    self.out_aval = {'a': pt_aval.x, 'b': (pt_aval.x, pt_aval.y)}
    self.params = {}
    super().__init__()

  def expand(self, pt):
    return {'a': pt.x ** 2, 'b': (jnp.sin(pt.x), jnp.cos(pt.y))}

  def vjp_fwd(self, nzs_in, pt):
    return self(pt), pt

  def vjp_bwd_retval(self, pt, g):
    g = jax.tree.map(instantiate_zeros, g,
                     is_leaf=lambda x: isinstance(x, Zero))
    a_bar, (b0_bar, b1_bar) = g['a'], g['b']
    x_bar = 2 * pt.x * a_bar + jnp.cos(pt.x) * b0_bar
    y_bar = -jnp.sin(pt.y) * b1_bar
    return (Point(x_bar, y_bar),)

def f(pt):
  return FPt(jax.tree.map(jax.typeof, pt))(pt)

def fun(pt):
  dct = f(pt)
  return dct['a'] + dct['b'][0]

pt = Point(1., 2.)
print(f(pt))
print(grad(fun)(pt))
```

### Symbolic zeros

The example above snuck in a new detail: the cotangents passed to the
backward rule can contain *symbolic zeros*. When part of the primitive's
output doesn't affect the value being differentiated (here `fun` doesn't use
`dct['b'][1]`), the corresponding cotangent is not an array of zeros but an
instance of the special `Zero` class, which records only the type. That's in
contrast to `jax.custom_vjp`, where symbolic zeros are opt-in via
`symbolic_zeros=True`.

Symbolic zeros let a rule avoid doing work (or avoid errors, for
non-differentiable outputs like integer values). If you don't want to handle
them, instantiate them into actual zero arrays with `instantiate_zeros`, as
above.

On the input side, the `nzs_in` argument to `vjp_fwd` reports symbolically
which inputs are being differentiated: it's a tuple of booleans, one per
input, where `False` means that input's tangent is symbolically zero, so no
cotangent for it will be used. A rule can use that to avoid saving unneeded
residuals:

```{code-cell}
class Mul2(VJPHiPrimitive):
  def __init__(self, x_aval, y_aval):
    self.in_avals = (x_aval, y_aval)
    self.out_aval = x_aval
    self.params = {}
    super().__init__()

  def expand(self, x, y):
    return x * y

  def vjp_fwd(self, nzs_in, x, y):
    x_nz, y_nz = nzs_in
    return self(x, y), (x if y_nz else None, y if x_nz else None)

  def vjp_bwd_retval(self, res, g):
    x, y = res
    return (g * y if y is not None else None,
            x * g if x is not None else None)

def mul2(x, y):
  return Mul2(jax.typeof(x), jax.typeof(y))(x, y)

print(grad(mul2, 0)(2., 3.))  # nzs_in == (True, False), saves only y
print(grad(mul2, 1)(2., 3.))  # nzs_in == (False, True), saves only x
```

Notice that the backward rule can return `None` for an input whose cotangent
isn't needed.

Symbolic zeros appear on the forward-mode side too: the tangents passed to a
`jvp` rule can be `Zero`s, for example when a primitive is applied to a mix
of differentiated inputs and constants. That's true whether the rule is
invoked directly by `jax.jvp` or partially evaluated by
`linearize_from_jvp`. As with cotangents, a rule can handle them explicitly
to exploit the sparsity, or clean them up with `instantiate_zeros`.

(jax-301-bwd-logging)=

### Logging data out of the backward pass

A `vjp_bwd` rule can also *return* something: a dict of named pytrees to
log out of the backward pass. To receive the logs, apply the VJP function
via its `with_logs` method: `f_vjp.with_logs(out_ct)` returns a pair
`(arg_cts, logs)`, where `logs` merges the dicts returned by all the rules
that ran. Logging is drop-by-default: a plain `f_vjp(out_ct)` call ignores
the logs, and under `jit` the logging computation is dead-code-eliminated,
so it costs nothing unless asked for. That makes it a lightweight way to
observe a backward pass — cotangent values, their norms, where a `nan`
first appears — without changing any function signatures:

```{code-cell}
class Square(VJPHiPrimitive):
  def __init__(self, x_aval, tag):
    self.in_avals = (x_aval,)
    self.out_aval = x_aval
    self.params = dict(tag=tag)
    super().__init__()

  def expand(self, x):
    return x ** 2

  def vjp_fwd(self, nzs_in, x):
    return self(x), x

  def vjp_bwd(self, x, g, x_acc):
    x_acc.accum(2. * x * g)
    return {self.tag: {'x': x, 'ct_in': g}}  # logged out of the backward pass

def square(x, tag='sq'):
  return Square(jax.typeof(x), tag)(x)

y, f_vjp = jax.vjp(square, 3.)
arg_cts, logs = f_vjp.with_logs(1.)
print(arg_cts)
print(logs)
```

The rules' dicts are merged with clobber semantics: on a key collision, the
entry logged earlier in forward order (that is, later in the backward pass)
wins. Distinct keys keep entries separate — here the tag is a static
parameter, so one primitive class can log under different names:

```{code-cell}
f = lambda x: square(square(x, 'inner'), 'outer')
_, f_vjp = jax.vjp(f, 2.)
_, logs = f_vjp.with_logs(1.)
print(logs)
```

Logs flow out through transposed control flow, taking a shape that reflects
the forward computation:

* out of a transposed `jit`, unchanged (and `with_logs` can itself be
  traced, e.g. called under a `jit`);
* out of a transposed `scan`, stacked leaf-wise along a leading axis,
  index-aligned with the forward iterations;
* out of a transposed `cond`, as a *sum*, represented as a tagged product:
  each logged key maps to a `CondSum` recording which branch ran, with one
  slot per branch (example below);
* out of a transposed `shard_map`, stacked along a leading mesh axis (a
  per-shard scalar becomes a vector of shape `(num_shards,)`).

For example, with a `scan`:

```{code-cell}
def f(xs):
  c, _ = jax.lax.scan(lambda c, x: (c + square(x), None), 0., xs)
  return c

xs = jnp.arange(1., 4.)
_, f_vjp = jax.vjp(f, xs)
_, logs = f_vjp.with_logs(1.)
print(logs)
```

And with a `cond`, only one branch runs, so its logs form a sum type — this
branch's logs *or* that branch's:

```{code-cell}
def g(x):
  return jax.lax.cond(x > 0,
                      lambda x: square(x, 'pos'),
                      lambda x: 2. * square(x, 'neg'),
                      x)

_, g_vjp = jax.vjp(g, 3.)
_, logs = g_vjp.with_logs(1.)
print(logs['pos'])
print(logs['neg'])
```

Each logged key gets its own `CondSum`: `index` records which branch ran
(`cond`'s false branch is 0 and its true branch is 1), and `branches` has
one slot per branch — the live value for the branch that ran, zeros for a
branch that logs the key but wasn't taken (here `'neg'`), and `None` for a
branch that doesn't log the key at all. Branches needn't agree on keys or
types, and nested `cond`s nest their `CondSum`s.

A rule's log return must be a dict (or `None`, meaning no logs). A few
transposed contexts don't yet plumb logs through — `jax.remat`,
`jax.custom_vjp` backward rules, and `lax.while_loop` — and logs inside
them are silently dropped, consistent with drop-by-default. For plumbing
data out of a backward pass with mutable refs instead, see {doc}`refs`.

### What we haven't covered

Custom derivatives are only part of the hijax story. Hijax primitives can
also:

* introduce new types beyond arrays, by subclassing `HiType` (immutable) or
  `MutableHiType` and registering them with `register_hitype`, with the
  primitive's `in_avals`/`out_aval` mentioning the new types — see
  {ref}`jax-301-hijax-types`;
* define a `transpose` rule, for primitives that are linear in some inputs;
* customize rematerialization via a `remat` method, and dead code
  elimination via a `dce` method.

Those deserve documents of their own. In the meantime, `tests/hijax_test.py`
is a good source of worked examples.
