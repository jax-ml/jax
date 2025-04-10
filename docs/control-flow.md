---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"id": "rg4CpMZ8c3ri"}

(control-flow)=
# Control flow and logical operators with JIT

<!--* freshness: { reviewed: '2024-11-11' } *-->

When executing eagerly (outside of `jit`), JAX code works with Python control flow and logical operators just like Numpy code. Using control flow and logical operators with `jit` is more complicated. 

In a nutshell, Python control flow and logical operators are evaluated at JIT compile time, such that the compiled function represents a single path through the [control flow graph](https://en.wikipedia.org/wiki/Control-flow_graph) (logical operators affect the path via short-circuiting). If the path depends on the values of the inputs, the function (by default) cannot be JIT compiled. The path may depend on the shape or dtype of the inputs, and the function is re-compiled every time it is called on an input with a new shape or dtype.

```{code-cell}
from jax import grad, jit
import jax.numpy as jnp
```

For example, this works:

```{code-cell}
:id: OZ_BJX0CplNC
:outputId: 60c902a2-eba1-49d7-c8c8-2f68616d660c

@jit
def f(x):
  for i in range(3):
    x = 2 * x
  return x

print(f(3))
```

+++ {"id": "22RzeJ4QqAuX"}

So does this:

```{code-cell}
:id: pinVnmRWp6w6
:outputId: 25e06cf2-474f-4782-af7c-4f5514b64422

@jit
def g(x):
  y = 0.
  for i in range(x.shape[0]):
    y = y + x[i]
  return y

print(g(jnp.array([1., 2., 3.])))
```

+++ {"id": "TStltU2dqf8A"}

But this doesn't, at least by default:

```{code-cell}
:id: 9z38AIKclRNM
:outputId: 38dd2075-92fc-4b81-fee0-b9dff8da1fac
:tags: [raises-exception]

@jit
def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

# This will fail!
f(2)
```

Neither does this:

```{code-cell}
:tags: [raises-exception]

@jit
def g(x):
  return (x > 0) and (x < 3)

# This will fail!
g(2)
```

+++ {"id": "pIbr4TVPqtDN"}

__What gives!?__

When we `jit`-compile a function, we usually want to compile a version of the function that works for many different argument values, so that we can cache and reuse the compiled code. That way we don't have to re-compile on each function evaluation.

For example, if we evaluate an `@jit` function on the array `jnp.array([1., 2., 3.], jnp.float32)`, we might want to compile code that we can reuse to evaluate the function on `jnp.array([4., 5., 6.], jnp.float32)` to save on compile time.

To get a view of your Python code that is valid for many different argument values, JAX traces it with the `ShapedArray` abstraction as input, where each abstract value represents the set of all array values with a fixed shape and dtype. For example, if we trace using the abstract value `ShapedArray((3,), jnp.float32)`, we get a view of the function that can be reused for any concrete value in the corresponding set of arrays. That means we can save on compile time.

But there's a tradeoff here: if we trace a Python function on a `ShapedArray((), jnp.float32)` that isn't committed to a specific concrete value, when we hit a line like `if x < 3`, the expression `x < 3` evaluates to an abstract `ShapedArray((), jnp.bool_)` that represents the set `{True, False}`. When Python attempts to coerce that to a concrete `True` or `False`, we get an error: we don't know which branch to take, and can't continue tracing! The tradeoff is that with higher levels of abstraction we gain a more general view of the Python code (and thus save on re-compilations), but we require more constraints on the Python code to complete the trace.

The good news is that you can control this tradeoff yourself. By having `jit` trace on more refined abstract values, you can relax the traceability constraints. For example, using the `static_argnames` (or `static_argnums`) argument to `jit`, we can specify to trace on concrete values of some arguments. Here's that example function again:

```{code-cell}
:id: -Tzp0H7Bt1Sn
:outputId: f7f664cb-2cd0-4fd7-c685-4ec6ba1c4b7a

def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

f = jit(f, static_argnames='x')

print(f(2.))
```

+++ {"id": "MHm1hIQAvBVs"}

Here's another example, this time involving a loop:

```{code-cell}
:id: iwY86_JKvD6b
:outputId: 48f9b51f-bd32-466f-eac1-cd23444ce937

def f(x, n):
  y = 0.
  for i in range(n):
    y = y + x[i]
  return y

f = jit(f, static_argnames='n')

f(jnp.array([2., 3., 4.]), 2)
```

+++ {"id": "nSPTOX8DvOeO"}

In effect, the loop gets statically unrolled.  JAX can also trace at _higher_ levels of abstraction, like `Unshaped`, but that's not currently the default for any transformation

+++ {"id": "wWdg8LTYwCW3"}

️⚠️ **functions with argument-__value__ dependent shapes**

These control-flow issues also come up in a more subtle way: numerical functions we want to __jit__ can't specialize the shapes of internal arrays on argument _values_ (specializing on argument __shapes__ is ok).  As a trivial example, let's make a function whose output happens to depend on the input variable `length`.

```{code-cell}
:id: Tqe9uLmUI_Gv
:outputId: 989be121-dfce-4bb3-c78e-a10829c5f883

def example_fun(length, val):
  return jnp.ones((length,)) * val
# un-jit'd works fine
print(example_fun(5, 4))
```

```{code-cell}
:id: fOlR54XRgHpd
:outputId: cf31d798-a4ce-4069-8e3e-8f9631ff4b71
:tags: [raises-exception]

bad_example_jit = jit(example_fun)
# this will fail:
bad_example_jit(10, 4)
```

```{code-cell}
:id: kH0lOD4GgFyI
:outputId: d009fcf5-c9f9-4ce6-fc60-22dc2cf21ade

# static_argnames tells JAX to recompile on changes at these argument positions:
good_example_jit = jit(example_fun, static_argnames='length')
# first compile
print(good_example_jit(10, 4))
# recompiles
print(good_example_jit(5, 4))
```

+++ {"id": "MStx_r2oKxpp"}

`static_argnames` can be handy if `length` in our example rarely changes, but it would be disastrous if it changed a lot!

Lastly, if your function has global side-effects, JAX's tracer can cause weird things to happen. A common gotcha is trying to print arrays inside __jit__'d functions:

```{code-cell}
:id: m2ABpRd8K094
:outputId: 4f7ebe17-ade4-4e18-bd8c-4b24087c33c3

@jit
def f(x):
  print(x)
  y = 2 * x
  print(y)
  return y
f(2)
```

+++ {"id": "uCDcWG4MnVn-"}

## Structured control flow primitives

There are more options for control flow in JAX. Say you want to avoid re-compilations but still want to use control flow that's traceable, and that avoids un-rolling large loops. Then you can use these 4 structured control flow primitives:

 - `lax.cond`  _differentiable_
 - `lax.while_loop` __fwd-mode-differentiable__
 - `lax.fori_loop` __fwd-mode-differentiable__ in general; __fwd and rev-mode differentiable__ if endpoints are static.
 - `lax.scan` _differentiable_

+++ {"id": "Sd9xrLMXeK3A"}

### `cond`
python equivalent:

```python
def cond(pred, true_fun, false_fun, operand):
  if pred:
    return true_fun(operand)
  else:
    return false_fun(operand)
```

```{code-cell}
:id: SGxz9JOWeiyH
:outputId: 942a8d0e-5ff6-4702-c499-b3941f529ca3

from jax import lax

operand = jnp.array([0.])
lax.cond(True, lambda x: x+1, lambda x: x-1, operand)
# --> array([1.], dtype=float32)
lax.cond(False, lambda x: x+1, lambda x: x-1, operand)
# --> array([-1.], dtype=float32)
```

+++ {"id": "lIYdn1woOS1n"}

`jax.lax` provides two other functions that allow branching on dynamic predicates:

- [`lax.select`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.select.html) is
  like a batched version of `lax.cond`, with the choices expressed as pre-computed arrays
  rather than as functions.
- [`lax.switch`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html) is
  like `lax.cond`, but allows switching between any number of callable choices.

In addition, `jax.numpy` provides several numpy-style interfaces to these functions:

- [`jnp.where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html) with
  three arguments is the numpy-style wrapper of `lax.select`.
- [`jnp.piecewise`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.piecewise.html)
  is a numpy-style wrapper of `lax.switch`, but switches on a list of boolean conditions rather than a single scalar index.
- [`jnp.select`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.select.html) has
  an API similar to `jnp.piecewise`, but the choices are given as pre-computed arrays rather
  than as functions. It is implemented in terms of multiple calls to `lax.select`.

+++ {"id": "xkOFAw24eOMg"}

### `while_loop`

python equivalent:
```
def while_loop(cond_fun, body_fun, init_val):
  val = init_val
  while cond_fun(val):
    val = body_fun(val)
  return val
```

```{code-cell}
:id: jM-D39a-c436
:outputId: 552fe42f-4d32-4e25-c8c2-b951160a3f4e

init_val = 0
cond_fun = lambda x: x < 10
body_fun = lambda x: x+1
lax.while_loop(cond_fun, body_fun, init_val)
# --> array(10, dtype=int32)
```

+++ {"id": "apo3n3HAeQY_"}

### `fori_loop`
python equivalent:
```
def fori_loop(start, stop, body_fun, init_val):
  val = init_val
  for i in range(start, stop):
    val = body_fun(i, val)
  return val
```

```{code-cell}
:id: dt3tUpOmeR8u
:outputId: 7819ca7c-1433-4d85-b542-f6159b0e8380

init_val = 0
start = 0
stop = 10
body_fun = lambda i,x: x+i
lax.fori_loop(start, stop, body_fun, init_val)
# --> array(45, dtype=int32)
```

+++ {"id": "SipXS5qiqk8e"}

### Summary

$$
\begin{array} {r|rr}
\hline \
\textrm{construct}
& \textrm{jit}
& \textrm{grad} \\
\hline \
\textrm{if} & ❌ & ✔ \\
\textrm{for} & ✔* & ✔\\
\textrm{while} & ✔* & ✔\\
\textrm{lax.cond} & ✔ & ✔\\
\textrm{lax.while_loop} & ✔ & \textrm{fwd}\\
\textrm{lax.fori_loop} & ✔ & \textrm{fwd}\\
\textrm{lax.scan} & ✔ & ✔\\
\hline
\end{array}
$$

<center>

$\ast$ = argument-<b>value</b>-independent loop condition - unrolls the loop

</center>

## Logical operators

`jax.numpy` provides `logical_and`, `logical_or`, and `logical_not`, which operate element-wise on arrays and can be evaluated under `jit` without recompiling. Like their Numpy counterparts, the binary operators do not short circuit. Bitwise operators (`&`, `|`, `~`) can also be used with `jit`.

For example, consider a function that checks if its input is a positive even integer. The pure Python and JAX versions give the same answer when the input is scalar.

```{code-cell}
def python_check_positive_even(x):
  is_even = x % 2 == 0
  # `and` short-circults, so when `is_even` is `False`, `x > 0` is not evaluated.
  return is_even and (x > 0)

@jit
def jax_check_positive_even(x):
  is_even = x % 2 == 0
  # `logical_and` does not short circuit, so `x > 0` is always evaluated.
  return jnp.logical_and(is_even, x > 0)

print(python_check_positive_even(24))
print(jax_check_positive_even(24))
```

When the JAX version with `logical_and` is applied to an array, it returns elementwise values.

```{code-cell}
x = jnp.array([-1, 2, 5])
print(jax_check_positive_even(x))
```

Python logical operators error when applied to JAX arrays of more than one element, even without `jit`. This replicates NumPy's behavior.

```{code-cell}
:tags: [raises-exception]

print(python_check_positive_even(x))
```

+++ {"id": "izLTvT24dAq0"}

## Python control flow + autodiff

Remember that the above constraints on control flow and logical operators are relevant only with `jit`. If you just want to apply `grad` to your python functions, without `jit`, you can use regular Python control-flow constructs with no problems, as if you were using [Autograd](https://github.com/hips/autograd) (or Pytorch or TF Eager).

```{code-cell}
:id: aAx0T3F8lLtu
:outputId: 383b7bfa-1634-4d23-8497-49cb9452ca52

def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

print(grad(f)(2.))  # ok!
print(grad(f)(4.))  # ok!
```
