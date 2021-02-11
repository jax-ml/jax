---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Writing custom Jaxpr interpreters in JAX

JAX offers several composable function transformations (`jit`, `grad`, `vmap`,
etc.) that enable writing concise, accelerated code. 

Here we show how to add your own function transformations to the system, by writing a custom Jaxpr interpreter. And we'll get composability with all the other transformations for free.

**This example uses internal JAX APIs, which may break at any time. Anything not in [the API Documentation](https://jax.readthedocs.io/en/latest/jax.html) should be assumed internal.**

```{code-cell}
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax import random
```

## What is JAX doing?

JAX provides a NumPy-like API for numerical computing which can be used as is, but JAX's true power comes from composable function transformations. Take the `jit` function transformation, which takes in a function and returns a semantically identical function but is lazily compiled by XLA for accelerators.

```{code-cell}
x = random.normal(random.PRNGKey(0), (5000, 5000))
def f(w, b, x):
  return jnp.tanh(jnp.dot(x, w) + b)
fast_f = jit(f)
```

When we call `fast_f`, what happens? JAX traces the function and constructs an XLA computation graph. The graph is then JIT-compiled and executed. Other transformations work similarly in that they first trace the function and handle the output trace in some way. To learn more about Jax's tracing machinery, you can refer to the ["How it works"](https://github.com/google/jax#how-it-works) section in the README.

## Jaxpr tracer

A tracer of special importance in Jax is the Jaxpr tracer, which records ops into a Jaxpr (Jax expression). A Jaxpr is a data structure that can be evaluated like a mini functional programming language and 
thus Jaxprs are a useful intermediate representation
for function transformation. 


To get a first look at Jaxprs, consider the `make_jaxpr` transformation. `make_jaxpr` is essentially a "pretty-printing" transformation:
it transforms a function into one that, given example arguments, produces a Jaxpr representation of its computation.
Although we can't generally use the Jaxprs that it returns, it is useful for debugging and introspection.
Let's use it to look at how some example Jaxprs
are structured.

```{code-cell}
def examine_jaxpr(typed_jaxpr):
  jaxpr = typed_jaxpr.jaxpr
  print("invars:", jaxpr.invars)
  print("outvars:", jaxpr.outvars)
  print("constvars:", jaxpr.constvars)
  for eqn in jaxpr.eqns:
    print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
  print()
  print("jaxpr:", jaxpr)

def foo(x):
  return x + 1
print("foo")
print("=====")
examine_jaxpr(jax.make_jaxpr(foo)(5))

print()

def bar(w, b, x):
  return jnp.dot(w, x) + b + jnp.ones(5), x
print("bar")
print("=====")
examine_jaxpr(jax.make_jaxpr(bar)(jnp.ones((5, 10)), jnp.ones(5), jnp.ones(10)))
```

* `jaxpr.invars` - the `invars` of a Jaxpr are a list of the input variables to Jaxpr, analogous to arguments in Python functions
* `jaxpr.outvars` - the `outvars` of a Jaxpr are the variables that are returned by the Jaxpr. Every Jaxpr has multiple outputs.
* `jaxpr.constvars` - the `constvars` are a list of variables that are also inputs to the Jaxpr, but correspond to constants from the trace (we'll go over these in more detail later)
* `jaxpr.eqns` - a list of equations, which are essentially let-bindings. Each equation is list of input variables, a list of output variables, and a *primitive*, which is used to evaluate inputs to produce outputs. Each equation also has a `params`, a dictionary of parameters.

All together, a Jaxpr encapsulates a simple program that can be evaluated with inputs to produce an output. We'll go over how exactly to do this later. The important thing to note now is that a Jaxpr is a data structure that can be manipulated and evaluated in whatever way we want.

### Why are Jaxprs useful?

Jaxprs are simple program representations that are easy to transform. And because Jax lets us stage out Jaxprs from Python functions, it gives us a way to transform numerical programs written in Python.

## Your first interpreter: `invert`

Let's try to implement a simple function "inverter", which takes in the output of the original function and returns the inputs that produced those outputs. For now, let's focus on simple, unary functions which are composed of other invertible unary functions.

Goal:
```python
def f(x):
  return jnp.exp(jnp.tanh(x))
f_inv = inverse(f)
assert jnp.allclose(f_inv(f(1.0)), 1.0)
```

The way we'll implement this is by (1) tracing `f` into a Jaxpr, then (2) interpreting the Jaxpr *backwards*. While interpreting the Jaxpr backwards, for each equation we'll look up the primitive's inverse in a table and apply it.

### 1. Tracing a function

We can't use `make_jaxpr` for this, because we need to pull out constants created during the trace to pass into the Jaxpr. However, we can write a function that does something very similar to `make_jaxpr`.

```{code-cell}
# Importing Jax functions useful for tracing/interpreting.
import numpy as np
from functools import wraps

from jax import core
from jax import lax
from jax._src.util import safe_map
```

This function first flattens its arguments into a list, which are the abstracted and wrapped as partial values. The `pe.trace_to_jaxpr` function is used to then trace a function into a Jaxpr
from a list of partial value inputs.

```{code-cell}
def f(x):
  return jnp.exp(jnp.tanh(x))

closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(5))
print(closed_jaxpr)
print(closed_jaxpr.literals)
```

### 2. Evaluating a Jaxpr


Before we write a custom Jaxpr interpreter, let's first implement the "default" interpreter, `eval_jaxpr`, which evaluates the Jaxpr as-is, computing the same values that the original, un-transformed Python function would. 

To do this, we first create an environment to store the values for each of the variables, and update the environment with each equation we evaluate in the Jaxpr.

```{code-cell}
def eval_jaxpr(jaxpr, consts, *args):
  # Mapping from variable -> value
  env = {}
  
  def read(var):
    # Literals are values baked into the Jaxpr
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val

  # Bind args and consts to environment
  write(core.unitvar, core.unit)
  safe_map(write, jaxpr.invars, args)
  safe_map(write, jaxpr.constvars, consts)

  # Loop through equations and evaluate primitives using `bind`
  for eqn in jaxpr.eqns:
    # Read inputs to equation from environment
    invals = safe_map(read, eqn.invars)  
    # `bind` is how a primitive is called
    outvals = eqn.primitive.bind(*invals, **eqn.params)
    # Primitives may return multiple outputs or not
    if not eqn.primitive.multiple_results: 
      outvals = [outvals]
    # Write the results of the primitive into the environment
    safe_map(write, eqn.outvars, outvals) 
  # Read the final result of the Jaxpr from the environment
  return safe_map(read, jaxpr.outvars) 
```

```{code-cell}
closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(5))
eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, jnp.ones(5))
```

Notice that `eval_jaxpr` will always return a flat list even if the original function does not.

Furthermore, this interpreter does not handle `subjaxprs`, which we will not cover in this guide. You can refer to `core.eval_jaxpr` ([link](https://github.com/google/jax/blob/master/jax/core.py#L185-L212)) to see the edge cases that this interpreter does not cover.


### Custom `inverse` Jaxpr interpreter

An `inverse` interpreter doesn't look too different from `eval_jaxpr`. We'll first set up the registry which will map primitives to their inverses. We'll then write a custom interpreter that looks up primitives in the registry.

It turns out that this interpreter will also look similar to the "transpose" interpreter used in reverse-mode autodifferentiation [found here](https://github.com/google/jax/blob/master/jax/interpreters/ad.py#L141-L187).

```{code-cell}
inverse_registry = {}
```

We'll now register inverses for some of the primitives. By convention, primitives in Jax end in `_p` and a lot of the popular ones live in `lax`.

```{code-cell}
inverse_registry[lax.exp_p] = jnp.log
inverse_registry[lax.tanh_p] = jnp.arctanh
```

`inverse` will first trace the function, then custom-interpret the Jaxpr. Let's set up a simple skeleton.

```{code-cell}
def inverse(fun):
  @wraps(fun)
  def wrapped(*args, **kwargs):
    # Since we assume unary functions, we won't
    # worry about flattening and
    # unflattening arguments
    closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
    out = inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
    return out[0]
  return wrapped
```

Now we just need to define `inverse_jaxpr`, which will walk through the Jaxpr backward and invert primitives when it can.

```{code-cell}
def inverse_jaxpr(jaxpr, consts, *args):
  env = {}
  
  def read(var):
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val
  # Args now correspond to Jaxpr outvars
  write(core.unitvar, core.unit)
  safe_map(write, jaxpr.outvars, args)
  safe_map(write, jaxpr.constvars, consts)

  # Looping backward
  for eqn in jaxpr.eqns[::-1]:
    #  outvars are now invars 
    invals = safe_map(read, eqn.outvars)
    if eqn.primitive not in inverse_registry:
      raise NotImplementedError("{} does not have registered inverse.".format(
          eqn.primitive
      ))
    # Assuming a unary function 
    outval = inverse_registry[eqn.primitive](*invals)
    safe_map(write, eqn.invars, [outval])
  return safe_map(read, jaxpr.invars)
```

That's it!

```{code-cell}
def f(x):
  return jnp.exp(jnp.tanh(x))

f_inv = inverse(f)
assert jnp.allclose(f_inv(f(1.0)), 1.0)
```

Importantly, you can trace through a Jaxpr interpreter.

```{code-cell}
jax.make_jaxpr(inverse(f))(f(1.))
```

That's all it takes to add a new transformation to a system, and you get composition with all the others for free! For example, we can use `jit`, `vmap`, and `grad` with `inverse`!

```{code-cell}
jit(vmap(grad(inverse(f))))((jnp.arange(5) + 1.) / 5.)
```

## Exercises for the reader

* Handle primitives with multiple arguments where inputs are partially known, for example `lax.add_p`, `lax.mul_p`.
* Handle `xla_call` and `xla_pmap` primitives, which will not work with both `eval_jaxpr` and `inverse_jaxpr` as written.
