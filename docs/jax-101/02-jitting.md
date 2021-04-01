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
  name: python3
---

+++ {"id": "O-SkdlPxvETZ"}

# Just In Time Compilation with JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/master/docs/jax-101/02-jitting.ipynb)

*Authors: Rosalia Schneider & Vladimir Mikulik*

In this section, we will further explore how JAX works, and how we can make it performant.
We will discuss the `jax.jit()` transform, which will perform *Just In Time* (JIT) compilation
of a JAX Python function so it can be executed efficiently in XLA.

## How JAX transforms work

In the previous section, we discussed that JAX allows us to transform Python functions. This is done by first converting the Python function into a simple intermediate language called jaxpr. The transformations then work on the jaxpr representation. 

We can show a representation of the jaxpr of a function by using `jax.make_jaxpr`:

```{code-cell}
:id: P9Xj77Wx3Z2P
:outputId: 77d28eec-bfbb-487b-b10f-e47bc8f7dd0d

import jax
import jax.numpy as jnp

global_list = []

def log2(x):
  global_list.append(x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2)(3.0))
```

+++ {"id": "jiDsT7y0RwIp"}

The [Understanding Jaxprs](https://jax.readthedocs.io/en/latest/jaxpr.html) section of the documentation provides more information on the meaning of the above output.

Importantly, note how the jaxpr does not capture the side-effect of the function: there is nothing in it corresponding to `global_list.append(x)`. This is a feature, not a bug: JAX is designed to understand side-effect-free (a.k.a. functionally pure) code. If *pure function* and *side-effect* are unfamiliar terms, this is explained in a little more detail in [🔪 JAX - The Sharp Bits 🔪: Pure Functions](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions).

Of course, impure functions can still be written and even run, but JAX gives no guarantees about their behaviour once converted to jaxpr. However, as a rule of thumb, you can expect (but shouldn't rely on) the side-effects of a JAX-transformed function to run once (during the first call), and never again. This is because of the way that JAX generates jaxpr, using a process called 'tracing'.

When tracing, JAX wraps each argument by a *tracer* object. These tracers then record all JAX operations performed on them during the function call (which happens in regular Python). Then, JAX uses the tracer records to reconstruct the entire function. The output of that reconstruction is the jaxpr. Since the tracers do not record the Python side-effects, they do not appear in the jaxpr. However, the side-effects still happen during the trace itself.

Note: the Python `print()` function is not pure: the text output is a side-effect of the function. Therefore, any `print()` calls will only happen during tracing, and will not appear in the jaxpr:

```{code-cell}
:id: JxV2p7e2RawC
:outputId: af7472de-df57-412f-83cb-e468c4b02d64

def log2_with_print(x):
  print("printed x:", x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2_with_print)(3.))
```

+++ {"id": "f6W_YYwRRwGp"}

See how the printed `x` is a `Traced` object? That's the JAX internals at work.

The fact that the Python code runs at least once is strictly an implementation detail, and so shouldn't be relied upon. However, it's useful to understand as you can use it when debugging to print out intermediate values of a computation.

+++ {"id": "PgVqi6NlRdWZ"}

A key thing to understand is that jaxpr captures the function as executed on the parameters given to it. For example, if we have a conditional, jaxpr will only know about the branch we take:

```{code-cell}
:id: hn0CuphEZKZm
:outputId: 669c5880-790d-4e5c-b635-f3acc21a0e02

def log2_if_rank_2(x):
  if x.ndim == 2:
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2
  else:
    return x

print(jax.make_jaxpr(log2_if_rank_2)(jax.numpy.array([1, 2, 3])))
```

+++ {"id": "Qp3WhqaqvHyD"}

## JIT compiling a function

As explained before, JAX enables operations to execute on CPU/GPU/TPU using the same code.
Let's look at an example of computing a *Scaled Exponential Linear Unit*
([SELU](https://proceedings.neurips.cc/paper/6698-self-normalizing-neural-networks.pdf)), an
operation commonly used in deep learning:

```{code-cell}
:id: JAXFYtlRvD6p
:outputId: e850862d-c9b4-4cfe-85a2-fb707178a957

import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)
%timeit selu(x).block_until_ready()
```

+++ {"id": "ecN5lEXe6ncy"}

The code above is sending one operation at a time to the accelerator. This limits the ability of the XLA compiler to optimize our functions.

Naturally, what we want to do is give the XLA compiler as much code as possible, so it can fully optimize it. For this purpose, JAX provides the `jax.jit` transformation, which will JIT compile a JAX-compatible function. The example below shows how to use JIT to speed up the previous function.

```{code-cell}
:id: nJVEwPcH6bQX
:outputId: 64b5eeab-0861-4cac-9c48-34586aa14fb6

selu_jit = jax.jit(selu)

# Warm up
selu_jit(x).block_until_ready()

%timeit selu_jit(x).block_until_ready()
```

+++ {"id": "hMNKi1mYXQg5"}

Here's what just happened:

1) We defined `selu_jit` as the compiled version of `selu`.

2) We ran `selu_jit` once on `x`. This is where JAX does its tracing -- it needs to have some inputs to wrap in tracers, after all. The jaxpr is then compiled using XLA into very efficient code optimized for your GPU or TPU. Subsequent calls to `selu_jit` will now use that code, skipping our old Python implementation entirely. 

(If we didn't include the warm-up call separately, everything would still work, but then the compilation time would be included in the benchmark. It would still be faster, because we run many loops in the benchmark, but it wouldn't be a fair comparison.)

3) We timed the execution speed of the compiled version. (Note the use of `block_until_ready()`, which is required due to JAX's [Asynchronous execution](https://jax.readthedocs.io/en/latest/async_dispatch.html) model).

+++ {"id": "DRJ6R6-d9Q_U"}

## Why can't we just JIT everything?

After going through the example above, you might be wondering whether we should simply apply `jax.jit` to every function. To understand why this is not the case, and when we should/shouldn't apply `jit`, let's first check some cases where JIT doesn't work.

```{code-cell}
:id: GO1Mwd_3_W6g
:outputId: be3ebdb8-1c38-48a5-be62-c8bb730481b5
:tags: [raises-exception]

# Condition on value of x.

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

f_jit = jax.jit(f)
f_jit(10)  # Should raise an error. 
```

```{code-cell}
:id: LHlipkIMFUhi
:outputId: 39fc5739-c74c-4deb-bc11-fde0e3b95625
:tags: [raises-exception]

# While loop conditioned on x and n.

def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

g_jit = jax.jit(g)
g_jit(10, 20)  # Should raise an error. 
```

+++ {"id": "isz2U_XX_wH2"}

The problem is that we tried to condition on the *value* of an input to the function being jitted. The reason we can't do this is related to the fact mentioned above that jaxpr depends on the actual values used to trace it. 

The more specific information about the values we use in the trace, the more we can use standard Python control flow to express ourselves. However, being too specific means we can't reuse the same traced function for other values. JAX solves this by tracing at different levels of abstraction for different purposes.

For `jax.jit`, the default level is `ShapedArray` -- that is, each tracer has a concrete shape (which we're allowed to condition on), but no concrete value. This allows the compiled function to work on all possible inputs with the same shape -- the standard use case in machine learning. However, because the tracers have no concrete value, if we attempt to condition on one, we get the error above.

In `jax.grad`, the constraints are more relaxed, so you can do more. If you compose several transformations, however, you must satisfy the constraints of the most strict one. So, if you `jit(grad(f))`, `f` mustn't condition on value. For more detail on the interaction between Python control flow and JAX, see [🔪 JAX - The Sharp Bits 🔪: Control Flow](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow).

One way to deal with this problem is to rewrite the code to avoid conditionals on value. Another is to use special [control flow operators](https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators) like `jax.lax.cond`. However, sometimes that is impossible. In that case, you can consider jitting only part of the function. For example, if the most computationally expensive part of the function is inside the loop, we can JIT just that inner part (though make sure to check the next section on caching to avoid shooting yourself in the foot):

```{code-cell}
:id: OeR8hF-NHAML
:outputId: 2b15ebbc-0fd8-4de1-e37f-a25fb5d4a522

# While loop conditioned on x and n with a jitted body.

@jax.jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

g_inner_jitted(10, 20)
```

+++ {"id": "5XUT2acoHBz-"}

If we really need to JIT a function that has a condition on the value of an input, we can tell JAX to help itself to a less abstract tracer for a particular input by specifying `static_argnums`. The cost of this is that the resulting jaxpr is less flexible, so JAX will have to re-compile the function for every new value of the specified input. It is only a good strategy if the function is guaranteed to get limited different values.

```{code-cell}
:id: 2yQmQTDNAenY
:outputId: ab9737b2-b991-47fe-c3d7-ed4a39c38832

f_jit_correct = jax.jit(f, static_argnums=0)
print(f_jit_correct(10))
```

```{code-cell}
:id: R4SXUEu-M-u1
:outputId: 7b704cea-188e-438b-9497-a6369a147f0d

g_jit_correct = jax.jit(g, static_argnums=1)
print(g_jit_correct(10, 20))
```

+++ {"id": "LczjIBt2X2Ms"}

## When to use JIT

In many of the the examples above, jitting is not worth it:

```{code-cell}
:id: uMOqsNnqYApD
:outputId: 96e4348d-0702-41f8-a097-c64e5f1da1dc

print("g jitted:")
%timeit g_jit_correct(10, 20).block_until_ready()

print("g:")
%timeit g(10, 20)
```

+++ {"id": "cZmGYq80YP0j"}

This is because `jax.jit` introduces some overhead itself. Therefore, it usually only saves time if the compiled function is complex and you will run it numerous times. Fortunately, this is common in machine learning, where we tend to compile a large, complicated model, then run it for millions of iterations.

Generally, you want to jit the largest possible chunk of your computation; ideally, the entire update step. This gives the compiler maximum freedom to optimise.

+++ {"id": "hJMjUlRcIzVS"}

## Caching

It's important to understand the caching behaviour of `jax.jit`.

Suppose I define `f = jax.jit(g)`. When I first invoke `f`, it will get compiled, and the resulting XLA code will get cached. Subsequent calls of `f` will reuse the cached code. This is how `jax.jit` makes up for the up-front cost of compilation.

If I specify `static_argnums`, then the cached code will be used only for the same values of arguments labelled as static. If any of them change, recompilation occurs. If there are many values, then your program might spend more time compiling than it would have executing ops one-by-one.

Avoid calling `jax.jit` inside loops. Doing that effectively creates a new `f` at each call, which will get compiled each time instead of reusing the same cached function:

```{code-cell}
:id: 6MDSXCfmSZVZ
:outputId: cb955855-0d24-4ff1-fcf7-872dc46ee565

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_poorly(x, n):
  i = 0
  while i < n:
    # Don't do this!
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

print("jit called outside the loop:")
%timeit g_inner_jitted(10, 20).block_until_ready()

print("jit called inside the loop:")
%timeit g_inner_jitted_poorly(10, 20).block_until_ready()
```
