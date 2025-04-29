# `jax.remat` / `jax.checkpoint` changes: what you need to know


## Contents


* [What's going on?](#whats-going-on)
* [How can I disable the change, and go back to the old behavior for
  now?](#how-can-i-disable-the-change-and-go-back-to-the-old-behavior-for-now)
* [Why are we doing this?](#why-are-we-doing-this)
* [What are the possible issues after the upgrade?](#what-are-the-possible-issues-after-the-upgrade)



## What’s going on?

As of [#11830](https://github.com/jax-ml/jax/pull/11830) we're switching on a new implementation of {func}`jax.checkpoint`, aka {func}`jax.remat` (the two names are aliases of one another). **For most code, there will be no changes.** But there may be some observable differences in edge cases; see [What are the possible issues after the upgrade?](#what-are-the-possible-issues-after-the-upgrade)


## How can I disable the change, and go back to the old behavior for now?

In case you have a problem with this change, **through version `jax==0.3.16`** it is possible to switch off the new implementation by setting the `jax_new_checkpoint` config option to be False, in any one of these ways:

1. set the shell environment variable `JAX_NEW_CHECKPOINT=0`;
2. execute `jax.config.update('jax_new_checkpoint', False)`;
3. if you parse flags with `absl`, pass the `--jax_new_checkpoint=False` option.

If you need to revert to the old implementation, **please reach out** on a GitHub issue so that we can make the new implementation work for you.

As of `jax==0.3.17` the `jax_new_checkpoint` config option is no longer
available. If you have an issue, please reach out on [the issue
tracker](https://github.com/jax-ml/jax/issues) so we can help fix it!


## Why are we doing this?

At the time of writing, JAX has two parallel implementations of `jax.checkpoint`. The new one has been used for months (e.g. by Pax and Flaxformer/T5X) on an opt-in basis. But it hasn't been on-by-default.

We want to switch the new implementation to be on-by-default, and then delete the old implementation. Using the new implementation, and removing the old implementation, gives users several benefits.


### User-customizable rematerialization policies

The main upside of the new implementation is a new feature corresponding to the `policy` argument. The idea is to give precise user control over what intermediates get saved (versus rematerialized) during the forward pass of automatic differentiation. By exercising this control over the memory-usage vs recomputation tradeoff, users can get significant performance wins, especially in large models and in our LLM MLPerf submission!

The full documentation for this feature is still forthcoming, but here's a quick example:


```python
from functools import partial
import jax

def apply_layer(W, x):
  return jnp.sin(jnp.dot(W, x))

@partial(jax.checkpoint, policy=jax.checkpoint_policies.checkpoint_dots)
def predict(params, x):
  for W in params[:-1]:
    x = apply_layer(W, x)
  return jnp.dot(params[-1], x)
```


By applying `jax.checkpoint` with `policy=jax.checkpoint_policies.checkpoint_dots` here, we ensure that only the results of matrix multiplies are allowed to be saved during the forward pass. The Jacobian coefficient values from `cos` applications, and the values of `sin` applications needed to compute them, are not saved from the forward pass and are instead recomputed during the backward pass. (Policies like this one can be effective on TPUs, where elementwise computations are effectively free but results from the matrix unit are worth saving.)


### Ability to rematerialize constants, not just operations with data dependence on arguments

The old `jax.checkpoint` implementation couldn't actually rematerialize computations without a data dependence on arguments to the decorated function. Consider this toy example:


```python
@jax.checkpoint
def f(x):
  a = some_function(jnp.arange(10_000_000))  # `a` does not depend on `x`
  return a * x
```


The old `jax.checkpoint` implementation was forced to save the value of `a`, which could require a lot of memory. The new `jax.checkpoint` implementation can rematerialize rather than save the value of `a`.


### Significantly less Python overhead in some cases

The new `jax.checkpoint` incurs significantly less Python overhead in some cases. [Simple overhead benchmarks](https://github.com/jax-ml/jax/blob/88636d2b649bfa31fa58a30ea15c925f35637397/benchmarks/api_benchmark.py#L511-L539) got 10x faster. These overheads only arise in eager op-by-op execution, so in the common case of using a `jax.checkpoint` under a `jax.jit` or similar the speedups aren't relevant. But still, nice!


### Enabling new JAX features by simplifying internals

This change unlocks big future user benefits too, like custom batching rules (the `vmap` analogue of `custom_vjp`) and a forward-differentiable upgrade to `custom_vjp`. It also significantly reduces complexity in parts of the JAX codebase, which will be good for maintainability and bug-fixing in general.


## What are the possible issues after the upgrade?


### Innocuous numerical changes

Because the new implementation can rematerialize more computations, including those of potentially large constants, some code may see small numerical changes. The magnitude of any numerical changes should be within the range we expect from changing compiler optimizations, like reordering of floating point operations. But some overly tight test tolerances may need to be slightly relaxed.


### The `concrete=True` option is removed.

The old `jax.checkpoint` implementation had a boolean `concrete` option, which allowed tracing on concrete Python values (rather than delaying all computations and only tracing on abstracted values). That option was seldom used, and in the cases where it was used there were much simpler alternatives. So we removed the option in the new `jax.checkpoint`.

For example, the overwhelmingly common use of `concrete=True` in Google code was to support passing an argument like `is_training`:


```python
@partial(jax.checkpoint, concrete=True)  # OLD jax.checkpoint API
def foo(x, is_training):
  if is_training:
    return g(x)
  else:
    return h(x)
```


With the new `jax.checkpoint` implementation, we can accomplish the same using the `static\_argnums` option:


```python
@partial(jax.checkpoint, static_argnums=(1,))  # NEW jax.checkpoint API
def foo(x, is_training):
  if is_training:
    ...
```


If `jax.numpy` operations need to be performed on static arguments, with their numerical results computed during Python tracing rather than delayed, we can use `static_argnums` with `jax.ensure_compile_time_eval()`. But it seems unlikely that you'd need this!

