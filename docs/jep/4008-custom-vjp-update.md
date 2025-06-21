# `custom_vjp` and `nondiff_argnums` update guide
_mattjj@_
_Oct 14 2020_

This doc assumes familiarity with `jax.custom_vjp`, as described in the [Custom
derivative rules for JAX-transformable Python
functions](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
notebook.

## What to update

After JAX [PR #4008](https://github.com/jax-ml/jax/pull/4008), the arguments
passed into a `custom_vjp` function's `nondiff_argnums` can't be `Tracer`s (or
containers of `Tracer`s), which basically means to allow for
arbitrarily-transformable code `nondiff_argnums` shouldn't be used for
array-valued arguments. Instead, `nondiff_argnums` should be used only for
non-array values, like Python callables or shape tuples or strings.

Wherever we used to use `nondiff_argnums` for array values, we should just pass
those as regular arguments. In the `bwd` rule, we need to produce values for them,
but we can just produce `None` values to indicate there's no corresponding
gradient value.

For example, here's the **old** way to write `clip_gradient`, which won't work
when `hi` and/or `lo` are `Tracer`s from some JAX transformation.

```python
from functools import partial
import jax

@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def clip_gradient(lo, hi, x):
  return x  # identity function

def clip_gradient_fwd(lo, hi, x):
  return x, None  # no residual values to save

def clip_gradient_bwd(lo, hi, _, g):
  return (jnp.clip(g, lo, hi),)

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)
```

Here's the **new**, awesome way, which supports arbitrary transformations:

```python
import jax

@jax.custom_vjp  # no nondiff_argnums!
def clip_gradient(lo, hi, x):
  return x  # identity function

def clip_gradient_fwd(lo, hi, x):
  return x, (lo, hi)  # save lo and hi values as residuals

def clip_gradient_bwd(res, g):
  lo, hi = res
  return (None, None, jnp.clip(g, lo, hi))  # return None for lo and hi

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)
```

If you use the old way instead of the new way, you'll get a loud error in any
case where something might go wrong (namely when there's a `Tracer` passed into
a `nondiff_argnums` argument).

Here's a case where we actually need `nondiff_argnums` with `custom_vjp`:

```python
from functools import partial
import jax

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def skip_app(f, x):
  return f(x)

def skip_app_fwd(f, x):
  return skip_app(f, x), None

def skip_app_bwd(f, _, g):
  return (g,)

skip_app.defvjp(skip_app_fwd, skip_app_bwd)
```


## Explanation

Passing `Tracer`s into `nondiff_argnums` arguments was always buggy. While there
were some cases that worked correctly, others would lead to complex and
confusing error messages.

The essence of the bug was that `nondiff_argnums` was implemented in a way that
acted very much like lexical closure. But lexical closure over `Tracer`s wasn't
at the time intended to work with `custom_jvp`/`custom_vjp`. Implementing
`nondiff_argnums` that way was a mistake!

**[PR #4008](https://github.com/jax-ml/jax/pull/4008) fixes all lexical closure
issues with `custom_jvp` and `custom_vjp`.** Woohoo! That is, now `custom_jvp`
and `custom_vjp` functions and rules can close over `Tracer`s to our hearts'
content. For all non-autodiff transformations, things will Just Work. For
autodiff transformations, we'll get a clear error message about why we can't
differentiate with respect to values over which a `custom_jvp` or `custom_vjp`
closes:

> Detected differentiation of a custom_jvp function with respect to a closed-over
value. That isn't supported because the custom JVP rule only specifies how to
differentiate the custom_jvp function with respect to explicit input parameters.
>
> Try passing the closed-over value into the custom_jvp function as an argument,
and adapting the custom_jvp rule.

In tightening up and robustifying `custom_jvp` and `custom_vjp` in this way, we
found that allowing `custom_vjp` to accept `Tracer`s in its `nondiff_argnums`
would take a significant amount of bookkeeping: we'd need to rewrite the user's
`fwd` function to return the values as residuals, and rewrite the user's `bwd`
function to accept them as normal residuals (rather than accepting them as
special leading arguments, as happens with `nondiff_argnums`). This seems maybe
manageable, until you think through how we have to handle arbitrary pytrees!
Moreover, that complexity isn't necessary: if user code treats array-like
non-differentiable arguments just like regular arguments and residuals,
everything already works. (Before
[#4039](https://github.com/jax-ml/jax/pull/4039) JAX might've complained about
involving integer-valued inputs and outputs in autodiff, but after
[#4039](https://github.com/jax-ml/jax/pull/4039) those will just work!)

Unlike `custom_vjp`, it was easy to make `custom_jvp` work with
`nondiff_argnums` arguments that were `Tracer`s. So these updates only need to
happen with `custom_vjp`.
