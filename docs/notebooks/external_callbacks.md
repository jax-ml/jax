---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "7XNMxdTwURqI"}

# External Callbacks in JAX

+++ {"id": "h6lXo6bSUYGq"}

This guide is a work-in-progress outlining the uses of various callback functions, which allow JAX code to execute certain commands on the host, even while running under `jit`, `vmap`, `grad`, or another transformation.

This is a work-in-progress, and will be updated soon.

*TODO(jakevdp, sharadmv): fill-in some simple examples of {func}`jax.pure_callback`, {func}`jax.debug.callback`, {func}`jax.debug.print`, and others.*

+++ {"id": "dF7hoWGQUneJ"}

## Example: `pure_callback` with `custom_jvp`

One powerful way to take advantage of {func}`jax.pure_callback` is to combine it with {class}`jax.custom_jvp` (see [Custom derivative rules](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html) for more details on `custom_jvp`).
Suppose we want to create a JAX-compatible wrapper for a scipy or numpy function that is not yet available in the `jax.scipy` or `jax.numpy` wrappers.

Here, we'll consider creating a wrapper for the Bessel function of the first kind, implemented in `scipy.special.jv`.
We can start by defining a straightforward `pure_callback`:

```{code-cell}
:id: Ge4fNPZdVSJY

import jax
import jax.numpy as jnp
import scipy.special

def jv(v, z):
  v, z = jnp.asarray(v), jnp.asarray(z)

  # Require the order v to be integer type: this simplifies
  # the JVP rule below.
  assert jnp.issubdtype(v.dtype, jnp.integer)

  # Promote the input to inexact (float/complex).
  # Note that jnp.result_type() accounts for the enable_x64 flag.
  z = z.astype(jnp.result_type(float, z.dtype))

  # Wrap scipy function to return the expected dtype.
  _scipy_jv = lambda v, z: scipy.special.jv(v, z).astype(z.dtype)

  # Define the expected shape & dtype of output.
  result_shape_dtype = jax.ShapeDtypeStruct(
      shape=jnp.broadcast_shapes(v.shape, z.shape),
      dtype=z.dtype)

  # We use vectorize=True because scipy.special.jv handles broadcasted inputs.
  return jax.pure_callback(_scipy_jv, result_shape_dtype, v, z, vectorized=True)
```

+++ {"id": "vyjQj-0QVuoN"}

This lets us call into `scipy.special.jv` from transformed JAX code, including when transformed by `jit` and `vmap`:

```{code-cell}
:id: f4e46670f4e4

from functools import partial
j1 = partial(jv, 1)
z = jnp.arange(5.0)
```

```{code-cell}
:id: 6svImqFHWBwj
:outputId: bc8c778a-6c10-443b-9be2-c0f28e2ac1a9

print(j1(z))
```

+++ {"id": "d48eb4f2d48e"}

Here is the same result with `jit`:

```{code-cell}
:id: txvRqR9DWGdC
:outputId: d25f3476-23b1-48e4-dda1-3c06d32c3b87

print(jax.jit(j1)(z))
```

+++ {"id": "d861a472d861"}

And here is the same result again with `vmap`:

```{code-cell}
:id: BS-Ve5u_WU0C
:outputId: 08cecd1f-6953-4853-e9db-25a03eb5b000

print(jax.vmap(j1)(z))
```

+++ {"id": "SCH2ii_dWXP6"}

However, if we call `jax.grad`, we see an error because there is no autodiff rule defined for this function:

```{code-cell}
:id: q3qh_4DrWxdQ
:outputId: c46b0bfa-96f3-4629-b9af-a4d4f3ccb870
:tags: [raises-exception]

jax.grad(j1)(z)
```

+++ {"id": "PtYeJ_xUW09v"}

Let's define a custom gradient rule for this. Looking at the definition of the [Bessel Function of the First Kind](https://en.wikipedia.org/?title=Bessel_function_of_the_first_kind), we find that there is a relatively straightforward recurrence relationship for the derivative with respect to the argument `z`:

$$
d J_\nu(z) = \left\{
\begin{eqnarray}
-J_1(z),\ &\nu=0\\
[J_{\nu - 1}(z) - J_{\nu + 1}(z)]/2,\ &\nu\ne 0
\end{eqnarray}\right.
$$

The gradient with respect to $\nu$ is more complicated, but since we've restricted the `v` argument to integer types we don't need to worry about its gradient for the sake of this example.

We can use `jax.custom_jvp` to define this automatic differentiation rule for our callback function:

```{code-cell}
:id: BOVQnt05XvLs

jv = jax.custom_jvp(jv)

@jv.defjvp
def _jv_jvp(primals, tangents):
  v, z = primals
  _, z_dot = tangents  # Note: v_dot is always 0 because v is integer.
  jv_minus_1, jv_plus_1 = jv(v - 1, z), jv(v + 1, z)
  djv_dz = jnp.where(v == 0, -jv_plus_1, 0.5 * (jv_minus_1 - jv_plus_1))
  return jv(v, z), z_dot * djv_dz
```

+++ {"id": "W1SxcvQSX44c"}

Now computing the gradient of our function will work correctly:

```{code-cell}
:id: sCGceBs-X8nL
:outputId: 71c5589f-f996-44a0-f09a-ca8bb40c167a

j1 = partial(jv, 1)
print(jax.grad(j1)(2.0))
```

+++ {"id": "gWQ4phN5YB26"}

Further, since we've defined our gradient in terms of `jv` itself, JAX's architecture means that we get second-order and higher derivatives for free:

```{code-cell}
:id: QTe5mRAvYQBh
:outputId: d58ecff3-9419-422a-fd0e-14a7d9cf2cc3

jax.hessian(j1)(2.0)
```

+++ {"id": "QEXGxU4uYZii"}

Keep in mind that although this all works correctly with JAX, each call to our callback-based `jv` function will result in passing the input data from the device to the host, and passing the output of `scipy.special.jv` from the host back to the device.
When running on accelerators like GPU or TPU, this data movement and host synchronization can lead to significant overhead each time `jv` is called.
However, if you are running JAX on a single CPU (where the "host" and "device" are on the same hardware), JAX will generally do this data transfer in a fast, zero-copy fashion, making this pattern is a relatively straightforward way extend JAX's capabilities.
