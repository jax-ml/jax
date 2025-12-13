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
  name: python3
---

# Complex numbers and differentiation

JAX is great at complex numbers and differentiation. To support both [holomorphic and non-holomorphic differentiation](https://en.wikipedia.org/wiki/Holomorphic_function), it helps to think in terms of JVPs and VJPs.

Consider a complex-to-complex function $f: \mathbb{C} \to \mathbb{C}$ and identify it with a corresponding function $g: \mathbb{R}^2 \to \mathbb{R}^2$,

```{code-cell}
import jax.numpy as jnp

def f(z):
  x, y = jnp.real(z), jnp.imag(z)
  return u(x, y) + v(x, y) * 1j

def g(x, y):
  return (u(x, y), v(x, y))
```

That is, we've decomposed $f(z) = u(x, y) + v(x, y) i$ where $z = x + y i$, and identified $\mathbb{C}$ with $\mathbb{R}^2$ to get $g$.

Since $g$ only involves real inputs and outputs, we already know how to write a Jacobian-vector product for it, say given a tangent vector $(c, d) \in \mathbb{R}^2$, namely:

$\begin{bmatrix} \partial_0 u(x, y) & \partial_1 u(x, y) \\ \partial_0 v(x, y) & \partial_1 v(x, y) \end{bmatrix}
\begin{bmatrix} c \\ d \end{bmatrix}$.

To get a JVP for the original function $f$ applied to a tangent vector $c + di \in \mathbb{C}$, we just use the same definition and identify the result as another complex number, 

$\partial f(x + y i)(c + d i) =
\begin{matrix} \begin{bmatrix} 1 & i \end{bmatrix} \\ ~ \end{matrix}
\begin{bmatrix} \partial_0 u(x, y) & \partial_1 u(x, y) \\ \partial_0 v(x, y) & \partial_1 v(x, y) \end{bmatrix}
\begin{bmatrix} c \\ d \end{bmatrix}$.

That's our definition of the JVP of a $\mathbb{C} \to \mathbb{C}$ function! Notice it doesn't matter whether or not $f$ is holomorphic: the JVP is unambiguous.

Here's a check:

```{code-cell}
from jax import random, grad, jvp

def check(seed):
  key = random.key(seed)

  # random coeffs for u and v
  key, subkey = random.split(key)
  a, b, c, d = random.uniform(subkey, (4,))

  def fun(z):
    x, y = jnp.real(z), jnp.imag(z)
    return u(x, y) + v(x, y) * 1j

  def u(x, y):
    return a * x + b * y

  def v(x, y):
    return c * x + d * y

  # primal point
  key, subkey = random.split(key)
  x, y = random.uniform(subkey, (2,))
  z = x + y * 1j

  # tangent vector
  key, subkey = random.split(key)
  c, d = random.uniform(subkey, (2,))
  z_dot = c + d * 1j

  # check jvp
  _, ans = jvp(fun, (z,), (z_dot,))
  expected = (grad(u, 0)(x, y) * c +
              grad(u, 1)(x, y) * d +
              grad(v, 0)(x, y) * c * 1j+
              grad(v, 1)(x, y) * d * 1j)
  print(jnp.allclose(ans, expected))
```

```{code-cell}
check(0)
check(1)
check(2)
```

What about VJPs? We do something pretty similar: for a cotangent vector $c + di \in \mathbb{C}$ we define the VJP of $f$ as

$(c + di)^* \; \partial f(x + y i) =
\begin{matrix} \begin{bmatrix} c & -d \end{bmatrix} \\ ~ \end{matrix}
\begin{bmatrix} \partial_0 u(x, y) & \partial_1 u(x, y) \\ \partial_0 v(x, y) & \partial_1 v(x, y) \end{bmatrix}
\begin{bmatrix} 1 \\ -i \end{bmatrix}$.

What's with the negatives? They're just to take care of complex conjugation, and the fact that we're working with covectors.

Here's a check of the VJP rules:

```{code-cell}
from jax import vjp

def check(seed):
  key = random.key(seed)

  # random coeffs for u and v
  key, subkey = random.split(key)
  a, b, c, d = random.uniform(subkey, (4,))

  def fun(z):
    x, y = jnp.real(z), jnp.imag(z)
    return u(x, y) + v(x, y) * 1j

  def u(x, y):
    return a * x + b * y

  def v(x, y):
    return c * x + d * y

  # primal point
  key, subkey = random.split(key)
  x, y = random.uniform(subkey, (2,))
  z = x + y * 1j

  # cotangent vector
  key, subkey = random.split(key)
  c, d = random.uniform(subkey, (2,))
  z_bar = jnp.array(c + d * 1j)  # for dtype control

  # check vjp
  _, fun_vjp = vjp(fun, z)
  ans, = fun_vjp(z_bar)
  expected = (grad(u, 0)(x, y) * c +
              grad(v, 0)(x, y) * (-d) +
              grad(u, 1)(x, y) * c * (-1j) +
              grad(v, 1)(x, y) * (-d) * (-1j))
  assert jnp.allclose(ans, expected, atol=1e-5, rtol=1e-5)
```

```{code-cell}
check(0)
check(1)
check(2)
```

What about convenience wrappers like {func}`jax.grad`, {func}`jax.jacfwd`, and {func}`jax.jacrev`?

For $\mathbb{R} \to \mathbb{R}$ functions, recall we defined `grad(f)(x)` as being `vjp(f, x)[1](1.0)`, which works because applying a VJP to a `1.0` value reveals the gradient (i.e. Jacobian, or derivative). We can do the same thing for $\mathbb{C} \to \mathbb{R}$ functions: we can still use `1.0` as the cotangent vector, and we just get out a complex number result summarizing the full Jacobian:

```{code-cell}
def f(z):
  x, y = jnp.real(z), jnp.imag(z)
  return x**2 + y**2

z = 3. + 4j
grad(f)(z)
```

For general $\mathbb{C} \to \mathbb{C}$ functions, the Jacobian has 4 real-valued degrees of freedom (as in the 2x2 Jacobian matrices above), so we can't hope to represent all of them within a complex number. But we can for holomorphic functions! A holomorphic function is precisely a $\mathbb{C} \to \mathbb{C}$ function with the special property that its derivative can be represented as a single complex number. (The [Cauchy-Riemann equations](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations) ensure that the above 2x2 Jacobians have the special form of a scale-and-rotate matrix in the complex plane, i.e. the action of a single complex number under multiplication.) And we can reveal that one complex number using a single call to `vjp` with a covector of `1.0`.

Because this only works for holomorphic functions, to use this trick we need to promise JAX that our function is holomorphic; otherwise, JAX will raise an error when {func}`jax.grad` is used for a complex-output function:

```{code-cell}
def f(z):
  return jnp.sin(z)

z = 3. + 4j
grad(f, holomorphic=True)(z)
```

All the `holomorphic=True` promise does is disable the error when the output is complex-valued. We can still write `holomorphic=True` when the function isn't holomorphic, but the answer we get out won't represent the full Jacobian. Instead, it'll be the Jacobian of the function where we just discard the imaginary part of the output:

```{code-cell}
def f(z):
  return jnp.conjugate(z)

z = 3. + 4j
grad(f, holomorphic=True)(z)  # f is not actually holomorphic!
```

There are some useful upshots for how {func}`jax.grad` works here:

1. We can use {func}`jax.grad` on holomorphic $\mathbb{C} \to \mathbb{C}$ functions.
2. We can use {func}`jax.grad` to optimize $f : \mathbb{C} \to \mathbb{R}$ functions, like real-valued loss functions of complex parameters `x`, by taking steps in the direction of the conjugate of `grad(f)(x)`.
3. If we have an $\mathbb{R} \to \mathbb{R}$ function that just happens to use some complex-valued operations internally (some of which must be non-holomorphic, e.g. FFTs used in convolutions) then {func}`jax.grad` still works and we get the same result that an implementation using only real values would have given.

In any case, JVPs and VJPs are always unambiguous. And if we wanted to compute the full Jacobian matrix of a non-holomorphic $\mathbb{C} \to \mathbb{C}$ function, we can do it with JVPs or VJPs!


You should expect complex numbers to work everywhere in JAX. Here's differentiating through a Cholesky decomposition of a complex matrix:

```{code-cell}
A = jnp.array([[5.,    2.+3j,    5j],
              [2.-3j,   7.,  1.+7j],
              [-5j,  1.-7j,    12.]])

def f(X):
    L = jnp.linalg.cholesky(X)
    return jnp.sum((L - jnp.sin(L))**2)

grad(f, holomorphic=True)(A)
```
