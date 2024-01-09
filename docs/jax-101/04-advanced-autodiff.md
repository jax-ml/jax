---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "kORMl5KmfByI"}

# Advanced Automatic Differentiation in JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/04-advanced-autodiff.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/jax-101/04-advanced-autodiff.ipynb)

*Authors: Vlatimir Mikulik & Matteo Hessel*

Computing gradients is a critical part of modern machine learning methods. This section considers a few advanced topics in the areas of automatic differentiation as it relates to modern machine learning.

While understanding how automatic differentiation works under the hood isn't crucial for using JAX in most contexts, we encourage the reader to check out this quite accessible [video](https://www.youtube.com/watch?v=wG_nF1awSSY)  to get a deeper sense of what's going on.

[The Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) is a more advanced and more detailed explanation of how these ideas are implemented in the JAX backend. It's not necessary to understand this to do most things in JAX. However, some features (like defining [custom derivatives](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)) depend on understanding this, so it's worth knowing this explanation exists if you ever need to use them.

+++ {"id": "qx50CO1IorCc"}

## Higher-order derivatives

JAX's autodiff makes it easy to compute higher-order derivatives, because the functions that compute derivatives are themselves differentiable. Thus, higher-order derivatives are as easy as stacking transformations.

We illustrate this in the single-variable case:

The derivative of $f(x) = x^3 + 2x^2 - 3x + 1$ can be computed as:

```{code-cell} ipython3
:id: Kqsbj98UTVdi

import jax

f = lambda x: x**3 + 2*x**2 - 3*x + 1

dfdx = jax.grad(f)
```

+++ {"id": "ItEt15OGiiAF"}

The higher-order derivatives of $f$ are:

$$
\begin{array}{l}
f'(x) = 3x^2 + 4x -3\\
f''(x) = 6x + 4\\
f'''(x) = 6\\
f^{iv}(x) = 0
\end{array}
$$

Computing any of these in JAX is as easy as chaining the `grad` function:

```{code-cell} ipython3
:id: 5X3yQqLgimqH

d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)
d4fdx = jax.grad(d3fdx)
```

+++ {"id": "fVL2P_pcj8T1"}

Evaluating the above in $x=1$ would give us:

$$
\begin{array}{l}
f'(1) = 4\\
f''(1) = 10\\
f'''(1) = 6\\
f^{iv}(1) = 0
\end{array}
$$

Using JAX:

```{code-cell} ipython3
:id: tJkIp9wFjxL3
:outputId: 581ecf87-2d20-4c83-9443-5befc1baf51d

print(dfdx(1.))
print(d2fdx(1.))
print(d3fdx(1.))
print(d4fdx(1.))
```

+++ {"id": "3-fTelU7LHRr"}

In the multivariable case, higher-order derivatives are more complicated. The second-order derivative of a function is represented by its [Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix), defined according to

$$(\mathbf{H}f)_{i,j} = \frac{\partial^2 f}{\partial_i\partial_j}.$$

The Hessian of a real-valued function of several variables, $f: \mathbb R^n\to\mathbb R$, can be identified with the Jacobian of its gradient. JAX provides two transformations for computing the Jacobian of a function, `jax.jacfwd` and `jax.jacrev`, corresponding to forward- and reverse-mode autodiff. They give the same answer, but one can be more efficient than the other in different circumstances – see the [video about autodiff](https://www.youtube.com/watch?v=wG_nF1awSSY) linked above for an explanation.

```{code-cell} ipython3
:id: ILhkef1rOB6_

def hessian(f):
  return jax.jacfwd(jax.grad(f))
```

+++ {"id": "xaENwADXOGf_"}

Let's double check this is correct on the dot-product $f: \mathbf{x} \mapsto \mathbf{x} ^\top \mathbf{x}$.

if $i=j$, $\frac{\partial^2 f}{\partial_i\partial_j}(\mathbf{x}) = 2$. Otherwise, $\frac{\partial^2 f}{\partial_i\partial_j}(\mathbf{x}) = 0$.

```{code-cell} ipython3
:id: Xm3A0QdWRdJl
:outputId: e1e8cba9-b567-439b-b8fc-34b21497e67f

import jax.numpy as jnp

def f(x):
  return jnp.dot(x, x)

hessian(f)(jnp.array([1., 2., 3.]))
```

+++ {"id": "7_gbi34WSUsD"}

Often, however, we aren't interested in computing the full Hessian itself, and doing so can be very inefficient. [The Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) explains some tricks, like the Hessian-vector product, that allow to use it without materialising the whole matrix.

If you plan to work with higher-order derivatives in JAX, we strongly recommend reading the Autodiff Cookbook.

+++ {"id": "zMT2qAi-SvcK"}

## Higher order optimization

Some meta-learning techniques, such as Model-Agnostic Meta-Learning ([MAML](https://arxiv.org/abs/1703.03400)), require differentiating through gradient updates. In other frameworks this can be quite cumbersome, but in JAX it's much easier:

```python
def meta_loss_fn(params, data):
  """Computes the loss after one step of SGD."""
  grads = jax.grad(loss_fn)(params, data)
  return loss_fn(params - lr * grads, data)

meta_grads = jax.grad(meta_loss_fn)(params, data)
```

+++ {"id": "3h9Aj3YyuL6P"}

## Stopping gradients

Auto-diff enables automatic computation of the gradient of a function with respect to its inputs. Sometimes, however, we might want some additional control: for instance, we might want to avoid back-propagating gradients through some subset of the computational graph.

Consider for instance the TD(0) ([temporal difference](https://en.wikipedia.org/wiki/Temporal_difference_learning)) reinforcement learning update. This is used to learn to estimate the *value* of a state in an environment from experience of interacting with the environment. Let's assume the value estimate $v_{\theta}(s_{t-1}$) in a state $s_{t-1}$ is parameterised by a linear function.

```{code-cell} ipython3
:id: fjLqbCb6SiOm

# Value function and initial parameters
value_fn = lambda theta, state: jnp.dot(theta, state)
theta = jnp.array([0.1, -0.1, 0.])
```

+++ {"id": "85S7HBo1tBzt"}

Consider a transition from a state $s_{t-1}$ to a state $s_t$ during which we observed the reward $r_t$

```{code-cell} ipython3
:id: T6cRPau6tCSE

# An example transition.
s_tm1 = jnp.array([1., 2., -1.])
r_t = jnp.array(1.)
s_t = jnp.array([2., 1., 0.])
```

+++ {"id": "QO5CHA9_Sk01"}

The TD(0) update to the network parameters is:

$$
\Delta \theta = (r_t + v_{\theta}(s_t) - v_{\theta}(s_{t-1})) \nabla v_{\theta}(s_{t-1})
$$

This update is not the gradient of any loss function.

However, it can be **written** as the gradient of the pseudo loss function

$$
L(\theta) = - \frac{1}{2} [r_t + v_{\theta}(s_t) - v_{\theta}(s_{t-1})]^2
$$

if the dependency of the target $r_t + v_{\theta}(s_t)$ on the parameter $\theta$ is ignored.

How can we implement this in JAX? If we write the pseudo loss naively we get:

```{code-cell} ipython3
:id: uMcFny2xuOwz
:outputId: 79c10af9-10b8-4e18-9753-a53918b9d72d

def td_loss(theta, s_tm1, r_t, s_t):
  v_tm1 = value_fn(theta, s_tm1)
  target = r_t + value_fn(theta, s_t)
  return -0.5 * ((target - v_tm1) ** 2)

td_update = jax.grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

delta_theta
```

+++ {"id": "CPnjm59GG4Gq"}

But `td_update` will **not** compute a TD(0) update, because the gradient computation will include the dependency of `target` on $\theta$.

We can use `jax.lax.stop_gradient` to force JAX to ignore the dependency of the target on $\theta$:

```{code-cell} ipython3
:id: MKeq7trKPS4V
:outputId: 0f27d754-a871-4c47-8e3a-a961418a24cc

def td_loss(theta, s_tm1, r_t, s_t):
  v_tm1 = value_fn(theta, s_tm1)
  target = r_t + value_fn(theta, s_t)
  return -0.5 * ((jax.lax.stop_gradient(target) - v_tm1) ** 2)

td_update = jax.grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

delta_theta
```

+++ {"id": "JOnjm59GG4Gq"}

This will treat `target` as if it did **not** depend on the parameters $\theta$ and compute the correct update to the parameters.

Now, let's also calculate $\Delta \theta$ using the original TD(0) update expression, to cross-check our work. You may wish to try and implement this yourself using jax.grad and your knowledge so far. Here's our solution:

```{code-cell} ipython3
:id: WCeq7trKPS4V
:outputId: 0f19d754-a871-4c47-8e3a-a961418a24cc

s_grad = jax.grad(value_fn)(theta, s_tm1)
delta_theta_original_calculation = (r_t + value_fn(theta, s_t) - value_fn(theta, s_tm1)) * s_grad

delta_theta_original_calculation # [1.2, 2.4, -1.2], same as `delta_theta`
```

+++ {"id": "TNF0CkwOTKpD"}

`jax.lax.stop_gradient` may also be useful in other settings, for instance if you want the gradient from some loss to only affect a subset of the parameters of the neural network (because, for instance, the other parameters are trained using a different loss).

+++ {"id": "UMY0IyuOTKpG"}

## Straight-through estimator using `stop_gradient`

The straight-through estimator is a trick for defining a 'gradient' of a function that is otherwise non-differentiable. Given a non-differentiable function $f : \mathbb{R}^n \to \mathbb{R}^n$ that is used as part of a larger function that we wish to find a gradient of, we simply pretend during the backward pass that $f$ is the identity function. This can be implemented neatly using `jax.lax.stop_gradient`:

```{code-cell} ipython3
:id: hdORJENmVHvX
:outputId: f0839541-46a4-45a9-fce7-ead08f20046b

def f(x):
  return jnp.round(x)  # non-differentiable

def straight_through_f(x):
  # Create an exactly-zero expression with Sterbenz lemma that has
  # an exactly-one gradient.
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(f(x))

print("f(x): ", f(3.2))
print("straight_through_f(x):", straight_through_f(3.2))

print("grad(f)(x):", jax.grad(f)(3.2))
print("grad(straight_through_f)(x):", jax.grad(straight_through_f)(3.2))
```

+++ {"id": "Wx3RNE0Sw5mn"}

## Per-example gradients

While most ML systems compute gradients and updates from batches of data, for reasons of computational efficiency and/or variance reduction, it is sometimes necessary to have access to the gradient/update associated with each specific sample in the batch.

For instance, this is needed to prioritise data based on gradient magnitude, or to apply clipping / normalisations on a sample by sample basis.

In many frameworks (PyTorch, TF, Theano) it is often not trivial to compute per-example gradients, because the library directly accumulates the gradient over the batch. Naive workarounds, such as computing a separate loss per example and then aggregating the resulting gradients are typically very inefficient.

In JAX we can define the code to compute the gradient per-sample in an easy but efficient way.

Just combine the `jit`, `vmap` and `grad` transformations together:

```{code-cell} ipython3
:id: tFLyd9ifw4GG
:outputId: bf3ad4a3-102d-47a6-ece0-f4a8c9e5d434

perex_grads = jax.jit(jax.vmap(jax.grad(td_loss), in_axes=(None, 0, 0, 0)))

# Test it:
batched_s_tm1 = jnp.stack([s_tm1, s_tm1])
batched_r_t = jnp.stack([r_t, r_t])
batched_s_t = jnp.stack([s_t, s_t])

perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```

+++ {"id": "VxvYVEYQYiS_"}

Let's walk through this one transformation at a time.

First, we apply `jax.grad` to `td_loss` to obtain a function that computes the gradient of the loss w.r.t. the parameters on single (unbatched) inputs:

```{code-cell} ipython3
:id: rPO67QQrY5Bk
:outputId: fbb45b98-2dbf-4865-e6e5-87dc3eef5560

dtdloss_dtheta = jax.grad(td_loss)

dtdloss_dtheta(theta, s_tm1, r_t, s_t)
```

+++ {"id": "cU36nVAlcnJ0"}

This function computes one row of the array above.

+++ {"id": "c6DQF0b3ZA5u"}

Then, we vectorise this function using `jax.vmap`. This adds a batch dimension to all inputs and outputs. Now, given a batch of inputs, we produce a batch of outputs -- each output in the batch corresponds to the gradient for the corresponding member of the input batch.

```{code-cell} ipython3
:id: 5agbNKavaNDM
:outputId: ab081012-88ab-4904-a367-68e9f81445f0

almost_perex_grads = jax.vmap(dtdloss_dtheta)

batched_theta = jnp.stack([theta, theta])
almost_perex_grads(batched_theta, batched_s_tm1, batched_r_t, batched_s_t)
```

+++ {"id": "K-v34yLuan7k"}

This isn't quite what we want, because we have to manually feed this function a batch of `theta`s, whereas we actually want to use a single `theta`. We fix this by adding `in_axes` to the `jax.vmap`, specifying theta as `None`, and the other args as `0`. This makes the resulting function add an extra axis only to the other arguments, leaving `theta` unbatched, as we want:

```{code-cell} ipython3
:id: S6kd5MujbGrr
:outputId: d3d731ef-3f7d-4a0a-ce91-7df57627ddbd

inefficient_perex_grads = jax.vmap(dtdloss_dtheta, in_axes=(None, 0, 0, 0))

inefficient_perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```

+++ {"id": "O0hbsm70be5T"}

Almost there! This does what we want, but is slower than it has to be. Now, we wrap the whole thing in a `jax.jit` to get the compiled, efficient version of the same function:

```{code-cell} ipython3
:id: Fvr709FcbrSW
:outputId: 627db899-5620-4bed-8d34-cd1364d3d187

perex_grads = jax.jit(inefficient_perex_grads)

perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```

```{code-cell} ipython3
:id: FH42yzbHcNs2
:outputId: c8e52f93-615a-4ce7-d8ab-fb6215995a39

%timeit inefficient_perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t).block_until_ready()
%timeit perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t).block_until_ready()
```
