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

# Higher-order derivatives

## Taking gradients (part 2)

JAX's autodiff makes it easy to compute higher-order derivatives, because the functions that compute derivatives are themselves differentiable. Thus, higher-order derivatives are as easy as stacking transformations.

The single-variable case was covered in the {ref}`automatic-differentiation` tutorial, where the example showed how to use {func}`jax.grad` to compute the derivative of $f(x) = x^3 + 2x^2 - 3x + 1$.

In the multivariable case, higher-order derivatives are more complicated. The second-order derivative of a function is represented by its [Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix), defined according to:

$$(\mathbf{H}f)_{i,j} = \frac{\partial^2 f}{\partial_i\partial_j}.$$

The Hessian of a real-valued function of several variables, $f: \mathbb R^n\to\mathbb R$, can be identified with the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of its gradient.

JAX provides two transformations for computing the Jacobian of a function, {func}`jax.jacfwd` and {func}`jax.jacrev`, corresponding to forward- and reverse-mode autodiff. They give the same answer, but one can be more efficient than the other in different circumstances – refer to the [video about autodiff](https://www.youtube.com/watch?v=wG_nF1awSSY).

```{code-cell}
import jax

def hessian(f):
  return jax.jacfwd(jax.grad(f))
```

Let's double check this is correct on the dot-product $f: \mathbf{x} \mapsto \mathbf{x} ^\top \mathbf{x}$.

if $i=j$, $\frac{\partial^2 f}{\partial_i\partial_j}(\mathbf{x}) = 2$. Otherwise, $\frac{\partial^2 f}{\partial_i\partial_j}(\mathbf{x}) = 0$.

```{code-cell}
import jax.numpy as jnp

def f(x):
  return jnp.dot(x, x)

hessian(f)(jnp.array([1., 2., 3.]))
```

## Higher-order derivative applications

Some meta-learning techniques, such as Model-Agnostic Meta-Learning ([MAML](https://arxiv.org/abs/1703.03400)), require differentiating through gradient updates. In other frameworks this can be quite cumbersome, but in JAX it's much easier:

```python
def meta_loss_fn(params, data):
  """Computes the loss after one step of SGD."""
  grads = jax.grad(loss_fn)(params, data)
  return loss_fn(params - lr * grads, data)

meta_grads = jax.grad(meta_loss_fn)(params, data)
```

(stopping-gradients)=
### Stopping gradients

Autodiff enables automatic computation of the gradient of a function with respect to its inputs. Sometimes, however, you might want some additional control: for instance, you might want to avoid backpropagating gradients through some subset of the computational graph.

Consider for instance the TD(0) ([temporal difference](https://en.wikipedia.org/wiki/Temporal_difference_learning)) reinforcement learning update. This is used to learn to estimate the *value* of a state in an environment from experience of interacting with the environment. Let's assume the value estimate $v_{\theta}(s_{t-1}$) in a state $s_{t-1}$ is parameterised by a linear function.

```{code-cell}
# Value function and initial parameters
value_fn = lambda theta, state: jnp.dot(theta, state)
theta = jnp.array([0.1, -0.1, 0.])
```

Consider a transition from a state $s_{t-1}$ to a state $s_t$ during which you observed the reward $r_t$

```{code-cell}
# An example transition.
s_tm1 = jnp.array([1., 2., -1.])
r_t = jnp.array(1.)
s_t = jnp.array([2., 1., 0.])
```

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

How can you implement this in JAX? If you write the pseudo loss naively, you get:

```{code-cell}
def td_loss(theta, s_tm1, r_t, s_t):
  v_tm1 = value_fn(theta, s_tm1)
  target = r_t + value_fn(theta, s_t)
  return -0.5 * ((target - v_tm1) ** 2)

td_update = jax.grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

delta_theta
```

But `td_update` will **not** compute a TD(0) update, because the gradient computation will include the dependency of `target` on $\theta$.

You can use {func}`jax.lax.stop_gradient` to force JAX to ignore the dependency of the target on $\theta$:

```{code-cell}
def td_loss(theta, s_tm1, r_t, s_t):
  v_tm1 = value_fn(theta, s_tm1)
  target = r_t + value_fn(theta, s_t)
  return -0.5 * ((jax.lax.stop_gradient(target) - v_tm1) ** 2)

td_update = jax.grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

delta_theta
```

This will treat `target` as if it did **not** depend on the parameters $\theta$ and compute the correct update to the parameters.

Now, let's also calculate $\Delta \theta$ using the original TD(0) update expression, to cross-check our work. You may wish to try and implement this yourself using {func}`jax.grad` and your knowledge so far. Here's our solution:

```{code-cell}
s_grad = jax.grad(value_fn)(theta, s_tm1)
delta_theta_original_calculation = (r_t + value_fn(theta, s_t) - value_fn(theta, s_tm1)) * s_grad

delta_theta_original_calculation # [1.2, 2.4, -1.2], same as `delta_theta`
```

`jax.lax.stop_gradient` may also be useful in other settings, for instance if you want the gradient from some loss to only affect a subset of the parameters of the neural network (because, for instance, the other parameters are trained using a different loss).

### Straight-through estimator using `stop_gradient`

The straight-through estimator is a trick for defining a 'gradient' of a function that is otherwise non-differentiable. Given a non-differentiable function $f : \mathbb{R}^n \to \mathbb{R}^n$ that is used as part of a larger function that we wish to find a gradient of, we simply pretend during the backward pass that $f$ is the identity function. This can be implemented neatly using `jax.lax.stop_gradient`:

```{code-cell}
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

### Per-example gradients

While most ML systems compute gradients and updates from batches of data, for reasons of computational efficiency and/or variance reduction, it is sometimes necessary to have access to the gradient/update associated with each specific sample in the batch.

For instance, this is needed to prioritize data based on gradient magnitude, or to apply clipping / normalisations on a sample by sample basis.

In many frameworks (PyTorch, TF, Theano) it is often not trivial to compute per-example gradients, because the library directly accumulates the gradient over the batch. Naive workarounds, such as computing a separate loss per example and then aggregating the resulting gradients are typically very inefficient.

In JAX, you can define the code to compute the gradient per-sample in an easy but efficient way.

Just combine the {func}`jax.jit`, {func}`jax.vmap` and {func}`jax.grad` transformations together:

```{code-cell}
perex_grads = jax.jit(jax.vmap(jax.grad(td_loss), in_axes=(None, 0, 0, 0)))

# Test it:
batched_s_tm1 = jnp.stack([s_tm1, s_tm1])
batched_r_t = jnp.stack([r_t, r_t])
batched_s_t = jnp.stack([s_t, s_t])

perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```

Let's go through this one transformation at a time.

First, you apply {func}`jax.grad` to `td_loss` to obtain a function that computes the gradient of the loss w.r.t. the parameters on single (unbatched) inputs:

```{code-cell}
dtdloss_dtheta = jax.grad(td_loss)

dtdloss_dtheta(theta, s_tm1, r_t, s_t)
```

This function computes one row of the array above.

Then, you vectorise this function using {func}`jax.vmap`. This adds a batch dimension to all inputs and outputs. Now, given a batch of inputs, you produce a batch of outputs — each output in the batch corresponds to the gradient for the corresponding member of the input batch.

```{code-cell}
almost_perex_grads = jax.vmap(dtdloss_dtheta)

batched_theta = jnp.stack([theta, theta])
almost_perex_grads(batched_theta, batched_s_tm1, batched_r_t, batched_s_t)
```

This isn't quite what we want, because we have to manually feed this function a batch of `theta`s, whereas we actually want to use a single `theta`. We fix this by adding `in_axes` to the {func}`jax.vmap`, specifying theta as `None`, and the other args as `0`. This makes the resulting function add an extra axis only to the other arguments, leaving `theta` unbatched, as we want:

```{code-cell}
inefficient_perex_grads = jax.vmap(dtdloss_dtheta, in_axes=(None, 0, 0, 0))

inefficient_perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```

This does what we want, but is slower than it has to be. Now, you wrap the whole thing in a {func}`jax.jit` to get the compiled, efficient version of the same function:

```{code-cell}
perex_grads = jax.jit(inefficient_perex_grads)

perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```

```{code-cell}
%timeit inefficient_perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t).block_until_ready()
%timeit perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t).block_until_ready()
```

### Hessian-vector products with `jax.grad`-of-`jax.grad`

One thing you can do with higher-order {func}`jax.grad` is build a Hessian-vector product function. (Later on you'll write an even more efficient implementation that mixes both forward- and reverse-mode, but this one will use pure reverse-mode.)

A Hessian-vector product function can be useful in a [truncated Newton Conjugate-Gradient algorithm](https://en.wikipedia.org/wiki/Truncated_Newton_method) for minimizing smooth convex functions, or for studying the curvature of neural network training objectives (e.g. [1](https://arxiv.org/abs/1406.2572), [2](https://arxiv.org/abs/1811.07062), [3](https://arxiv.org/abs/1706.04454), [4](https://arxiv.org/abs/1802.03451)).

For a scalar-valued function $f : \mathbb{R}^n \to \mathbb{R}$ with continuous second derivatives (so that the Hessian matrix is symmetric), the Hessian at a point $x \in \mathbb{R}^n$ is written as $\partial^2 f(x)$. A Hessian-vector product function is then able to evaluate

$\qquad v \mapsto \partial^2 f(x) \cdot v$

for any $v \in \mathbb{R}^n$.

The trick is not to instantiate the full Hessian matrix: if $n$ is large, perhaps in the millions or billions in the context of neural networks, then that might be impossible to store.

Luckily, {func}`jax.grad` already gives us a way to write an efficient Hessian-vector product function. You just have to use the identity:

$\qquad \partial^2 f (x) v = \partial [x \mapsto \partial f(x) \cdot v] = \partial g(x)$,

where $g(x) = \partial f(x) \cdot v$ is a new scalar-valued function that dots the gradient of $f$ at $x$ with the vector $v$. Notice that you're only ever differentiating scalar-valued functions of vector-valued arguments, which is exactly where you know {func}`jax.grad` is efficient.

In JAX code, you can just write this:

```{code-cell}
def hvp(f, x, v):
    return jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)
```

This example shows that you can freely use lexical closure, and JAX will never get perturbed or confused.

You will check this implementation a few cells down, once you learn how to compute dense Hessian matrices. You'll also write an even better version that uses both forward-mode and reverse-mode.

### Jacobians and Hessians using `jax.jacfwd` and `jax.jacrev`

You can compute full Jacobian matrices using the {func}`jax.jacfwd` and {func}`jax.jacrev` functions:

```{code-cell}
from jax import jacfwd, jacrev

# Define a sigmoid function.
def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)

# Build a toy dataset.
inputs = jnp.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])

# Initialize random model coefficients
key = jax.random.key(0)
key, W_key, b_key = jax.random.split(key, 3)
W = jax.random.normal(W_key, (3,))
b = jax.random.normal(b_key, ())

# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

J = jacfwd(f)(W)
print("jacfwd result, with shape", J.shape)
print(J)

J = jacrev(f)(W)
print("jacrev result, with shape", J.shape)
print(J)
```

These two functions compute the same values (up to machine numerics), but differ in their implementation: {func}`jax.jacfwd` uses forward-mode automatic differentiation, which is more efficient for "tall" Jacobian matrices (more outputs than inputs), while {func}`jax.jacrev` uses reverse-mode, which is more efficient for "wide" Jacobian matrices (more inputs than outputs). For matrices that are near-square, {func}`jax.jacfwd` probably has an edge over {func}`jax.jacrev`.

You can also use {func}`jax.jacfwd` and {func}`jax.jacrev` with container types:

```{code-cell}
def predict_dict(params, inputs):
    return predict(params['W'], params['b'], inputs)

J_dict = jax.jacrev(predict_dict)({'W': W, 'b': b}, inputs)
for k, v in J_dict.items():
    print("Jacobian from {} to logits is".format(k))
    print(v)
```

For more details on forward- and reverse-mode, as well as how to implement {func}`jax.jacfwd` and {func}`jax.jacrev` as efficiently as possible, read on!

Using a composition of two of these functions gives us a way to compute dense Hessian matrices:

```{code-cell}
def hessian(f):
    return jax.jacfwd(jax.jacrev(f))

H = hessian(f)(W)
print("hessian, with shape", H.shape)
print(H)
```

This shape makes sense: if you start with a function $f : \mathbb{R}^n \to \mathbb{R}^m$, then at a point $x \in \mathbb{R}^n$ you expect to get the shapes:

* $f(x) \in \mathbb{R}^m$, the value of $f$ at $x$,
* $\partial f(x) \in \mathbb{R}^{m \times n}$, the Jacobian matrix at $x$,
* $\partial^2 f(x) \in \mathbb{R}^{m \times n \times n}$, the Hessian at $x$,

and so on.

To implement `hessian`, you could have used `jacfwd(jacrev(f))` or `jacrev(jacfwd(f))` or any other composition of these two. But forward-over-reverse is typically the most efficient. That's because in the inner Jacobian computation we're often differentiating a function wide Jacobian (maybe like a loss function $f : \mathbb{R}^n \to \mathbb{R}$), while in the outer Jacobian computation we're differentiating a function with a square Jacobian (since $\nabla f : \mathbb{R}^n \to \mathbb{R}^n$), which is where forward-mode wins out.
