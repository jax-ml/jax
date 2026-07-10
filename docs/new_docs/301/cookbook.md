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

(jax-301-cookbook)=
# The Autodiff Cookbook

<!--* freshness: { reviewed: '2026-07-10' } *-->

JAX has a pretty general automatic differentiation system. In this document
we'll go through a whole bunch of neat autodiff ideas that you can cherry-pick
for your own work. The basics of `jax.grad` — `argnums`, differentiating with
respect to containers, `value_and_grad`, checking derivatives against finite
differences — are covered in the 101 docs ({ref}`jax-101-transformations`), so
here we start one level up: Hessian-vector products, full Jacobians, the
`jvp`/`vjp` machinery underneath it all, and differentiation with complex
numbers.

```{code-cell}
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.key(0)
```

## Setup: a running example

As a running example, we'll use the same linear logistic regression model as
the 101 docs:

```{code-cell}
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
targets = jnp.array([True, True, False, True])

# Training loss is the negative log-likelihood of the training examples.
def loss(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs))

# Initialize random model coefficients
key, W_key, b_key = random.split(key, 3)
W = random.normal(W_key, (3,))
b = random.normal(b_key, ())
```

## Gradients

### Hessian-vector products with `grad`-of-`grad`

One thing we can do with higher-order `grad` is build a Hessian-vector product function. (Later on we'll write an even more efficient implementation that mixes both forward- and reverse-mode, but this one will use pure reverse-mode.)

A Hessian-vector product function can be useful in a [truncated Newton Conjugate-Gradient algorithm](https://en.wikipedia.org/wiki/Truncated_Newton_method) for minimizing smooth convex functions, or for studying the curvature of neural network training objectives (e.g. [1](https://arxiv.org/abs/1406.2572), [2](https://arxiv.org/abs/1811.07062), [3](https://arxiv.org/abs/1706.04454), [4](https://arxiv.org/abs/1802.03451)).

For a scalar-valued function $f : \mathbb{R}^n \to \mathbb{R}$ with continuous second derivatives (so that the Hessian matrix is symmetric), the Hessian at a point $x \in \mathbb{R}^n$ is written as $\partial^2 f(x)$. A Hessian-vector product function is then able to evaluate

$\qquad v \mapsto \partial^2 f(x) \cdot v$

for any $v \in \mathbb{R}^n$.

The trick is not to instantiate the full Hessian matrix: if $n$ is large, perhaps in the millions or billions in the context of neural networks, then that might be impossible to store.

Luckily, `grad` already gives us a way to write an efficient Hessian-vector product function. We just have to use the identity

$\qquad \partial^2 f (x) v = \partial [x \mapsto \partial f(x) \cdot v] = \partial g(x)$,

where $g(x) = \partial f(x) \cdot v$ is a new scalar-valued function that dots the gradient of $f$ at $x$ with the vector $v$. Notice that we're only ever differentiating scalar-valued functions of vector-valued arguments, which is exactly where we know `grad` is efficient.

In JAX code, we can just write this:

```{code-cell}
def hvp(f, x, v):
    return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)
```

This example shows that you can freely use lexical closure, and JAX will never get perturbed or confused.

We'll check this implementation a few cells down, once we see how to compute dense Hessian matrices. We'll also write an even better version that uses both forward-mode and reverse-mode.

+++

### Jacobians and Hessians using `jacfwd` and `jacrev`

+++

You can compute full Jacobian matrices using the `jacfwd` and `jacrev` functions:

```{code-cell}
from jax import jacfwd, jacrev

# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

J = jacfwd(f)(W)
print("jacfwd result, with shape", J.shape)
print(J)

J = jacrev(f)(W)
print("jacrev result, with shape", J.shape)
print(J)
```

These two functions compute the same values (up to machine numerics), but differ in their implementation: `jacfwd` uses forward-mode automatic differentiation, which is more efficient for "tall" Jacobian matrices (more outputs than inputs), while `jacrev` uses reverse-mode, which is more efficient for "wide" Jacobian matrices (more inputs than outputs). For matrices that are near-square, `jacfwd` probably has an edge over `jacrev`.

+++

You can also use `jacfwd` and `jacrev` with container types:

```{code-cell}
def predict_dict(params, inputs):
    return predict(params['W'], params['b'], inputs)

J_dict = jacrev(predict_dict)({'W': W, 'b': b}, inputs)
for k, v in J_dict.items():
    print("Jacobian from {} to logits is".format(k))
    print(v)
```

For more details on forward- and reverse-mode, as well as how to implement `jacfwd` and `jacrev` as efficiently as possible, read on!

+++

Using a composition of two of these functions gives us a way to compute dense Hessian matrices:

```{code-cell}
def hessian(f):
    return jacfwd(jacrev(f))

H = hessian(f)(W)
print("hessian, with shape", H.shape)
print(H)
```

This shape makes sense: if we start with a function $f : \mathbb{R}^n \to \mathbb{R}^m$, then at a point $x \in \mathbb{R}^n$ we expect to get the shapes

* $f(x) \in \mathbb{R}^m$, the value of $f$ at $x$,
* $\partial f(x) \in \mathbb{R}^{m \times n}$, the Jacobian matrix at $x$,
* $\partial^2 f(x) \in \mathbb{R}^{m \times n \times n}$, the Hessian at $x$,

and so on.

To implement `hessian`, we could have used `jacfwd(jacrev(f))` or `jacrev(jacfwd(f))` or any other composition of the two. But forward-over-reverse is typically the most efficient. That's because in the inner Jacobian computation we're often differentiating a function wide Jacobian (maybe like a loss function $f : \mathbb{R}^n \to \mathbb{R}$), while in the outer Jacobian computation we're differentiating a function with a square Jacobian (since $\nabla f : \mathbb{R}^n \to \mathbb{R}^n$), which is where forward-mode wins out.

+++

## How it's made: two foundational autodiff functions

+++

(jax-301-jvp)=

### Jacobian-Vector products (JVPs, aka forward-mode autodiff)

JAX includes efficient and general implementations of both forward- and reverse-mode automatic differentiation. The familiar `grad` function is built on reverse-mode, but to explain the difference in the two modes, and when each can be useful, we need a bit of math background.

#### JVPs in math

Mathematically, given a function $f : \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian of $f$ evaluated at an input point $x \in \mathbb{R}^n$, denoted $\partial f(x)$, is often thought of as a matrix in $\mathbb{R}^m \times \mathbb{R}^n$:

$\qquad \partial f(x) \in \mathbb{R}^{m \times n}$.

But we can also think of $\partial f(x)$ as a linear map, which maps the tangent space of the domain of $f$ at the point $x$ (which is just another copy of $\mathbb{R}^n$) to the tangent space of the codomain of $f$ at the point $f(x)$ (a copy of $\mathbb{R}^m$):

$\qquad \partial f(x) : \mathbb{R}^n \to \mathbb{R}^m$.

This map is called the [pushforward map](https://en.wikipedia.org/wiki/Pushforward_(differential)) of $f$ at $x$. The Jacobian matrix is just the matrix for this linear map in a standard basis.

If we don't commit to one specific input point $x$, then we can think of the function $\partial f$ as first taking an input point and returning the Jacobian linear map at that input point:

$\qquad \partial f : \mathbb{R}^n \to \mathbb{R}^n \to \mathbb{R}^m$.

In particular, we can uncurry things so that given input point $x \in \mathbb{R}^n$ and a tangent vector $v \in \mathbb{R}^n$, we get back an output tangent vector in $\mathbb{R}^m$. We call that mapping, from $(x, v)$ pairs to output tangent vectors, the *Jacobian-vector product*, and write it as

$\qquad (x, v) \mapsto \partial f(x) v$

#### JVPs in JAX code

Back in Python code, JAX's `jvp` function models this transformation. Given a Python function that evaluates $f$, JAX's `jvp` is a way to get a Python function for evaluating $(x, v) \mapsto (f(x), \partial f(x) v)$.

```{code-cell}
from jax import jvp

# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

key, subkey = random.split(key)
v = random.normal(subkey, W.shape)

# Push forward the vector `v` along `f` evaluated at `W`
y, u = jvp(f, (W,), (v,))
```

In terms of [Haskell-like type signatures](https://wiki.haskell.org/Type_signature),
we could write

```haskell
jvp :: (a -> b) -> a -> T a -> (b, T b)
```

where we use `T a` to denote the type of the tangent space for `a`. In words, `jvp` takes as arguments a function of type `a -> b`, a value of type `a`, and a tangent vector value of type `T a`. It gives back a pair consisting of a value of type `b` and an output tangent vector of type `T b`.

+++

The `jvp`-transformed function is evaluated much like the original function, but paired up with each primal value of type `a` it pushes along tangent values of type `T a`. For each primitive numerical operation that the original function would have applied, the `jvp`-transformed function executes a "JVP rule" for that primitive that both evaluates the primitive on the primals and applies the primitive's JVP at those primal values.

That evaluation strategy has some immediate implications about computational complexity: since we evaluate JVPs as we go, we don't need to store anything for later, and so the memory cost is independent of the depth of the computation. In addition, the FLOP cost of the `jvp`-transformed function is about 3x the cost of just evaluating the function (one unit of work for evaluating the original function, for example `sin(x)`; one unit for linearizing, like `cos(x)`; and one unit for applying the linearized function to a vector, like `cos_x * v`). Put another way, for a fixed primal point $x$, we can evaluate $v \mapsto \partial f(x) \cdot v$ for about the same marginal cost as evaluating $f$.

That memory complexity sounds pretty compelling! So why don't we see forward-mode very often in machine learning?

To answer that, first think about how you could use a JVP to build a full Jacobian matrix. If we apply a JVP to a one-hot tangent vector, it reveals one column of the Jacobian matrix, corresponding to the nonzero entry we fed in. So we can build a full Jacobian one column at a time, and to get each column costs about the same as one function evaluation. That will be efficient for functions with "tall" Jacobians, but inefficient for "wide" Jacobians.

If you're doing gradient-based optimization in machine learning, you probably want to minimize a loss function from parameters in $\mathbb{R}^n$ to a scalar loss value in $\mathbb{R}$. That means the Jacobian of this function is a very wide matrix: $\partial f(x) \in \mathbb{R}^{1 \times n}$, which we often identify with the Gradient vector $\nabla f(x) \in \mathbb{R}^n$. Building that matrix one column at a time, with each call taking a similar number of FLOPs to evaluate the original function, sure seems inefficient! In particular, for training neural networks, where $f$ is a training loss function and $n$ can be in the millions or billions, this approach just won't scale.

To do better for functions like this, we just need to use reverse-mode.

+++

(jax-301-vjp)=

### Vector-Jacobian products (VJPs, aka reverse-mode autodiff)

Where forward-mode gives us back a function for evaluating Jacobian-vector products, which we can then use to build Jacobian matrices one column at a time, reverse-mode is a way to get back a function for evaluating vector-Jacobian products (equivalently Jacobian-transpose-vector products), which we can use to build Jacobian matrices one row at a time.

#### VJPs in math

Let's again consider a function $f : \mathbb{R}^n \to \mathbb{R}^m$.
Starting from our notation for JVPs, the notation for VJPs is pretty simple:

$\qquad (x, v) \mapsto v \partial f(x)$,

where $v$ is an element of the cotangent space of $f$ at $x$ (isomorphic to another copy of $\mathbb{R}^m$). When being rigorous, we should think of $v$ as a linear map $v : \mathbb{R}^m \to \mathbb{R}$, and when we write $v \partial f(x)$ we mean function composition $v \circ \partial f(x)$, where the types work out because $\partial f(x) : \mathbb{R}^n \to \mathbb{R}^m$. But in the common case we can identify $v$ with a vector in $\mathbb{R}^m$ and use the two almost interchangeably, just like we might sometimes flip between "column vectors" and "row vectors" without much comment.

With that identification, we can alternatively think of the linear part of a VJP as the transpose (or adjoint conjugate) of the linear part of a JVP:

$\qquad (x, v) \mapsto \partial f(x)^\mathsf{T} v$.

For a given point $x$, we can write the signature as

$\qquad \partial f(x)^\mathsf{T} : \mathbb{R}^m \to \mathbb{R}^n$.

The corresponding map on cotangent spaces is often called the [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry))
of $f$ at $x$. The key for our purposes is that it goes from something that looks like the output of $f$ to something that looks like the input of $f$, just like we might expect from a transposed linear function.

#### VJPs in JAX code

Switching from math back to Python, the JAX function `vjp` can take a Python function for evaluating $f$ and give us back a Python function for evaluating the VJP $(x, v) \mapsto (f(x), v^\mathsf{T} \partial f(x))$.

```{code-cell}
from jax import vjp

# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

y, vjp_fun = vjp(f, W)

key, subkey = random.split(key)
u = random.normal(subkey, y.shape)

# Pull back the covector `u` along `f` evaluated at `W`
v = vjp_fun(u)
```

In terms of [Haskell-like type signatures](https://wiki.haskell.org/Type_signature),
we could write

```haskell
vjp :: (a -> b) -> a -> (b, CT b -> CT a)
```

where we use `CT a` to denote the type for the cotangent space for `a`. In words, `vjp` takes as arguments a function of type `a -> b` and a point of type `a`, and gives back a pair consisting of a value of type `b` and a linear map of type `CT b -> CT a`.

This is great because it lets us build Jacobian matrices one row at a time, and the FLOP cost for evaluating $(x, v) \mapsto (f(x), v^\mathsf{T} \partial f(x))$ is only about three times the cost of evaluating $f$. In particular, if we want the gradient of a function $f : \mathbb{R}^n \to \mathbb{R}$, we can do it in just one call. That's how `grad` is efficient for gradient-based optimization, even for objectives like neural network training loss functions on millions or billions of parameters.

There's a cost, though: though the FLOPs are friendly, memory scales with the depth of the computation. Also, the implementation is traditionally more complex than that of forward-mode, though JAX has some tricks up its sleeve (that's a story for a future notebook!).

For more on how reverse-mode works, see [this tutorial video from the Deep Learning Summer School in 2017](http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/).

+++

### Vector-valued gradients with VJPs

If you're interested in taking vector-valued gradients (like `tf.gradients`):

```{code-cell}
from jax import vjp

def vgrad(f, x):
  y, vjp_fn = vjp(f, x)
  return vjp_fn(jnp.ones(y.shape))[0]

print(vgrad(lambda x: 3*x**2, jnp.ones((2, 2))))
```

### Hessian-vector products using both forward- and reverse-mode

+++

In a previous section, we implemented a Hessian-vector product function just using reverse-mode (assuming continuous second derivatives):

```{code-cell}
def hvp(f, x, v):
    return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)
```

That's efficient, but we can do even better and save some memory by using forward-mode together with reverse-mode.

Mathematically, given a function $f : \mathbb{R}^n \to \mathbb{R}$ to differentiate, a point $x \in \mathbb{R}^n$ at which to linearize the function, and a vector $v \in \mathbb{R}^n$, the Hessian-vector product function we want is

$(x, v) \mapsto \partial^2 f(x) v$

Consider the helper function $g : \mathbb{R}^n \to \mathbb{R}^n$ defined to be the derivative (or gradient) of $f$, namely $g(x) = \partial f(x)$. All we need is its JVP, since that will give us

$(x, v) \mapsto \partial g(x) v = \partial^2 f(x) v$.

We can translate that almost directly into code:

```{code-cell}
from jax import jvp, grad

# forward-over-reverse
def hvp(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1]
```

Even better, since we didn't have to call `jnp.dot` directly, this `hvp` function works with arrays of any shape and with arbitrary container types (like vectors stored as nested lists/dicts/tuples), and doesn't even have a dependence on `jax.numpy`.

Here's an example of how to use it:

```{code-cell}
def f(X):
  return jnp.sum(jnp.tanh(X)**2)

key, subkey1, subkey2 = random.split(key, 3)
X = random.normal(subkey1, (30, 40))
V = random.normal(subkey2, (30, 40))

ans1 = hvp(f, (X,), (V,))
ans2 = jnp.tensordot(hessian(f)(X), V, 2)

print(jnp.allclose(ans1, ans2, 1e-4, 1e-4))
```

Another way you might consider writing this is using reverse-over-forward:

```{code-cell}
# reverse-over-forward
def hvp_revfwd(f, primals, tangents):
  g = lambda primals: jvp(f, primals, tangents)[1]
  return grad(g)(primals)
```

That's not quite as good, though, because forward-mode has less overhead than reverse-mode, and since the outer differentiation operator here has to differentiate a larger computation than the inner one, keeping forward-mode on the outside works best:

```{code-cell}
# reverse-over-reverse, only works for single arguments
def hvp_revrev(f, primals, tangents):
  x, = primals
  v, = tangents
  return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)


print("Forward over reverse")
%timeit -n10 -r3 hvp(f, (X,), (V,))
print("Reverse over forward")
%timeit -n10 -r3 hvp_revfwd(f, (X,), (V,))
print("Reverse over reverse")
%timeit -n10 -r3 hvp_revrev(f, (X,), (V,))

print("Naive full Hessian materialization")
%timeit -n10 -r3 jnp.tensordot(hessian(f)(X), V, 2)
```

## Composing VJPs, JVPs, and `vmap`

+++

### Jacobian-Matrix and Matrix-Jacobian products

Now that we have `jvp` and `vjp` transformations that give us functions to push-forward or pull-back single vectors at a time, we can use JAX's `vmap` [transformation](https://github.com/jax-ml/jax#auto-vectorization-with-vmap) to push and pull entire bases at once. In particular, we can use that to write fast matrix-Jacobian and Jacobian-matrix products.

```{code-cell}
# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

# Pull back the covectors `m_i` along `f`, evaluated at `W`, for all `i`.
# First, use a list comprehension to loop over rows in the matrix M.
def loop_mjp(f, x, M):
    y, vjp_fun = vjp(f, x)
    return jnp.vstack([jnp.asarray(vjp_fun(mi)) for mi in M])

# Now, use vmap to build a computation that does a single fast matrix-matrix
# multiply, rather than an outer loop over vector-matrix multiplies.
def vmap_mjp(f, x, M):
    y, vjp_fun = vjp(f, x)
    outs, = vmap(vjp_fun)(M)
    return outs

key = random.key(0)
num_covecs = 128
U = random.normal(key, (num_covecs,) + y.shape)

loop_vs = loop_mjp(f, W, M=U)
print('Non-vmapped Matrix-Jacobian product')
%timeit -n10 -r3 loop_mjp(f, W, M=U)

print('\nVmapped Matrix-Jacobian product')
vmap_vs = vmap_mjp(f, W, M=U)
%timeit -n10 -r3 vmap_mjp(f, W, M=U)

assert jnp.allclose(loop_vs, vmap_vs), 'Vmap and non-vmapped Matrix-Jacobian Products should be identical'
```

```{code-cell}
def loop_jmp(f, W, M):
    # jvp immediately returns the primal and tangent values as a tuple,
    # so we'll compute and select the tangents in a list comprehension
    return jnp.vstack([jvp(f, (W,), (mi,))[1] for mi in M])

def vmap_jmp(f, W, M):
    _jvp = lambda s: jvp(f, (W,), (s,))[1]
    return vmap(_jvp)(M)

num_vecs = 128
S = random.normal(key, (num_vecs,) + W.shape)

loop_vs = loop_jmp(f, W, M=S)
print('Non-vmapped Jacobian-Matrix product')
%timeit -n10 -r3 loop_jmp(f, W, M=S)
vmap_vs = vmap_jmp(f, W, M=S)
print('\nVmapped Jacobian-Matrix product')
%timeit -n10 -r3 vmap_jmp(f, W, M=S)

assert jnp.allclose(loop_vs, vmap_vs), 'Vmap and non-vmapped Jacobian-Matrix products should be identical'
```

### The implementation of `jacfwd` and `jacrev`

+++

Now that we've seen fast Jacobian-matrix and matrix-Jacobian products, it's not hard to guess how to write `jacfwd` and `jacrev`. We just use the same technique to push-forward or pull-back an entire standard basis (isomorphic to an identity matrix) at once.

```{code-cell}
from jax import jacrev as builtin_jacrev

def our_jacrev(f):
    def jacfun(x):
        y, vjp_fun = vjp(f, x)
        # Use vmap to do a matrix-Jacobian product.
        # Here, the matrix is the Euclidean basis, so we get all
        # entries in the Jacobian at once.
        J, = vmap(vjp_fun, in_axes=0)(jnp.eye(len(y)))
        return J
    return jacfun

assert jnp.allclose(builtin_jacrev(f)(W), our_jacrev(f)(W)), 'Incorrect reverse-mode Jacobian results!'
```

```{code-cell}
from jax import jacfwd as builtin_jacfwd

def our_jacfwd(f):
    def jacfun(x):
        _jvp = lambda s: jvp(f, (x,), (s,))[1]
        Jt = vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun

assert jnp.allclose(builtin_jacfwd(f)(W), our_jacfwd(f)(W)), 'Incorrect forward-mode Jacobian results!'
```

Interestingly, [Autograd](https://github.com/hips/autograd) couldn't do this. Our [implementation](https://github.com/HIPS/autograd/blob/96a03f44da43cd7044c61ac945c483955deba957/autograd/differential_operators.py#L60) of reverse-mode `jacobian` in Autograd had to pull back one vector at a time with an outer-loop `map`. Pushing one vector at a time through the computation is much less efficient than batching it all together with `vmap`.

+++

Another thing that Autograd couldn't do is `jit`. Interestingly, no matter how much Python dynamism you use in your function to be differentiated, we could always use `jit` on the linear part of the computation. For example:

```{code-cell}
def f(x):
    try:
        if x < 3:
            return 2 * x ** 3
        else:
            raise ValueError
    except ValueError:
        return jnp.pi * x

y, f_vjp = vjp(f, 4.)
print(jit(f_vjp)(1.))
```

## Complex numbers and differentiation

Complex differentiation has a reputation for confusion: must the function be
holomorphic? Where do the conjugates go? What does `grad` even mean at a
complex input? In JAX, one organizing principle dissolves nearly all of it:

**JVPs and VJPs are unambiguous, for any function, holomorphic or not. The
only subtlety lives in the convenience wrapper `grad`, which must pick a
packaging convention — and JAX's choice involves a conjugate you should know
about.**

### The unambiguous part: JVPs and VJPs

The key move is to remember what $\mathbb{C}$ is: $\mathbb{R}^2$ with some
extra (multiplicative) structure. Any function $f : \mathbb{C} \to
\mathbb{C}$ determines an ordinary real function $F : \mathbb{R}^2 \to
\mathbb{R}^2$ by

$\qquad f(x + y i) = u(x, y) + v(x, y) i
\quad\leftrightarrow\quad F(x, y) = (u(x, y), v(x, y)),$

and $F$ has an ordinary real Jacobian at each point — a $2 \times 2$ matrix
with four independent real entries, no holomorphy required:

$\qquad J = \begin{bmatrix} \partial_0 u & \partial_1 u \\ \partial_0 v & \partial_1 v \end{bmatrix}.$

The **JVP** of $f$ is nothing more than this real linear map applied to the
tangent pair, with complex numbers serving as containers for pairs of reals:
for a tangent $t = t_1 + t_2 i$, the output tangent is $J (t_1, t_2)$,
repackaged as a complex number. Here's a check, on a decidedly
non-holomorphic function:

```{code-cell}
def u(x, y): return x**2 + jnp.sin(y)
def v(x, y): return x * y

def fun(z):  # not holomorphic!
  x, y = jnp.real(z), jnp.imag(z)
  return u(x, y) + v(x, y) * 1j

z = 1.5 + 0.5j
x, y = jnp.real(z), jnp.imag(z)
J = jnp.array([[grad(u, 0)(x, y), grad(u, 1)(x, y)],
               [grad(v, 0)(x, y), grad(v, 1)(x, y)]])

t = 0.7 - 0.3j
_, t_out = jvp(fun, (z,), (t,))
t_pair = J @ jnp.array([jnp.real(t), jnp.imag(t)])
print(jnp.allclose(t_out, t_pair[0] + t_pair[1] * 1j))
```

No ambiguity, and no conjugates: the JVP is the pushforward by the real
derivative, in complex packaging. (If $f$ happens to be holomorphic, the
Cauchy–Riemann equations make $J$ the rotate-and-scale matrix of complex
multiplication by $f'(z)$, and the JVP becomes $t \mapsto f'(z)\, t$.)

The **VJP** is just as unambiguous, but seeing why takes one more idea —
and that idea is also where `grad`'s conjugation convention will come from,
so it's worth spelling out.

What a VJP fundamentally is, is the *dual map* (or pullback) of the
derivative: cotangents are linear functionals on tangents, and the
derivative's dual map carries output functionals to input functionals by
composition, $\varphi \mapsto \varphi \circ \partial F(x)$. That definition
involves no choices at all. The familiar matrix transpose is what the dual
map looks like *after* you identify each space with its dual — and the
identification is where a choice enters. A nondegenerate pairing
$\langle\cdot,\cdot\rangle$ is exactly such an identification (it matches
the vector $w$ with the functional $\langle w, \cdot\rangle$), and relative
to pairings on the two spaces, the transpose of a linear map $A$ is
characterized by

$\qquad \langle w, A t \rangle = \langle A^\mathsf{T} w, t \rangle
\quad \text{for all } t, w.$

For $\mathbb{R}^n$ with the standard dot product this recovers
swap-the-matrix-indices transposition, but change the pairing and "the
transpose" changes with it.

Now, JAX's `vjp` hands you cotangents as complex numbers — the same type as
the primal values — so somewhere an identification of covectors with
vectors has been made. And note what the scalar field is in this whole
story: it's $\mathbb{R}$, not $\mathbb{C}$. Since $f$ needn't be
holomorphic, its derivative is only guaranteed to be $\mathbb{R}$-linear,
so we must treat $\mathbb{C}$ and its (co)tangent spaces as vector spaces
*over the reals*. Linear functionals are accordingly $\mathbb{R}$-valued —
which is why the pairings below produce real numbers, not complex ones.
(A real-linear map on $\mathbb{C}$ is complex-linear exactly when its
matrix is a rotate-and-scale — that is, exactly when $f$ is holomorphic.
Holomorphy is precisely the special case in which the scalar field could
be upgraded to $\mathbb{C}$.)

On complex numbers viewed as packed pairs of reals, there are two natural
real-valued pairings to make the identification with:

* the **bilinear** pairing $\langle w, t \rangle = \operatorname{Re}(w t)$,
  which in real coordinates is $w_1 t_1 - w_2 t_2$; and
* the **sesquilinear** pairing
  $\langle w, t \rangle = \operatorname{Re}(\bar{w} t) = w_1 t_1 + w_2 t_2$
  — the real part of the standard complex inner product, i.e. the ordinary
  dot product on $\mathbb{R}^2$.

They differ by conjugating one slot, so the two transposes they define
differ by an elementwise conjugation. A convention is required, and **JAX
uses the bilinear one**: its `vjp` is the unique map satisfying

$\qquad \operatorname{Re}(w \cdot \texttt{jvp}(t)) \; = \;
\operatorname{Re}(\texttt{vjp}(w) \cdot t)
\qquad \text{for all } t, w,$

with plain products and no conjugations in sight:

```{code-cell}
w = -0.2 + 1.1j

_, fun_vjp = vjp(fun, z)
w_out, = fun_vjp(w)

print(jnp.allclose(jnp.real(w * t_out), jnp.real(w_out * t)))       # True
print(jnp.allclose(jnp.real(jnp.conj(w) * t_out),
                   jnp.real(jnp.conj(w_out) * t)))                  # False!
```

The second `print` shows the fork in the road: the sesquilinear version of
the same identity is *false* for JAX's `vjp` — it's the identity the other
convention would satisfy instead. In coordinates, JAX's VJP works out to
$w \mapsto \overline{J^\mathsf{T} \bar{w}}$ — but the conjugations are
packaging artifacts, not extra derivative content. Writing $\eta =
\operatorname{diag}(1, -1)$, conjugation of a packed pair *is* $\eta$, and
$\operatorname{Re}(w t) = w^\mathsf{T} \eta\, t$, so
$\overline{J^\mathsf{T} \bar{w}} = \eta J^\mathsf{T} \eta\, w$: precisely
the transpose of $J$ relative to the bilinear pairing. Equivalently, the
underlying operation is plain $J^\mathsf{T}$ on pairs of reals, with
covectors *encoded* as complex numbers via the conjugated identification.

One payoff of this convention: for a *holomorphic* $f$, both maps become the
same plain complex multiplication, with no conjugate in sight:

$\qquad \texttt{jvp}(t) = f'(z)\,t, \qquad \texttt{vjp}(w) = f'(z)\,w.$

```{code-cell}
_, t_out = jvp(jnp.sin, (z,), (t,))
_, sin_vjp = vjp(jnp.sin, z)
w_out, = sin_vjp(w)
print(jnp.allclose(t_out, jnp.cos(z) * t))
print(jnp.allclose(w_out, jnp.cos(z) * w))
```

Under the sesquilinear convention, the VJP of a holomorphic function would
instead be multiplication by $\overline{f'(z)}$. That's why we chose the
bilinear one.

A general (non-holomorphic) $\mathbb{C} \to \mathbb{C}$ function has four
real degrees of freedom in its derivative, so no single complex number can
summarize it. A single JVP or VJP call reveals a two-real-number slice; two
calls — say with tangents (or cotangents) $1$ and $i$ — reveal everything,
exactly as for an $\mathbb{R}^2 \to \mathbb{R}^2$ function. And for
functions with a real side, one call is a full summary: a single `jvp`
reveals everything about an $\mathbb{R} \to \mathbb{C}$ scalar function, and
a single `vjp` (or `grad` — read on) reveals everything about a
$\mathbb{C} \to \mathbb{R}$ function. When in doubt about what a complex
derivative "means", drop down to `jvp` and `vjp`: they never lie and never
raise.

### `grad` at complex inputs: mind the conjugate

`grad(f)(x)` is defined as `vjp(f, x)[1](1.0)`. For a function $f :
\mathbb{C} \to \mathbb{R}$, writing $f(x + yi) = u(x, y)$, this works with
no flags, and by the transpose formula above it evaluates to

$\qquad \texttt{grad}(f)(z) = \partial_0 u(x, y) - \partial_1 u(x, y)\, i.$

Note the minus sign: this is the *conjugate* of the vector
$(\partial_0 u, \partial_1 u)$, which is the direction of steepest ascent in
the plane. Under JAX's convention, directional derivatives use the plain
product with no conjugation,

$\qquad \lim_{\epsilon \to 0} \tfrac{1}{\epsilon}(f(z + \epsilon t) - f(z))
= \operatorname{Re}(\texttt{grad}(f)(z) \cdot t),$

and the flip side of that tidiness is that **for optimization you must
conjugate the gradient**: the steepest-descent step is $z \leftarrow z -
\eta\, \overline{\texttt{grad}(f)(z)}$. Using the unconjugated gradient
flips the sign of the update's imaginary part — it isn't descent at all:

```{code-cell}
def f(z):
  x, y = jnp.real(z), jnp.imag(z)
  return x**2 + y**2      # |z|^2, minimized at z = 0

print(grad(f)(3. + 4j))   # 6 - 8j: conjugate of the steepest-ascent 6 + 8j
```

```{code-cell}
z = 3. + 4j
for _ in range(100):
  z = z - 0.05 * jnp.conj(grad(f)(z))   # with the conjugate: descends
print(f(z))

z = 3. + 4j
for _ in range(100):
  z = z - 0.05 * grad(f)(z)             # without: the imaginary part grows!
print(f(z))
```

This is a convention, not a law of nature: defining `grad` via the
sesquilinear pairing instead would build the conjugate in, at the cost of
breaking the plain-product identities above (and, for holomorphic functions,
making `grad` return $\overline{f'(z)}$ rather than $f'(z)$). The practical
takeaways under JAX's convention:

1. To optimize a real-valued loss of complex parameters, step along the
   *conjugate* of `grad` — and audit any optimizer library you use on
   complex parameters for exactly this, since an optimizer written with real
   parameters in mind won't conjugate.
2. Directional derivatives and first-order Taylor expansions use the plain
   product: $f(z + t) \approx f(z) + \operatorname{Re}(\texttt{grad}(f)(z) \cdot t)$.

### Relation to Wirtinger derivatives

If you've seen complex autodiff described elsewhere — for example in
PyTorch's documentation — you may have met *Wirtinger derivatives*:

$\qquad \frac{\partial}{\partial z} = \tfrac{1}{2}\left(\frac{\partial}{\partial x} - i \frac{\partial}{\partial y}\right), \qquad
\frac{\partial}{\partial \bar z} = \tfrac{1}{2}\left(\frac{\partial}{\partial x} + i \frac{\partial}{\partial y}\right).$

These are nothing exotic: they're a change of coordinates on the same real
$2 \times 2$ Jacobian we've been using all along. Instead of four real
numbers, the derivative of a $\mathbb{C} \to \mathbb{C}$ function is
encoded as two complex ones, $\partial f/\partial z$ and $\partial
f/\partial \bar z$, and the JVP takes the tidy form

$\qquad \texttt{jvp}(t) = \frac{\partial f}{\partial z}\, t + \frac{\partial f}{\partial \bar z}\, \bar{t}.$

```{code-cell}
z = 1.5 + 0.5j                # the point where we computed J above

fx = J[0, 0] + J[1, 0] * 1j   # df/dx = du/dx + i dv/dx
fy = J[0, 1] + J[1, 1] * 1j   # df/dy = du/dy + i dv/dy
dfdz    = 0.5 * (fx - 1j * fy)
dfdzbar = 0.5 * (fx + 1j * fy)

_, t_out = jvp(fun, (z,), (t,))
print(jnp.allclose(t_out, dfdz * t + dfdzbar * jnp.conj(t)))
```

In this vocabulary: a function is holomorphic exactly when $\partial
f/\partial \bar z = 0$ (that *is* the Cauchy–Riemann equations), in which
case $\partial f/\partial z = f'(z)$. And for a real-valued $f$, JAX's
convention gives $\texttt{grad}(f)(z) = 2\, \partial f/\partial z$, while
the steepest-ascent vector is its conjugate, $2\, \partial f/\partial \bar
z$. PyTorch and TensorFlow adopt that other convention: PyTorch's autograd
notes define the gradient of a real-valued loss as $\partial L/\partial
z^*$, the steepest-descent-ready choice, so their optimizers step without
an explicit conjugate — whereas JAX computes $\partial L/\partial z$, and
you conjugate when stepping, as above. It's the same real derivative
underneath; the conventions differ only in which complex packaging of it
the convenience wrapper hands you.

### Holomorphic functions and `grad(f, holomorphic=True)`

For a $\mathbb{C} \to \mathbb{C}$ function, `grad` raises an error on the
complex output — in general four real numbers of derivative information
can't be summarized in one complex number, and the error message suggests
the fix: use `jax.vjp` (or `jax.jvp`) directly.

But a holomorphic function is precisely a $\mathbb{C} \to \mathbb{C}$
function whose derivative *is* a single complex number, $f'(z)$: the
Cauchy–Riemann equations force the $2\times2$ Jacobian to be a
rotate-and-scale, the action of one complex number under multiplication. By
passing `holomorphic=True` you promise JAX that's the case, and `grad`
returns exactly $f'(z)$ (it's `vjp`'s $w \mapsto f'(z) w$ applied to $w =
1.0$):

```{code-cell}
print(grad(jnp.sin, holomorphic=True)(3. + 4j))
print(jnp.cos(3. + 4j))
```

The `holomorphic=True` promise does nothing but disable the
complex-output error. If the function isn't actually holomorphic, you'll
silently get the wrong-looking answer: the gradient of the function with the
imaginary part of its output discarded:

```{code-cell}
def f(z):
  return jnp.conjugate(z)   # not holomorphic!

grad(f, holomorphic=True)(3. + 4j)
```

You should expect complex numbers to work everywhere in JAX. Here's
differentiating through a Cholesky decomposition of a complex matrix:

```{code-cell}
A = jnp.array([[5.,    2.+3j,    5j],
              [2.-3j,   7.,  1.+7j],
              [-5j,  1.-7j,    12.]])

def f(X):
    L = jnp.linalg.cholesky(X)
    return jnp.sum((L - jnp.sin(L))**2)

grad(f, holomorphic=True)(A)
```

## More advanced autodiff

We worked through some easy, and then progressively more complicated,
applications of automatic differentiation in JAX. We hope you now feel that
taking derivatives in JAX is easy and powerful. The rest of this
documentation section goes deeper:

- {doc}`sharding-ad` — how autodiff interacts with explicit sharding: the
  same cotangent-type reasoning as this page, extended to distributed
  arrays.
- {doc}`custom-derivatives` — defining your own derivative rules with hijax
  primitives (including efficient derivatives at fixed points), and
  {doc}`custom-jvp-vjp` for the classic decorator APIs.
- {doc}`refs` — how autodiff interacts with mutable arrays: plumbing values
  out of backward passes and accumulating gradients in place.
- {doc}`remat` — controlling what autodiff saves versus recomputes, to trade
  memory for FLOPs.
- {ref}`jax-301-hijax-types` — differentiating with respect to entirely new
  types.
