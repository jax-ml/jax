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

```{code-cell}
:tags: [remove-cell]

# This ensures that code cell tracebacks appearing below will be concise.
%xmode minimal
```

(jax-101-transformations)=
# Transformations: `grad` and `vmap`

<!--* freshness: { reviewed: '2026-07-09' } *-->

The heart of JAX is a set of *function transformations*: higher-order
functions that take a numerical function you've written and return a new
function that computes something related. The two most common are:

- {func}`jax.grad`, which transforms a function into one that computes its
  gradient via automatic differentiation;
- {func}`jax.vmap`, which transforms a function written for single examples
  into one that operates efficiently over batches via automatic vectorization.

There's a third canonical transformation, {func}`jax.jit`, which compiles a
function to make it run fast. Since `jax.jit` changes only performance and not
what a function computes, we defer it to the performance and scaling docs (see
{ref}`jax-201-jit`), but everything on this page applies to it too.

Transformations compose: you can differentiate a vectorized function, vectorize
a derivative, and compile any of it. This page introduces `grad` and `vmap`,
then explains the tracing mechanism that makes all transformations work.

## Automatic differentiation with `jax.grad`

{func}`jax.grad` takes a scalar-valued function and returns a new function
that computes its gradient:

```{code-cell}
import jax
import jax.numpy as jnp

grad_tanh = jax.grad(jnp.tanh)
print(grad_tanh(2.0))
```

If `f` is a Python function that evaluates the mathematical function $f$, then
`jax.grad(f)` is a Python function that evaluates $\nabla f$, so `grad(f)(x)`
is the gradient value $\nabla f(x)$.

Since `jax.grad` maps functions to functions, you can apply it repeatedly to
take higher-order derivatives:

```{code-cell}
f = lambda x: x**3 + 2*x**2 - 3*x + 1

dfdx = jax.grad(f)
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)

print(dfdx(1.0))   # 3x² + 4x - 3  evaluated at 1, i.e. 4
print(d2fdx(1.0))  # 6x + 4        evaluated at 1, i.e. 10
print(d3fdx(1.0))  # 6
```

### Differentiating with respect to different arguments

By default `jax.grad` differentiates with respect to the first argument. The
`argnums` parameter selects other arguments, or several at once. Here's a
linear logistic regression model, where we might want gradients with respect
to the weights `W`, the bias `b`, or both:

```{code-cell}
def sigmoid(x):
  return 0.5 * (jnp.tanh(x / 2) + 1)

# Outputs probability of a label being true.
def predict(W, b, inputs):
  return sigmoid(jnp.dot(inputs, W) + b)

# A toy dataset, and some parameter values.
inputs = jnp.array([[0.52, 1.12,  0.77],
                    [0.88, -1.08, 0.15],
                    [0.52, 0.06, -1.30],
                    [0.74, -2.49, 1.39]])
targets = jnp.array([True, True, False, True])
W = jnp.array([0.1, 0.4, -0.3])
b = 0.5

# Training loss is the negative log-likelihood of the training examples.
def loss(W, b):
  preds = predict(W, b, inputs)
  label_probs = preds * targets + (1 - preds) * (1 - targets)
  return -jnp.sum(jnp.log(label_probs))
```

```{code-cell}
# Differentiate `loss` with respect to the first positional argument:
W_grad = jax.grad(loss, argnums=0)(W, b)
print(f'{W_grad=}')

# Since argnums=0 is the default, this does the same thing:
W_grad = jax.grad(loss)(W, b)
print(f'{W_grad=}')

# But you can choose different values too, and drop the keyword:
b_grad = jax.grad(loss, 1)(W, b)
print(f'{b_grad=}')

# Including tuple values:
W_grad, b_grad = jax.grad(loss, (0, 1))(W, b)
print(f'{W_grad=}')
print(f'{b_grad=}')
```

Real models don't keep their parameters in separate positional arguments.
Instead they use nested containers like dictionaries. `jax.grad` handles those
natively; that's the subject of the next page, {ref}`jax-101-pytrees`.

### `value_and_grad` and auxiliary outputs

You usually want the loss value as well as its gradient, for example to log
training progress. {func}`jax.value_and_grad` computes both in one pass:

```{code-cell}
loss_value, Wb_grad = jax.value_and_grad(loss, (0, 1))(W, b)
print(loss_value)
print(Wb_grad)
```

And sometimes a function naturally computes intermediate results worth
returning alongside the scalar being differentiated. Just provide a function
that produces a `(scalar_output, aux_data)` pair as output and pass
`has_aux=True`:

```{code-cell}
def loss_and_preds(W, b):
  preds = predict(W, b, inputs)
  label_probs = preds * targets + (1 - preds) * (1 - targets)
  return -jnp.sum(jnp.log(label_probs)), preds

W_grad, preds = jax.grad(loss_and_preds, has_aux=True)(W, b)
print(preds)
```

### Checking derivatives numerically

Derivatives are easy to check against finite differences:

```{code-cell}
eps = 1e-4
b_grad_numerical = (loss(W, b + eps / 2.) - loss(W, b - eps / 2.)) / eps
print('b_grad_numerical', b_grad_numerical)
print('b_grad_autodiff', jax.grad(loss, 1)(W, b))
```

JAX ships a convenience that does this automatically, to any order:

```{code-cell}
from jax.test_util import check_grads

check_grads(loss, (W, b), order=2)  # check up to 2nd order derivatives
```

`jax.grad` is one entry point into a much deeper autodiff system built on the
fundamental {func}`jax.jvp` and {func}`jax.vjp` transformations. Those are
covered in the advanced autodiff docs; see {ref}`jax-301`.

## Automatic vectorization with `jax.vmap`

{func}`jax.vmap` transforms a function written for single inputs into one that
works efficiently on batches of inputs. Consider a function that computes the
convolution of two one-dimensional vectors:

```{code-cell}
x = jnp.arange(5.0)
w = jnp.array([2., 3., 4.])

def convolve(x, w):
  output = []
  for i in range(1, len(x) - 1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

convolve(x, w)
```

Suppose we want to apply this function to a whole batch of `x`s and `w`s:

```{code-cell}
xs = jnp.stack([x, x])
ws = jnp.stack([w, w])
```

The naive option is a Python loop over the batch:

```{code-cell}
def manually_batched_convolve(xs, ws):
  output = []
  for i in range(xs.shape[0]):
    output.append(convolve(xs[i], ws[i]))
  return jnp.stack(output)

manually_batched_convolve(xs, ws)
```

This produces the correct result, but it processes one example at a time, which
performs poorly on hardware built for array-level parallelism. To batch the
computation efficiently you'd normally rewrite the function by hand so that
every operation works over the batch dimension. That's manageable here, but
messy and error-prone for realistic functions.

`jax.vmap` does this rewrite automatically:

```{code-cell}
auto_batch_convolve = jax.vmap(convolve)

auto_batch_convolve(xs, ws)
```

The transformed function behaves *as if* `convolve` were called on each
example, but under the hood every operation inside it acts on the whole batch
at once. No Python loop, no manual rewrite.

### Choosing which axes to map with `in_axes` and `out_axes`

By default, `vmap` maps over the leading axis of every input. The `in_axes` and
`out_axes` arguments override this. For example, if your data has the batch as
the *second* axis:

```{code-cell}
auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)

auto_batch_convolve_v2(xst, wst)
```

An `in_axes` entry of `None` means "don't map this argument"; instead, it's
broadcast to every call. Here we convolve a batch of `x`s against one shared
`w`:

```{code-cell}
batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])

batch_convolve_v3(xs, w)
```

### Composing `vmap`

Like all JAX transformations, `vmap` composes. Nesting `vmap` gives concise
expressions for "all pairs" computations:

```{code-cell}
def dist(x, y):
  return jnp.sqrt(jnp.sum((x - y) ** 2))

def all_pairs(f):
  return jax.vmap(jax.vmap(f, in_axes=(None, 0)), in_axes=(0, None))

points = jnp.array([[0., 0.], [1., 0.], [0., 2.]])
all_pairs(dist)(points, points)
```

Composing `vmap` with `grad` answers questions that are awkward to express
otherwise. For example: the gradient of our logistic regression loss for
*each example separately* (rather than summed over the batch). Write the
per-example loss, differentiate it, then vectorize the derivative:

```{code-cell}
def example_loss(W, b, x, y):
  pred = predict(W, b, x)
  label_prob = pred * y + (1 - pred) * (1 - y)
  return -jnp.log(label_prob)

per_example_grads = jax.vmap(jax.grad(example_loss, (0, 1)),
                             in_axes=(None, None, 0, 0))
W_grads, b_grads = per_example_grads(W, b, inputs, targets)
print(W_grads)  # one gradient per example
print(b_grads)
```

Each transformation did one conceptually simple job, and composition did the
rest. This is the characteristic JAX pattern: write the mathematically natural,
single-example function, and build everything else out of transformations.

(jax-101-tracing)=
## How transformations work: tracing

To transform a function, JAX has to know what the function *does*. It learns
this by **tracing**: calling your Python function with special *tracer* objects
in place of arrays, and overloading every JAX operation applied to them.

You can see tracers directly by printing an argument inside a transformed
function:

```{code-cell}
def f(x):
  print("x =", x)
  print("jax.typeof(x) =", jax.typeof(x))
  return x * 2

result = jax.vmap(f)(jnp.arange(3))
```

The printed value isn't an array, but a tracer, a stand-in for an element of
`jnp.arange(3)`. A tracer always has the same *JAX type* as the value it
stands for, which you can query with {func}`jax.typeof`. Here `x`'s type is
`int32[]`: inside the transformed function, `x` is a rank-0 `int32` array
(shape `()`, hence the empty brackets), exactly as `f` is written to expect.
The tracer's printout also reveals `vmap`'s bookkeeping: a batch of three such
values. As the traced function runs, each operation like `x * 2` is
intercepted by the transformation. What happens next depends on the
transformation being applied: `vmap` replaces each intercepted operation with
a batched version, and `grad` applies each operation's derivative rule.

Another thing a transformation can do is simply record all the operations and
their data dependencies. The result represents the computation performed by
the traced function, specialized to the JAX types of the inputs that were
provided. JAX's datatype for representing such a computation is a *jaxpr*. To
see one, we can borrow {func}`jax.jit`, the compilation transformation we'll
meet properly in the performance docs, and ask it to run only its tracing
step:

```{code-cell}
def g(x):
  return jnp.sin(x) * 2.0

jax.jit(g).trace(1.0).jaxpr
```

We'll use this `.trace(...).jaxpr` idiom whenever we want to see what a
function traces to. Notice what appears in the jaxpr: just the JAX operations,
with every variable annotated with its JAX type (`f32[]` abbreviates
`float32[]`). Anything else
about your Python function (variable names, comments, and importantly, any
*non-JAX-intercepted* side effects) is not recorded.

Jaxprs also let us see precisely what a transformation does to a program.
Here's the recording of `convolve` from earlier, applied to a single example:

```{code-cell}
jax.jit(convolve).trace(x, w).jaxpr
```

Three windows, three dot products, one concatenate. Now the `vmap`ed version,
applied to the batch:

```{code-cell}
jax.jit(jax.vmap(convolve)).trace(xs, ws).jaxpr
```

This is the same program, operation for operation. The only change is that
every operation gained a batch axis: each `f32[5]` became `f32[2,5]`, and each
`dot_general` picked up a batch dimension. That's what "`vmap` replaces each
intercepted operation with a batched version" means concretely: the batching
happens *inside* each operation, where array-level hardware parallelism lives,
and the program stays the same size no matter the batch.

Compare the Python-loop version, `manually_batched_convolve`, whose recording
contains a full copy of the body *per batch element*: six `dot_general`s, not
three, and growing linearly with the batch size:

```{code-cell}
jax.jit(manually_batched_convolve).trace(xs, ws).jaxpr
```

Two big consequences follow from this design.

### Consequence 1: transformations require traceable functions

JAX transformations only work on operations that the tracing machinery
intercepts. The usual sufficient condition for traceability is *functional
purity*: outputs depend only on inputs (arguments and closed-over values);
outputs are produced by applying JAX operations; and no side effects occur.

Purity is valuable even without a tracing implementation: it makes code easier
for the user to reason about, and easier for the compiler to optimize,
parallelize, and scale. It also gives the transformations simple, clear
meanings. A pure function denotes a mathematical function, and the
transformations are mathematical operators on it:
`jax.grad(f)` means $\nabla f$; `jax.vmap(f)` means "$f$ applied to each
element" without worrying about whether or in what order side-effects might
occur; `jax.jit(f)` can promise to return exactly what `f` returns, while
caching and optimizing freely.

Tracing then turns this from good advice into a working requirement. Side
effects in your function, like built-in Python `print` calls, happen at *trace
time*, not when the transformed computation runs.
Our `vmap` example above already showed this: the batch had three elements, but
`print` ran only once, because `vmap` traces the function a single time,
transforming each operation as it's intercepted. Under `jax.jit`, the effect
is even sharper: traces are cached, so a side effect might happen on the first
call and then never again.
If you want to print *runtime* values from transformed code, there's a
purpose-built tool: {func}`jax.debug.print`; see {ref}`jax-201-debugging`.

Reading mutable state has the same problem in reverse: a global value is likely
baked in at trace time, so later updates to it are silently ignored by
transformed code.

For more about traceability, see {ref}`jax-101-state`.

### Consequence 2: traced code can't always specialize on data values

How much does a tracer know about the value it stands in for? That depends on
the transformation. `vmap`'s tracers actually know quite a lot: they carry
the whole batch of values along as the trace proceeds. We can even peek at
the batch mid-trace, through the tracer's `.val` attribute:

```{code-cell}
# warning: `.val` is unsupported internals, so don't rely on it in real code!
jax.vmap(lambda x: print(x.val))(jnp.arange(3.0))
```

(Reaching into a tracer's internals like `.val` is unsafe — it's an
implementation detail, not an API, and in real code you'd use
{func}`jax.debug.print`. We're doing it here purely for a look inside.)

So the values may well be present. What traced code can't do is *specialize* on
them. The operations applied must work for every element of the batch, so the
traced code can't take one particular control-flow branch based on one
particular element's value. Each element might want a different branch, and
the trace can only record one:

```{code-cell}
:tags: [raises-exception]

def absolute(x):
  if x > 0:      # needs one answer, but x stands for a whole batch of values
    return x
  else:
    return -x

jax.vmap(absolute)(jnp.arange(-2.0, 3.0))
```

The error message links to the fix: express data-dependent choices as array
operations, like {func}`jax.numpy.where`, which compute a per-element answer
instead of forcing a single branch:

```{code-cell}
def absolute(x):
  return jnp.where(x > 0, x, -x)

jax.vmap(absolute)(jnp.arange(-2.0, 3.0))
```

(For data-dependent loops and more elaborate control flow, JAX provides
structured control-flow operations like `jax.lax.cond` and `jax.lax.scan`; see
{ref}`jax-201-control-flow`.)

`jax.jit` is the extreme case: its tracers carry no values at all, only the
JAX type, because its recorded program must serve *every* value of that type.
The same constraint follows, for an even stronger reason: there are no values
to consult in the first place.

On the other hand, anything that depends only on JAX types works freely during
tracing, because those are ordinary Python values at trace time.
Python `for` loops over a fixed range, `if` statements on shapes, shape
arithmetic — all fine. Our `convolve` function above used a Python loop whose
bounds came from `len(x)`: that loop simply unrolls during tracing, and the
recorded operations are as if we'd written the unrolled version by hand. This
is also why keeping `import numpy as np` around is useful: `np` operations on
shapes execute immediately at trace time, cleanly separating "computations on
static values" from "computations being traced" (`jnp`).

### `grad` works with Python control flow

If a `vmap` tracer stands for a whole batch of values, and a `jit` tracer
stands for any value of the right JAX type, `jax.grad`'s tracers occupy the
opposite pole: applied on its own, `jax.grad` evaluates your function with
tracers that carry exactly one concrete value alongside the derivative
bookkeeping. With a single value, data-dependent Python control flow is
unambiguous, so if you're just using `grad`, you can apply data-dependent `if`,
`while`, recursion, or anything else:

```{code-cell}
def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4. * x

print(jax.grad(f)(2.0))  # differentiates the 3x² branch: 6x at x=2
print(jax.grad(f)(4.0))  # differentiates the -4x branch
```

Each call differentiates the branch actually taken, giving exactly the
piecewise derivative you'd write by hand. Even loops whose trip count depends
on the data are fine. Here's differentiating through Newton's method for
square roots, where the number of iterations depends on the input:

```{code-cell}
def sqrt_newton(a):
  x = a
  while abs(x * x - a) > 1e-6:   # data-dependent loop
    x = 0.5 * (x + a / x)
  return x

print(sqrt_newton(2.0))
print(jax.grad(sqrt_newton)(2.0))  # 1/(2√2) ≈ 0.3536
```

This flexibility is part of JAX's lineage: JAX's autodiff grew out of
[Autograd](https://github.com/hips/autograd), whose whole point was
differentiating ordinary, idiomatic Python and NumPy code, including branches,
loops, and closures.

The constraints return the moment you compose with a transformation that traces
abstractly: `jax.jit(jax.grad(f))` and `jax.vmap(jax.grad(f))` see the `if`
above fail again, and want it rewritten with `jnp.where`, `lax.cond`, and
friends ({ref}`jax-201-control-flow`). That trade is this documentation's split
in miniature: `grad` by itself maximizes what you can express, and it's
compiling for speed that asks you to make control flow explicit.

### Where `jit` fits in

{func}`jax.jit` uses this same tracing machinery, but instead of replacing
each intercepted operation on the fly, it records them all and hands the
recording to the XLA compiler to produce fast fused machine code, caching the
compiled result keyed on the JAX types of its inputs. In use, it's most often
applied as a decorator:

```{code-cell}
@jax.jit          # equivalent to sum_sq = jax.jit(sum_sq)
def sum_sq(x):
  return jnp.sum(x ** 2)

sum_sq(jnp.arange(3.0))
```

The first call with a given set of input JAX types pays for tracing and
compilation; later calls skip straight to the compiled code. What's new with
`jit` is performance: compilation, caching and retracing, static arguments,
asynchronous dispatch. That story starts the performance and scaling docs; see
{ref}`jax-201-jit`.

## Next steps

Real programs pass around richer structures than just arrays: dictionaries of
parameters, lists of batches, nested configurations. The next page,
{ref}`jax-101-pytrees`, shows how JAX treats those structures as first-class
citizens.
