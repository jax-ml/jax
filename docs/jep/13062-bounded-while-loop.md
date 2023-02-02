# Bounded while loops

- *Author: patrick-kidger (GitHub) / kidger (Google)*

- *Date: November 2022*

## Summary

We propose to introduce "bounded while loops" into JAX: these are while loops which exit unconditionally after some maximum number of iterations (are "bounded"), and are reverse-mode autodifferentiable.

An implementation is available at https://github.com/google/jax/pull/13062.

## Background

### JAX limitations

In order to reverse-mode autodifferentiate an operation, then residuals must be stored between the forward and backward pass. In the case of a while loop, which may iterate for an arbitrary number of steps, then these residuals may be of potentially unbounded size. However, it is a limitation of JAX (inherited from XLA) that the shape of all arrays be known statically in advance. For this reason, `lax.while_loop` is not reverse-mode autodifferentiable (or to be precise, not transposable).

If a bound were placed on the maximum number of steps (after which the while loop exits unconditionally), then this would change: residuals may be stored in arrays of length `max_steps`, which will be partially filled according to how many steps are actually taken.

### Use cases

Differential equation solvers are a major use-case for this feature. A differential equation solver will typically take some variable number of steps, as determined on-the-fly by an adaptive step size controller. Differentiating through the solver is desirable: this makes it possible to fit model parameters to data via gradient updates, as is also done in deep learning. (Indeed, modern techniques frequently hybridise the two.)

Other use-cases include linear and nonlinear solvers, as many of these solvers are iterative in nature, e.g. linear conjugate gradient or (quasi-)Newton methods. Sparse numerical linear algebra, such as sparse truncated SVD, is another example.

(  
Some of these other use cases can admit other approaches. For example, if run to convergence then a linear or nonlinear solver may be autodifferentiated as a higher-order primitive, e.g. via the implicit function theorem. For this reason we emphasise differential equation solvers as our main use-case of interest, as they typically admit no satisfactory alternative.
)

As such, bounded while loops have become a frequent feature request:
https://github.com/google/jax/issues/2139  
https://github.com/google/jax/issues/2469  
https://github.com/google/jax/issues/5642  
https://github.com/google/jax/discussions/8375  
https://github.com/google/jax/issues/9960  
etc.

For this reason we would like to add support for this feature into JAX.

Note that having a bound on the maximum number of steps is typically not a limitation in practice. Such a bound is often either known for a particular problem (e.g. the conjugate gradient algorithm must necessarily converge in a number of iterations bounded by the dimensionality of the linear solve) or a user-specified maximum number of steps is already desirable (in differential equation solvers, we frequently prefer to raise an error rather than take an arbitrary number of steps integrating too-stiff dynamics).

## Implementation

It turns out that this is an easy feature to add into core JAX, but very difficult to emulate using existing JAX operations.

### Can this be done by composing existing JAX operations?

What follows is a high-level summary of the best-possible approach available today. (With further references, and an implementation, provided below.)

A first naive implementation would be to nest `scan-cond`. However:

1. This needlessly iterates for `max_steps - actual_steps` many identity functions (on both the forward and backward pass).
2. This misbehaves under `jax.vmap` -- as each `cond` then becomes a `select`, and the computational benefits of early-exit are lost.

These issues can be addressed. By nesting `scan`s then only logarithmically many identity steps need be taken, and by introducing new primitives it can be arranged that `vmap(cond)` runs if and only if one of the batch elements of the predicate are truthy.

Unfortunately, these fixes then introduce further issues of their own:

1. Backpropagation scales as `O(max_steps)` rather than `O(actual_steps)`. We hypothesise that this is due to the XLA compiler being unable to recognise the structure of the nested scans, and that as such it is unable to efficiently pass residuals between the forward and backward passes.
2. XLA:CPU fails to optimise away in-place updates, again likely due to the nested structure. (https://github.com/google/jax/issues/8192)

The issue of backpropagation can be partially ameliorated by `checkpoint`ing each step (and thus removing the possibility of residuals). However this implies extra computational work, which unnecessarily slows things down.

Moreover, this is asymptotically inefficient, costing `O(actual_steps * log(max_steps))`. Contrast direct backpropagation, which is `O(actual_steps)`, and recursive binary checkpointing (also known as 'treeverse' or 'optimal checkpointing') which is `O(actual_steps log(actual_steps))`. [A proof of the asymptotic inefficiency is available here](https://github.com/patrick-kidger/diffrax/blob/b8475527eacdba81328130edd7dadd08a0b34063/docs/devdocs/bounded_while_loop.md). This result may be understood intuitively by recognising that each step of the bounded while loop is placed under multiple levels of checkpointing (from each level of the `scan`), and thus is computed `depth = log(max_steps)` many times during backpropagation. Even so, to the best of the author's knowledge, this is the asymptotically optimal approach available when composing existing JAX operations. (See previous reference.) In particular, note that optimal checkpointing is not possible without compile-time blow-up, as this requires unrolling the loop.

Returning to the issue of in-place updates. This can be partially ameliorated by hoisting the in-place update out of the `cond`, i.e. by performing `array.at[i].set(cond(...))` rather than `cond(..., lambda ...: array.at[i].set(...), ...)`. The downside of this is that this requires changing the function signature of the body function, so as to pass the additional information (on what is and what isn't an in-place update) to the bounded while loop implementation. This is inelegant, and if standard in-place updates are used in the body function then this introduces a silent performance footgun.

Nonetheless -- once these changes are made, then a third round of issues arise.

1. When nesting such "fake bounded while loops", then some manner of XLA bug is hit, such that dead code must be introduced to produce efficient run times. ([See here](https://github.com/patrick-kidger/diffrax/blob/2b4e4d863c15abc7143919bac7825090bbfe50be/diffrax/integrate.py#L256).)
2. The extra tracing work required to resolve this nested structure implies a measurable amount of additional tracing and compilation time. (E.g. 28 seconds -> 5 seconds on one microbenchmark.)

These issues have no workaround; poor performance is the result.

In summary, the best that is available today suffers from:

1. Asymptotically suboptimal scaling, with actual runtime slowdowns observed in practice;
2. An API different to the current `while_loop`, requiring manual marking of precisely which arrays need an in-place update;
3. A silent performance footgun if an in-place update is made anyway;
4. Longer compilation times, due to the complexity of the computation;
5. A need to navigate several obscure XLA bugs, whose behaviour may change in the future.

Further details about what is possible in JAX today can be found in [this comment](https://github.com/google/jax/issues/2139#issuecomment-1039293633) and [in this documentation](https://github.com/patrick-kidger/diffrax/blob/b8475527eacdba81328130edd7dadd08a0b34063/docs/devdocs/bounded_while_loop.md). An implementation is available [as part of the Diffrax library](https://github.com/patrick-kidger/diffrax/blob/b8475527eacdba81328130edd7dadd08a0b34063/diffrax/misc/bounded_while_loop.py), which typically lowers to a nested `scan-checkpoint-unvmap-cond-scan-checkpoint-unvmap-cond-scan-unvmap-cond`.

From the point of view of XLA, the above implementation is "morally wrong". Both the forward and backward passes through a bounded while loop should be implementable as a single while loop each. The above complexity is a reflection of a lack of expressivity on JAX's part.

For these reasons, we would like to move to a cleaner implementation, in which bounded while loops are a first-class citizen in JAX.

### Bounded while loops as a first-class citizen

We propose extending the `while_loop` API with a `max_steps` parameter, after which the computation will halt unconditionally.

To be able to reverse-mode autodifferentiate (or transpose) a bounded while loop, we must store the residuals as extensive outputs. These then become extensive inputs to the backward pass, which would be implemented as another while loop that reads from these residuals.

This is now essentially the form of a `scan`, which consumes and produces extensive inputs and outputs. As such, we propose to leverage the existing `scan` implementation by introducing a `scan(..., early_exit=...)` option, which halts the scan once `early_exit(carry) == True`.

`while_loop(..., max_steps=...)` may then be implemented by prepending the following code to the start of the existing `while_loop` implementation:
```python
if max_steps is not None:
  def f(carry, _):
    return body_fun(carry), None

  def early_exit(carry):
    return lax.bitwise_not(cond_fun(carry))

  final_val, _ = scan(f, init_val, xs=None, length=max_steps, early_exit=early_exit)
  return final_val
```

Implenting `scan(..., early_exit=...)` itself is straightforward. After all transformation rules, `scan` is already implemented as a lowering to `while_loop`, with a conditional that counts the number of steps. `early_exit` may be implemented by adjusting this conditional from `step < max_steps` to `(step < max_steps) & bitwise_not(early_exit(carry))`, and then plumbing this additional `early_exit` argument through each of the transformation rules of `scan`.

This is what is done in https://github.com/google/jax/pull/13062.

Some minor technical notes:

- The primitive `scan_p` now takes three extra inputs: a `start` and a `stop` specifying the interval to iterate over, and a `done` specifying the initial value of `early_exit`. A user-facing call to `scan` will always have `start = 0`, `stop = length`, and `done = early_exit(init)`. The transpose computation will then have `start = 0`, `stop = <final step of primal computation>`, `done = False`, `early_exit = lambda _: False`.
    - Both `start` and a `stop` are needed to support the `reverse=True` case.

- In practice, we append the early-exit value to the output of `f` in `scan(f, ...)`, and treat this value as a carry. This simplifies the diff relative to the existing JAX implementation. It is not necessary to thread through an additional jaxpr for `early_exit`, nor must we treat closed-over constants separately to those of `f`, nor do we need to adjust the resolution of fixed-points (for symbolic zeros, partial values, DCE), etc. etc.
    - The only downside is that this means tracing/compiling `early_exit` twice: once for the initial value, and once in the body. In practice this is not a severe limitation, as such functions are typically small and therefore fast to trace and compile.
