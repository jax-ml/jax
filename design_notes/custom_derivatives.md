# Custom JVP/VJP rules for JAX-transformable functions

This is a design document, explaining some of the thinking behind the design and
implementation of `jax.custom_jvp` and `jax.custom_vjp`. For user-oriented
documentation, see [the tutorial notebook](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).

There are two ways to define differentiation rules in JAX:
1. using `jax.custom_jvp` and `jax.custom_vjp` to define custom differentiation
   rules for Python functions that are already JAX-transformable; and
2. defining new `core.Primitive` instances along with all their transformation
   rules, for example to call into functions from other systems like solvers,
   simulators, or general numerical computing systems.

This document is about #1 only.

### Contents

* [Goals](#goals)
* [Non-goals](#non-goals)
* [Main problem descriptions](#main-problem-descriptions)
  * [The vmap-removes-custom-jvp semantics problem](#the-vmap-removes-custom-jvp-semantics-problem)
  * [The Python flexibility problem](#the-python-flexibility-problem)
* [Solution idea](#solution-idea)
* [Implementation notes](#implementation-notes)

## Goals

We want **users** to customize the forward- and/or reverse-mode differentiation
behavior of their code. This customization
1. should have a _clear and consistent semantics_ in how it works and how it
   composes with other JAX transformations; and
2. should be _flexible_ in supporting use cases and workflows like in
   [Autograd](https://github.com/hips/autograd) and
   [PyTorch](https://pytorch.org), including cases involving differentiation of
   Python control flow and workflows for NaN debugging.

As **JAX developers** we want to write library functions, like
[`logit`](https://github.com/google/jax/blob/01039299304b148b405ef9b9fa5e82bbb527471d/jax/scipy/special.py#L83)
and
[`expit`](https://github.com/google/jax/blob/01039299304b148b405ef9b9fa5e82bbb527471d/jax/scipy/special.py#L91),
that are defined in terms of other primitives, but for the purposes of
differentiation have primitive-like behavior in the sense that we want to define
custom differentiation rules for them, which may be more numerically stable or
performant. In particular, we don't want to have to specify `vmap` or `jit`
rules for functions like `logit` and `expit`.

As a stretch goal, we’d like to make JAX a great environment for power users
looking to add custom differentiation rules for higher-order functions like
`fixed_point`, `odeint`, etc.; this design doc won’t solve that problem, but we
want to be confident we’re not going to preclude good solutions to that problem.

That is, our primary goals are
1. solve the vmap-removes-custom-jvp semantics problem ([#1249](https://github.com/google/jax/issues/1249)), and
2. allow Python in custom VJPs, e.g. to debug NaNs
   ([#1275](https://github.com/google/jax/issues/1275)).

Secondary goals are
3. clean up and simplify user experience (symbolic zeros, kwargs, etc)
4. make progress towards a world where users can easily add `fixed_point`,
   `odeint`, `root`, etc.

Overall, we want to close
[#116](https://github.com/google/jax/issues/116),
[#1097](https://github.com/google/jax/issues/1097),
[#1249](https://github.com/google/jax/issues/1249),
[#1275](https://github.com/google/jax/issues/1275),
[#1366](https://github.com/google/jax/issues/1366),
[#1723](https://github.com/google/jax/issues/1723),
[#1670](https://github.com/google/jax/issues/1670),
[#1875](https://github.com/google/jax/issues/1875),
[#1938](https://github.com/google/jax/issues/1938),
and replace the custom_transforms machinery (from
[#636](https://github.com/google/jax/issues/636),
[#818](https://github.com/google/jax/issues/818),
and others).

## Non-goals

Here are objectives we're **not** aiming to achieve:
1. The `custom_transforms` machinery aimed to provide a transformation-generic
   mechanism for customizing behavior, in principle (though never really used in
   practice) allowing users to customize rules for any transformation while
   somehow inheriting the “transparent” behavior for others. **We are instead
   only going to solve the customization problem for differentiation (JVP and
   VJP, separately).** Differentiation is the only case actually requested, and
   by specializing to differentiation we can reduce complexity and improve
   flexibility. To control all rules one can just write a primitive.
2. **We’re not going to prioritize mathematical aesthetics** over flexibility
   and clarity on the user side, and simplicity on the implementation side. In
   particular, while the custom VJP signature `a -> (b, CT b --o CT a)` is
   mathematically pleasing, if it’s hard to implement in a Python mechanism
   because of the closure in the return type, we’re fine doing something that
   handles residuals more explicitly.
3. **Serialization support**, of the form where the staged-out serialized
   program representation can be loaded and further JAX-transformed as opposed
   to just evaluated, is currently out of scope for these custom JVP/VJP
   transformation rules. Serialization may be useful not only for researchers
   who want to save some representation of their computation (and transform it
   after loading it), but also for future considerations like having jaxpr
   transformations implemented outside Python, or having jaxprs as an MLIR
   dialect. By defining this as a non-goal for the purpose of this design, we
   have fewer constraints on where we can stash Python callables.

## Main problem descriptions

### The vmap-removes-custom-jvp semantics problem

The vmap-removes-custom-jvp semantics problem is that vmap does not compose
properly with differentiation of functions with `custom_transforms` rules:

```python
# old custom_transforms api to be replaced
@jax.custom_transforms
def f(x):
  return 2. * x

# f_vjp :: a -> (b, CT b --o CT a)
def f_vjp(x):
  return f(x), lambda g: 3. * x  # 3 instead of 2

jax.defvjp_all(f, f_vjp)

grad(f)(1.)  # 3.
vmap(grad(f))(np.ones(4))  # [3., 3., 3., 3.]
grad(lambda x: vmap(f)(x).sum())(np.ones(4))  # [2., 2., 2., 2.]
```

The last grad-of-vmap line has an unexpected result! In general, applying
`vmap`, or really any non-differentiation transformation, has the effect of
removing the custom differentiation rule. (Applying `jvp` causes a failure when
a custom VJP rule is defined.)

The problem exists because transformations are like rewrites, and the `vmap`
transformation effectively rewrites the function to no longer call the
newly-introduced primitive for which there is a custom rule (and hence `grad`
then doesn’t produce the custom rule’s result). In more detail, the
`custom_transforms` machinery sets things up so that evaluating `f(x)` applies
the function

```
{ lambda  ; ; a.
  let b = f_primitive a
  in [b] }
```

where `f_primitive` is a new primitive (introduced for every `custom_transforms`
function and in fact for every call of the function) to which the custom VJP
rule is associated. When we evaluate `grad(f)(x)`, the differentiation machinery
encounters `f_primitive` and processes it with the custom rule.

However, because `f_primitive` is _transparent_ to `vmap`, in the sense that
`vmap` operates on (effectively by inlining) the definition of `f_primitive`,
the function `vmap(f)` is effectively

```
{ lambda  ; ; a.
  let b = mul 2. a
  in [b] }
```

In words, `vmap` rewrites the function in terms of its underlying primitives and
their transformation rules, removing `f_primitive` entirely.


More generally, **because `vmap(f)` has semantics defined in terms of calls to
f, it is semantically inconsistent to remove the custom derivative rule**. That
is, since we define

```python
vmap(f)(xs) == np.stack([f(x) for x in xs])
```

we must have

```python
jvp(vmap(f))(xs) == jvp(lambda xs: np.stack([f(x) for x in xs]))
```

yet this property is not observed when `f` has a custom derivative rule defined,
as the custom derivative rule is used in the right-hand version but not the
left-hand one.

This issue isn’t specific to `vmap`; it applies to all transformations for which
the semantics of transforming a function `f` are defined in terms of calls to
the function `f`, rather than rewriting it into another function. The `mask`
transformation also falls into this class. Differentiation transforms and the
hypothetical all-unary-functions-become-cosine transform are not in this class.

(The interaction between additional custom rules, like custom `vmap` rules, is
likely to get even more complex, suggesting the problem framing of
`custom_transforms` is too broad.)

### The Python flexibility problem

In JAX, as in [Autograd](https://github.com/hips/autograd) and
[PyTorch](https://pytorch.org) but not TF1, differentiation of a Python function
is performed while the function is being executed and traced. This behavior
delights users for a few reasons.

**First and most importantly, it enables pdb-based workflows, e.g. for
inspecting numerics or catching NaNs.** That is, users can employ the standard
Python debugger and other Python-native tools to debug their code, even being
able to inspect runtime values to understand numerical behavior on examples and
to catch fundamentally runtime errors like NaNs. In fact, just while working on
the PR corresponding to this design, especially on the `odeint` primitive, I
used runtime value inspection to debug issues many times, increasing my
confidence that this is a key user workflow in Python. One especially handy
trick, which I’ve used in both JAX and Autograd many times, is the ability to
insert a debugger breakpoint in a custom VJP rule to enter a debugger at a
specific point in the backward pass.

**Second, it allows differentiation of Python native control flow.** We’re not
sure how often this is used in practice in finalized software artifacts, but
when users first poke around JAX or Autograd they’re often impressed by this
freedom. There’s a reason we include it at the top of our JAX and Autograd
READMEs, slide decks, and demos. Ceding this capability would be a step backward
from Autograd. We want JAX to have the best automatic differentiation.

However, the `custom_transforms` machinery does not provide this Python-support
flexibility. That is, because it’s implemented in terms of up-front jaxpr
formation from the Python code for both the user function and custom
differentiation rules, code like this leads to an abstract value tracing error:

```python
# old custom_transforms api to be replaced
@jax.custom_transforms
def f(x):
  if x > 0:
    return x
  else:
    return 0.

def f_vjp(x):
  return ...

jax.defvjp_all(f, f_vjp)

grad(f)(1.)  # Error!
```

## Solution idea

The main idea is that **[dougalm@](https://github.com/dougalm) already solved
these problems with `core.call`**. That is, we can frame the task of specifying
a custom JVP rule for a user function in terms of a new Python-level call
primitive (not to be added to the jaxpr language; see below). This new call
primitive has a user Python function associated with it just like `core.call`,
but additionally has a second Python callable representing the JVP rule. Let’s
refer to this new call primitive as `custom_jvp_call`.

Transformations like `vmap` interact with `custom_jvp_call` as with `core.call`:
they effectively pass right through it and are applied to the underlying Python
callables. Schematically, writing in terms of curried versions of the primitives
for convenience, analogously to how `vmap` interacts with `core.call` by
applying to the function to be called:

```python
vmap(call(f)) == call(vmap(f))
```

for the new primitive `custom_jvp_call` we simply apply `vmap` to the two
functions it entails:

```python
vmap(custom_jvp_call(f, f_jvp)) == custom_jvp_call(vmap(f), vmap(f_jvp))
```

This behavior means we’ve solved the [vmap-removes-custom-jvp semantics
problem](the-vmap-removes-custom-jvp-semantics-problem).

The `jvp` transformation interacts as one might expect: it just calls `f_jvp`,


```python
jvp(call(f)) == call(jvp(f))

jvp(custom_jvp_call(f, f_jvp)) == f_jvp
```

Because `custom_jvp_call` acts like `core.call` (and not like `xla.xla_call`) in
that it doesn’t raise the abstraction level of its inputs (because it’s not
delaying anything or staging anything out), it means we’ve solved [the Python
flexibility problem](the-python-flexibility-problem): there are no constraints
on the user Python function (above the usual functional programming constraints
required by `jvp` or `vjp`).

What about evaluation and compilation? These are two ways to “exit” the JAX
system, in the sense that no additional transformations can be applied after
these steps. As a result, their rules are trivial:

```python
eval(call(f)) == eval(f)
jit(call(f)) == hlo_call(jit(f))

eval(custom_jvp_call(f, f_jvp)) == eval(f)
jit(custom_jvp_call(f, f_jvp)) == hlo_call(jit(f))
```

In words, if a JVP rule hasn’t already rewritten `custom_jvp_call(f, f_jvp)`
into `f_jvp`, when we get to the point of evaluation with `eval` or staging out
to XLA with `jit`, differentiation is never going to be applied, so we just
ignore `f_jvp` and behave just like `core.call`. However, due to the wrinkle
discussed next, the partial eval rule for `custom_jvp_call` must be a bit more
complex, since partial evaluation isn’t just used to stage out to XLA with
`jit`.

The only remaining wrinkle has to do with “initial-style” jaxpr-forming
primitives, like `lax.scan`, and their transformation rules. These represent a
different kind of “staging out to a jaxpr” than that for compilation because we
can perform additional transformations on the staged-out jaxpr. That is, when
`lax.scan` forms a jaxpr, it does not exit the transformation system, since when
we apply a jvp or vmap to a `lax.scan` we need to apply it to the function
represented by the jaxpr.

Another way to state the wrinkle is that initial-style primitives like `lax.scan`
rely on the ability to round-trip to a jaxpr and back to a Python callable while
preserving semantics. That must mean preserving custom differentiation rule
semantics too.

The solution is to use a bit of dynamic scoping: when we're staging out to a
jaxpr for an initial-style primitive, like those in lax_control_flow.py, we set
a bit on the global trace state. When that bit is set, instead of using the
final-style `custom_jvp_call` primitive, we use an initial-style
`custom_jvp_call_jaxpr` primitive, and trace the functions `f` and `f_jvp` to
jaxprs up-front to make initial-style processing easier. The
`custom_jvp_call_jaxpr` primitive is otherwise similar to the final-style
version.

(Footnote: while morally we form jaxprs for both `f` and `f_jvp` before binding
`custom_jvp_call_jaxpr`, we need to delay the formation of the jaxpr of `f_jvp`
because it may call the custom-JVP function and thus eager processing would lead
to an infinite recursion. We delay that jaxpr formation in a thunk.)

If we gave up on [the Python flexibility
problem](the-python-flexibility-problem), we could get away with only having
`custom_jvp_call_jaxpr` and not having the separate Python-level primitive
`custom_jvp_call`.


## API

The custom JVP for an `a -> b` function is specified with an `(a, Ta) -> (b, T
b)` function:

```python
# f :: a -> b
@jax.custom_jvp
def f(x):
  return np.sin(x)

# f_jvp :: (a, T a) -> (b, T b)
def f_jvp(primals, tangents):
  x, = primals
  t, = tangents
  return f(x), np.cos(x) * t

f.defjvp(f_jvp)
```

(Interesting autodiff aside: for the rule to apply to higher-order
differentiation, one must call `f` in the body of `f_jvp`; that precludes some
kinds of work sharing between the internals of `f` and the tangent calculation.)

The custom VJP for an `a -> b` function is specified with an `a -> (b, c)` forward
pass function paired with a `(c, CT b) -> CT` a backward pass function:

```python
# f :: a -> b
@jax.custom_vjp
def f(x):
  return np.sin(x)

# f_fwd :: a -> (b, c)
def f_fwd(x):
  return f(x), np.cos(x)

# f_bwd :: (c, CT b) -> CT a
def f_bwd(cos_x, g):
  return (cos_x * g,)

f.defvjp(f_fwd, f_bwd)
```

The signature `a -> (b, CT b --o CT a)` is more aesthetically pleasing, but
supporting it would make the implementation more complex and might require
compromising expressibility desiderata. The basic reason that Python callables
are opaque (unless we trace them to a jaxpr eagerly, which places expressiveness
constraints), and in this case we may be returning a callable with `vmap` tracers
inside its closure that we need to know about during the forward pass.

We could add convenience wrappers, for example to define the JVP rule for a
single argument at a time (like we do internally for primitives). But because
this proposal is complicated enough as it is, I decided against convenience
layers; let’s keep things minimal for now.

There are some other bells and whistles to the API:
* Inputs and output types `a`, `b`, and `c` can be arbitrary pytrees of
  jaxtypes.
* Passing arguments by name (keyword arguments) is supported when they can be
  resolved to positions using the `inspect` module. This is a bit of an experiment
  with Python 3’s improved ability to programmatically inspect argument
  signatures. I believe it is sound but not complete, which is a fine place to be.
  (See also [#2069](https://github.com/google/jax/issues/2069).)
* Arguments can be marked non-differentiable using `nondiff_argnums`, and as with
  `jit`’s `static_argnums` these arguments don’t have to be JAX types. We need to
  set a convention for how these arguments are passed to the rules. For a primal
  function with type signature `(d, a) -> b` where `d` represents the
  non-differentiable type, the JVP rule’s signature is `(a, T a, d) -> T b` and
  the VJP rule’s reverse component signature is `(d, c, CT b) -> CT a`. That is,
  the non-differentiable arguments are passed in order after `primals` and
  `tangents` for a custom JVP rule, and passed in order preceding the residuals in
  a custom VJP rule’s reverse function.

## Implementation notes

* Updated `jax.experimental.odeint`
  * Since `odeint` is a pretty complex user of a custom VJP rule, in addition to
  just updating it to work at all, I wanted to revise it to be a canonical
  user of the new custom VJP API as a way to test that the API was a good one.
  * Along the way I made other improvements to the `odeint` implementation:
    * remove raveling/unraveling boilerplate
    * make use of `lax.scan` to remove the index-update logic
    * speed up by 20+% on the simple pendulum benchmark
* Added a custom bind method on each transform for the custom derivative call
  primitives, `custom_jvp_call` and `custom_vjp_call`. It’s like
  `core.call_bind`, except we don’t process env traces: those are just errors.
* Added `custom_lin` primitive, which gets staged out into linear jaxprs to be
  transposed when using a custom VJP rule.
  * Because our reverse-mode autodiff is decomposed into linearization, partial
  evaluation, and transposition, our custom VJP rules are processed in two
  separate steps: one during linearization and one during transposition.
  * The linearization step, i.e. the JVP rule for `custom_vjp_call`, applies
  `custom_lin` to the tangent values; `custom_lin` carries with it the user’s
  custom backward-pass function, and as a primitive it only has a transpose
  rule.
  * This mechanism is described more in [#636](https://github.com/google/jax/issues/636).
* To prevent 
