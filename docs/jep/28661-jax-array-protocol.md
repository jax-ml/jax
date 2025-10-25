# JEP 28661: Supporting the `__jax_array__` protocol

[@jakevdp](http://github.com/jakevdp), *May 2025*

An occasional user request is for the ability to define custom array-like objects that
work with jax APIs. JAX currently has a partial implementation of a mechanism that does
this via a `__jax_array__` method defined on the custom object. This was never intended
to be a load-bearing public API (see the discussion at {jax-issue}`#4725`), but has
become essential to packages like Keras and flax, which explicitly document the ability
to use their custom array objects with jax functions. This JEP proposes a design for
full, documented support of the `__jax_array__` protocol.

## Levels of array extensibility
Requests for extensibility of JAX arrays come in a few flavors:

### Level 1 Extensibility: polymorphic inputs
What I’ll call "Level 1" extensibility is the desire that JAX APIs accept polymorphic inputs.
That is, a user desires behavior like this:

```python
class CustomArray:
  data: numpy.ndarray
  ...

x = CustomArray(np.arange(5))
result = jnp.sin(x)  # Converts `x` to JAX array and returns a JAX array
```

Under this extensibility model, JAX functions would accept CustomArray objects as inputs,
implicitly converting them to `jax.Array` objects for the sake of computation.
This is similar to the functionality offered by NumPy via the `__array__` method, and in
JAX (in many but not all cases) via the `__jax_array__` method.

This is the mode of extensibility that has been requested by the maintainers of `flax.nnx`
and others. The current implementation is also used by JAX internally for the case of
symbolic dimensions.

### Level 2 extensibility: polymorphic outputs
What I’ll call "Level 2" extensibility is the desire that JAX APIs should not only accept
polymorphic inputs, but also wrap outputs to match the class of the input.
That is, a user desires behavior like this:

```python
class CustomArray:
  data: numpy.ndarray
  ...

x = CustomArray(np.arange(5))
result = jnp.sin(x)  # returns a new CustomArray
```

Under this extensibility model, JAX functions would not only accept custom objects
as inputs, but have some protocol to determine how to correctly re-wrap outputs with
the same class. In NumPy, this sort of functionality is offered in varying degrees by
the special `__array_ufunc__`, `__array_wrap__`, and `__array_function__` protocols,
which allow user-defined objects to customize how NumPy API functions operate on
arbitrary inputs and map input types to outputs.
JAX does not currently have any equivalent to these interfaces in NumPy.

This is the mode of extensibility that has been requested by the maintainers of `keras`,
among others.

### Level 3 extensibility: subclassing `Array`

What I’ll call "Level 3" extensibility is the desire that the JAX array object itself
could be subclassable. NumPy provides some APIs that allow this
(see [Subclassing ndarray](https://numpy.org/devdocs/user/basics.subclassing.html)) but
this sort of approach would take some extra thought in JAX due to the need for
representing array objects abstractly via tracing.

This mode of extensibility has occasionally been requested by users who want to add
special metadata to JAX arrays, such as units of measurement.

## Synopsis

For the sake of this proposal, we will stick with the simplest, level 1 extensibility
model. The proposed interface is the one currently non-uniformly supported by a number
of JAX APIs, the `__jax_array__` method. Its usage looks something like this:

```python
import jax
import jax.numpy as jnp
import numpy as np

class CustomArray:
  data: np.ndarray

  def __init__(self, data: np.ndarray):
    self.data = data

  def __jax_array__(self) -> jax.Array:
    return jnp.asarray(self.data)

arr = CustomArray(np.arange(5))
result = jnp.multiply(arr, 2)
print(repr(result))
# Array([0, 2, 4, 6, 8], dtype=int32)
```

We may revisit other extensibility levels in the future.

## Design challenges

JAX presents some interesting design challenges related to this kind of extensibility,
which have not been fully explored previously. We’ll discuss them in turn here:

### Priority of `__jax_array__` vs. PyTree flattening
JAX already has a supported mechanism for registering custom objects, namely pytree
registration (see [Extending pytrees](https://docs.jax.dev/en/latest/pytrees.html#extending-pytrees)).
If we also support __jax_array__, which one should take precedence?

To put this more concretely, what should be the result of this code?

```python
@jax.jit
def f(x):
  print("is JAX array:", isinstance(x, jax.Array))

f(CustomArray(...))
```

If we choose to prioritize `__jax_array__` at the JIT boundary, then the output of this
function would be:
```
is JAX array: True
```
That is, at the JIT boundary, the `CustomArray` object would be converted into a
`__jax_array__`, and its shape and dtype would be used to construct a standard JAX
tracer for the function.

If we choose to prioritize pytree flattening at the JIT boundary, then the output of
this function would be:
```
type(x)=CustomArray
```
That is, at the JIT boundary, the `CustomArray` object is flattened, and then unflattened
before being passed to the JIT-compiled function for tracing. If `CustomArray` has been
registered as a pytree, it will generally contain traced arrays as its attributes, and
when x is passed to any JAX API that supports `__jax_array__`, these traced attributes
will be converted to a single traced array according to the logic specified in the method.

There are deeper consequences here for how other transformations like vmap and grad work
when encountering custom objects: for example, if we prioritize pytree flattening, vmap
would operate over the dimensions of the flattened contents of the custom object, while
if we prioritize `__jax_array__`, vmap would operate over the converted array dimensions.

This also has consequences when it comes to JIT invariance: consider a function like this:
```python
def f(x):
  if isinstance(x, CustomArray):
    return x.custom_method()
  else:
    # do something else
    ...

result1 = f(x)
result2 = jax.jit(f)(x)
```
If `jit` consumes `x` via pytree flattening, the results should agree for a well-specified
flattening rule. If `jit` consumes `x` via `__jax_array__`, the results will differ because
`x` is no longer a CustomArray within the JIT-compiled version of the function.

#### Synopsis
As of JAX v0.6.0, transformations prioritize `__jax_array__` when it is available. This status
quo can lead to confusion around lack of JIT invariance, and the current implementation in practice
leads to subtle bugs in the case of automatic differentiation, where the forward and backward pass
do not treat inputs consistently.

Because the pytree extensibility mechanism already exists for the case of customizing
transformations, it seems most straightforward if transformations act only via this
mechanism: that is, **we propose to remove `__jax_array__` parsing during abstractification.**
This approach will preserve object identity through transformations, and give the user the
most possible flexibility. If the user wants to opt-in to array conversion semantics, that
is always possible by explicitly casting their input via jnp.asarray, which will trigger the 
`__jax_array__` protocol.

### Which APIs should support `__jax_array__`?
JAX has a number of different levels of API, from the level of explicit primitive binding
(e.g. `jax.lax.add_p.bind(x, y)`) to the `jax.lax` APIs (e.g. `jax.lax.add(x, y)`) to the
`jax.numpy` APIs (e.g. `jax.numpy.add(x, y)`). Which of these API categories should handle
implicit conversion via `__jax_array__`?

In order to limit the scope of the change and the required testing, I propose that `__jax_array__`
only be explicitly supported in `jax.numpy` APIs: after all, it is inspired by the` __array__`
protocol which is supported by the NumPy package. We could always expand this in the future to
`jax.lax` APIs if needed.

This is in line with the current state of the package, where `__jax_array__` handling is mainly
within the input validation utilities used by `jax.numpy` APIs.

## Implementation
With these design choices in mind, we plan to implement this as follows:

- **Adding runtime support to `jax.numpy`**: This is likely the easiest part, as most
  `jax.numpy` functions use a common internal utility (`ensure_arraylike`) to validate
  inputs and convert them to array. This utility already supports `__jax_array__`, and
  so most jax.numpy APIs are already compliant.
- **Adding test coverage**:  To ensure compliance across the APIs, we should add a new
  test scaffold that calls every `jax.numpy` API with custom inputs and validates correct
  behavior.
- **Deprecating `__jax_array__` during abstractification**: Currently JAX's abstractification
  pass, used in `jit` and other transformations, does parse the `__jax_array__` protocol,
  and this is not the behavior we want long-term. We need to deprecate this behavior, and
  ensure that downstream packages that rely on it can move toward pytree registration or
  explicit array conversion where necessary.
- **Adding type annotations**: the type interface for jax.numpy functions is in
  `jax/numpy/__init__.pyi`, and we’ll need to change each input type from `ArrayLike` to
  `ArrayLike | SupportsJAXArray`, where the latter is a protocol with a `__jax_array__`
  method. We cannot add this directly to the `ArrayLike` definition, because `ArrayLike`
  is used in contexts where `__jax_array__` should not be supported.
- **Documentation**: once the above support is added, we should add a documentation section
  on array extensibility that outlines exactly what to expect regarding the `__jax_array__`
  protocol, with examples of how it can be used in conjunction with pytree registration
  in order to effectively work with user-defined types.
