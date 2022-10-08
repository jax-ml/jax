---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  name: python
  file_extension: .py
---

(argument_annotations)=

# Argument Annotations

Many transformations in JAX require providing some additional metadata about
the transformed functions input arguments.
We refer to this metadata as *argument annotations*, which are generally
supplied using `*_argnums` and `*_argnames` arguments to the transformation.

## Example - {func}`~jax.jit` and static arguments

```{code-cell}
import jax

def fun(a, /, b, *, c, **kwargs):
    # fun demonstrates different argument types
    # a is positional only (a=... not allowed)
    # b is positional or keyword
    # c is keyword only
    # kwargs is variable keywords
    # *args is also possible, though not combined with keyword only arguments
    ...
```

Say we wish to JIT compile `fun`, but mark `b` as a _static_ argument.
This can be done by annotating `b` as static using either:

```{code-cell}
fun_jit = jax.jit(fun, static_argnums=(1,))
# or
fun_jit = jax.jit(fun, static_argnames=("b",))
```

## `argnums` mechanism

- Negative `argnums` are allowed similar to negative slice indices in Python
- Variable positional arguments through `*args` are allowed

## `argnames` mechanism

- Variable keyword arguments through `**kwargs` are allowed

## Validation

JAX does its best to validate argument annotations such that typos fail loudly.
However, as validation is done at the time of transformation, some validations
cannot be made.

For example:
```{code-cell}
:tags: ["raises-exception"]
def fun(a, b, c, *args):
  ...

jitted = jax.jit(fun, static_argnums=(2, -7))
jitted(1, 2, 3, 4, 5)  # Too few argnums to find -7th arg.
# In this case c will be static and -7th static argument annotation is ignored.
```

In those rare cases, which always involve variable arguments (`*args, **kwargs`),
invalid annotations are ignored.

## Supported functions

Getting consistent annotation behaviour across all JAX transformations is
currently work in progress (tracked in: [#10614](https://github.com/google/jax/issues/10614)).
