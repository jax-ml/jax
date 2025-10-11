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

(pytrees-custom-pytree-nodes)=
# Custom pytrees and initialization with unexpected values

Another common gotcha with user-defined pytree objects is that JAX transformations occasionally initialize them with unexpected values, so that any input validation done at initialization may fail. For example:

```{code-cell}
:tags: [raises-exception]

class MyTree:
  def __init__(self, a):
    self.a = jnp.asarray(a)

register_pytree_node(MyTree, lambda tree: ((tree.a,), None),
    lambda _, args: MyTree(*args))

tree = MyTree(jnp.arange(5.0))

jax.vmap(lambda x: x)(tree)      # Error because object() is passed to `MyTree`.
```

```{code-cell}
:tags: [raises-exception]

jax.jacobian(lambda x: x)(tree)  # Error because MyTree(...) is passed to `MyTree`.
```

- In the first case with `jax.vmap(...)(tree)`, JAX’s internals use arrays of `object()` values to infer the structure of the tree
- In the second case with `jax.jacobian(...)(tree)`, the Jacobian of a function mapping a tree to a tree is defined as a tree of trees.

**Potential solution 1:**

- The `__init__` and `__new__` methods of custom pytree classes should generally avoid doing any array conversion or other input validation, or else anticipate and handle these special cases. For example:

```{code-cell}
class MyTree:
  def __init__(self, a):
    if not (type(a) is object or a is None or isinstance(a, MyTree)):
      a = jnp.asarray(a)
    self.a = a
```

**Potential solution 2:**

- Structure your custom `tree_unflatten` function so that it avoids calling `__init__`. If you choose this route, make sure that your `tree_unflatten` function stays in sync with `__init__` if and when the code is updated. Example:

```{code-cell}
def tree_unflatten(aux_data, children):
  del aux_data  # Unused in this class.
  obj = object.__new__(MyTree)
  obj.a = a
  return obj
```
