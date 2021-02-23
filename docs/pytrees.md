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

(pytrees)=

# Pytrees

## What is a pytree?

In JAX, we use the term *pytree* to refer to a tree-like structure built out of
container-like Python objects. Classes are considered container-like if they
are in the pytree registry, which by default includes lists, tuples, and dicts.
That is:

1. any object whose type is *not* in the pytree container registry is
   considered a *leaf* pytree;
2. any object whose type is in the pytree container registry, and which
   contains pytrees, is considered a pytree.

For each entry in the pytree container registry, a container-like type is
registered with a pair of functions which specify how to convert an instance of
the container type to a `(children, metadata)` pair and how to convert such a
pair back to an instance of the container type. Using these functions, JAX can
canonicalize any tree of registered container objects into tuples.

Example pytrees:

```
[1, "a", object()]  # 3 leaves

(1, (2, 3), ())  # 3 leaves

[1, {"k1": 2, "k2": (3, 4)}, 5]  # 5 leaves
```

JAX can be extended to consider other container types as pytrees; see
[Extending pytrees](extending pytrees) below.

## Pytrees and JAX functions

Many JAX functions, like {func}`jax.lax.scan`, operate over pytrees of arrays.
JAX function transformations can be applied to functions that accept as input
and produce as output pytrees of arrays.

## Applying optional parameters to pytrees

Some JAX function transformations take optional parameters that specify how
certain input or output values should be treated (e.g. the `in_axes` and
`out_axes` arguments to {func}`~jax.vmap`). These parameters can also be pytrees,
and their structure must correspond to the pytree structure of the corresponding
arguments. In particular, to be able to "match up" leaves in these parameter
pytrees with values in the argument pytrees, the parameter pytrees are often
constrained to be tree prefixes of the argument pytrees.

For example, if we pass the following input to {func}`~jax.vmap` (note that the input
arguments to a function considered a tuple):

```
(a1, {"k1": a2, "k2": a3})
```

We can use the following `in_axes` pytree to specify that only the `k2`
argument is mapped (`axis=0`) and the rest aren't mapped over
(`axis=None`):

```
(None, {"k1": None, "k2": 0})
```

The optional parameter pytree structure must match that of the main input
pytree. However, the optional parameters can optionally be specified as a
"prefix" pytree, meaning that a single leaf value can be applied to an entire
sub-pytree. For example, if we have the same {func}`~jax.vmap` input as above,
but wish to only map over the dictionary argument, we can use:

```
(None, 0)  # equivalent to (None, {"k1": 0, "k2": 0})
```

Or, if we want every argument to be mapped, we can simply write a single leaf
value that is applied over the entire argument tuple pytree:

```
0
```

This happens to be the default `in_axes` value for {func}`~jax.vmap`!

The same logic applies to other optional parameters that refer to specific input
or output values of a transformed function, e.g. `vmap`'s `out_axes`.

## Viewing the pytree definition of an object

To view the pytree definition of an arbitrary `object` for debugging purposes, you can use:

```
from jax.tree_util import tree_structure
print(tree_structure(object))
```

## Developer information

*This is primarily JAX internal documentation, end-users are not supposed to need
to understand this to use JAX, except when registering new user-defined
container types with JAX. Some of these details may change.*

### Internal pytree handling

JAX flattens pytrees into lists of leaves at the `api.py` boundary (and also
in control flow primitives). This keeps downstream JAX internals simpler:
transformations like {func}`~jax.grad`, {func}`~jax.jit`, and {func}`~jax.vmap`
can handle user functions that accept and return the myriad different Python
containers, while all the other parts of the system can operate on functions
that only take (multiple) array arguments and always return a flat list of arrays.

When JAX flattens a pytree it will produce a list of leaves and a `treedef`
object that encodes the structure of the original value. The `treedef` can
then be used to construct a matching structured value after transforming the
leaves. Pytrees are tree-like, rather than DAG-like or graph-like, in that we
handle them assuming referential transparency and that they can't contain
reference cycles.

Here is a simple example:

```{code-cell}
:tags: [remove-cell]

# Execute this to consume & hide the GPU warning.
import jax.numpy as _jnp
_jnp.arange(10)
```

```{code-cell}
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node
import jax.numpy as jnp

# The structured value to be transformed
value_structured = [1., (2., 3.)]

# The leaves in value_flat correspond to the `*` markers in value_tree
value_flat, value_tree = tree_flatten(value_structured)
print("value_flat={}\nvalue_tree={}".format(value_flat, value_tree))

# Transform the flat value list using an element-wise numeric transformer
transformed_flat = list(map(lambda v: v * 2., value_flat))
print("transformed_flat={}".format(transformed_flat))

# Reconstruct the structured output, using the original
transformed_structured = tree_unflatten(value_tree, transformed_flat)
print("transformed_structured={}".format(transformed_structured))
```

By default, pytree containers can be lists, tuples, dicts, namedtuple, None,
OrderedDict. Other types of values, including numeric and ndarray values, are
treated as leaves:

```{code-cell}
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])

example_containers = [
    (1., [2., 3.]),
    (1., {'b': 2., 'a': 3.}),
    1.,
    None,
    jnp.zeros(2),
    Point(1., 2.)
]
def show_example(structured):
  flat, tree = tree_flatten(structured)
  unflattened = tree_unflatten(tree, flat)
  print("structured={}\n  flat={}\n  tree={}\n  unflattened={}".format(
      structured, flat, tree, unflattened))

for structured in example_containers:
  show_example(structured)
```

### Extending pytrees

By default, any part of a structured value that is not recognized as an
internal pytree node (i.e. container-like) is treated as a leaf:

```{code-cell}
class Special(object):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __repr__(self):
    return "Special(x={}, y={})".format(self.x, self.y)


show_example(Special(1., 2.))
```

The set of Python types that are considered internal pytree nodes is extensible,
through a global registry of types. Values of registered types are traversed
recursively:

```{code-cell}
class RegisteredSpecial(Special):
  def __repr__(self):
    return "RegisteredSpecial(x={}, y={})".format(self.x, self.y)

def special_flatten(v):
  """Specifies a flattening recipe.

  Params:
    v: the value of registered type to flatten.
  Returns:
    a pair of an iterable with the children to be flattened recursively,
    and some opaque auxiliary data to pass back to the unflattening recipe.
    The auxiliary data is stored in the treedef for use during unflattening.
    The auxiliary data could be used, e.g., for dictionary keys.
  """
  children = (v.x, v.y)
  aux_data = None
  return (children, aux_data)

def special_unflatten(aux_data, children):
  """Specifies an unflattening recipe.

  Params:
    aux_data: the opaque data that was specified during flattening of the
      current treedef.
    children: the unflattened children

  Returns:
    a re-constructed object of the registered type, using the specified
    children and auxiliary data.
  """
  return RegisteredSpecial(*children)

# Global registration
register_pytree_node(
    RegisteredSpecial,
    special_flatten,    # tell JAX what are the children nodes
    special_unflatten   # tell JAX how to pack back into a RegisteredSpecial
)

show_example(RegisteredSpecial(1., 2.))
```

JAX needs sometimes to compare `treedef` for equality. Therefore, care must be
taken to ensure that the auxiliary data specified in the flattening recipe
supports a meaningful equality comparison.

The whole set of functions for operating on pytrees are in {mod}`jax.tree_util`.
