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
# Custom pytree nodes

This section explains how in JAX you can extend the set of Python types that will be considered _internal nodes_ in pytrees (pytree nodes) by using {func}`jax.tree_util.register_pytree_node` with {func}`jax.tree.map`.

Why would you need this? In the previous examples, pytrees were shown as lists, tuples, and dicts, with everything else as pytree leaves. This is because if you define your own container class, it will be considered to be a pytree leaf unless you _register_ it with JAX. This is also the case even if your container class has trees inside it. For example:

```{code-cell}
import jax

class Special(object):
  def __init__(self, x, y):
    self.x = x
    self.y = y

jax.tree.leaves([
    Special(0, 1),
    Special(2, 4),
])
```

Accordingly, if you try to use a {func}`jax.tree.map` expecting the leaves to be elements inside the container, you will get an error:

```{code-cell}
:tags: [raises-exception]

jax.tree.map(lambda x: x + 1,
  [
    Special(0, 1),
    Special(2, 4)
  ])
```

As a solution, JAX allows to extend the set of types to be considered internal pytree nodes through a global registry of types. Additionally, the values of registered types are traversed recursively.

First, register a new type using {func}`jax.tree_util.register_pytree_node`:

```{code-cell}
from jax.tree_util import register_pytree_node

class RegisteredSpecial(Special):
  def __repr__(self):
    return "RegisteredSpecial(x={}, y={})".format(self.x, self.y)

def special_flatten(v):
  """Specifies a flattening recipe.

  Params:
    v: The value of the registered type to flatten.
  Returns:
    A pair of an iterable with the children to be flattened recursively,
    and some opaque auxiliary data to pass back to the unflattening recipe.
    The auxiliary data is stored in the treedef for use during unflattening.
    The auxiliary data could be used, for example, for dictionary keys.
  """
  children = (v.x, v.y)
  aux_data = None
  return (children, aux_data)

def special_unflatten(aux_data, children):
  """Specifies an unflattening recipe.

  Params:
    aux_data: The opaque data that was specified during flattening of the
      current tree definition.
    children: The unflattened children

  Returns:
    A reconstructed object of the registered type, using the specified
    children and auxiliary data.
  """
  return RegisteredSpecial(*children)

# Global registration
register_pytree_node(
    RegisteredSpecial,
    special_flatten,    # Instruct JAX what are the children nodes.
    special_unflatten   # Instruct JAX how to pack back into a `RegisteredSpecial`.
)
```

Now you can traverse the special container structure:

```{code-cell}
jax.tree.map(lambda x: x + 1,
  [
   RegisteredSpecial(0, 1),
   RegisteredSpecial(2, 4),
  ])
```

Alternatively, you can define appropriate `tree_flatten` and `tree_unflatten` methods
on your class and decorate it with {func}`~jax.tree_util.register_pytree_node_class`:

```{code-cell}
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class RegisteredSpecial2(Special):
  def __repr__(self):
    return "RegisteredSpecial2(x={}, y={})".format(self.x, self.y)

  def tree_flatten(self):
    children = (self.x, self.y)
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)


def show_example(structured):
  flat, tree = structured.tree_flatten()
  unflattened = RegisteredSpecial2.tree_unflatten(tree, flat)
  print(f"{structured=}\n  {flat=}\n  {tree=}\n  {unflattened=}")


show_example(RegisteredSpecial2(1., 2.))
```

Modern Python comes equipped with helpful tools to make defining containers easier. Some will work with JAX out-of-the-box, but others require more care.

For instance, a Python `NamedTuple` subclass doesn't need to be registered to be considered a pytree node type:

```{code-cell}
from typing import NamedTuple, Any

class MyOtherContainer(NamedTuple):
  name: str
  a: Any
  b: Any
  c: Any

# NamedTuple subclasses are handled as pytree nodes, so
# this will work out-of-the-box.
jax.tree.leaves([
    MyOtherContainer('Alice', 1, 2, 3),
    MyOtherContainer('Bob', 4, 5, 6)
])
```

Notice that the `name` field now appears as a leaf, because all tuple elements are children. This is what happens when you don't have to register the class the hard way.

When defining unflattening functions, in general `children` should contain all the
dynamic elements of the data structure (arrays, dynamic scalars, and pytrees), while
`aux_data` should contain all the static elements that will be rolled into the `treedef`
structure. JAX sometimes needs to compare `treedef` for equality, or compute its hash
for use in the JIT cache, and so care must be taken to ensure that the auxiliary data
specified in the flattening recipe supports meaningful hashing and equality comparisons.

Unlike `NamedTuple` subclasses, classes decorated with `@dataclass` are not automatically pytrees. However, they can be registered as pytrees using the {func}`jax.tree_util.register_dataclass` decorator:

```{code-cell}
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
import functools

@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['a', 'b', 'c'],
                   meta_fields=['name'])
@dataclass
class MyDataclassContainer(object):
  name: str
  a: Any
  b: Any
  c: Any

# MyDataclassContainer is now a pytree node.
jax.tree.leaves([
  MyDataclassContainer('apple', 5.3, 1.2, jnp.zeros([4])),
  MyDataclassContainer('banana', np.array([3, 4]), -1., 0.)
])
```

Notice that the `name` field does not appear as a leaf. This is because we included it in the `meta_fields` argument to {func}`jax.tree_util.register_dataclass`, indicating that it should be treated as metadata/auxiliary data, just like `aux_data` in `RegisteredSpecial` above. Now instances of `MyDataclassContainer` can be passed into JIT-ed functions, and `name` will be treated as static (see {ref}`jit-marking-arguments-as-static` for more information on static args):

```{code-cell}
@jax.jit
def f(x: MyDataclassContainer | MyOtherContainer):
  return x.a + x.b

# Works fine! `mdc.name` is static.
mdc = MyDataclassContainer('mdc', 1, 2, 3)
y = f(mdc)
```

Contrast this with `MyOtherContainer`, the `NamedTuple` subclass. Since the `name` field is a pytree leaf, JIT expects it to be convertible to {class}`jax.Array`, and the following raises an error:

```{code-cell}
:tags: [raises-exception]

moc = MyOtherContainer('moc', 1, 2, 3)
y = f(moc)
```

The whole set of functions for operating on pytrees are in {mod}`jax.tree_util`.

## Custom pytrees and initialization with unexpected values

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
  obj.a = children[0]
  return obj
```

## Internal pytree handling

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
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

# The structured value to be transformed
value_structured = [1., (2., 3.)]

# The leaves in value_flat correspond to the `*` markers in value_tree
value_flat, value_tree = tree_flatten(value_structured)
print(f"{value_flat=}\n{value_tree=}")

# Transform the flat value list using an element-wise numeric transformer
transformed_flat = list(map(lambda v: v * 2., value_flat))
print(f"{transformed_flat=}")

# Reconstruct the structured output, using the original
transformed_structured = tree_unflatten(value_tree, transformed_flat)
print(f"{transformed_structured=}")
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
  print(f"{structured=}\n  {flat=}\n  {tree=}\n  {unflattened=}")

for structured in example_containers:
  show_example(structured)
```
