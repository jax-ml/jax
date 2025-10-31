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

```{code-cell}
:tags: [remove-cell]

# This ensures that code cell tracebacks appearing below will be concise.
%xmode minimal
```

(pytrees)=
(working-with-pytrees)=
# Pytrees

<!--* freshness: { reviewed: '2025-10-15' } *-->

JAX has built-in support for objects that look like dictionaries (dicts) of arrays, or lists of lists of dicts, or other nested structures — in JAX these are called pytrees.
This section will explain how to use them, provide useful code examples, and point out common "gotchas" and patterns.

For an explanation of how to create custom pytrees, see {doc}`custom_pytrees`.

(pytrees-what-is-a-pytree)=
## What is a pytree?

A pytree is a container-like structure built out of container-like Python objects — “leaf” pytrees and/or more pytrees. A pytree can include lists, tuples, and dicts. A leaf is anything that’s not a pytree, such as an array, but a single leaf is also a pytree.

In the context of machine learning (ML), a pytree can contain:

- Model parameters
- Dataset entries
- Reinforcement learning agent observations

When working with datasets, you can often come across pytrees (such as lists of lists of dicts).

Below is an example of a simple pytree. In JAX, you can use {func}`jax.tree.leaves`, to extract the flattened leaves from the trees, as demonstrated here:

```{code-cell}
import jax
import jax.numpy as jnp

example_trees = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]

# Print how many leaves the pytrees have.
for pytree in example_trees:
  # This `jax.tree.leaves()` method extracts the flattened leaves from the pytrees.
  leaves = jax.tree.leaves(pytree)
  print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")
```

Any tree-like structure built out of container-like Python objects can be treated as a pytree in JAX.
Classes are considered container-like if they are in the pytree registry, which by default includes lists, tuples, and dicts. Any object whose type is *not* in the pytree container registry will be treated as a leaf node in the tree.

The pytree registry can be extended to include user-defined container classes by registering the class
with functions that specify how to flatten the tree; see {ref}`pytrees-custom-pytree-nodes` below.

(pytrees-common-pytree-functions)=
## Common pytree functions

JAX provides a number of utilities to operate over pytrees. These can be found in the {mod}`jax.tree_util` subpackage;
for convenience many of these have aliases in the {mod}`jax.tree` module.

### Common function: `jax.tree.map`

The most commonly used pytree function is {func}`jax.tree.map`. It works analogously to Python's native `map`, but transparently operates over entire pytrees.

Here's an example:

```{code-cell}
list_of_lists = [
    [1, 2, 3],
    [1, 2],
    [1, 2, 3, 4]
]

jax.tree.map(lambda x: x*2, list_of_lists)
```

{func}`jax.tree.map` also allows mapping a [N-ary](https://en.wikipedia.org/wiki/N-ary) function over multiple arguments. For example:

```{code-cell}
another_list_of_lists = list_of_lists
jax.tree.map(lambda x, y: x+y, list_of_lists, another_list_of_lists)
```

When using multiple arguments with {func}`jax.tree.map`, the structure of the inputs must exactly match. That is, lists must have the same number of elements, dicts must have the same keys, etc.

(pytrees-example-jax-tree-map-ml)=
### Example of `jax.tree.map` with ML model parameters

This example demonstrates how pytree operations can be useful when training a simple [multi-layer perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron).

Begin with defining the initial model parameters:

```{code-cell}
import numpy as np

def init_mlp_params(layer_widths):
  params = []
  for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
    params.append(
        dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
             biases=np.ones(shape=(n_out,))
            )
    )
  return params

params = init_mlp_params([1, 128, 128, 1])
```

Use {func}`jax.tree.map` to check the shapes of the initial parameters:

```{code-cell}
jax.tree.map(lambda x: x.shape, params)
```

Next, define the functions for training the MLP model:

```{code-cell}
# Define the forward pass.
def forward(params, x):
  *hidden, last = params
  for layer in hidden:
    x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
  return x @ last['weights'] + last['biases']

# Define the loss function.
def loss_fn(params, x, y):
  return jnp.mean((forward(params, x) - y) ** 2)

# Set the learning rate.
LEARNING_RATE = 0.0001

# Using the stochastic gradient descent, define the parameter update function.
# Apply `@jax.jit` for JIT compilation (speed).
@jax.jit
def update(params, x, y):
  # Calculate the gradients with `jax.grad`.
  grads = jax.grad(loss_fn)(params, x, y)
  # Note that `grads` is a pytree with the same structure as `params`.
  # `jax.grad` is one of many JAX functions that has
  # built-in support for pytrees.
  # This is useful - you can apply the SGD update using JAX pytree utilities.
  return jax.tree.map(
      lambda p, g: p - LEARNING_RATE * g, params, grads
  )
```

## Viewing the pytree definition of an object

To view the pytree definition of an arbitrary `object` for debugging purposes, you can use:

```{code-cell}
from jax.tree_util import tree_structure
print(tree_structure(object))
```

(pytree-and-jax-transformations)=
## Pytrees and JAX transformations

Many JAX functions, like {func}`jax.lax.scan`, operate over pytrees of arrays. In addition, all JAX function transformations can be applied to functions that accept as input and produce as output pytrees of arrays.

Some JAX function transformations take optional parameters that specify how certain input or output values should be treated (such as the `in_axes` and `out_axes` arguments to {func}`jax.vmap`). These parameters can also be pytrees, and their structure must correspond to the pytree structure of the corresponding arguments. In particular, to be able to “match up” leaves in these parameter pytrees with values in the argument pytrees, the parameter pytrees are often constrained to be tree prefixes of the argument pytrees.

For example, if you pass the following input to {func}`jax.vmap` (note that the input arguments to a function are considered a tuple):

```python
vmap(f, in_axes=(a1, {"k1": a2, "k2": a3}))
```

then you can use the following `in_axes` pytree to specify that only the `k2` argument is mapped (`axis=0`), and the rest aren’t mapped over (`axis=None`):

```python
vmap(f, in_axes=(None, {"k1": None, "k2": 0}))
```

The optional parameter pytree structure must match that of the main input pytree. However, the optional parameters can optionally be specified as a “prefix” pytree, meaning that a single leaf value can be applied to an entire sub-pytree.

For example, if you have the same {func}`jax.vmap` input as above, but wish to only map over the dictionary argument, you can use:

```python
vmap(f, in_axes=(None, 0))  # equivalent to (None, {"k1": 0, "k2": 0})
```

Alternatively, if you want every argument to be mapped, you can write a single leaf value that is applied over the entire argument tuple pytree:

```python
vmap(f, in_axes=0)  # equivalent to (0, {"k1": 0, "k2": 0})
```

This happens to be the default `in_axes` value for {func}`jax.vmap`.

The same logic applies to other optional parameters that refer to specific input or output values of a transformed function, such as `out_axes` in {func}`jax.vmap`.

(pytrees-explicity-key-paths)=
## Explicit key paths

In a pytree each leaf has a _key path_. A key path for a leaf is a `list` of _keys_, where the length of the list is equal to the depth of the leaf in the pytree . Each _key_ is a [hashable object](https://docs.python.org/3/glossary.html#term-hashable) that represents an index into the corresponding pytree node type. The type of the key depends on the pytree node type; for example, the type of keys for `dict`s is different from the type of keys for `tuple`s.

For built-in pytree node types, the set of keys for any pytree node instance is unique. For a pytree comprising nodes with this property, the key path for each leaf is unique.

JAX has the following `jax.tree_util.*` methods for working with key paths:

- {func}`jax.tree_util.tree_flatten_with_path`: Works similarly to {func}`jax.tree.flatten`, but returns key paths.
- {func}`jax.tree_util.tree_map_with_path`: Works similarly to {func}`jax.tree.map`, but the function also takes key paths as arguments.
- {func}`jax.tree_util.keystr`: Given a general key path, returns a reader-friendly string expression.

For example, one use case is to print debugging information related to a certain leaf value:

```{code-cell}
import collections

ATuple = collections.namedtuple("ATuple", ('name'))

tree = [1, {'k1': 2, 'k2': (3, 4)}, ATuple('foo')]
flattened, _ = jax.tree_util.tree_flatten_with_path(tree)

for key_path, value in flattened:
  print(f'Value of tree{jax.tree_util.keystr(key_path)}: {value}')
```

To express key paths, JAX provides a few default key types for the built-in pytree node types, namely:

*  `SequenceKey(idx: int)`: For lists and tuples.
*  `DictKey(key: Hashable)`: For dictionaries.
*  `GetAttrKey(name: str)`: For `namedtuple`s and preferably custom pytree nodes (more in the next section)

You are free to define your own key types for your custom nodes. They will work with {func}`jax.tree_util.keystr` as long as their `__str__()` method is also overridden with a reader-friendly expression.

```{code-cell}
for key_path, _ in flattened:
  print(f'Key path of tree{jax.tree_util.keystr(key_path)}: {repr(key_path)}')
```

(pytrees-common-pytree-gotchas)=
## Common pytree gotchas

This section covers some of the most common problems ("gotchas") encountered when using JAX pytrees.

### Mistaking pytree nodes for leaves

A common gotcha to look out for is accidentally introducing _tree nodes_ instead of _leaves_:

```{code-cell}
a_tree = [jnp.zeros((2, 3)), jnp.zeros((3, 4))]

# Try to make another pytree with ones instead of zeros.
shapes = jax.tree.map(lambda x: x.shape, a_tree)
jax.tree.map(jnp.ones, shapes)
```

What happened here is that the `shape` of an array is a tuple, which is a pytree node, with its elements as leaves. Thus, in the map, instead of calling `jnp.ones` on e.g. `(2, 3)`, it's called on `2` and `3`.

The solution will depend on the specifics, but there are two broadly applicable options:

- Rewrite the code to avoid the intermediate {func}`jax.tree.map`.
- Convert the tuple into a NumPy array (`np.array`) or a JAX NumPy array (`jnp.array`), which makes the entire sequence a leaf.

### Handling of `None` by `jax.tree_util`

`jax.tree_util` functions treat `None` as the absence of a pytree node, not as a leaf:

```{code-cell}
jax.tree.leaves([None, None, None])
```

To treat `None` as a leaf, you can use the `is_leaf` argument:

```{code-cell}
jax.tree.leaves([None, None, None], is_leaf=lambda x: x is None)
```

(pytrees-common-pytree-patterns)=
## Common pytree patterns

This section covers some of the most common patterns with JAX pytrees.

### Transposing pytrees with `jax.tree.map` and `jax.tree.transpose`

To transpose a pytree (turn a list of trees into a tree of lists), JAX has two functions: {func}`jax.tree.map` (more basic) and {func}`jax.tree.transpose` (more flexible, complex and verbose).

**Option 1:** Use {func}`jax.tree.map`. Here's an example:

```{code-cell}
def tree_transpose(list_of_trees):
  """
  Converts a list of trees of identical structure into a single tree of lists.
  """
  return jax.tree.map(lambda *xs: list(xs), *list_of_trees)

# Convert a dataset from row-major to column-major.
episode_steps = [dict(t=1, obs=3), dict(t=2, obs=4)]
tree_transpose(episode_steps)
```

**Option 2:** For more complex transposes, use {func}`jax.tree.transpose`, which is more verbose, but allows you specify the structure of the inner and outer pytree for more flexibility. For example:

```{code-cell}
jax.tree.transpose(
  outer_treedef = jax.tree.structure([0 for e in episode_steps]),
  inner_treedef = jax.tree.structure(episode_steps[0]),
  pytree_to_transpose = episode_steps
)
```
