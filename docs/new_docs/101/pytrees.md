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

(jax-101-pytrees)=
# Pytrees

<!--* freshness: { reviewed: '2026-07-09' } *-->

JAX functions and transformations fundamentally operate on arrays, but in
practice programs pass around richer structures: a neural network's parameters
might live in a dictionary of arrays with meaningful names, a dataset might be
a list of dicts, and so on. JAX has built-in support for such nested
structures, which it calls **pytrees**, and every JAX transformation handles
them natively. This page explains the pytree abstraction, the utilities for
working with pytrees, and some common gotchas and patterns.

## What is a pytree?

A pytree is a container-like structure built out of container-like Python
objects. A pytree can include lists, tuples, and dicts, nested arbitrarily. A
*leaf* is anything that's not a container — such as an array, or a scalar. (A
single leaf on its own also counts as a pytree.)

Here are some example pytrees, using {func}`jax.tree.leaves` to extract the
flattened leaves from each:

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
  leaves = jax.tree.leaves(pytree)
  print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")
```

Which types count as containers is determined by the *pytree registry*, which
by default includes lists, tuples, and dicts (plus a few relatives like
`namedtuple` and `OrderedDict`). Any object whose type is not in the registry
is treated as a leaf. The registry can be extended with user-defined classes —
that's how libraries like Flax and Equinox make entire models into pytrees.
We'll see how to register your own in {ref}`jax-101-custom-pytrees` below.

Conceptually, any pytree can be split into two parts: its **leaves** (the
data) and its **treedef** (the structure). {func}`jax.tree.flatten` performs
the split, and {func}`jax.tree.unflatten` reassembles:

```{code-cell}
params = {'W': jnp.zeros((2, 3)), 'b': jnp.zeros(3)}

leaves, treedef = jax.tree.flatten(params)
print(leaves)
print(treedef)
print(jax.tree.unflatten(treedef, leaves))
```

This flatten/unflatten decomposition is exactly how JAX transformations
support pytrees: internally they operate on the flat list of arrays, then
reassemble your structure around the results.

## Common pytree functions

The pytree utilities live in {mod}`jax.tree` (with lower-level versions in
{mod}`jax.tree_util`). The one you'll use most is {func}`jax.tree.map`, which
works like Python's `map` but operates over entire pytrees:

```{code-cell}
list_of_lists = [
    [1, 2, 3],
    [1, 2],
    [1, 2, 3, 4]
]

jax.tree.map(lambda x: x * 2, list_of_lists)
```

{func}`jax.tree.map` also supports mapping a function over multiple pytrees at
once. The structures must match exactly — lists with the same lengths, dicts
with the same keys:

```{code-cell}
another_list_of_lists = list_of_lists
jax.tree.map(lambda x, y: x + y, list_of_lists, another_list_of_lists)
```

Other useful functions include {func}`jax.tree.reduce` for reductions over
leaves and {func}`jax.tree.structure` for extracting a treedef; see the
{mod}`jax.tree` documentation for the full set.

## Pytrees and JAX transformations

All JAX transformations accept functions whose inputs and outputs are pytrees
of arrays. Differentiating with respect to a dictionary of parameters just
works — and the gradient comes back as a dictionary with the same structure:

```{code-cell}
def loss(params, x):
  pred = jnp.dot(x, params['W']) + params['b']
  return jnp.sum(pred ** 2)

params = {'W': jnp.ones(3), 'b': 0.5}
x = jnp.array([[1., 2., 3.],
               [4., 5., 6.]])

jax.grad(loss)(params, x)
```

This is the pattern that makes JAX practical for machine learning: parameters
go in a pytree, `jax.grad` produces a matching pytree of gradients, and
`jax.tree.map` applies the update. Here's a complete example, training a small
multi-layer perceptron:

```{code-cell}
import numpy as np

def init_mlp_params(layer_widths):
  params = []
  for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
    params.append(
        dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
             biases=np.ones(shape=(n_out,)))
    )
  return params

params = init_mlp_params([1, 128, 128, 1])
```

We can use `jax.tree.map` to check the shapes of what we built:

```{code-cell}
jax.tree.map(lambda x: x.shape, params)
```

Then define the forward pass, the loss, and the update step:

```{code-cell}
def forward(params, x):
  *hidden, last = params
  for layer in hidden:
    x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
  return x @ last['weights'] + last['biases']

def loss_fn(params, x, y):
  return jnp.mean((forward(params, x) - y) ** 2)

LEARNING_RATE = 0.0001

def update(params, x, y):
  # `grads` is a pytree with the same structure as `params`.
  grads = jax.grad(loss_fn)(params, x, y)
  # The SGD update is one tree.map over the two matching pytrees.
  return jax.tree.map(
      lambda p, g: p - LEARNING_RATE * g, params, grads
  )
```

```{code-cell}
x = np.random.normal(size=(128, 1))
y = x ** 2

for _ in range(100):
  params = update(params, x, y)

print(loss_fn(params, x, y))
```

(In a real training loop you'd wrap `update` in `jax.jit` to make it fast; see
{ref}`jax-201-jit`.)

### Transformation parameters can be pytrees too

Some transformation parameters that refer to inputs — like `in_axes` and
`out_axes` for {func}`jax.vmap` — can themselves be pytrees, matched up
against the argument structure. For example, with a function whose second
argument is a dict:

```python
vmap(f, in_axes=(0, {"k1": 0, "k2": None}))
```

maps over the leading axis of the first argument and of `k1`, while
broadcasting `k2`. These parameter pytrees may also be *prefixes* of the
argument structure, in which case a single value applies to the whole subtree:

```python
vmap(f, in_axes=(0, 0))   # equivalent to (0, {"k1": 0, "k2": 0})
vmap(f, in_axes=0)        # equivalent to (0, {"k1": 0, "k2": 0}) as well
```

The single-leaf spec `in_axes=0` is the familiar default: map everything along
its leading axis.

## Explicit key paths

Each leaf in a pytree has a *key path*: the sequence of keys you'd follow to
reach it from the root. This is useful for debugging and for anything that
needs leaf names, like per-parameter logging. The key-path utilities live in
{mod}`jax.tree_util`:

```{code-cell}
import collections

ATuple = collections.namedtuple("ATuple", ('name',))

tree = [1, {'k1': 2, 'k2': (3, 4)}, ATuple('foo')]
flattened, _ = jax.tree_util.tree_flatten_with_path(tree)

for key_path, value in flattened:
  print(f'Value of tree{jax.tree_util.keystr(key_path)}: {value}')
```

{func}`jax.tree_util.tree_map_with_path` similarly works like
{func}`jax.tree.map` with the key path passed as an extra argument.

(jax-101-custom-pytrees)=
## Custom pytree nodes

By default, any type not in the pytree registry is treated as a leaf — even if
it's a container-like class holding arrays inside:

```{code-cell}
class Special:
  def __init__(self, x, y):
    self.x = x
    self.y = y

jax.tree.leaves([Special(0, 1), Special(2, 4)])
```

The two `Special` objects themselves are the leaves. So mapping over what you
*meant* to be the contents fails:

```{code-cell}
:tags: [raises-exception]

jax.tree.map(lambda x: x + 1, [Special(0, 1), Special(2, 4)])
```

To make your own class act as a container, register it with
{func}`jax.tree_util.register_pytree_node`, supplying a pair of functions: one
that *flattens* an instance into `(children, aux_data)`, and one that
*unflattens* those pieces back into an instance:

```{code-cell}
from jax.tree_util import register_pytree_node

class RegisteredSpecial(Special):
  def __repr__(self):
    return f"RegisteredSpecial(x={self.x}, y={self.y})"

def special_flatten(v):
  children = (v.x, v.y)  # the dynamic contents, traversed recursively
  aux_data = None        # static metadata, stored in the treedef
  return children, aux_data

def special_unflatten(aux_data, children):
  return RegisteredSpecial(*children)

register_pytree_node(RegisteredSpecial, special_flatten, special_unflatten)

jax.tree.map(lambda x: x + 1, [RegisteredSpecial(0, 1), RegisteredSpecial(2, 4)])
```

The division of labor matters: `children` should hold the *dynamic* values
(arrays and sub-pytrees), while `aux_data` holds any *static* metadata.
Auxiliary data becomes part of the treedef, which JAX compares and hashes (for
example, when deciding whether two pytrees have the same structure), so it
must support meaningful equality and hashing.

Once registered, your type works with everything pytrees work with —
including transformations. Here's `jax.grad` differentiating with respect to
a `RegisteredSpecial` input, returning a matching `RegisteredSpecial` of
gradients:

```{code-cell}
jax.grad(lambda s: s.x ** 2 + s.y)(RegisteredSpecial(3.0, 4.0))
```

Some standard Python containers come pre-registered. A `NamedTuple` subclass,
for example, works with no registration at all — but note that *every* field
becomes a child, including ones you may have meant as metadata:

```{code-cell}
from typing import NamedTuple, Any

class MyOtherContainer(NamedTuple):
  name: str
  a: Any
  b: Any

jax.tree.leaves([MyOtherContainer('Alice', 1, 2),
                 MyOtherContainer('Bob', 4, 5)])
```

The names `'Alice'` and `'Bob'` show up as leaves, which becomes a problem as
soon as a transformation tries to treat them as array data.

### Registering dataclasses

Unlike `NamedTuple` subclasses, classes decorated with `@dataclass` are *not*
automatically pytree nodes. But they're easy to register, with
{func}`jax.tree_util.register_dataclass` — and it fixes the metadata problem
above, too, by letting you say explicitly which fields are data and which are
static metadata:

```{code-cell}
from dataclasses import dataclass
import functools

@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['a', 'b'],
                   meta_fields=['name'])
@dataclass
class MyDataclassContainer:
  name: str
  a: Any
  b: Any

jax.tree.leaves([
    MyDataclassContainer('apple', 5.3, 1.2),
    MyDataclassContainer('banana', jnp.zeros(4), -1.0),
])
```

The `name` field doesn't appear among the leaves: as a `meta_field`, it's
carried in the treedef, like `aux_data` above (and so it must be hashable).
This distinction pays off again with `jax.jit`, where meta fields are
automatically treated as static arguments — see {ref}`jax-201-jit-static-arguments`.

One caution when writing custom pytree nodes: JAX transformations sometimes
build instances of your type with placeholder objects standing in for the
real contents, so `__init__` and your unflatten function should avoid input
validation or array conversion. For this and more (including a method-based
registration API, {func}`jax.tree_util.register_pytree_node_class`), see
{doc}`/custom_pytrees`.

## Common pytree gotchas

### Mistaking pytree nodes for leaves

Watch out for accidentally treating *nodes* as *leaves*. For example, an
array's `.shape` is a tuple — which is a pytree node, not a leaf:

```{code-cell}
a_tree = [jnp.zeros((2, 3)), jnp.zeros((3, 4))]

# Try to make another pytree with ones instead of zeros.
shapes = jax.tree.map(lambda x: x.shape, a_tree)
jax.tree.map(jnp.ones, shapes)
```

Instead of calling `jnp.ones` on `(2, 3)`, this called it on `2` and `3`
separately, because the tuples became part of the tree structure. The fix
depends on the goal: avoid the intermediate `tree.map`, or make the shape a
leaf by converting it to an array.

### `None` is an empty node, not a leaf

`jax.tree` functions treat `None` as the absence of a node — it has no leaves:

```{code-cell}
jax.tree.leaves([None, None, None])
```

To treat `None` values as leaves, use the `is_leaf` argument:

```{code-cell}
jax.tree.leaves([None, None, None], is_leaf=lambda x: x is None)
```

### Dictionary keys must be sortable

Dictionaries are flattened by *sorted* key order, so that pytree structure
depends only on the set of keys and not insertion order. That means mixing key
types with no ordering between them, like `int` and `str`, is an error:

```{code-cell}
:tags: [raises-exception]

jax.tree.map(lambda x: x + 1, {1: 7, "y": 42})
```

If you need unordered keys, `collections.OrderedDict` flattens in insertion
order without sorting, or you can register a custom node type.

## Common pytree patterns

### Transposing a list of trees into a tree of lists

To turn a list of trees into a tree of lists, the idiomatic trick is
`jax.tree.map` with a variadic function:

```{code-cell}
def tree_transpose(list_of_trees):
  """Converts a list of trees of identical structure into a single tree of lists."""
  return jax.tree.map(lambda *xs: list(xs), *list_of_trees)

# Convert a dataset from row-major to column-major.
episode_steps = [dict(t=1, obs=3), dict(t=2, obs=4)]
tree_transpose(episode_steps)
```

For more complex transposes, {func}`jax.tree.transpose` lets you specify the
inner and outer structure explicitly.

## Next steps

With arrays, transformations, and pytrees, you can express most pure
computations in JAX. Two pieces of the expressiveness story remain:
pseudorandom numbers ({ref}`jax-101-random`) and stateful computations
({ref}`jax-101-state`).
