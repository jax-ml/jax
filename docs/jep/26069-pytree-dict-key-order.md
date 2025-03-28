(jep-pytree-key-order)=
# JEP 26069: PyTree preservation of dict key order

A frequent JAX feature request/bug report (e.g. {jax-issue}`#4085`, {jax-issue}`#5948`,
{jax-issue}`#8419`, {jax-issue}`#11817`, {jax-issue}`#24398`, {jax-issue}`#)
concerns the fact that JAX {ref}`PyTrees <pytrees>` do not preserve dictionary key
order; for example:
```python
>>> dct = {'b': 1, 'c': 2, 'a': 3}
>>> print(list(dct.keys()))
['b', 'c', 'a']

>>> import jax
>>> leaves, tree = jax.tree.flatten(dct)
>>> dct2 = jax.tree.unflatten(tree, leaves)
>>> print(list(dct2.keys()))
['a', 'b', 'c']
```
The keys of the unflattened dict are sorted, and this fact sometimes
surprises users.

This JEP explores options for modifying {mod}`jax.tree_util` in order
to preserve dict key order.

## Why sort keys at all?

The root of this behavior is the fact that `tree_flatten` sorts dict keys:
```python
>>> import jax
>>> dct = {'b': 'b_leaf', 'c': 'c_leaf', 'a': 'a_leaf'}
>>> leaves, treedef = jax.tree.flatten(dct)
>>> print(leaves)
['a_leaf', 'b_leaf', 'c_leaf']
```
Prior to Python 3.7, Python dicts did not guarantee preservation of insertion
order in standard dicts, and so when JAX was first implemented this sorting was
necessary to ensure predictable and repeatable runtime behavior.

In the era of order-preserving dicts, key sorting still provides a benefit in
JAX; namely it means the JIT cache is insensitive to the insertion order of
its inputs.
```python
>>> @jax.jit
... def f(x):
...   print('tracing')
...   return x['a'] + x['b']
...
>>> dct1 = {'a': 1, 'b': 2}
>>> print("result:", f(dct1))  # initial trace
tracing
result: 3

>>> dct2 = {'b': 1, 'a': 2}
>>> print("result:", f(dct1))  # hits cache: no trace
result: 3

```
Reordering dict keys doesn't lead to a cache miss, and this is a very good
thing, becuase it avoids unexpected recompilations.

## Possible approaches for preserving key order

### 1. Preserve order when flattening
The simplest option here may be to preserve order when flattening, meaning
`{'a': 1, 'b': 2}` would lower to `[1, 2]`, while `{'b': 2, 'a': 1}` would
lower to `[2, 1]`. The biggest downside of this approach would be that
equivalent dicts passed to a function could break the JIT cache, leading to
extra compilations in some cases. This is an important enough downside that
this is a non-option.

### 2. Restore order when unflattening
Because flattening to a sorted order is important for caching of compiled
computations, perhaps instead we could store the key order in the `PyTreeDef`,
and use that to restore the original order when unflattening.
Unfortunately, this would still lead to cache misses: if we store the key
permutation in the `PyTreeDef`, then different dict orders would lead to
different `PyTreeDef` structures, and the hash of the `PyTreeDef` affects
the JIT cache. This is a non-option for the same reason as option 1.

### 3. Restore order when unflattening & specialize the hash of PyTreeDef
Building on the weakness of option 2, perhaps we could flatten in sorted
order, store the permutation in the `PyTreeDef`, but *ignore that permutation*
when computing the hash of the `PyTreeDef` and the equality between two
`PyTreeDef` objects. This sounds promising, but unfortunately would lead
to context-dependent execution when equivalent dicts with different orders
are passed to a JIT-compiled function. For example, consider this function:
```python
@jax.jit
def f(x):
  return list(x)[0] == 'a'

x = {'a': 1, 'b': 1}
y = {'b': 1, 'a': 1}
```
Under this approach, `f(y)` would return `False` if it were the first call to `f`,
and would return `True` if `f(x)` had been called previously. This is a contrived
example, but with this approach, any code that depends on key order when wrapped
by `jit` or another transformation would have similar characteristics.
Such context-dependent "action at a distance" can lead to very tricky bugs in user
code, and so this approach is a non-option.

## Possible alternatives: Preserving key order in some contexts
We see that due to corner cases in JIT caching, there is no suitable avenue toward
maintaining key order during flattening or restoring key order during unflattening.
But not all hope is lost: perhaps we could keep the existing behavior in calls to
{func}`jax.tree.flatten` used by `jit` and other transforms, but modify the approach
when used in other functions, like {func}`jax.tree.map`.

### 1. Add a `sort_dict_keys` parameter to `tree.flatten`
One option would be to add an extra `sort_dict_keys` parameter to `tree.flatten`
that could be set to `False` if the user wants unsorted keys. In flattening at
function boundaries it would be set to `True` (to not adversely affect the cache),
but in routines like `tree.map`, it could be set to `False` to preserve key order.
The only downside here is added complexity in the `tree.flatten` API.

### 2. Add an `unflatten_as` function
Another option would be to follow the approach of the
[`dm-tree`](https://tree.readthedocs.io/) package and define a function similar to
[`unflatten_as`](https://tree.readthedocs.io/en/latest/api.html#tree.unflatten_as),
in which the first argument is not a `PyTreeDef` but rather the actual Python
structure to be filled; by convention the key order of this structure is applied
to the key order of the output. This `unflatten_as` function could be used to
implement a general `tree.map` function that preserves dict key order.
The only downside here is in added complexity of the implementation. Additionally,
if this were adopted, it's likely that we'd eventually need a mechanism to register
`unflatten_as` handlers for user-defined types.

## Evaluation
It appears that there is no way to preserve key order in the general JAX
transformations without having an adverse impact on caching and/or debuggability.
We could preserve key order in the special case of `tree.map` with a bit of effort;
next steps would be to explore alternatives 1 and 2.
