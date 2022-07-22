# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import difflib
import functools
from functools import partial
import operator as op
from typing import (Any, Callable, Hashable, Iterable, Optional, Tuple, List,
                    Dict, Type, TypeVar, overload, TYPE_CHECKING, NamedTuple)
import textwrap
import warnings

from jax._src.lib import pytree
from jax._src.lib import xla_extension
from jax._src.lib import xla_extension_version

from jax._src.util import safe_zip, unzip2

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

T = TypeVar("T")
U = TypeVar("U")

if TYPE_CHECKING or xla_extension_version >= 78:
  PyTreeDef = pytree.PyTreeDef
else:
  PyTreeDef = xla_extension.PyTreeDef  # pytype: disable=module-attr


def tree_flatten(tree, is_leaf: Optional[Callable[[Any], bool]] = None):
  """Flattens a pytree.

  Args:
    tree: a pytree to flatten.
    is_leaf: an optionally specified function that will be called at each
      flattening step. It should return a boolean, with true stopping the
      traversal and the whole subtree being treated as a leaf, and false
      indicating the flattening should traverse the current object.
  Returns:
    A pair where the first element is a list of leaf values and the second
    element is a treedef representing the structure of the flattened tree.
  """
  return pytree.flatten(tree, is_leaf)


def tree_unflatten(treedef, leaves):
  """Reconstructs a pytree from the treedef and the leaves.

  The inverse of :func:`tree_flatten`.

  Args:
    treedef: the treedef to reconstruct
    leaves: the list of leaves to use for reconstruction. The list must match
      the leaves of the treedef.

  Returns:
    The reconstructed pytree, containing the ``leaves`` placed in the structure
    described by ``treedef``.
  """
  return treedef.unflatten(leaves)

def tree_leaves(tree, is_leaf: Optional[Callable[[Any], bool]] = None):
  """Gets the leaves of a pytree."""
  return pytree.flatten(tree, is_leaf)[0]

def tree_structure(tree, is_leaf: Optional[Callable[[Any], bool]] = None):
  """Gets the treedef for a pytree."""
  return pytree.flatten(tree, is_leaf)[1]

def treedef_tuple(treedefs):
  """Makes a tuple treedef from a list of child treedefs."""
  return pytree.tuple(list(treedefs))

def treedef_children(treedef):
  return treedef.children()

def treedef_is_leaf(treedef):
  return treedef.num_nodes == 1

def treedef_is_strict_leaf(treedef):
  return treedef.num_nodes == 1 and treedef.num_leaves == 1

def all_leaves(iterable, is_leaf: Optional[Callable[[Any], bool]] = None):
  """Tests whether all elements in the given iterable are all leaves.

  >>> tree = {"a": [1, 2, 3]}
  >>> assert all_leaves(jax.tree_util.tree_leaves(tree))
  >>> assert not all_leaves([tree])

  This function is useful in advanced cases, for example if a library allows
  arbitrary map operations on a flat list of leaves it may want to check if
  the result is still a flat list of leaves.

  Args:
    iterable: Iterable of leaves.

  Returns:
    A boolean indicating if all elements in the input are leaves.
  """
  if is_leaf is None:
    return pytree.all_leaves(iterable)
  else:
    lst = list(iterable)
    return lst == tree_leaves(lst, is_leaf)


_Children = TypeVar("_Children", bound=Iterable[Any])
_AuxData = TypeVar("_AuxData", bound=Hashable)

def register_pytree_node(nodetype: Type[T],
                         flatten_func: Callable[[T], Tuple[_Children, _AuxData]],
                         unflatten_func: Callable[[_AuxData, _Children], T]):
  """Extends the set of types that are considered internal nodes in pytrees.

  See `example usage <pytrees.html>`_.

  Args:
    nodetype: a Python type to treat as an internal pytree node.
    flatten_func: a function to be used during flattening, taking a value of
      type ``nodetype`` and returning a pair, with (1) an iterable for the
      children to be flattened recursively, and (2) some hashable auxiliary
      data to be stored in the treedef and to be passed to the
      ``unflatten_func``.
    unflatten_func: a function taking two arguments: the auxiliary data that was
      returned by ``flatten_func`` and stored in the treedef, and the
      unflattened children. The function should return an instance of
      ``nodetype``.
  """
  pytree.register_node(nodetype, flatten_func, unflatten_func)
  _registry[nodetype] = _RegistryEntry(flatten_func, unflatten_func)

def register_pytree_node_class(cls):
  """Extends the set of types that are considered internal nodes in pytrees.

  This function is a thin wrapper around ``register_pytree_node``, and provides
  a class-oriented interface::

    @register_pytree_node_class
    class Special:
      def __init__(self, x, y):
        self.x = x
        self.y = y
      def tree_flatten(self):
        return ((self.x, self.y), None)
      @classmethod
      def tree_unflatten(cls, aux_data, children):
        return cls(*children)
  """
  register_pytree_node(cls, op.methodcaller('tree_flatten'), cls.tree_unflatten)
  return cls

def tree_map(f: Callable[..., Any], tree: Any, *rest: Any,
             is_leaf: Optional[Callable[[Any], bool]] = None) -> Any:
  """Maps a multi-input function over pytree args to produce a new pytree.

  Args:
    f: function that takes ``1 + len(rest)`` arguments, to be applied at the
      corresponding leaves of the pytrees.
    tree: a pytree to be mapped over, with each leaf providing the first
      positional argument to ``f``.
    rest: a tuple of pytrees, each of which has the same structure as ``tree``
      or has ``tree`` as a prefix.
    is_leaf: an optionally specified function that will be called at each
      flattening step. It should return a boolean, which indicates whether
      the flattening should traverse the current object, or if it should be
      stopped immediately, with the whole subtree being treated as a leaf.

  Returns:
    A new pytree with the same structure as ``tree`` but with the value at each
    leaf given by ``f(x, *xs)`` where ``x`` is the value at the corresponding
    leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in
    ``rest``.

  Examples:

    >>> import jax.tree_util
    >>> jax.tree_util.tree_map(lambda x: x + 1, {"x": 7, "y": 42})
    {'x': 8, 'y': 43}

    If multiple inputs are passed, the structure of the tree is taken from the
    first input; subsequent inputs need only have ``tree`` as a prefix:

    >>> jax.tree_util.tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]
  """
  leaves, treedef = tree_flatten(tree, is_leaf)
  all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
  return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))

def tree_multimap(*args, **kwargs):
  """Deprecated alias of :func:`jax.tree_util.tree_map`"""
  warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
                'instead as a drop-in replacement.', FutureWarning)
  return tree_map(*args, **kwargs)

def build_tree(treedef, xs):
  return treedef.from_iterable_tree(xs)

def tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose):
  """Transform a tree having tree structure (outer, inner) into one having structure
  (inner, outer).
  """
  flat, treedef = tree_flatten(pytree_to_transpose)
  inner_size = inner_treedef.num_leaves
  outer_size = outer_treedef.num_leaves
  if treedef.num_leaves != (inner_size * outer_size):
    expected_treedef = outer_treedef.compose(inner_treedef)
    raise TypeError(f"Mismatch\n{treedef}\n != \n{expected_treedef}")
  flat = iter(flat)
  lol = [[next(flat) for _ in range(inner_size)] for __ in range(outer_size)]
  transposed_lol = zip(*lol)
  subtrees = map(partial(tree_unflatten, outer_treedef), transposed_lol)
  return tree_unflatten(inner_treedef, subtrees)

# TODO(mattjj): remove the Python-side registry when the C++-side registry is
# sufficiently queryable that we can express _replace_nones. That may mean once
# we have a flatten_one function.
_RegistryEntry = collections.namedtuple("_RegistryEntry", ["to_iter", "from_iter"])
_registry = {
    tuple: _RegistryEntry(lambda xs: (xs, None), lambda _, xs: tuple(xs)),
    list: _RegistryEntry(lambda xs: (xs, None), lambda _, xs: list(xs)),
    dict: _RegistryEntry(lambda xs: unzip2(sorted(xs.items()))[::-1],
                         lambda keys, xs: dict(zip(keys, xs))),
    type(None): _RegistryEntry(lambda z: ((), None), lambda _, xs: None),
}
def _replace_nones(sentinel, tree):
  """Replaces ``None`` in ``tree`` with ``sentinel``."""
  if tree is None:
    return sentinel
  else:
    handler = _registry.get(type(tree))
    if handler:
      children, metadata = handler.to_iter(tree)
      proc_children = [_replace_nones(sentinel, child) for child in children]
      return handler.from_iter(metadata, proc_children)
    elif isinstance(tree, tuple) and hasattr(tree, '_fields'):
      # handle namedtuple as a special case, based on heuristic
      children = iter(tree)
      proc_children = [_replace_nones(sentinel, child) for child in children]
      return type(tree)(*proc_children)
    else:
      return tree

no_initializer = object()

@overload
def tree_reduce(function: Callable[[T, Any], T],
                tree: Any) -> T:
    ...

@overload
def tree_reduce(function: Callable[[T, Any], T],
                tree: Any,
                initializer: T) -> T:
    ...

def tree_reduce(function: Callable[[T, Any], T],
                tree: Any,
                initializer: Any = no_initializer) -> T:
  if initializer is no_initializer:
    return functools.reduce(function, tree_leaves(tree))
  else:
    return functools.reduce(function, tree_leaves(tree), initializer)

def tree_all(tree):
  return all(tree_leaves(tree))

register_pytree_node(
  collections.OrderedDict,
  lambda x: (tuple(x.values()), tuple(x.keys())),
  lambda keys, values: collections.OrderedDict(safe_zip(keys, values)))

register_pytree_node(
  collections.defaultdict,
  lambda x: (tuple(x.values()), (x.default_factory, tuple(x.keys()))),
  lambda s, values: collections.defaultdict(s[0], safe_zip(s[1], values)))  # type: ignore[index]



class _HashableCallableShim:
  """Object that delegates __call__, __hash__, and __eq__ to another object."""
  def __init__(self, fun):
    self.fun = fun

  def __call__(self, *args, **kw):
    return self.fun(*args, **kw)

  def __hash__(self):
    return hash(self.fun)

  def __eq__(self, other):
    if isinstance(other, _HashableCallableShim):
      return self.fun == other.fun
    return self.fun == other


class Partial(functools.partial):
  """A version of functools.partial that works in pytrees.

  Use it for partial function evaluation in a way that is compatible with JAX's
  transformations, e.g., ``Partial(func, *args, **kwargs)``.

  (You need to explicitly opt-in to this behavior because we didn't want to give
  functools.partial different semantics than normal function closures.)

  For example, here is a basic usage of ``Partial`` in a manner similar to
  ``functools.partial``:

  >>> import jax.numpy as jnp
  >>> add_one = Partial(jnp.add, 1)
  >>> add_one(2)
  DeviceArray(3, dtype=int32, weak_type=True)

  Pytree compatibility means that the resulting partial function can be passed
  as an argument within transformed JAX functions, which is not possible with a
  standard ``functools.partial`` function:

  >>> from jax import jit
  >>> @jit
  ... def call_func(f, *args):
  ...   return f(*args)
  ...
  >>> call_func(add_one, 2)
  DeviceArray(3, dtype=int32, weak_type=True)

  Passing zero arguments to ``Partial`` effectively wraps the original function,
  making it a valid argument in JAX transformed functions:

  >>> call_func(Partial(jnp.add), 1, 2)
  DeviceArray(3, dtype=int32, weak_type=True)

  Had we passed ``jnp.add`` to ``call_func`` directly, it would have resulted in a
  ``TypeError``.

  Note that if the result of ``Partial`` is used in the context where the
  value is traced, it results in all bound arguments being traced when passed
  to the partially-evaluated function:

  >>> print_zero = Partial(print, 0)
  >>> print_zero()
  0
  >>> call_func(print_zero)
  Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
  """
  def __new__(klass, func, *args, **kw):
    # In Python 3.10+, if func is itself a functools.partial instance,
    # functools.partial.__new__ would merge the arguments of this Partial
    # instance with the arguments of the func. We box func in a class that does
    # not (yet) have a `func` attribute to defeat this optimization, since we
    # care exactly which arguments are considered part of the pytree.
    if isinstance(func, functools.partial):
      original_func = func
      func = _HashableCallableShim(original_func)
      out = super().__new__(klass, func, *args, **kw)
      func.func = original_func.func
      func.args = original_func.args
      func.keywords = original_func.keywords
      return out
    else:
      return super().__new__(klass, func, *args, **kw)


register_pytree_node(
    Partial,
    lambda partial_: ((partial_.args, partial_.keywords), partial_.func),
    lambda func, xs: Partial(func, *xs[0], **xs[1]),  # type: ignore[index]
)


def broadcast_prefix(prefix_tree: Any, full_tree: Any,
                     is_leaf: Optional[Callable[[Any], bool]] = None
                     ) -> List[Any]:
  # If prefix_tree is not a tree prefix of full_tree, this code can raise a
  # ValueError; use prefix_errors to find disagreements and raise more precise
  # error messages.
  result = []
  num_leaves = lambda t: tree_structure(t).num_leaves
  add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))
  tree_map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
  return result

def flatten_one_level(pytree: Any) -> Tuple[List[Any], Hashable]:
  handler = _registry.get(type(pytree))
  if handler:
    children, meta = handler.to_iter(pytree)
    return list(children), meta
  elif isinstance(pytree, tuple) and hasattr(pytree, '_fields'):
    return list(pytree), None
  else:
    raise ValueError(f"can't tree-flatten type: {type(pytree)}")

def prefix_errors(prefix_tree: Any, full_tree: Any,
                  is_leaf: Optional[Callable[[Any], bool]] = None,
                  ) -> List[Callable[[str], ValueError]]:
  return list(_prefix_error(KeyPath(()), prefix_tree, full_tree, is_leaf))

class KeyPathEntry(NamedTuple):
  key: Any
  def pprint(self) -> str:
    assert False  # must override

class KeyPath(NamedTuple):
  keys: Tuple[KeyPathEntry, ...]
  def __add__(self, other):
    if isinstance(other, KeyPathEntry):
      return KeyPath(self.keys + (other,))
    raise TypeError(type(other))
  def pprint(self) -> str:
    if not self.keys:
      return ' tree root'
    return ''.join(k.pprint() for k in self.keys)

class GetitemKeyPathEntry(KeyPathEntry):
  def pprint(self) -> str:
    return f'[{repr(self.key)}]'

class AttributeKeyPathEntry(KeyPathEntry):
  def pprint(self) -> str:
    return f'.{self.key}'

class FlattenedKeyPathEntry(KeyPathEntry):  # fallback
  def pprint(self) -> str:
    return f'[<flat index {self.key}>]'

def _child_keys(pytree: Any) -> List[KeyPathEntry]:
  assert not treedef_is_strict_leaf(tree_structure(pytree))
  handler = _keypath_registry.get(type(pytree))
  if handler:
    return handler(pytree)
  elif isinstance(pytree, tuple) and hasattr(pytree, '_fields'):
    # handle namedtuple as a special case, based on heuristic
    return [AttributeKeyPathEntry(s) for s in pytree._fields]
  else:
    num_children = len(treedef_children(tree_structure(pytree)))
    return [FlattenedKeyPathEntry(i) for i in range(num_children)]

_keypath_registry: Dict[Type, Callable[[Any], List[KeyPathEntry]]] = {}

def register_keypaths(ty: Type, handler: Callable[[Any], List[KeyPathEntry]]
                      ) -> None:
  _keypath_registry[ty] = handler

register_keypaths(tuple,
                  lambda tup: [GetitemKeyPathEntry(i) for i in range(len(tup))])
register_keypaths(list,
                  lambda lst: [GetitemKeyPathEntry(i) for i in range(len(lst))])
register_keypaths(dict,
                  lambda dct: [GetitemKeyPathEntry(k) for k in sorted(dct)])

def _prefix_error(key_path: KeyPath, prefix_tree: Any, full_tree: Any,
                  is_leaf: Optional[Callable[[Any], bool]] = None,
                  ) -> Iterable[Callable[[str], ValueError]]:
  # A leaf is a valid prefix of any tree:
  if treedef_is_strict_leaf(tree_structure(prefix_tree, is_leaf=is_leaf)): return

  # The subtrees may disagree because their roots are of different types:
  if type(prefix_tree) != type(full_tree):
    yield lambda name: ValueError(
      "pytree structure error: different types at key path\n"
      f"    {{name}}{key_path.pprint()}\n"
      f"At that key path, the prefix pytree {{name}} has a subtree of type\n"
      f"    {type(prefix_tree)}\n"
      f"but at the same key path the full pytree has a subtree of different type\n"
      f"    {type(full_tree)}.".format(name=name))
    return  # don't look for more errors in this subtree

  # Or they may disagree if their roots have different numbers of children (note
  # that because both prefix_tree and full_tree have the same type at this
  # point, and because prefix_tree is not a leaf, each can be flattened once):
  prefix_tree_children, prefix_tree_meta = flatten_one_level(prefix_tree)
  full_tree_children, full_tree_meta = flatten_one_level(full_tree)
  if len(prefix_tree_children) != len(full_tree_children):
    yield lambda name: ValueError(
      "pytree structure error: different numbers of pytree children at key path\n"
      f"    {{name}}{key_path.pprint()}\n"
      f"At that key path, the prefix pytree {{name}} has a subtree of type\n"
      f"    {type(prefix_tree)}\n"
      f"with {len(prefix_tree_children)} children, "
      f"but at the same key path the full pytree has a subtree of the same "
      f"type but with {len(full_tree_children)} children.".format(name=name))
    return  # don't look for more errors in this subtree

  # Or they may disagree if their roots have different pytree metadata:
  if prefix_tree_meta != full_tree_meta:
    prefix_tree_meta_str = str(prefix_tree_meta)
    full_tree_meta_str = str(full_tree_meta)
    metadata_diff = textwrap.indent(
        '\n'.join(difflib.ndiff(prefix_tree_meta_str.splitlines(),
                                full_tree_meta_str.splitlines())),
        prefix="    ")
    yield lambda name: ValueError(
      "pytree structure error: different pytree metadata at key path\n"
      f"    {{name}}{key_path.pprint()}\n"
      f"At that key path, the prefix pytree {{name}} has a subtree of type\n"
      f"    {type(prefix_tree)}\n"
      f"with metadata\n"
      f"    {prefix_tree_meta_str}\n"
      f"but at the same key path the full pytree has a subtree of the same "
      f"type but with metadata\n"
      f"    {full_tree_meta_str}\n"
      f"so the diff in the metadata at these pytree nodes is\n"
      f"{metadata_diff}".format(name=name))
    return  # don't look for more errors in this subtree

  # If the root types and numbers of children agree, there must be an error
  # in a subtree, so recurse:
  keys = _child_keys(prefix_tree)
  keys_ = _child_keys(full_tree)
  assert keys == keys_, \
    f"equal pytree nodes gave differing keys: {keys} and {keys_}"
  for k, t1, t2 in zip(keys, prefix_tree_children, full_tree_children):
    yield from _prefix_error(key_path + k, t1, t2)


# TODO(jakevdp) remove these deprecated wrappers & their imports in jax/__init__.py
def _deprecate(f):
  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    warnings.warn(f"jax.{f.__name__} is deprecated, and will be removed in a future release. "
                  f"Use jax.tree_util.{f.__name__} instead.",
                  category=FutureWarning, stacklevel=2)
    return f(*args, **kwargs)
  return wrapped

def __getattr__(name):
  prefix = "_deprecated_"
  if name.startswith(prefix):
    name = name[len(prefix):]
    return _deprecate(globals()[name])
  else:
    raise AttributeError(f"module {__name__} has no attribute {name!r}")
