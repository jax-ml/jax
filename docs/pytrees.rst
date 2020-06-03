Pytrees
========

What is a pytree?
^^^^^^^^^^^^^^^^^

In JAX, a pytree is **a container of leaf elements and/or more pytrees**.
Containers include lists, tuples, and dicts (JAX can be extended to consider
other container types as pytrees, see `Extending pytrees`_ below). A leaf
element is anything that's not a pytree, e.g. an array. In other words, a pytree
is just **a possibly-nested standard or user-registered Python container**.  If
nested, note that the container types do not need to match. A single "leaf",
i.e. a non-container object, is also considered a pytree.

Example pytrees::

  [1, "a", object()] # 3 leaves

  (1, (2, 3), ()) # 3 leaves

  [1, {"k1": 2, "k2": (3, 4)}, 5] # 5 leaves

Pytrees and JAX functions
^^^^^^^^^^^^^^^^^^^^^^^^^

Many JAX functions, including all function transformations, operate over pytrees
of arrays (other leaf types are sometimes allowed as well). Transformations are
only applied to the leaf arrays while preserving the original pytree structure;
for example, ``vmap`` and ``pmap`` only map over arrays, but automatically map
over arrays inside of standard Python sequences, and can return mapped Python
sequences.

Applying optional parameters to pytrees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some JAX function transformations take optional parameters that specify how
certain input or output values should be treated (e.g. the ``in_axes`` and
``out_axes`` arguments to ``vmap``). These parameters are also pytrees, and the
leaf values are "matched up" with the corresponding input or output leaf arrays.
For example, if we pass the following input to vmap (note that the input
arguments to a function are considered a tuple)::

  (a1, {"k1": a2, "k2": a3})

We can use the following ``in_axes`` pytree to specify that only the "k2"
argument is mapped (axis=0) and the rest aren't mapped over (axis=None)::

  (None, {"k1": None, "k2": 0})

Note that the optional parameter pytree structure must match that of the main
input pytree. However, the optional parameters can optionally be specified as a
"prefix" pytree, meaning that a single leaf value can be applied to an entire
sub-pytree. For example, if we have the same ``vmap`` input as above, but wish
to only map over the dictionary argument, we can use::

  (None, 0)  # equivalent to (None, {"k1": 0, "k2": 0})

Or, if want every argument to be mapped, we can simply write a single leaf value
that is applied over the entire argument tuple pytree::

  0

This happens to be the default ``in_axes`` value!

The same logic applies to other optional parameters that refer to specific input
or output values of a transformed function, e.g. ``vmap``'s ``out_axes`` and
``pmaps``'s ``in_axes``.


Developer information
^^^^^^^^^^^^^^^^^^^^^^

*This is primarily JAX internal documentation, end-users are not supposed to need
to understand this to use JAX, except when registering new user-defined
container types with JAX. Some of these details may change.*

Internal pytree handling
------------------------

JAX canonicalizes pytrees into flat lists of numeric or array types at the
`api.py` boundary (and also in control flow primitives). This keeps downstream
JAX internals simpler: `vmap` etc. can handle user functions that accept and
return Python containers, while all the other parts of the system can operate on
functions that only take (multiple) array arguments and always return a flat
list of arrays.

When JAX flattens a pytree it will produce a list of leaves and a `treedef`
object that encodes the structure of the original value. The `treedef` can then
be used to construct a matching structured value after transforming the
leaves. Pytrees are tree-like, rather than DAG-like or graph-like, in that we
handle them assuming referential transparency and that they can't contain
reference cycles.

Here is a simple example::

  from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node
  from jax import numpy as np

  # The structured value to be transformed
  value_structured = [1., (2., 3.)]

  # The leaves in value_flat correspond to the `*` markers in value_tree
  value_flat, value_tree = tree_flatten(value_structured)
  print("value_flat={}\nvalue_tree={}".format(value_flat, value_tree))

  # Transform the flt value list using an element-wise numeric transformer
  transformed_flat = list(map(lambda v: v * 2., value_flat))
  print("transformed_flat={}".format(transformed_flat))

  # Reconstruct the structured output, using the original
  transformed_structured = tree_unflatten(value_tree, transformed_flat)
  print("transformed_structured={}".format(transformed_structured))

  # Output:
  # value_flat=[1.0, 2.0, 3.0]
  # value_tree=PyTreeDef(list, [*,PyTreeDef(tuple, [*,*])])
  # transformed_flat=[2.0, 4.0, 6.0]
  # transformed_structured=[2.0, (4.0, 6.0)]

By default, Pytrees containers can be lists, tuples, dicts, namedtuple, None,
OrderedDict. Other types of values, including numeric and ndarray values, are
treated as leaves::

  from collections import namedtuple
  Point = namedtuple('Point', ['x', 'y'])

  example_containers = [
      (1., [2., 3.]),
      (1., {'b': 2., 'a': 3.}),
      1.,
      None,
      np.zeros(2),
      Point(1., 2.)
  ]
  def show_example(structured):
    flat, tree = tree_flatten(structured)
    unflattened = tree_unflatten(tree, flat)
    print("structured={}\n  flat={}\n  tree={}\n  unflattened={}".format(
        structured, flat, tree, unflattened))

  for structured in example_containers:
    show_example(structured)

  # Output:
  # structured=(1.0, [2.0, 3.0])
  #   flat=[1.0, 2.0, 3.0]
  #   tree=PyTreeDef(tuple, [*,PyTreeDef(list, [*,*])])
  #   unflattened=(1.0, [2.0, 3.0])
  # structured=(1.0, {'b': 2.0, 'a': 3.0})
  #   flat=[1.0, 3.0, 2.0]
  #   tree=PyTreeDef(tuple, [*,PyTreeDef(dict[['a', 'b']], [*,*])])
  #   unflattened=(1.0, {'a': 3.0, 'b': 2.0})
  # structured=1.0
  #   flat=[1.0]
  #   tree=*
  #   unflattened=1.0
  # structured=None
  #   flat=[]
  #   tree=PyTreeDef(None, [])
  #   unflattened=None
  # structured=[0. 0.]
  #   flat=[DeviceArray([0., 0.], dtype=float32)]
  #   tree=*
  #   unflattened=[0. 0.]
  # structured=Point(x=1.0, y=2.0)
  #   flat=[1.0, 2.0]
  #   tree=PyTreeDef(namedtuple[<class '__main__.Point'>], [*,*])
  #   unflattened=Point(x=1.0, y=2.0)

Extending pytrees
-----------------

By default, any part of a structured value that is not recognized as an internal
pytree node is treated as a leaf (and such containers could not be passed to
JAX-traceable functions)::

  class Special(object):
    def __init__(self, x, y):
      self.x = x
      self.y = y

    def __repr__(self):
      return "Special(x={}, y={})".format(self.x, self.y)


  show_example(Special(1., 2.))

  # Output:
  # structured=Special(x=1.0, y=2.0)
  #   flat=[Special(x=1.0, y=2.0)]
  #   tree=*
  #   unflattened=Special(x=1.0, y=2.0)

The set of Python types that are considered internal pytree nodes is extensible,
through a global registry of types. Values of registered types are traversed
recursively::

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

  # Output:
  # structured=RegisteredSpecial(x=1.0, y=2.0)
  #   flat=[1.0, 2.0]
  #   tree=PyTreeDef(<class '__main__.RegisteredSpecial'>[None], [*,*])
  #   unflattened=RegisteredSpecial(x=1.0, y=2.0)

JAX needs sometimes to compare treedef for equality. Therefore care must be
taken to ensure that the auxiliary data specified in the flattening recipe
supports a meaningful equality comparison.

The whole set of functions for operating on pytrees are in `tree_util module
<https://jax.readthedocs.io/en/latest/jax.tree_util.html>`_.
