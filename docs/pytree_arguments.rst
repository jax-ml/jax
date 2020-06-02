Pytree arguments to JAX functions
=================================

What is a pytree?
^^^^^^^^^^^^^^^^^

In JAX, a pytree is **a container of leaf elements and/or more pytrees**.
Containers include lists, tuples, and dicts (JAX can be extended to consider
other container types as pytrees, see the :doc:`notebooks/JAX_pytrees`
notebook). A leaf element is anything that's not a pytree, e.g. an array. In
other words, a pytree is just **a possibly-nested standard or user-registered
Python container**.  If nested, note that the container types do not need to
match. A single "leaf", i.e. a non-container object, is also considered a
pytree.

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
