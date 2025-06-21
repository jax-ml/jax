``jax.experimental.sparse`` module
==================================

.. note::

   The methods in ``jax.experimental.sparse`` are experimental reference
   implementations, and not recommended for use in performance-critical
   applications.

.. automodule:: jax.experimental.sparse

.. currentmodule:: jax.experimental.sparse

Sparse API Reference
--------------------
.. autosummary::
   :toctree: _autosummary

   sparsify
   grad
   value_and_grad
   empty
   eye
   todense
   random_bcoo
   JAXSparse


BCOO Data Structure
~~~~~~~~~~~~~~~~~~~
:class:`BCOO` is the *Batched COO format*, and is the main sparse data structure
implemented in :mod:`jax.experimental.sparse`. 
Its operations are compatible with JAX's core transformations, including batching
(e.g. :func:`jax.vmap`) and autodiff (e.g. :func:`jax.grad`).

.. autosummary::
   :toctree: _autosummary

   BCOO
   bcoo_broadcast_in_dim
   bcoo_concatenate
   bcoo_dot_general
   bcoo_dot_general_sampled
   bcoo_dynamic_slice
   bcoo_extract
   bcoo_fromdense
   bcoo_gather
   bcoo_multiply_dense
   bcoo_multiply_sparse
   bcoo_update_layout
   bcoo_reduce_sum
   bcoo_reshape
   bcoo_slice
   bcoo_sort_indices
   bcoo_squeeze
   bcoo_sum_duplicates
   bcoo_todense
   bcoo_transpose


BCSR Data Structure
~~~~~~~~~~~~~~~~~~~
:class:`BCSR` is the *Batched Compressed Sparse Row* format, and is under development.
Its operations are compatible with JAX's core transformations, including batching
(e.g. :func:`jax.vmap`) and autodiff (e.g. :func:`jax.grad`).

.. autosummary::
   :toctree: _autosummary

   BCSR
   bcsr_dot_general
   bcsr_extract
   bcsr_fromdense
   bcsr_todense


Other Sparse Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Other sparse data structures include :class:`COO`, :class:`CSR`, and :class:`CSC`. These are
reference implementations of simple sparse structures with a few core operations implemented.
Their operations are generally compatible with autodiff transformations such as :func:`jax.grad`,
but not with batching transforms like :func:`jax.vmap`.

.. autosummary::
   :toctree: _autosummary

   COO
   CSC
   CSR
   coo_fromdense
   coo_matmat
   coo_matvec
   coo_todense
   csr_fromdense
   csr_matmat
   csr_matvec
   csr_todense

``jax.experimental.sparse.linalg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jax.experimental.sparse.linalg

.. currentmodule:: jax.experimental.sparse.linalg

.. autosummary::
   :toctree: _autosummary

   spsolve
   lobpcg_standard