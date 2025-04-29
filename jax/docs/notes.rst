.. _notes:

Notes
-----
This section contains shorter notes on topics relevant to using JAX; see also the
longer design discussions in :doc:`jep/index`.

Dependencies and version compatibility:
  - :doc:`api_compatibility` outlines JAX's policies with regard to API compatibility across releases.
  - :doc:`deprecation` outlines JAX's policies with regard to compatibility with Python and NumPy.

Memory and computation usage:
  - :doc:`async_dispatch` describes JAX's asynchronous dispatch model.
  - :doc:`concurrency` describes how JAX interacts with other Python concurrency.
  - :doc:`gpu_memory_allocation` describes how JAX interacts with memory allocation on GPU.

Programmer guardrails:
  - :doc:`rank_promotion_warning` describes how to configure :mod:`jax.numpy` to avoid implicit rank promotion.

Arrays and data types:
  - :doc:`type_promotion` describes JAX's implicit type promotion for functions of two or more values.
  - :doc:`default_dtypes` describes how JAX determines the default dtype for array creation functions.


.. toctree::
   :hidden:
   :maxdepth: 1

   api_compatibility
   deprecation
   async_dispatch
   concurrency
   gpu_memory_allocation
   rank_promotion_warning
   type_promotion
   default_dtypes
