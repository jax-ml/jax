``jax.experimental.pallas.fuser`` module
========================================

.. automodule:: jax.experimental.pallas.fuser

Classes
-------

.. autosummary::
   :toctree: _autosummary

   BlockIndexTransform
   Fusion

Functions
---------

.. autosummary::
   :toctree: _autosummary

   custom_fusion
   evaluate
   fuse
   fusible

Block specs
-----------

.. autosummary::
   :toctree: _autosummary

   block_spec_with_prefetch
   filter_no_block_specs
   make_scalar_prefetch_handler
   pull_block_spec
   push_block_spec
   push_pull_block_spec

Fusion values
-------------

.. autosummary::
   :toctree: _autosummary

   get_fusion_values
   get_stateful_input_fusion_values
   get_stateful_output_fusion_values
