.. _pallas:

Pallas: a JAX kernel language
=============================

Pallas is an extension to JAX for writing custom kernels for GPU and TPU.
Use it to achieve peak performance when XLA's automated optimizations
fall short—specifically when you need to:

- Write custom fusions that XLA cannot generate automatically.
- Take explicit control over memory movement and pipelining.
- Directly access low-level hardware features.

Both backends build on shared ideas (``Ref``\s, ``BlockSpec``\s, and pipelining)
but provide distinct, hardware-specific APIs.

.. note::
   Pallas is under active development. See the :ref:`pallas-changelog`
   for recent changes.

.. grid:: 2
   :gutter: 3
   :margin: 0
   :padding: 0

   .. grid-item-card:: Quickstart for GPU
      :link: pallas-gpu-quickstart
      :link-type: ref

      Custom kernels for NVIDIA Hopper (H100) and Blackwell (B200) GPUs.

   .. grid-item-card:: Quickstart for TPU
      :link: pallas-tpu-quickstart
      :link-type: ref

      Custom kernels for TensorCore and SparseCore on TPUs.

.. toctree::
   :caption: TPU backend guide
   :hidden:
   :maxdepth: 2

   tpu/index

.. toctree::
   :caption: Mosaic GPU backend guide
   :hidden:
   :maxdepth: 2

   gpu/index

.. toctree::
   :hidden:
   :caption: Guides

   quickstart
   grid_blockspec
   pipelining

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   API reference <../jax.experimental.pallas>

.. toctree::
   :caption: Design Notes
   :hidden:
   :maxdepth: 2

   design/index

.. toctree::
   :caption: Other
   :hidden:
   :maxdepth: 1

   CHANGELOG
