.. _pallas-tpu-tutorial:

TPU Tutorial
============

This multi-chapter tutorial walks through writing custom TPU kernels in Pallas
from the ground up using :func:`~jax.experimental.pallas.kernel` (``pl.kernel``)
and ``pltpu.emit_pipeline``. Starting from TPU hardware fundamentals, the
chapters build progressively toward pipelined matrix multiplication, in-place
mutation, dynamic SMEM indexing, manual asynchronous DMA, and SparseCore
programming.

.. note::
   Looking for a 5-minute overview first? Start with the
   :ref:`TPU Quickstart <pallas-tpu-quickstart>`. For a compact lookup reference
   of supported operations, vector layouts, and hardware specifications, see
   :doc:`Writing TPU kernels with Pallas <../details>` and the
   :ref:`TPU Hardware Reference <pallas_tpu_hardware_reference>`.

.. toctree::
   :caption: Part 1: The Basics
   :maxdepth: 2

   01_tpus

.. toctree::
   :caption: Part 3: Advanced Features
   :maxdepth: 2

   08_async_refs
