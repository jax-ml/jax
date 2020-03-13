Change Log
==========

.. This is a comment.
   Remember to leave an empty line before the start of an itemized list,
   and to align the itemized text with the first line of an item.

These are the release notes for JAX.


jax 0.1.60 (unreleased)
-----------------------

.. PLEASE REMEMBER TO CHANGE THE '..master' WITH AN ACTUAL TAG in GITHUB LINK.

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.59...master>`_.
* New features:

  * :py:func:`jax.pmap` has ``static_broadcast_argnums`` argument which allows the user to
    specify arguments that should be treated as compile-time constants and
    should be broadcasted to all devices. It works analogously to
    ``static_argnums`` in :py:func:`jax.jit`.
  * Improved error messages for when tracers are mistakenly saved in global state.
  * Added :py:func:`jax.nn.one_hot` utility function.
* The minimum jaxlib version is now 0.1.40.

jaxlib 0.1.40 (March 4, 2020)
--------------------------------

* Adds experimental support in Jaxlib for TensorFlow profiler, which allows
  tracing of CPU and GPU computations from TensorBoard.
* Includes prototype support for multihost GPU computations that communicate via
  NCCL.
* Improves performance of NCCL collectives on GPU.
* Adds TopK, CustomCallWithoutLayout, CustomCallWithLayout, IGammaGradA and
  RandomGamma implementations.
* Supports device assignments known at XLA compilation time.

jax 0.1.59 (February 11, 2020)
------------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.58...jax-v0.1.59>`_.
* Breaking changes

  * The minimum jaxlib version is now 0.1.38.
  * Simplified :py:class:`Jaxpr` by removing the ``Jaxpr.freevars`` and
    ``Jaxpr.bound_subjaxprs``. The call primitives (``xla_call``, ``xla_pmap``,
    ``sharded_call``, and ``remat_call``) get a new parameter ``call_jaxpr`` with a
    fully-closed (no ``constvars``) JAXPR. Also, added a new field ``call_primitive``
    to primitives.
* New features:

  * Reverse-mode automatic differentiation (e.g. ``grad``) of ``lax.cond``, making it
    now differentiable in both modes (https://github.com/google/jax/pull/2091)
  * JAX now supports DLPack, which allows sharing CPU and GPU arrays in a
    zero-copy way with other libraries, such as PyTorch.
  * JAX GPU DeviceArrays now support ``__cuda_array_interface__``, which is another
    zero-copy protocol for sharing GPU arrays with other libraries such as CuPy
    and Numba.
  * JAX CPU device buffers now implement the Python buffer protocol, which allows
    zero-copy buffer sharing between JAX and NumPy.
  * Added JAX_SKIP_SLOW_TESTS environment variable to skip tests known as slow.

jaxlib 0.1.39 (February 11, 2020)
--------------------------------

* Updates XLA.


jaxlib 0.1.38 (January 29, 2020)
--------------------------------

* CUDA 9.0 is no longer supported.
* CUDA 10.2 wheels are now built by default.

jax 0.1.58 (January 28, 2020)
-----------------------------

* `GitHub commits <https://github.com/google/jax/compare/46014da21...jax-v0.1.58>`_.
* Breaking changes

  * JAX has dropped Python 2 support, because Python 2 reached its end of life on
    January 1, 2020. Please update to Python 3.5 or newer.
* New features

    * Forward-mode automatic differentiation (`jvp`) of while loop
      (https://github.com/google/jax/pull/1980)
    * New NumPy and SciPy functions:

      * :py:func:`jax.numpy.fft.fft2`
      * :py:func:`jax.numpy.fft.ifft2`
      * :py:func:`jax.numpy.fft.rfft`
      * :py:func:`jax.numpy.fft.irfft`
      * :py:func:`jax.numpy.fft.rfft2`
      * :py:func:`jax.numpy.fft.irfft2`
      * :py:func:`jax.numpy.fft.rfftn`
      * :py:func:`jax.numpy.fft.irfftn`
      * :py:func:`jax.numpy.fft.fftfreq`
      * :py:func:`jax.numpy.fft.rfftfreq`
      * :py:func:`jax.numpy.linalg.matrix_rank`
      * :py:func:`jax.numpy.linalg.matrix_power`
      * :py:func:`jax.scipy.special.betainc`
    * Batched Cholesky decomposition on GPU now uses a more efficient batched
      kernel.


Notable bug fixes
^^^^^^^^^^^^^^^^^

* With the Python 3 upgrade, JAX no longer depends on ``fastcache``, which should
  help with installation.
