Change Log
==========

.. This is a comment.
   Remember to leave an empty line before the start of an itemized list,
   and to align the itemized text with the first line of an item.

.. PLEASE REMEMBER TO CHANGE THE '..master' WITH AN ACTUAL TAG in GITHUB LINK.

These are the release notes for JAX.

jax 0.1.73 (July 22, 2020)
--------------------------
* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.72...jax-v0.1.73>`_.
* The minimum jaxlib version is now 0.1.51.

* New Features:
  * jax.image.resize. (#3703)
  * hfft and ihfft (#3664)
  * jax.numpy.intersect1d (#3726)
  * jax.numpy.lexsort (#3812)

* Bug Fixes:
  * Fix reduction repeated axis error (#3618)
  * Fix shape rule for lax.pad for input dimensions of size 0. (#3608)
  * make psum transpose handle zero cotangents (#3653)
  * Fix shape error when taking JVP of reduce-prod over size 0 axis. (#3729)
  * Support differentiation through jax.lax.all_to_all (#3733)
  * address nan issue in jax.scipy.special.zeta (#3777)

* Improvements:
  * Many improvements to jax2tf
  * Reimplement argmin/argmax using a single pass variadic reduction. (#3611)
  * Enable XLA SPMD partitioning by default. (#3151)
  * Add support for 0d transpose convolution (#3643)
  * Make LU gradient work for low-rank matrices (#3610)
  * support multiple_results and custom JVPs in jet (#3657)
  * Generalize reduce-window padding to support (lo, hi) pairs. (#3728)
  * Implement complex convolutions on CPU and GPU. (#3735)
  * Make jnp.take work for empty slices of empty arrays. (#3751)
  * Relax dimension ordering rules for dot_general. (#3778)
  * Enable buffer donation for GPU. (#3800)
  * Add support for base dilation and window dilation to reduce window op… (#3803)


jaxlib 0.1.51 (July 2, 2020)
------------------------------

* Update XLA.
* Add new runtime support for host_callback.

jax 0.1.72 (June 28, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.71...jax-v0.1.72>`_.

* Bug fixes:

  * Fix an odeint bug introduced in the previous release, see
    `#3587 <https://github.com/google/jax/pull/3587>`_.


jax 0.1.71 (June 25, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.70...jax-v0.1.71>`_.
* The minimum jaxlib version is now 0.1.48.

* Bug fixes:

  * Allow ``jax.experimental.ode.odeint`` dynamics functions to close over
    values with respect to which we're differentiating
    `#3562 <https://github.com/google/jax/pull/3562>`_.

jaxlib 0.1.50 (June 25, 2020)
------------------------------

* Add support for CUDA 11.0.
* Drop support for CUDA 9.2 (we only maintain support for the last four CUDA
  versions.)
* Update XLA.

jaxlib 0.1.49 (June 19, 2020)
------------------------------

* Bug fixes:

  * Fix build issue that could result in slow compiles
    (https://github.com/tensorflow/tensorflow/commit/f805153a25b00d12072bd728e91bb1621bfcf1b1)

jaxlib 0.1.48 (June 12, 2020)
------------------------------

* New features:

  * Adds support for fast traceback collection.
  * Adds preliminary support for on-device heap profiling.
  * Implements ``np.nextafter`` for ``bfloat16`` types.
  * Complex128 support for FFTs on CPU and GPU.

* Bugfixes:

  * Improved float64 ``tanh`` accuracy on GPU.
  * float64 scatters on GPU are much faster.
  * Complex matrix multiplication on CPU should be much faster.
  * Stable sorts on CPU should actually be stable now.
  * Concurrency bug fix in CPU backend.


jax 0.1.70 (June 8, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.69...jax-v0.1.70>`_.

* New features:

  * ``lax.switch`` introduces indexed conditionals with multiple
    branches, together with a generalization of the ``cond``
    primitive
    `#3318 <https://github.com/google/jax/pull/3318>`_.

jax 0.1.69 (June 3, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.68...jax-v0.1.69>`_.

jax 0.1.68 (May 21, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.67...jax-v0.1.68>`_.

* New features:

  * `lax.cond` supports a single-operand form, taken as the argument
    to both branches
    `#2993 <https://github.com/google/jax/pull/2993>`_.

* Notable changes:

  * The format of the `transforms` keyword for the `lax.experimental.host_callback.id_tap`
    primitive has changed `#3132 <https://github.com/google/jax/pull/3132>`_.


jax 0.1.67 (May 12, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.66...jax-v0.1.67>`_.

* New features:

  * Support for reduction over subsets of a pmapped axis using ``axis_index_groups``
    `#2382 <https://github.com/google/jax/pull/2382>`_.
  * Experimental support for printing and calling host-side Python function from
    compiled code. See `id_print and id_tap <https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html>`_
    (`#3006 <https://github.com/google/jax/pull/3006>`_).

* Notable changes:

  * The visibility of names exported from :py:module:`jax.numpy` has been
    tightened. This may break code that was making use of names that were
    previously exported accidentally.

jaxlib 0.1.47 (May 8, 2020)
------------------------------

* Fixes crash for outfeed.

jax 0.1.66 (May 5, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.65...jax-v0.1.66>`_.

* New features:

  * Support for ``in_axes=None`` on :func:`pmap`
    `#2896 <https://github.com/google/jax/pull/2896>`_.

jaxlib 0.1.46 (May 5, 2020)
------------------------------

* Fixes crash for linear algebra functions on Mac OS X (#432).
* Fixes an illegal instruction crash caused by using AVX512 instructions when
  an operating system or hypervisor disabled them (#2906).

jax 0.1.65 (April 30, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.64...jax-v0.1.65>`_.

* New features:

  * Differentiation of determinants of singular matrices
    `#2809 <https://github.com/google/jax/pull/2809>`_.

* Bug fixes:

  * Fix :func:`odeint` differentiation with respect to time of ODEs with
    time-dependent dynamics `#2817 <https://github.com/google/jax/pull/2817>`_,
    also add ODE CI testing.
  * Fix :func:`lax_linalg.qr` differentiation
    `#2867 <https://github.com/google/jax/pull/2867>`_.

jaxlib 0.1.45 (April 21, 2020)
------------------------------

* Fixes segfault: https://github.com/google/jax/issues/2755
* Plumb is_stable option on Sort HLO through to Python.

jax 0.1.64 (April 21, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.63...jax-v0.1.64>`_.
* New features:

  * Add syntactic sugar for functional indexed updates
    `#2684 <https://github.com/google/jax/issues/2684>`_.
  * Add :func:`jax.numpy.linalg.multi_dot` `#2726 <https://github.com/google/jax/issues/2726>`_.
  * Add :func:`jax.numpy.unique` `#2760 <https://github.com/google/jax/issues/2760>`_.
  * Add :func:`jax.numpy.rint` `#2724 <https://github.com/google/jax/issues/2724>`_.
  * Add :func:`jax.numpy.rint` `#2724 <https://github.com/google/jax/issues/2724>`_.
  * Add more primitive rules for :func:`jax.experimental.jet`.

* Bug fixes:

  * Fix :func:`logaddexp` and :func:`logaddexp2` differentiation at zero `#2107
    <https://github.com/google/jax/issues/2107>`_.
  * Improve memory usage in reverse-mode autodiff without :func:`jit`
    `#2719 <https://github.com/google/jax/issues/2719>`_.

* Better errors:

  * Improves error message for reverse-mode differentiation of :func:`lax.while_loop`
    `#2129 <https://github.com/google/jax/issues/2129>`_.


jaxlib 0.1.44 (April 16, 2020)
------------------------------

* Fixes a bug where if multiple GPUs of different models were present, JAX
  would only compile programs suitable for the first GPU.
* Bugfix for ``batch_group_count`` convolutions.
* Added precompiled SASS for more GPU versions to avoid startup PTX compilation
  hang.


jax 0.1.63 (April 12, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.62...jax-v0.1.63>`_.
* Added ``jax.custom_jvp`` and ``jax.custom_vjp`` from `#2026 <https://github.com/google/jax/pull/2026>`_, see the `tutorial notebook <https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_. Deprecated ``jax.custom_transforms`` and removed it from the docs (though it still works).
* Add ``scipy.sparse.linalg.cg`` `#2566 <https://github.com/google/jax/pull/2566>`_.
* Changed how Tracers are printed to show more useful information for debugging `#2591 <https://github.com/google/jax/pull/2591>`_.
* Made ``jax.numpy.isclose`` handle ``nan`` and ``inf`` correctly `#2501 <https://github.com/google/jax/pull/2501>`_.
* Added several new rules for ``jax.experimental.jet`` `#2537 <https://github.com/google/jax/pull/2537>`_.
* Fixed ``jax.experimental.stax.BatchNorm`` when ``scale``/``center`` isn't provided.
* Fix some missing cases of broadcasting in ``jax.numpy.einsum`` `#2512 <https://github.com/google/jax/pull/2512>`_.
* Implement ``jax.numpy.cumsum`` and ``jax.numpy.cumprod`` in terms of a parallel prefix scan `#2596 <https://github.com/google/jax/pull/2596>`_ and make ``reduce_prod`` differentiable to arbitray order `#2597 <https://github.com/google/jax/pull/2597>`_.
* Add ``batch_group_count`` to ``conv_general_dilated`` `#2635 <https://github.com/google/jax/pull/2635>`_.
* Add docstring for ``test_util.check_grads`` `#2656 <https://github.com/google/jax/pull/2656>`_.
* Add ``callback_transform`` `#2665 <https://github.com/google/jax/pull/2665>`_.
* Implement ``rollaxis``, ``convolve``/``correlate`` 1d & 2d, ``copysign``,
  ``trunc``, ``roots``, and ``quantile``/``percentile`` interpolation options.

jaxlib 0.1.43 (March 31, 2020)
------------------------------

* Fixed a performance regression for Resnet-50 on GPU.

jax 0.1.62 (March 21, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.61...jax-v0.1.62>`_.
* JAX has dropped support for Python 3.5. Please upgrade to Python 3.6 or newer.
* Removed the internal function ``lax._safe_mul``, which implemented the
  convention ``0. * nan == 0.``. This change means some programs when
  differentiated will produce nans when they previously produced correct
  values, though it ensures nans rather than silently incorrect results are
  produced for other programs. See #2447 and #1052 for details.
* Added an ``all_gather`` parallel convenience function.
* More type annotations in core code.

jaxlib 0.1.42 (March 19, 2020)
------------------------------

* jaxlib 0.1.41 broke cloud TPU support due to an API incompatibility. This
  release fixes it again.
* JAX has dropped support for Python 3.5. Please upgrade to Python 3.6 or newer.

jax 0.1.61 (March 17, 2020)
---------------------------
* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.60...jax-v0.1.61>`_.
* Fixes Python 3.5 support. This will be the last JAX or jaxlib release that
  supports Python 3.5.

jax 0.1.60 (March 17, 2020)
---------------------------

* `GitHub commits <https://github.com/google/jax/compare/jax-v0.1.59...jax-v0.1.60>`_.
* New features:

  * :py:func:`jax.pmap` has ``static_broadcast_argnums`` argument which allows
    the user to specify arguments that should be treated as compile-time
    constants and should be broadcasted to all devices. It works analogously to
    ``static_argnums`` in :py:func:`jax.jit`.
  * Improved error messages for when tracers are mistakenly saved in global state.
  * Added :py:func:`jax.nn.one_hot` utility function.
  * Added :py:module:`jax.experimental.jet` for exponentially faster
    higher-order automatic differentiation.
  * Added more correctness checking to arguments of :py:func:`jax.lax.broadcast_in_dim`.

* The minimum jaxlib version is now 0.1.41.

jaxlib 0.1.40 (March 4, 2020)
-------------------------------

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
    fully-closed (no ``constvars``) jaxpr. Also, added a new field ``call_primitive``
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
---------------------------------

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
