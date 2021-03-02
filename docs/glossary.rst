JAX Glossary of Terms
=====================

.. glossary::

    CPU
      Short for *Central Processing Unit*, CPUs are the standard computational architecture
      available in most computers. JAX can run computations on CPUs, but often can achieve
      much better performance on :term:`GPU` and :term:`TPU`.

    Device
      The generic name used to refer to the :term:`CPU`, :term:`GPU`, or :term:`TPU` used
      by JAX to perform computations.

    DeviceArray
      JAX's analog of the :class:`numpy.ndarray`. See :class:`jax.interpreters.xla.DeviceArray`.

    forward-mode autodiff
      See :term:`JVP`

    functional programming
      A programming paradigm in which programs are defined by applying and composing
      :term:`pure functions<pure function>`. JAX is designed for use with functional programs.

    GPU
      Short for *Graphical Processing Unit*, GPUs were originally specialized for operations
      related to rendering of images on screen, but now are much more general-purpose. JAX is
      able to target GPUs for fast operations on arrays (see also :term:`CPU` and :term:`TPU`).

    JIT
      Short for *Just In Time* compilation, JIT in JAX generally refers to the compilation of
      array operations to :term:`XLA`, most often accomplished using :func:`jax.jit`.

    JVP
      Short for *Jacobian Vector Product*, also sometimes known as *forward-mode* automatic
      differentiation. For more details, see :ref:`jacobian-vector-product`. In JAX, JVP is
      a :term:`transformation` that is implemented via :func:`jax.jvp`. See also :term:`VJP`.

    pure function
      A pure function is a function whose outputs are based only on its inputs, and which has
      no side-effects. JAX's :term:`transformation` model is designed to work with pure functions.
      See also :term:`functional programming`.

    reverse-mode autodiff
      See :term:`VJP`.

    static
      In a :term:`JIT` compilation, a value that is not traced (see :term:`Tracer`). Also
      sometimes refers to compile-time computations on static values.
    
    TPU
      Short for *Tensor Processing Unit*, TPUs are chips specifically engineered for fast operations
      on N-dimensional tensors used in deep learning applications. JAX is able to target TPUs for
      fast operations on arrays (see also :term:`CPU` and :term:`GPU`).

    Tracer
      An object used as a standin for a JAX :term:`DeviceArray` in order to determine the
      sequence of operations performed by a Python function. Internally, JAX implements this
      via the :class:`jax.core.Tracer` class.

    transformation
      A higher-order function: that is, a function that takes a function as input and outputs
      a transformed function. Examples in JAX include :func:`jax.jit`, :func:`jax.vmap`, and
      :func:`jax.grad`.

    VJP
      Short for *Vector Jacobian Product*, also sometimes known as *reverse-mode* automatic
      differentiation. For more details, see :ref:`vector-jacobian-product`. In JAX, VJP is
      a :term:`transformation` that is implemented via :func:`jax.vjp`. See also :term:`JVP`.

    XLA
      Short for *Accelerated Linear Algebra*, XLA is a domain-specific compiler for linear
      algebra operations that is the primary backend for :term:`JIT`-compiled JAX code.
      See https://www.tensorflow.org/xla/.
