jax.lax package
================

.. automodule:: jax.lax

:mod:`jax.lax` is a library of primitives operations that underpins libraries
such as :mod:`jax.numpy`. Transformation rules, such as JVP and batching rules,
are typically defined as transformations on :mod:`jax.lax` primitives.

Many of the primitives are thin wrappers around equivalent XLA operations,
described by the `XLA operation semantics
<https://www.tensorflow.org/xla/operation_semantics>`_ documentation. In a few
cases JAX diverges from XLA, usually to ensure that the set of operations is
closed under the operation of JVP and transpose rules.

Where possible, prefer to use libraries such as :mod:`jax.numpy` instead of
using :mod:`jax.lax` directly. The :mod:`jax.numpy` API follows NumPy, and is
therefore more stable and less likely to change than the :mod:`jax.lax` API.

Operators
---------

.. autosummary::
  :toctree: _autosummary

    abs
    add
    acos
    argmax
    argmin
    asin
    atan
    atan2
    batch_matmul
    bessel_i0e
    bessel_i1e
    betainc
    bitcast_convert_type
    bitwise_not
    bitwise_and
    bitwise_or
    bitwise_xor
    population_count
    broadcast
    broadcasted_iota
    broadcast_in_dim
    ceil
    clamp
    collapse
    complex
    concatenate
    conj
    conv
    convert_element_type
    conv_general_dilated
    conv_with_general_padding
    conv_transpose
    cos
    cosh
    digamma
    div
    dot
    dot_general
    dynamic_index_in_dim
    dynamic_slice
    dynamic_slice_in_dim
    dynamic_update_index_in_dim
    dynamic_update_slice_in_dim
    eq
    erf
    erfc
    erf_inv
    exp
    expand_dims
    expm1
    fft
    floor
    full
    full_like
    gather
    ge
    gt
    igamma
    igammac
    imag
    index_in_dim
    index_take
    iota
    is_finite
    le
    lt
    lgamma
    log
    log1p
    max
    min
    mul
    ne
    neg
    nextafter
    pad
    pow
    real
    reciprocal
    reduce
    reduce_window
    reshape
    rem
    rev
    round
    rsqrt
    scatter
    scatter_add
    select
    shift_left
    shift_right_arithmetic
    shift_right_logical
    slice
    slice_in_dim
    sign
    sin
    sinh
    sort
    sort_key_val
    sqrt
    square
    squeeze
    sub
    tan
    tie_in
    top_k
    transpose


Control flow operators
----------------------

.. autosummary::
  :toctree: _autosummary

    cond
    fori_loop
    map
    scan
    while_loop

Custom gradient operators
-------------------------

.. autosummary::
  :toctree: _autosummary

    stop_gradient
    custom_linear_solve
    custom_root

Parallel operators
------------------

Parallelism support is experimental.

.. autosummary::
  :toctree: _autosummary

    all_gather
    all_to_all
    psum
    pmax
    pmin
    pmean
    ppermute
    pswapaxes
    axis_index
