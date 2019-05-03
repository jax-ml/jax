jax.lax package
================

.. automodule:: jax.lax

`lax` is a library of primitives that underpins libraries such as `jax.numpy`.

Many of the primitives are thin wrappers around equivalent XLA operations,
described by the `XLA operation semantics
<https://www.tensorflow.org/xla/operation_semantics>`_ documentation.

Where possible, prefer to use libraries such as `jax.numpy` instead of using `jax.lax` directly.

Operators
---------

.. autosummary::
  :toctree: _autosummary

    abs
    add
    acos
    acosh
    asin
    asinh
    atan
    atanh
    atan2
    batch_matmul
    bitcast_convert_type
    bitwise_not
    bitwise_and
    bitwise_or
    bitwise_xor
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
    expm1
    fft
    floor
    full
    full_like
    gather
    ge
    gt
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
    shaped_identity
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
    stop_gradient
    sub
    tan
    tie_in
    transpose


Control flow operators
----------------------

.. autosummary::
  :toctree: _autosummary

    cond
    fori_loop
    scan
    while_loop


Parallel operators
------------------

Parallelism support is experimental.

.. autosummary::
  :toctree: _autosummary

    pcollect
    pmax
    psplit
    psplit_like
    psum
    pswapaxes
