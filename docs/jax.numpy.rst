
jax.numpy package
=================

.. currentmodule:: jax.numpy

.. automodule:: jax.numpy

Implements the NumPy API, using the primitives in :mod:`jax.lax`.

While JAX tries to follow the NumPy API as closely as possible, sometimes JAX
cannot follow NumPy exactly.

* Notably, since JAX arrays are immutable, NumPy APIs that mutate arrays
  in-place cannot be implemented in JAX. However, often JAX is able to provide a
  alternative API that is purely functional. For example, instead of in-place
  array updates (:code:`x[i] = y`), JAX provides an alternative pure indexed
  update function :func:`jax.ops.index_update`.

* NumPy is very aggressive at promoting values to :code:`float64` type. JAX
  sometimes is less aggressive about type promotion.

A small number of NumPy operations that have data-dependent output shapes are
incompatible with :func:`jax.jit` compilation. The XLA compiler requires that
shapes of arrays be known at compile time. While it would be possible to provide
a JAX implementation of an API such as :func:`numpy.nonzero`, we would be unable
to JIT-compile it because the shape of its output depends on the contents of the
input data.

Not every function in NumPy is implemented; contributions are welcome!

.. autosummary::
  :toctree: _autosummary

    abs
    absolute
    add
    all
    allclose
    alltrue
    amax
    amin
    angle
    any
    append
    arange
    arccos
    arccosh
    arcsin
    arcsinh
    arctan
    arctan2
    arctanh
    argmax
    argmin
    argsort
    argwhere
    around
    array
    array_repr
    array_str
    asarray
    atleast_1d
    atleast_2d
    atleast_3d
    bartlett
    bincount
    bitwise_and
    bitwise_not
    bitwise_or
    bitwise_xor
    blackman
    block
    broadcast_arrays
    broadcast_to
    can_cast
    ceil
    clip
    column_stack
    compress
    concatenate
    conj
    conjugate
    convolve
    copysign
    corrcoef
    correlate
    cos
    cosh
    count_nonzero
    cov
    cross
    cumsum
    cumprod
    cumproduct
    deg2rad
    degrees
    diag
    diag_indices
    diagflat
    diagonal
    digitize
    divide
    divmod
    dot
    dsplit
    dstack
    ediff1d
    einsum
    equal
    empty
    empty_like
    exp
    exp2
    expand_dims
    expm1
    extract
    eye
    fabs
    fix
    flatnonzero
    flip
    fliplr
    flipud
    float_power
    floor
    floor_divide
    fmax
    fmin
    fmod
    frexp
    full
    full_like
    gcd
    geomspace
    greater
    greater_equal
    hamming
    hanning
    heaviside
    histogram
    histogram_bin_edges
    hsplit
    hstack
    hypot
    identity
    imag
    in1d
    indices
    inner
    isclose
    iscomplex
    isfinite
    isin
    isinf
    isnan
    isneginf
    isposinf
    isreal
    isscalar
    issubdtype
    issubsctype
    ix_
    kaiser
    kron
    lcm
    ldexp
    left_shift
    less
    less_equal
    linspace
    log
    log10
    log1p
    log2
    logaddexp
    logaddexp2
    logical_and
    logical_not
    logical_or
    logical_xor
    logspace
    matmul
    max
    maximum
    mean
    median
    meshgrid
    min
    minimum
    mod
    moveaxis
    msort
    multiply
    nan_to_num
    nanargmax
    nanargmin
    nancumprod
    nancumsum
    nanmax
    nanmin
    nanprod
    nansum
    negative
    nextafter
    nonzero
    not_equal
    ones
    ones_like
    outer
    packbits
    pad
    percentile
    polyadd
    polymul
    polysub
    polyval
    power
    positive
    prod
    product
    promote_types
    ptp
    quantile
    rad2deg
    radians
    ravel
    real
    reciprocal
    remainder
    repeat
    reshape
    result_type
    right_shift
    rint
    roll
    rollaxis
    roots
    rot90
    round
    row_stack
    searchsorted
    select
    sign
    signbit
    sin
    sinc
    sinh
    sometrue
    sort
    split
    sqrt
    square
    squeeze
    stack
    std
    subtract
    sum
    swapaxes
    take
    take_along_axis
    tan
    tanh
    tensordot
    tile
    trace
    transpose
    tri
    tril
    tril_indices
    tril_indices_from
    triu
    triu_indices
    true_divide
    trunc
    unique
    unpackbits
    unravel_index
    vander
    var
    vdot
    vsplit
    vstack
    where
    zeros
    zeros_like

jax.numpy.fft
-------------

.. automodule:: jax.numpy.fft

.. autosummary::
  :toctree: _autosummary

  fft
  ifft
  fft2
  ifft2
  fftn
  ifftn
  rfft
  irfft
  rfft2
  irfft2
  rfftn
  irfftn
  fftfreq
  rfftfreq
  fftshift
  ifftshift

jax.numpy.linalg
----------------

.. automodule:: jax.numpy.linalg

.. autosummary::
  :toctree: _autosummary

  cholesky
  cond
  det
  eig
  eigh
  eigvals
  eigvalsh
  inv
  lstsq
  matrix_power
  matrix_rank
  multi_dot
  norm
  pinv
  qr
  slogdet
  solve
  svd
  tensorinv
  tensorsolve
