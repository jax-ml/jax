
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

.. Generate the list below as follows:
   >>> import jax.numpy, numpy
   >>> fns = set(dir(numpy)) & set(dir(jax.numpy)) - set(jax.numpy._NOT_IMPLEMENTED)
   >>> print('\n'.join('    ' + x for x in fns if callable(getattr(jax.numpy, x))))  # doctest: +SKIP

   # Finally, sort the list using sort(1), which is different than Python's
   # sorted() function.

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
    apply_along_axis
    apply_over_axes
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
    array_equal
    array_equiv
    array_repr
    array_split
    array_str
    asarray
    atleast_1d
    atleast_2d
    atleast_3d
    average
    bartlett
    bincount
    bitwise_and
    bitwise_not
    bitwise_or
    bitwise_xor
    blackman
    block
    bool_
    broadcast_arrays
    broadcast_to
    can_cast
    cbrt
    cdouble
    ceil
    character
    choose
    clip
    column_stack
    complex_
    complex128
    complex64
    complexfloating
    ComplexWarning
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
    csingle
    cumprod
    cumproduct
    cumsum
    deg2rad
    degrees
    diag
    diagflat
    diag_indices
    diag_indices_from
    diagonal
    diff
    digitize
    divide
    divmod
    dot
    double
    dsplit
    dstack
    dtype
    ediff1d
    einsum
    einsum_path
    empty
    empty_like
    equal
    exp
    exp2
    expand_dims
    expm1
    extract
    eye
    fabs
    finfo
    fix
    flatnonzero
    flexible
    flip
    fliplr
    flipud
    float_
    float16
    float32
    float64
    floating
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
    gradient
    greater
    greater_equal
    hamming
    hanning
    heaviside
    histogram
    histogram_bin_edges
    histogram2d
    histogramdd
    hsplit
    hstack
    hypot
    i0
    identity
    iinfo
    imag
    in1d
    indices
    inexact
    inner
    int_
    int16
    int32
    int64
    int8
    integer
    interp
    intersect1d
    invert
    isclose
    iscomplex
    iscomplexobj
    isfinite
    isin
    isinf
    isnan
    isneginf
    isposinf
    isreal
    isrealobj
    isscalar
    issubdtype
    issubsctype
    iterable
    ix_
    kaiser
    kron
    lcm
    ldexp
    left_shift
    less
    less_equal
    lexsort
    linspace
    load
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
    mask_indices
    matmul
    max
    maximum
    mean
    median
    meshgrid
    min
    minimum
    mod
    modf
    moveaxis
    msort
    multiply
    nanargmax
    nanargmin
    nancumprod
    nancumsum
    nanmax
    nanmean
    nanmedian
    nanmin
    nanpercentile
    nanprod
    nanquantile
    nanstd
    nansum
    nan_to_num
    nanvar
    ndarray
    ndim
    negative
    nextafter
    nonzero
    not_equal
    number
    object_
    ones
    ones_like
    outer
    packbits
    pad
    percentile
    piecewise
    polyadd
    polyder
    polymul
    polysub
    polyval
    positive
    power
    prod
    product
    promote_types
    ptp
    quantile
    rad2deg
    radians
    ravel
    ravel_multi_index
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
    save
    savez
    searchsorted
    select
    set_printoptions
    shape
    sign
    signbit
    signedinteger
    sin
    sinc
    single
    sinh
    size
    sometrue
    sort
    sort_complex
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
    trapz
    tri
    tril
    tril_indices
    tril_indices_from
    trim_zeros
    triu
    triu_indices
    triu_indices_from
    true_divide
    trunc
    uint16
    uint32
    uint64
    uint8
    unique
    unpackbits
    unravel_index
    unsignedinteger
    unwrap
    vander
    var
    vdot
    vectorize
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
  fft2
  fftfreq
  fftn
  fftshift
  hfft
  ifft
  ifft2
  ifftn
  ifftshift
  ihfft
  irfft
  irfft2
  irfftn
  rfft
  rfft2
  rfftfreq
  rfftn

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
