``jax.numpy`` module
====================

.. currentmodule:: jax.numpy

.. automodule:: jax.numpy

Implements the NumPy API, using the primitives in :mod:`jax.lax`.

While JAX tries to follow the NumPy API as closely as possible, sometimes JAX
cannot follow NumPy exactly.

* Notably, since JAX arrays are immutable, NumPy APIs that mutate arrays
  in-place cannot be implemented in JAX. However, often JAX is able to provide
  an alternative API that is purely functional. For example, instead of in-place
  array updates (:code:`x[i] = y`), JAX provides an alternative pure indexed
  update function :code:`x.at[i].set(y)` (see :attr:`ndarray.at`).

* Relatedly, some NumPy functions often return views of arrays when possible
  (examples are :func:`transpose` and :func:`reshape`). JAX versions of such
  functions will return copies instead, although such are often optimized
  away by XLA when sequences of operations are compiled using :func:`jax.jit`.

* NumPy is very aggressive at promoting values to :code:`float64` type. JAX
  sometimes is less aggressive about type promotion (See :ref:`type-promotion`).

* Some NumPy routines have data-dependent output shapes (examples include
  :func:`unique` and :func:`nonzero`). Because the XLA compiler requires array
  shapes to be known at compile time, such operations are not compatible with
  JIT. For this reason, JAX adds an optional ``size`` argument to such functions
  which may be specified statically in order to use them with JIT.

Nearly all applicable NumPy functions are implemented in the ``jax.numpy``
namespace; they are listed below.


Array creation
--------------

.. autosummary::
  :toctree: _autosummary

    arange
    array
    asarray
    copy
    diag
    diagflat
    empty
    empty_like
    eye
    from_dlpack
    frombuffer
    fromfile
    fromfunction
    fromiter
    fromstring
    full
    full_like
    geomspace
    identity
    linspace
    logspace
    meshgrid
    mgrid
    ogrid
    ones
    ones_like
    zeros
    zeros_like

Array properties
----------------

.. autosummary::
  :toctree: _autosummary

    ndim
    shape
    size
    
Array manipulation
------------------

.. autosummary::
  :toctree: _autosummary

    append
    array_split
    astype
    atleast_1d
    atleast_2d
    atleast_3d
    block
    broadcast_arrays
    broadcast_to
    column_stack
    concat
    concatenate
    delete
    dsplit
    dstack
    expand_dims
    flip
    fliplr
    flipud
    hsplit
    hstack
    insert
    matrix_transpose
    moveaxis
    permute_dims
    ravel
    repeat
    reshape
    resize
    roll
    rollaxis
    rot90
    split
    squeeze
    stack
    swapaxes
    tile
    transpose
    trim_zeros
    unstack
    vsplit
    vstack

Elementwise functions
---------------------

.. autosummary::
  :toctree: _autosummary

    abs
    absolute
    acos
    acosh
    add
    angle
    arccos
    arccosh
    arcsin
    arcsinh
    arctan
    arctan2
    arctanh
    around
    asin
    asinh
    atan
    atan2
    atanh
    bitwise_and
    bitwise_count
    bitwise_invert
    bitwise_left_shift
    bitwise_not
    bitwise_or
    bitwise_right_shift
    bitwise_xor
    cbrt
    ceil
    clip
    conj
    conjugate
    copysign
    cos
    cosh
    deg2rad
    degrees
    divide
    divmod
    equal
    exp
    exp2
    expm1
    fabs
    float_power
    floor
    floor_divide
    fmax
    fmin
    fmod
    frexp
    gcd
    greater
    greater_equal
    heaviside
    hypot
    i0
    imag
    invert
    isclose
    isfinite
    isinf
    isnan
    isneginf
    isposinf
    lcm
    ldexp
    left_shift
    less
    less_equal
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
    maximum
    minimum
    mod
    modf
    multiply
    nan_to_num
    negative
    nextafter
    not_equal
    positive
    pow
    power
    rad2deg
    radians
    real
    reciprocal
    remainder
    right_shift
    rint
    round
    sign
    signbit
    sin
    sinc
    sinh
    spacing
    sqrt
    square
    subtract
    tan
    tanh
    true_divide
    trunc

Sorting and searching
---------------------

.. autosummary::
  :toctree: _autosummary

    argmax
    argmin
    argpartition
    argsort
    argwhere
    count_nonzero
    extract
    flatnonzero
    lexsort
    nanargmax
    nanargmin
    nonzero
    partition
    searchsorted
    select
    sort
    sort_complex
    where

Reductions and statistics
-------------------------

.. autosummary::
  :toctree: _autosummary

    all
    allclose
    amax
    amin
    any
    average
    cumprod
    cumsum
    cumulative_prod
    cumulative_sum
    max
    mean
    median
    min
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
    nanvar
    percentile
    prod
    ptp
    quantile
    std
    sum
    var

Indexing
--------

.. autosummary::
  :toctree: _autosummary

    ndarray.at
    diag_indices
    diag_indices_from
    index_exp
    indices
    mask_indices
    place
    put
    put_along_axis
    ravel_multi_index
    take
    take_along_axis
    tril_indices
    tril_indices_from
    triu_indices
    triu_indices_from
    unravel_index

Set-like operations
-------------------

.. autosummary::
  :toctree: _autosummary

    intersect1d
    isin
    setdiff1d
    setxor1d
    union1d
    unique
    unique_all
    unique_counts
    unique_inverse
    unique_values

Polynomial functions
--------------------

.. autosummary::
  :toctree: _autosummary

    poly
    polyadd
    polyder
    polydiv
    polyfit
    polyint
    polymul
    polysub
    polyval

Tensor products
---------------

.. autosummary::
  :toctree: _autosummary

    cross
    dot
    einsum
    einsum_path
    inner
    kron
    matmul
    matvec
    outer
    tensordot
    vdot
    vecdot
    vecmat

Data types and related
----------------------

.. autosummary::
  :toctree: _autosummary

    ComplexWarning
    bool_
    can_cast
    cdouble
    character
    complex128
    complex64
    complex_
    complexfloating
    csingle
    double
    dtype
    finfo
    flexible
    float16
    float32
    float64
    float_
    floating
    generic
    iinfo
    inexact
    int16
    int32
    int64
    int8
    int_
    integer
    isdtype
    issubdtype
    number
    object_
    promote_types
    result_type
    signedinteger
    single
    ufunc
    uint
    uint16
    uint32
    uint64
    uint8
    unsignedinteger

Other
-----

.. autosummary::
  :toctree: _autosummary

    apply_along_axis
    apply_over_axes
    array_equal
    array_equiv
    array_repr
    array_str
    bartlett
    bincount
    blackman
    broadcast_shapes
    c_
    choose
    compress
    convolve
    corrcoef
    correlate
    cov
    diagonal
    diff
    digitize
    ediff1d
    fill_diagonal
    frompyfunc
    get_printoptions
    gradient
    hamming
    hanning
    histogram
    histogram2d
    histogram_bin_edges
    histogramdd
    interp
    iscomplex
    iscomplexobj
    isreal
    isrealobj
    isscalar
    iterable
    ix_
    kaiser
    load
    ndarray
    packbits
    pad
    piecewise
    printoptions
    r_
    roots
    s_
    save
    savez
    set_printoptions
    trace
    trapezoid
    tri
    tril
    triu
    unpackbits
    unwrap
    vander
    vectorize


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
  cross
  det
  diagonal
  eig
  eigh
  eigvals
  eigvalsh
  inv
  lstsq
  matmul
  matrix_norm
  matrix_power
  matrix_rank
  matrix_transpose
  multi_dot
  norm
  outer
  pinv
  qr
  slogdet
  solve
  svd
  svdvals
  tensordot
  tensorinv
  tensorsolve
  trace
  vector_norm
  vecdot

JAX Array
---------
The JAX :class:`~jax.Array` (along with its alias, :class:`jax.numpy.ndarray`) is
the core array object in JAX: you can think of it as JAX's equivalent of a
:class:`numpy.ndarray`. Like :class:`numpy.ndarray`, most users will not need to
instantiate :class:`~jax.Array` objects manually, but rather will create them via
:mod:`jax.numpy` functions like :func:`~jax.numpy.array`, :func:`~jax.numpy.arange`,
:func:`~jax.numpy.linspace`, and others listed above.

Copying and Serialization
~~~~~~~~~~~~~~~~~~~~~~~~~
JAX :class:`~jax.Array` objects are designed to work seamlessly with Python
standard library tools where appropriate.

With the built-in :mod:`copy` module, when :func:`copy.copy` or :func:`copy.deepcopy`
encounder an :class:`~jax.Array`, it is equivalent to calling the
:meth:`~jax.Array.copy` method, which will create a copy of
the buffer on the same device as the original array. This will work correctly within
traced/JIT-compiled code, though copy operations may be elided by the compiler
in this context.

When the built-in :mod:`pickle` module encounters an :class:`~jax.Array`,
it will be serialized via a compact bit representation in a similar manner to pickled
:class:`numpy.ndarray` objects. When unpickled, the result will be a new
:class:`~jax.Array` object *on the default device.*
This is because in general, pickling and unpickling may take place in different runtime
environments, and there is no general way to map the device IDs of one runtime
to the device IDs of another. If :mod:`pickle` is used in traced/JIT-compiled code,
it will result in a :class:`~jax.errors.ConcretizationTypeError`.

.. _python-array-api:

Python Array API standard
-------------------------

.. note::

  Prior to JAX v0.4.32, you must ``import jax.experimental.array_api`` in order
  to enable the array API for JAX arrays. After JAX v0.4.32, importing this
  module is no longer required, and will raise a deprecation warning. After
  JAX v0.5.0, this import will raise an error.

Starting with JAX v0.4.32, :class:`jax.Array` and :mod:`jax.numpy` are compatible
with the `Python Array API Standard`_. You can access the Array API namespace via
:meth:`jax.Array.__array_namespace__`::

    >>> def f(x):
    ...   nx = x.__array_namespace__()
    ...   return nx.sin(x) ** 2 + nx.cos(x) ** 2

    >>> import jax.numpy as jnp
    >>> x = jnp.arange(5)
    >>> f(x).round()
    Array([1., 1., 1., 1., 1.], dtype=float32)

JAX departs from the standard in a few places, namely because JAX arrays are
immutable, in-place updates are not supported. Some of these incompatibilities
are being addressed via the `array-api-compat`_ module.

For more information, refer to the `Python Array API Standard`_ documentation.

.. _Python Array API Standard: https://data-apis.org/array-api
.. _array-api-compat: https://github.com/data-apis/array-api-compat