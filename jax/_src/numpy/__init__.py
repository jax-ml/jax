# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The following are a subset of the full jax.numpy functionality used by
# internal imports.

from jax._src.numpy import (
    fft as fft,
    linalg as linalg,
)

from jax._src.numpy.array_constructors import (
    asarray as asarray,
    array as array,
)

from jax._src.numpy.array_creation import (
    empty as empty,
    empty_like as empty_like,
    full as full,
    full_like as full_like,
    linspace as linspace,
    ones as ones,
    ones_like as ones_like,
    zeros as zeros,
    zeros_like as zeros_like,
)

from jax._src.numpy.indexing import (
    take as take,
)

from numpy import (
    nan as nan,
    inf as inf,
    floating as floating,
    integer as integer,
    inexact as inexact,
    complexfloating as complexfloating,
    number as number,
    character as character,
    generic as generic,
    dtype as dtype,
    unsignedinteger as unsignedinteger,
)
from jax._src import dtypes as dtypes
from jax._src.numpy.scalar_types import (
    bfloat16 as bfloat16,
    bool_ as bool,  # Array API alias for bool_  # noqa: F401
    bool_ as bool_,
    cdouble as cdouble,
    csingle as csingle,
    complex128 as complex128,
    complex64 as complex64,
    complex_ as complex_,
    double as double,
    float16 as float16,
    float32 as float32,
    float4_e2m1fn as float4_e2m1fn,
    float64 as float64,
    float8_e3m4 as float8_e3m4,
    float8_e4m3 as float8_e4m3,
    float8_e4m3b11fnuz as float8_e4m3b11fnuz,
    float8_e4m3fn as float8_e4m3fn,
    float8_e4m3fnuz as float8_e4m3fnuz,
    float8_e5m2 as float8_e5m2,
    float8_e5m2fnuz as float8_e5m2fnuz,
    float8_e8m0fnu as float8_e8m0fnu,
    float_ as float_,
    int2 as int2,
    int4 as int4,
    int8 as int8,
    int16 as int16,
    int32 as int32,
    int64 as int64,
    int_ as int_,
    single as single,
    uint as uint,
    uint2 as uint2,
    uint4 as uint4,
    uint8 as uint8,
    uint16 as uint16,
    uint32 as uint32,
    uint64 as uint64,
)

from jax._src.numpy.lax_numpy import (
    apply_along_axis as apply_along_axis,
    arange as arange,
    argmax as argmax,
    argmin as argmin,
    argsort as argsort,
    astype as astype,
    atleast_1d as atleast_1d,
    atleast_2d as atleast_2d,
    block as block,
    broadcast_arrays as broadcast_arrays,
    broadcast_shapes as broadcast_shapes,
    broadcast_to as broadcast_to,
    clip as clip,
    concatenate as concatenate,
    cov as cov,
    cross as cross,
    diag as diag,
    diag_indices as diag_indices,
    diff as diff,
    digitize as digitize,
    expand_dims as expand_dims,
    eye as eye,
    flip as flip,
    hstack as hstack,
    iscomplexobj as iscomplexobj,
    isscalar as isscalar,
    issubdtype as issubdtype,
    iinfo as iinfo,
    meshgrid as meshgrid,
    moveaxis as moveaxis,
    nonzero as nonzero,
    pad as pad,
    permute_dims as permute_dims,
    piecewise as piecewise,
    promote_types as promote_types,
    ravel as ravel,
    reshape as reshape,
    repeat as repeat,
    roll as roll,
    round as round,
    result_type as result_type,
    searchsorted as searchsorted,
    select as select,
    split as split,
    squeeze as squeeze,
    stack as stack,
    trace as trace,
    transpose as transpose,
    tril as tril,
    triu as triu,
    triu_indices as triu_indices,
    unravel_index as unravel_index,
    vstack as vstack,
    where as where,
)

from jax._src.numpy.index_tricks import (
    ogrid as ogrid,
)

from jax._src.numpy.polynomial import (
    polyval as polyval
)

from jax._src.numpy.reductions import (
    all as all,
    amax as amax,
    amin as amin,
    any as any,
    cumprod as cumprod,
    cumsum as cumsum,
    max as max,
    mean as mean,
    median as median,
    nanstd as nanstd,
    prod as prod,
    sum as sum,
)

from jax._src.numpy.setops import (
    unique as unique,
)

from jax._src.numpy.tensor_contractions import (
    dot as dot,
    matmul as matmul,
    vdot as vdot,
)

from jax._src.numpy.ufuncs import (
    abs as abs,
    arctan2 as arctan2,
    bitwise_and as bitwise_and,
    cbrt as cbrt,
    ceil as ceil,
    conj as conj,
    conjugate as conjugate,
    cos as cos,
    deg2rad as deg2rad,
    divide as divide,
    equal as equal,
    exp as exp,
    expm1 as expm1,
    floor as floor,
    floor_divide as floor_divide,
    fmod as fmod,
    greater as greater,
    greater_equal as greater_equal,
    hypot as hypot,
    imag as imag,
    isinf as isinf,
    isfinite as isfinite,
    isnan as isnan,
    less as less,
    less_equal as less_equal,
    log as log,
    log1p as log1p,
    log2 as log2,
    logaddexp as logaddexp,
    logical_and as logical_and,
    logical_not as logical_not,
    logical_or as logical_or,
    maximum as maximum,
    minimum as minimum,
    mod as mod,
    not_equal as not_equal,
    power as power,
    rad2deg as rad2deg,
    real as real,
    reciprocal as reciprocal,
    sign as sign,
    signbit as signbit,
    sin as sin,
    sqrt as sqrt,
    square as square,
    subtract as subtract,
    tanh as tanh
)
from jax._src.numpy.ufuncs import (
    multiply as multiply,
)
