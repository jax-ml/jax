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

from jax._src.numpy.lax_numpy import (
    apply_along_axis as apply_along_axis,
    arange as arange,
    argmax as argmax,
    argmin as argmin,
    argsort as argsort,
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
