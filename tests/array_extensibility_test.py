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

import functools
from typing import Any, Callable, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax._src import config
from jax._src import test_util as jtu


config.parse_flags_with_absl()


class JaxArrayWrapper:
  """Class that provides a __jax_array__ method."""
  x: ArrayLike

  def __init__(self, x: ArrayLike):
    self.x = x

  def __jax_array__(self) -> jax.Array:
    return jnp.asarray(self.x)


class DuckTypedArrayWithErroringJaxArray:
  """Duck-typed array that provides a __jax_array__ method which fails."""
  shape = (2, 3)
  dtype = np.dtype('float32')

  def __jax_array__(self):
    raise ValueError("jax array was called.")


class NumPyAPI(NamedTuple):
  fun: Callable[..., Any]
  args: list[jax.ShapeDtypeStruct]
  kwargs: dict[str, Any]
  skip_on_devices: list[str] | None

  def name(self):
    return self.fun.__name__

  def make_args(self, rng):
    rng = jtu.rand_default(rng)
    return jax.tree.map(lambda arg: rng(arg.shape, arg.dtype), self.args)

  def with_skip_on_devices(self, disabled_devices: list[str]) -> 'NumPyAPI':
    return self._replace(skip_on_devices=disabled_devices)

  @classmethod
  def sig(cls, fun: Callable[..., Any], *args: Any, **kwargs: Any) -> 'NumPyAPI':
    return cls(fun, args, kwargs, None)


class ShapeDtype:
  """Shortcut for specifying ShapeDtypeStruct."""
  def __init__(self, dtype):
    self.dtype = jax.dtypes.canonicalize_dtype(dtype)
  def __getitem__(self, shape) -> jax.ShapeDtypeStruct:
    if isinstance(shape, int):
      shape = (shape,)
    return jax.ShapeDtypeStruct(shape, self.dtype)

Bool = ShapeDtype(bool)
Int = ShapeDtype(int)
UInt = ShapeDtype('uint32')
Uint8 = ShapeDtype('uint8')
Float = ShapeDtype(float)
Complex = ShapeDtype(complex)


# NumPy namespace objects skipped in the enumeration below, mainly because
# they are not functions or do not take arrays as positional arguments.
SKIPPED_APIS = [
 'apply_along_axis',
 'apply_over_axes',
 'arange',
 'array_str',
 'array_repr',
 'astype',
 'bartlett',
 'bfloat16',
 'blackman',
 'block',
 'bool',
 'bool_',
 'broadcast_shapes',
 'c_',
 'can_cast',
 'cdouble',
 'character',
 'complex128',
 'complex64',
 'complex_',
 'complexfloating',
 'csingle',
 'diag_indices',
 'double',
 'dtype',
 'e',
 'einsum',
 'einsum_path',
 'euler_gamma',
 'empty',
 'eye',
 'finfo',
 'flexible',
 'float_',
 'float16',
 'float32',
 'float4_e2m1fn',
 'float64',
 'float8_e3m4',
 'float8_e4m3',
 'float8_e4m3b11fnuz',
 'float8_e4m3fn',
 'float8_e4m3fnuz',
 'float8_e5m2',
 'float8_e5m2fnuz',
 'float8_e8m0fnu',
 'floating',
 'from_dlpack',
 'frombuffer',
 'fromfile',
 'fromfunction',
 'fromiter',
 'frompyfunc',
 'fromstring',
 'full',
 'generic',
 'geomspace',
 'get_printoptions',
 'gradient',
 'hamming',
 'hanning',
 'identity',
 'iinfo',
 'index_exp',
 'indices',
 'inexact',
 'inf',
 'int16',
 'int2',
 'int32',
 'int4',
 'int64',
 'int8',
 'int_',
 'integer',
 'isdtype',
 'issubdtype'
 'iterable'
 'kaiser'
 'kron'
 'ix_',
 'linalg',
 'linspace',
 'load',
 'logspace',
 'mask_indices',
 'mgrid',
 'nan',
 'ndarray',
 'newaxis',
 'number',
 'object_',
 'ogrid',
 'ones',
 'pi',
 'printoptions',
 'promote_types'
 'r_',
 'result_type',
 's_',
 'save',
 'savez',
 'set_printoptions',
 'signedinteger',
 'single',
 'tri',
 'tril_indices',
 'triu_indices',
 'ufunc',
 'uint',
 'uint16',
 'uint2',
 'uint32',
 'uint4',
 'uint64',
 'uint8',
 'unsignedinteger',
 'vectorize',
 'zeros',
]

# TODO(jakevdp): commented APIs are ones which do not yet support
#   __jax_array__ on inputs. We should fix these!
NUMPY_APIS = [
  NumPyAPI.sig(jnp.abs, Float[5]),
  NumPyAPI.sig(jnp.absolute, Float[5]),
  NumPyAPI.sig(jnp.acos, Float[5]),
  NumPyAPI.sig(jnp.acosh, Float[5]),
  NumPyAPI.sig(jnp.add, Float[5], Float[5]),
  NumPyAPI.sig(jnp.all, Bool[5]),
  NumPyAPI.sig(jnp.allclose, Float[5], Float[5]),
  NumPyAPI.sig(jnp.amax, Float[5]),
  NumPyAPI.sig(jnp.amin, Float[5]),
  NumPyAPI.sig(jnp.angle, Float[5]),
  NumPyAPI.sig(jnp.any, Float[5]),
  NumPyAPI.sig(jnp.append, Float[10], Float[()]),
  NumPyAPI.sig(jnp.arccos, Float[5]),
  NumPyAPI.sig(jnp.arccosh, Float[5]),
  NumPyAPI.sig(jnp.arcsin, Float[5]),
  NumPyAPI.sig(jnp.arcsinh, Float[5]),
  NumPyAPI.sig(jnp.arctan, Float[5]),
  NumPyAPI.sig(jnp.arctan2, Float[5], Float[5]),
  NumPyAPI.sig(jnp.arctanh, Float[5]),
  NumPyAPI.sig(jnp.argmax, Float[10]),
  NumPyAPI.sig(jnp.argmin, Float[10]),
  NumPyAPI.sig(jnp.argpartition, Float[10], kth=5),
  NumPyAPI.sig(jnp.argsort, Float[10]),
  NumPyAPI.sig(jnp.argwhere, Float[10]),
  NumPyAPI.sig(jnp.around, Float[5]),
  NumPyAPI.sig(jnp.array, Float[5]),
  NumPyAPI.sig(jnp.array_equal, Float[5], Float[5]),
  NumPyAPI.sig(jnp.array_equiv, Float[5], Float[5]),
  NumPyAPI.sig(jnp.array_split, Float[9], indices_or_sections=3),
  NumPyAPI.sig(jnp.asarray, Float[5]),
  NumPyAPI.sig(jnp.asin, Float[5]),
  NumPyAPI.sig(jnp.asinh, Float[5]),
  NumPyAPI.sig(jnp.atan, Float[5]),
  NumPyAPI.sig(jnp.atan2, Float[5], Float[5]),
  NumPyAPI.sig(jnp.atanh, Float[5]),
  NumPyAPI.sig(jnp.atleast_1d, Float[5]),
  NumPyAPI.sig(jnp.atleast_2d, Float[5]),
  NumPyAPI.sig(jnp.atleast_3d, Float[5]),
  NumPyAPI.sig(jnp.average, Float[10]),
  NumPyAPI.sig(jnp.bincount, Int[10]),
  NumPyAPI.sig(jnp.bitwise_and, Int[5], Int[5]),
  NumPyAPI.sig(jnp.bitwise_count, Int[5]),
  NumPyAPI.sig(jnp.bitwise_invert, Int[5]),
  NumPyAPI.sig(jnp.bitwise_left_shift, Int[5], Int[5]),
  NumPyAPI.sig(jnp.bitwise_not, Int[5]),
  NumPyAPI.sig(jnp.bitwise_or, Int[5], Int[5]),
  NumPyAPI.sig(jnp.bitwise_right_shift, Int[5], Int[5]),
  NumPyAPI.sig(jnp.bitwise_xor, Int[5], Int[5]),
  NumPyAPI.sig(jnp.broadcast_arrays, Float[5]),
  NumPyAPI.sig(jnp.broadcast_to, Float[()], shape=(10,)),
  NumPyAPI.sig(jnp.cbrt, Float[5]),
  NumPyAPI.sig(jnp.ceil, Float[5]),
  NumPyAPI.sig(jnp.choose, Int[3], [Float[3], Float[3], Float[3]], mode='clip'),
  NumPyAPI.sig(jnp.clip, Float[5]),
  NumPyAPI.sig(jnp.column_stack, [Float[5], Float[5], Float[5]]),
  NumPyAPI.sig(jnp.compress, Float[10], Bool[10]),
  NumPyAPI.sig(jnp.concat, [Float[5], Float[5]]),
  NumPyAPI.sig(jnp.concatenate, [Float[5], Float[5]]),
  NumPyAPI.sig(jnp.conj, Float[5]),
  NumPyAPI.sig(jnp.conjugate, Float[5]),
  NumPyAPI.sig(jnp.convolve, Float[7], Float[3]),
  NumPyAPI.sig(jnp.copy, Float[5]),
  NumPyAPI.sig(jnp.copysign, Float[5], Float[5]),
  NumPyAPI.sig(jnp.corrcoef, Float[7], Float[7]),
  NumPyAPI.sig(jnp.correlate, Float[7], Float[3]),
  NumPyAPI.sig(jnp.cos, Float[5]),
  NumPyAPI.sig(jnp.cosh, Float[5]),
  NumPyAPI.sig(jnp.count_nonzero, Float[10]),
  NumPyAPI.sig(jnp.cov, Float[10]),
  NumPyAPI.sig(jnp.cross, Float[3], Float[3]),
  NumPyAPI.sig(jnp.cumprod, Float[5]),
  NumPyAPI.sig(jnp.cumsum, Float[5]),
  NumPyAPI.sig(jnp.cumulative_prod, Float[5]),
  NumPyAPI.sig(jnp.cumulative_sum, Float[5]),
  NumPyAPI.sig(jnp.deg2rad, Float[5]),
  NumPyAPI.sig(jnp.degrees, Float[5]),
  NumPyAPI.sig(jnp.delete, Float[5], Int[()]),
  NumPyAPI.sig(jnp.diag, Float[5]),
  NumPyAPI.sig(jnp.diag_indices_from, Float[5, 5]),
  NumPyAPI.sig(jnp.diagflat, Float[5]),
  NumPyAPI.sig(jnp.diagonal, Float[5, 5]),
  NumPyAPI.sig(jnp.diff, Float[5]),
  NumPyAPI.sig(jnp.digitize, Float[5], Float[5]),
  NumPyAPI.sig(jnp.divide, Float[5], Float[5]),
  NumPyAPI.sig(jnp.divmod, Float[5], Float[5]),
  NumPyAPI.sig(jnp.dot, Float[5], Float[5]),
  NumPyAPI.sig(jnp.dsplit, Float[3, 5, 6], indices_or_sections=2),
  NumPyAPI.sig(jnp.dstack, [Float[3, 5, 1], Float[3, 5, 3]]),
  NumPyAPI.sig(jnp.ediff1d, Float[5]),
  NumPyAPI.sig(jnp.empty_like, Float[5]),
  NumPyAPI.sig(jnp.equal, Float[5], Float[5]),
  NumPyAPI.sig(jnp.exp, Float[5]),
  NumPyAPI.sig(jnp.exp2, Float[5]),
  NumPyAPI.sig(jnp.expand_dims, Float[5], axis=0),
  NumPyAPI.sig(jnp.expm1, Float[5]),
  NumPyAPI.sig(jnp.extract, Bool[5], Float[5]),
  NumPyAPI.sig(jnp.fabs, Float[5]),
  NumPyAPI.sig(jnp.fft.fft, Float[5]),
  NumPyAPI.sig(jnp.fft.fft2, Float[5, 5]),
  NumPyAPI.sig(jnp.fft.ifft, Float[5]),
  NumPyAPI.sig(jnp.fft.ifft2, Float[5, 5]),
  NumPyAPI.sig(jnp.fill_diagonal, Float[5, 5], Float[()], inplace=False),
  NumPyAPI.sig(jnp.fix, Float[5]),
  NumPyAPI.sig(jnp.flatnonzero, Float[5]),
  NumPyAPI.sig(jnp.flip, Float[5]),
  NumPyAPI.sig(jnp.fliplr, Float[5, 5]),
  NumPyAPI.sig(jnp.flipud, Float[5, 5]),
  NumPyAPI.sig(jnp.float_power, Float[5], Float[5]),
  NumPyAPI.sig(jnp.floor, Float[5]),
  NumPyAPI.sig(jnp.floor_divide, Float[5], Float[5]),
  NumPyAPI.sig(jnp.fmax, Float[5], Float[5]),
  NumPyAPI.sig(jnp.fmin, Float[5], Float[5]),
  NumPyAPI.sig(jnp.fmod, Float[5], Float[5]),
  NumPyAPI.sig(jnp.frexp, Float[5]),
  NumPyAPI.sig(jnp.full_like, Float[5], Float[()]),
  NumPyAPI.sig(jnp.gcd, Int[5], Int[5]),
  NumPyAPI.sig(jnp.greater, Float[5], Float[5]),
  NumPyAPI.sig(jnp.greater_equal, Float[5], Float[5]),
  NumPyAPI.sig(jnp.heaviside, Float[5], Float[5]),
  NumPyAPI.sig(jnp.histogram, Float[5]),
  NumPyAPI.sig(jnp.histogram2d, Float[5], Float[5]),
  NumPyAPI.sig(jnp.histogram_bin_edges, Float[5]),
  NumPyAPI.sig(jnp.histogramdd, Float[5, 3]),
  NumPyAPI.sig(jnp.hsplit, Float[3, 6], indices_or_sections=2),
  NumPyAPI.sig(jnp.hstack, (Float[5], Float[5])),
  NumPyAPI.sig(jnp.hypot, Float[5], Float[5]),
  NumPyAPI.sig(jnp.i0, Float[5]),
  NumPyAPI.sig(jnp.imag, Complex[5]),
  NumPyAPI.sig(jnp.inner, Float[5], Float[5]),
  NumPyAPI.sig(jnp.insert, Float[5], Int[()], Float[2]),
  NumPyAPI.sig(jnp.interp, Float[10], Float[5], Float[5]),
  NumPyAPI.sig(jnp.intersect1d, Int[5], Int[5]),
  NumPyAPI.sig(jnp.invert, Int[5]),
  NumPyAPI.sig(jnp.isclose, Float[5], Float[5]),
  NumPyAPI.sig(jnp.iscomplex, Float[5]),
  NumPyAPI.sig(jnp.iscomplexobj, Complex[5]),
  NumPyAPI.sig(jnp.isfinite, Float[5]),
  NumPyAPI.sig(jnp.isin, Int[5], Int[10]),
  NumPyAPI.sig(jnp.isinf, Float[5]),
  NumPyAPI.sig(jnp.isnan, Float[5]),
  NumPyAPI.sig(jnp.isneginf, Float[5]),
  NumPyAPI.sig(jnp.isposinf, Float[5]),
  NumPyAPI.sig(jnp.isreal, Float[5]),
  NumPyAPI.sig(jnp.isrealobj, Float[5]),
  NumPyAPI.sig(jnp.isscalar, Float[()]),
  NumPyAPI.sig(jnp.lcm, Int[5], Int[5]),
  NumPyAPI.sig(jnp.ldexp, Float[5], Int[5]),
  NumPyAPI.sig(jnp.left_shift, Int[5], Int[5]),
  NumPyAPI.sig(jnp.less, Float[5], Float[5]),
  NumPyAPI.sig(jnp.less_equal, Float[5], Float[5]),
  NumPyAPI.sig(jnp.lexsort, [Float[5], Float[5]]),
  NumPyAPI.sig(jnp.log, Float[5]),
  NumPyAPI.sig(jnp.log10, Float[5]),
  NumPyAPI.sig(jnp.log1p, Float[5]),
  NumPyAPI.sig(jnp.log2, Float[5]),
  NumPyAPI.sig(jnp.logaddexp, Float[5], Float[5]),
  NumPyAPI.sig(jnp.logaddexp2, Float[5], Float[5]),
  NumPyAPI.sig(jnp.logical_and, Int[5], Int[5]),
  NumPyAPI.sig(jnp.logical_not, Int[5]),
  NumPyAPI.sig(jnp.logical_or, Int[5], Int[5]),
  NumPyAPI.sig(jnp.logical_xor, Int[5], Int[5]),
  NumPyAPI.sig(jnp.matmul, Float[5, 5], Float[5]),
  NumPyAPI.sig(jnp.matrix_transpose, Float[5, 6]),
  NumPyAPI.sig(jnp.matvec, Float[5, 5], Float[5]),
  NumPyAPI.sig(jnp.max, Float[5]),
  NumPyAPI.sig(jnp.maximum, Float[5], Float[5]),
  NumPyAPI.sig(jnp.mean, Float[5]),
  NumPyAPI.sig(jnp.median, Float[5]),
  NumPyAPI.sig(jnp.meshgrid, Float[5], Float[5]),
  NumPyAPI.sig(jnp.min, Float[5]),
  NumPyAPI.sig(jnp.minimum, Float[5], Float[5]),
  NumPyAPI.sig(jnp.mod, Float[5], Float[5]),
  NumPyAPI.sig(jnp.modf, Float[5]),
  NumPyAPI.sig(jnp.moveaxis, Float[5, 3], source=0, destination=1),
  NumPyAPI.sig(jnp.multiply, Float[5], Float[5]),
  NumPyAPI.sig(jnp.nan_to_num, Float[5]),
  NumPyAPI.sig(jnp.nanargmax, Float[5]),
  NumPyAPI.sig(jnp.nanargmin, Float[5]),
  NumPyAPI.sig(jnp.nancumprod, Float[5]),
  NumPyAPI.sig(jnp.nancumsum, Float[5]),
  NumPyAPI.sig(jnp.nanmax, Float[5]),
  NumPyAPI.sig(jnp.nanmean, Float[5]),
  NumPyAPI.sig(jnp.nanmedian, Float[5]),
  NumPyAPI.sig(jnp.nanmin, Float[5]),
  NumPyAPI.sig(jnp.nanpercentile, Float[5], q=75),
  NumPyAPI.sig(jnp.nanprod, Float[5]),
  NumPyAPI.sig(jnp.nanquantile, Float[5], q=0.75),
  NumPyAPI.sig(jnp.nanstd, Float[5]),
  NumPyAPI.sig(jnp.nansum, Float[5]),
  NumPyAPI.sig(jnp.nanvar, Float[5]),
  NumPyAPI.sig(jnp.ndim, Float[5]),
  NumPyAPI.sig(jnp.negative, Float[5]),
  NumPyAPI.sig(jnp.nextafter, Float[5], Float[5]),
  NumPyAPI.sig(jnp.nonzero, Float[5]),
  NumPyAPI.sig(jnp.not_equal, Float[5], Float[5]),
  NumPyAPI.sig(jnp.ones_like, Float[5]),
  NumPyAPI.sig(jnp.outer, Float[5], Float[5]),
  NumPyAPI.sig(jnp.packbits, Int[5]),
  NumPyAPI.sig(jnp.pad, Float[5], pad_width=2),
  NumPyAPI.sig(jnp.partition, Float[5], kth=3),
  NumPyAPI.sig(jnp.percentile, Float[5], q=75),
  NumPyAPI.sig(jnp.permute_dims, Float[3, 5], axes=(1, 0)),
  NumPyAPI.sig(jnp.piecewise, Float[5], [Bool[5], Bool[5]], funclist=[jnp.sin, jnp.cos]),
  NumPyAPI.sig(jnp.place, Float[5], Bool[5], Float[3], inplace=False),
  NumPyAPI.sig(jnp.poly, Float[5]),
  NumPyAPI.sig(jnp.polyadd, Float[5], Float[5]),
  NumPyAPI.sig(jnp.polyder, Float[5]),
  NumPyAPI.sig(jnp.polydiv, Float[5], Float[5]),
  NumPyAPI.sig(jnp.polyfit, Float[5], Float[5], deg=2),
  NumPyAPI.sig(jnp.polyint, Float[5]),
  NumPyAPI.sig(jnp.polymul, Float[5], Float[5]),
  NumPyAPI.sig(jnp.polysub, Float[5], Float[5]),
  NumPyAPI.sig(jnp.polyval, Float[5], Float[10]),
  NumPyAPI.sig(jnp.positive, Float[5]),
  NumPyAPI.sig(jnp.pow, Float[5], Float[5]),
  NumPyAPI.sig(jnp.power, Float[5], Float[5]),
  NumPyAPI.sig(jnp.prod, Float[5]),
  NumPyAPI.sig(jnp.ptp, Float[5]),
  NumPyAPI.sig(jnp.put, Float[5], Int[()], Float[()], inplace=False),
  NumPyAPI.sig(jnp.put_along_axis, Float[5], Int[1], Float[1], axis=0, inplace=False),
  NumPyAPI.sig(jnp.quantile, Float[5], q=0.75),
  NumPyAPI.sig(jnp.rad2deg, Float[5]),
  NumPyAPI.sig(jnp.radians, Float[5]),
  NumPyAPI.sig(jnp.ravel, Float[5]),
  NumPyAPI.sig(jnp.ravel_multi_index, [Uint8[5], Uint8[5]], dims=(8, 9)),
  NumPyAPI.sig(jnp.real, Complex[5]),
  NumPyAPI.sig(jnp.reciprocal, Float[5]),
  NumPyAPI.sig(jnp.remainder, Float[5], Float[5]),
  NumPyAPI.sig(jnp.repeat, Float[5], repeats=np.array([2, 3, 1, 5, 4])),
  NumPyAPI.sig(jnp.reshape, Float[6], shape=(2, 3)),
  NumPyAPI.sig(jnp.resize, Float[6], new_shape=(2, 3)),
  NumPyAPI.sig(jnp.right_shift, Int[5], Int[5]),
  NumPyAPI.sig(jnp.rint, Float[5]),
  NumPyAPI.sig(jnp.roll, Float[5], Int[1]),
  NumPyAPI.sig(jnp.rollaxis, Float[5, 4], axis=1),
  NumPyAPI.sig(jnp.roots, Float[5]).with_skip_on_devices(['tpu']),
  NumPyAPI.sig(jnp.rot90, Float[5, 3]),
  NumPyAPI.sig(jnp.round, Float[5]),
  NumPyAPI.sig(jnp.searchsorted, Float[5], Float[5]),
  NumPyAPI.sig(jnp.select, [Bool[5], Bool[5]], [Float[5], Float[5]], Float[()]),
  NumPyAPI.sig(jnp.setdiff1d, Int[5], Int[5]),
  NumPyAPI.sig(jnp.setxor1d, Int[5], Int[5]),
  NumPyAPI.sig(jnp.shape, Float[5, 3]),
  NumPyAPI.sig(jnp.sign, Float[5]),
  NumPyAPI.sig(jnp.signbit, Float[5]),
  NumPyAPI.sig(jnp.sin, Float[5]),
  NumPyAPI.sig(jnp.sinc, Float[5]),
  NumPyAPI.sig(jnp.sinh, Float[5]),
  NumPyAPI.sig(jnp.size, Float[5]),
  NumPyAPI.sig(jnp.sort, Float[5]),
  NumPyAPI.sig(jnp.sort_complex, Complex[5]),
  NumPyAPI.sig(jnp.spacing, Float[5]),
  NumPyAPI.sig(jnp.split, Float[6], indices_or_sections=2),
  NumPyAPI.sig(jnp.sqrt, Float[5]),
  NumPyAPI.sig(jnp.square, Float[5]),
  NumPyAPI.sig(jnp.squeeze, Float[5]),
  NumPyAPI.sig(jnp.stack, [Float[2, 3], Float[2, 3]], axis=1),
  NumPyAPI.sig(jnp.std, Float[5]),
  NumPyAPI.sig(jnp.subtract, Float[5], Float[5]),
  NumPyAPI.sig(jnp.sum, Float[5]),
  NumPyAPI.sig(jnp.swapaxes, Float[3, 5], axis1=1, axis2=0),
  NumPyAPI.sig(jnp.take, Float[5], Int[2]),
  NumPyAPI.sig(jnp.take_along_axis, Float[5], Int[2], axis=0),
  NumPyAPI.sig(jnp.tan, Float[5]),
  NumPyAPI.sig(jnp.tanh, Float[5]),
  NumPyAPI.sig(jnp.tensordot, Float[2, 3, 4], Float[3, 4, 5]),
  NumPyAPI.sig(jnp.tile, Float[5], reps=(2,)),
  NumPyAPI.sig(jnp.trace, Float[5, 5]),
  NumPyAPI.sig(jnp.transpose, Float[5, 6]),
  NumPyAPI.sig(jnp.trapezoid, Float[5]),
  NumPyAPI.sig(jnp.tril, Float[5, 6]),
  NumPyAPI.sig(jnp.tril_indices_from, Float[5, 6]),
  NumPyAPI.sig(jnp.trim_zeros, Float[5]),
  NumPyAPI.sig(jnp.triu, Float[5, 6]),
  NumPyAPI.sig(jnp.triu_indices_from, Float[5, 6]),
  NumPyAPI.sig(jnp.true_divide, Float[5], Float[5]),
  NumPyAPI.sig(jnp.trunc, Float[5]),
  NumPyAPI.sig(jnp.union1d, Int[5], Int[5]),
  NumPyAPI.sig(jnp.unique, Int[10]),
  NumPyAPI.sig(jnp.unique_all, Int[10]),
  NumPyAPI.sig(jnp.unique_counts, Int[10]),
  NumPyAPI.sig(jnp.unique_inverse, Int[10]),
  NumPyAPI.sig(jnp.unique_values, Int[10]),
  NumPyAPI.sig(jnp.unpackbits, Uint8[8]),
  NumPyAPI.sig(jnp.unravel_index, Int[5], shape=(2, 3)),
  NumPyAPI.sig(jnp.unstack, Float[5]),
  NumPyAPI.sig(jnp.unwrap, Float[5]),
  NumPyAPI.sig(jnp.vander, Float[5]),
  NumPyAPI.sig(jnp.var, Float[5]),
  NumPyAPI.sig(jnp.vdot, Float[5], Float[5]),
  NumPyAPI.sig(jnp.vecdot, Float[5], Float[5]),
  NumPyAPI.sig(jnp.vecmat, Float[5], Float[5, 3]),
  NumPyAPI.sig(jnp.vsplit, Float[6], indices_or_sections=2),
  NumPyAPI.sig(jnp.vstack, [Float[5], Float[2, 5]]),
  NumPyAPI.sig(jnp.where, Bool[5], Float[5], Float[5]),
  NumPyAPI.sig(jnp.zeros_like, Float[5]),
]


class JaxArrayTests(jtu.JaxTestCase):
  @parameterized.named_parameters(
      {'testcase_name': api.name(), 'api': api} for api in NUMPY_APIS)
  def test_numpy_api_supports_jax_array(self, api):
    if api.skip_on_devices and jtu.test_device_matches(api.skip_on_devices):
      self.skipTest(f'{api.name()} not supported on {api.skip_on_devices}')
    fun = api.fun
    args = api.make_args(self.rng())
    wrapped_args = jax.tree.map(JaxArrayWrapper, args)
    kwargs = api.kwargs

    expected = fun(*args, **kwargs)
    wrapped = fun(*wrapped_args, **kwargs)

    self.assertAllClose(wrapped, expected, atol=0, rtol=0)

  @parameterized.named_parameters(
    {'testcase_name': func.__name__, 'func': func}
    for func in [jnp.zeros_like, jnp.ones_like, jnp.empty_like, jnp.full_like]
  )
  def test_array_creation_from_duck_typed_array(self, func):
    # Ensure that jnp.*_like prefers shape/dtype over __jax_array__ when
    # both methods are available.
    if func is jnp.full_like:
      func = functools.partial(func, fill_value=2.0)
    obj = DuckTypedArrayWithErroringJaxArray()

    # The test relies on this failing
    with self.assertRaises(ValueError):
      jnp.asarray(obj)

    result = func(obj)
    self.assertIsInstance(result, jax.Array)
    self.assertEqual(result.shape, obj.shape)
    self.assertEqual(result.dtype, obj.dtype)

  @parameterized.named_parameters(
      {"testcase_name": "subscript-form", "args": ("jk,k->j", Float[5, 3], Float[3])},
      {"testcase_name": "index-form", "args": (Float[5, 3], (0, 1), Float[3], (1,), (0,))},
  )
  def test_einsum(self, args):
    rng = jtu.rand_default(self.rng())
    def make_arg(arg):
      if isinstance(arg, jax.ShapeDtypeStruct):
        return rng(arg.shape, arg.dtype)
      return arg
    args = jax.tree.map(make_arg, args)

    def wrap_array(arg):
      if isinstance(arg, (jax.Array, np.ndarray)):
        return JaxArrayWrapper(arg)
      return arg
    wrapped_args = jax.tree.map(wrap_array, args)

    expected = jnp.einsum(*args)
    actual = jnp.einsum(*wrapped_args)

    self.assertAllClose(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
