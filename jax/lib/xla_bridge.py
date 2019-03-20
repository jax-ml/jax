# Copyright 2018 Google LLC
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

"""Interface and utility functions to XLA.

This module wraps the XLA client(s) and builders to standardize their interfaces
and provide some automatic type mapping logic for converting between Numpy and
XLA. There are also a handful of related casting utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
from distutils.util import strtobool

from ..config import flags
import numpy as onp  # 'onp' rather than 'np' to distinguish from autograd.numpy

import jaxlib

# Check the jaxlib version before importing anything else from jaxlib.
def _check_jaxlib_version():
  minimum_version = (0, 1, 9)
  if hasattr(jaxlib, '__version__'):
    version = tuple(int(x) for x in jaxlib.__version__.split('.'))
  else:
    version = (0, 1, 9)  # The version before jaxlib.__version__ was added.
  if version < minimum_version:
    msg = 'jaxlib is version {}, but this version of jax requires version {}.'
    raise ValueError(msg.format('.'.join(map(str, version)),
                                '.'.join(map(str, minimum_version))))

_check_jaxlib_version()


from jaxlib import xla_client
from jaxlib import xla_data_pb2


FLAGS = flags.FLAGS
flags.DEFINE_bool('jax_enable_x64',
                  strtobool(os.getenv('JAX_ENABLE_X64', "False")),
                  'Enable 64-bit types to be used.')
flags.DEFINE_string(
    'jax_xla_backend', 'xla',
    'Either "xla" for the XLA service directly, or "xrt" for an XRT backend.')
flags.DEFINE_string(
    'jax_backend_target', 'local',
    'Either "local" or "rpc:address" to connect to a remote service target.')
flags.DEFINE_string(
    'jax_platform_name', '',
    'Platform name for XLA. The default is to attempt to use a '
    'GPU if available, but fall back to CPU otherwise. To set '
    'the platform manually, pass "Host" for CPU or "CUDA" for '
    'GPU.')


_platform_name = None  # set to the active platform name


def get_compile_options(num_replicas=None):
  """Returns the compile options to use, as derived from flag values."""
  compile_options = None
  if num_replicas is not None:
    compile_options = compile_options or get_xla_client().CompileOptions()
    compile_options.num_replicas = num_replicas
  return compile_options


def memoize(func):
  class memodict(dict):
    def __missing__(self, key):
      val = self[key] = func(key)
      return val
  return memodict().__getitem__


def memoize_thunk(func):
  cached = []
  return lambda: cached[0] if cached else (cached.append(func()) or cached[0])


@memoize_thunk
def get_xla_client():
  return _get_xla_client(FLAGS.jax_xla_backend, FLAGS.jax_platform_name)


def _get_xla_client(backend_name, platform_name):
  """Configures and returns a handle to the XLA client.

  Args:
    backend_name: backend name, 'xla' or 'xrt'
    platform_name: platform name for XLA backend

  Returns:
    A client library module, or an object that behaves identically to one.
  """
  global _platform_name
  if backend_name == 'xla':
    if platform_name:
      xla_client.initialize_platform_name(platform_name)
      _platform_name = platform_name
    else:
      try:
        xla_client.initialize_platform_name('CUDA')
        _platform_name = 'CUDA'
      except RuntimeError:
        warnings.warn('No GPU found, falling back to CPU.')
        xla_client.initialize_platform_name('Host')
        _platform_name = 'Host'
  return xla_client


_backends = {}

def register_backend(name, factory):
  _backends[name] = factory


if hasattr(xla_client, 'XlaLocalBackend'):
  register_backend('xla', lambda: xla_client.XlaLocalBackend())
  register_backend('xrt',
                   lambda: xla_client.XrtBackend(FLAGS.jax_backend_target))
else:
  # TODO(phawkins): this case is for cross-version compatibility. Delete this
  # case after a Jaxlib update.
  register_backend(
    'xla', lambda: xla_client.BackendSpec(xla_client.BackendType.XLA_LOCAL, ''))
  register_backend(
    'xrt', lambda: xla_client.BackendSpec(xla_client.BackendType.XRT,
                                          FLAGS.jax_backend_target))


@memoize_thunk
def _get_backend():
  backend = _backends.get(FLAGS.jax_xla_backend)
  if backend is None:
    msg = 'Unknown jax_xla_backend value "{}".'
    raise ValueError(msg.format(FLAGS.jax_xla_backend))
  return backend()


def device_count():
  _ = get_xla_client()  # ensure initialize_platform_name is called
  return _get_backend().device_count()


def device_put(pyval, device_num=0):
  client = get_xla_client()
  return client.LocalBuffer.from_pyval(pyval, device_num, backend=_get_backend())


Shape = xla_client.Shape        # pylint: disable=invalid-name


### utility functions

@memoize
def dtype_to_etype(dtype):
  """Convert from dtype to canonical etype (reading FLAGS.jax_enable_x64)."""
  return xla_client.DTYPE_TO_XLA_ELEMENT_TYPE[canonicalize_dtype(dtype)]

@memoize
def dtype_to_etype_exact(dtype):
  """Convert from dtype to exact etype (ignoring FLAGS.jax_enable_x64)."""
  return xla_client.dtype_to_etype(dtype)


_dtype_to_32bit_dtype = {
    str(onp.dtype('int64')): onp.dtype('int32'),
    str(onp.dtype('uint64')): onp.dtype('uint32'),
    str(onp.dtype('float64')): onp.dtype('float32'),
    str(onp.dtype('complex128')): onp.dtype('complex64'),
}


@memoize
def canonicalize_dtype(dtype):
  """Convert from a dtype to a canonical dtype based on FLAGS.jax_enable_x64."""
  dtype = onp.dtype(dtype)

  if FLAGS.jax_enable_x64:
    return str(dtype)
  else:
    return str(_dtype_to_32bit_dtype.get(str(dtype), dtype))


@memoize_thunk
def supported_numpy_dtypes():
  return {canonicalize_dtype(dtype)
          for dtype in xla_client.XLA_ELEMENT_TYPE_TO_DTYPE.values()}


def canonicalize_shape(shape):
  """Given an xla_client.Shape, return a new instance with canonical dtypes."""
  if shape.is_tuple():
    return Shape.tuple_shape(tuple(
        canonicalize_shape(s) for s in shape.tuple_shapes()))
  else:
    return Shape.array_shape(
        canonicalize_dtype(shape.element_type()),
        shape.dimensions(), shape.minor_to_major())


# TODO(mattjj,frostig): try to remove this function
def normalize_to_xla_dtypes(val):
  """Normalize dtypes in a value."""
  if hasattr(val, '__array__') or onp.isscalar(val):
    return onp.asarray(val, dtype=canonicalize_dtype(onp.result_type(val)))
  elif isinstance(val, (tuple, list)):
    return tuple(normalize_to_xla_dtypes(x) for x in val)
  raise TypeError('Can\'t convert to XLA: {}'.format(val))


# TODO(mattjj,frostig): try to remove this function
def shape_of(value):
  """Given a Python or XLA value, return its canonicalized XLA Shape."""
  if hasattr(value, 'shape') and hasattr(value, 'dtype'):
    return Shape.array_shape(canonicalize_dtype(value.dtype), value.shape)
  elif onp.isscalar(value):
    return shape_of(onp.asarray(value))
  elif isinstance(value, (tuple, list)):
    return Shape.tuple_shape(tuple(shape_of(elt) for elt in value))
  else:
    raise TypeError('Unexpected type: {}'.format(type(value)))


def infeed_put(replica_id, pyval):
  pyval = normalize_to_xla_dtypes(pyval)
  return get_xla_client().transfer_to_infeed(
      pyval, replica_number=replica_id)


class _JaxComputationBuilderBase(object):
  """Base class implementing all of JaxComputationBuilder.

  This class is intended to override and augment the interface of an XLA
  ComputationBuilder to form JaxComputationBuilder, as made clear by
  `get_jax_computation_builder_class`, which relies on Python's
  method-resolution order to set up inheritance-like behavior. The class
  inheritance setup is deferred because the choice of the XLA ComputationBuilder
  class is based on the result of `get_xla_client()`. That is, the choice is
  based at least on the setting of flags, which are available only after module
  initialization time.
  """
  # The JAXComputationBuilder is implemented using subclassing and inheritance
  # (via this base class), rather than a wrap-and-delegate style, simply to
  # avoid having to spell out all the methods to be forwarded to a wrapped
  # ComputationBuilder, especially since the underlying ComputationBuilders are
  # likely to be revised in the future. An alternative is to generate these
  # forwarding methods programmatically.

  # Method name case follows that of the XLA ComputationBuilder
  # pylint: disable=invalid-name

  def Build(self, *args, **kwargs):
    return super(_JaxComputationBuilderBase, self).Build(
        *args, backend=_get_backend(), **kwargs)

  def Parameter(self, value, name=None, parameter_num=None):
    return super(_JaxComputationBuilderBase, self).ParameterWithShape(
        shape_of(value), name=name, parameter_num=parameter_num)

  def NumpyArrayConstant(self, value, canonicalize_types=True):
    if canonicalize_types:
      value = normalize_to_xla_dtypes(value)
    return super(_JaxComputationBuilderBase, self).Constant(value)

  def ConstantLike(self, example_value, value, canonicalize_types=True):
    example_value = onp.asarray(example_value)
    return self.Constant(onp.array(value, dtype=example_value.dtype))

  def Constant(self, py_val, canonicalize_types=True):
    """Translate constant `py_val` to a constant for this ComputationBuilder.

    Args:
      py_val: a Python value to be translated to a constant.

    Returns:
      A representation of the constant, either a ComputationDataHandle or None
    """
    py_type = type(py_val)
    if py_type in _constant_handlers:
      return _constant_handlers[py_type](self, py_val, canonicalize_types)
    else:
      raise TypeError("No constant handler for type: {}".format(py_type))

  def AllToAll(self, operand, split_dimension, concat_dimension, replica_groups):
    """Workaround for AllToAll not being implemented on some backends."""
    if split_dimension == concat_dimension and len(replica_groups[0]) == 1:
      return operand
    else:
      return super(_JaxComputationBuilderBase, self).AllToAll(
          operand, split_dimension, concat_dimension, replica_groups)


@memoize_thunk
def get_jax_computation_builder_class():
  xla_base = get_xla_client().ComputationBuilder
  jax_base = _JaxComputationBuilderBase
  return type('JaxComputationBuilder', (jax_base, xla_base), {})


def make_computation_builder(name):
  return get_jax_computation_builder_class()(name)


def register_constant_handler(type_, handler_fun):
  _constant_handlers[type_] = handler_fun
_constant_handlers = {}


def _ndarray_constant_handler(c, val, canonicalize_types=True):
  """Constant handler for ndarray literals, handling zero-size strides.

  This function essentially calls c.NumpyArrayConstant(val) except it has
  special handling of arrays with any strides of size zero: for those, it
  generates appropriate calls to NumpyArrayConstant, Broadcast, and Transpose
  to avoid staging in large literals that might arise from np.zeros or np.ones
  or the output of lax.broadcast (which uses onp.broadcast_to which in turn
  uses size-zero strides).

  Args:
    c: XLA client ComputationBuilder.
    val: an ndarray.

  Returns:
    An XLA ComputationDataHandle / XlaOp representing the constant ndarray
    staged into the XLA Computation.
  """
  # TODO(mattjj): revise this to use c.BroadcastInDim rather than Transpose
  if onp.any(onp.equal(0, val.strides)) and val.size > 0:
    zero_stride_axes, = onp.where(onp.equal(0, val.strides))
    other_axes, = onp.where(onp.not_equal(0, val.strides))
    collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None)
                              for ax in range(val.ndim))]
    xla_val = c.Broadcast(
        c.NumpyArrayConstant(collapsed_val, canonicalize_types),
        onp.take(val.shape, zero_stride_axes))
    permutation = onp.argsort(tuple(zero_stride_axes) + tuple(other_axes))
    return c.Transpose(xla_val, permutation)
  else:
    return c.NumpyArrayConstant(val, canonicalize_types)
register_constant_handler(onp.ndarray, _ndarray_constant_handler)


def _scalar_constant_handler(c, val, canonicalize_types=True):
  return c.NumpyArrayConstant(val, canonicalize_types)

for scalar_type in [onp.int8, onp.int16, onp.int32, onp.int64,
                    onp.uint8, onp.uint16, onp.uint32, onp.uint64,
                    onp.float16, onp.float32, onp.float64, onp.float128,
                    float, int, bool, onp.bool_]:
  register_constant_handler(scalar_type, _scalar_constant_handler)
