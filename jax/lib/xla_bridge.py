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

from ..config import flags
import numpy as onp  # 'onp' rather than 'np' to distinguish from autograd.numpy

from jaxlib import xla_data_pb2
from jaxlib import xla_client


FLAGS = flags.FLAGS
flags.DEFINE_bool('jax_enable_x64', False, 'Enable 64-bit types to be used.')
flags.DEFINE_string('jax_dump_hlo_graph', None, 'Regexp of HLO graphs to dump.')
flags.DEFINE_bool('jax_hlo_profile', False, 'Enables HLO profiling mode.')
flags.DEFINE_string('jax_dump_hlo_unoptimized', None,
                    'Dirpath for unoptimized HLO dump.')
flags.DEFINE_string('jax_dump_hlo_optimized', None,
                    'Dirpath for optimized HLO dump.')
flags.DEFINE_string('jax_dump_hlo_per_pass', None,
                    'Dirpath for per-pass HLO dump.')
flags.DEFINE_integer('jax_replica_count', 1, 'Replica count for computations.')
flags.DEFINE_enum(
    'jax_xla_backend', 'xla', ['xla', 'xrt'],
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

# Prefix for HLO-dump flags indicating output should go into Sponge's
# visible output files.
_SPONGE_PREFIX = '/SPONGE/'


def _hlo_path(path, name):
  path = path.replace(_SPONGE_PREFIX,
                      os.getenv('TEST_UNDECLARED_OUTPUTS_DIR', ''))
  path = os.path.join(path, name)
  if not os.path.exists(path):
    os.mkdir(path)
  return path


def get_compile_options():
  """Returns the compile options to use, as derived from flag values."""
  compile_options = None
  if FLAGS.jax_dump_hlo_graph is not None:
    compile_options = get_xla_client().CompileOptions()
    compile_options.generate_hlo_graph = FLAGS.jax_dump_hlo_graph
  if FLAGS.jax_hlo_profile:
    compile_options = compile_options or get_xla_client().CompileOptions()
    compile_options.hlo_profile = True
  if FLAGS.jax_dump_hlo_unoptimized:
    compile_options = compile_options or get_xla_client().CompileOptions()
    path = _hlo_path(FLAGS.jax_dump_hlo_unoptimized, 'hlo_unoptimized')
    compile_options.dump_unoptimized_hlo_proto_to = path
  if FLAGS.jax_dump_hlo_optimized:
    compile_options = compile_options or get_xla_client().CompileOptions()
    path = _hlo_path(FLAGS.jax_dump_hlo_optimized, 'hlo_optimized')
    compile_options.dump_optimized_hlo_proto_to = path
  if FLAGS.jax_dump_hlo_per_pass:
    compile_options = compile_options or get_xla_client().CompileOptions()
    path = _hlo_path(FLAGS.jax_dump_hlo_per_pass, 'hlo_per_pass')
    compile_options.dump_per_pass_hlo_proto_to = path
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
  return _get_xla_client(FLAGS.jax_xla_backend,
                         FLAGS.jax_platform_name,
                         FLAGS.jax_replica_count)


def _get_xla_client(backend_name, platform_name, replica_count):
  """Configures and returns a handle to the XLA client.

  Args:
    backend_name: backend name, 'xla' or 'xrt'
    platform_name: platform name for XLA backend
    replica_count: number of computation replicas with which to configure the
      backend library.

  Returns:
    A client library module, or an object that behaves identically to one.
  """
  xla_client.initialize_replica_count(replica_count)
  if backend_name == 'xla':
    if platform_name:
      xla_client.initialize_platform_name(platform_name)
    else:
      try:
        xla_client.initialize_platform_name('CUDA')
      except RuntimeError:
        warnings.warn('No GPU found, falling back to CPU.')
        xla_client.initialize_platform_name('Host')
  return xla_client


def get_replica_count():
  return get_xla_client().get_replica_count()


_backend_flag_to_type = {
    'xla': xla_client.BackendType.XLA_LOCAL,
    'xrt': xla_client.BackendType.XRT,
}


@memoize_thunk
def _get_backend():
  return xla_client.BackendSpec(_backend_flag_to_type[FLAGS.jax_xla_backend],
                                FLAGS.jax_backend_target)


def device_put(pyval):
  # TODO(frostig): Accept a replica id for placement. For now, this places on
  # the first replica only.
  return get_xla_client().LocalBuffer.from_pyval(pyval, backend=_get_backend())


Shape = xla_client.Shape        # pylint: disable=invalid-name


### utility functions

# Similar or identical dtype-to-etype conversion tables exist in the XLA
# clients, but because their organization hasn't been made consistent across
# clients yet, we repeat the information here.
_etype_to_dtype = {
    xla_data_pb2.PRED: onp.dtype('bool'),
    xla_data_pb2.S8: onp.dtype('int8'),
    xla_data_pb2.S16: onp.dtype('int16'),
    xla_data_pb2.S32: onp.dtype('int32'),
    xla_data_pb2.S64: onp.dtype('int64'),
    xla_data_pb2.U8: onp.dtype('uint8'),
    xla_data_pb2.U16: onp.dtype('uint16'),
    xla_data_pb2.U32: onp.dtype('uint32'),
    xla_data_pb2.U64: onp.dtype('uint64'),
    xla_data_pb2.F16: onp.dtype('float16'),
    xla_data_pb2.F32: onp.dtype('float32'),
    xla_data_pb2.F64: onp.dtype('float64'),
    xla_data_pb2.C64: onp.dtype('complex64'),
}

# Note the conversion on the key. Numpy has a known issue wherein dtype hashing
# doesn't work as expected (https://github.com/numpy/numpy/issues/7242). Thus,
# when keying by dtype in this dict, we use the string form of dtypes.
_dtype_to_etype = {str(dt): et for et, dt in _etype_to_dtype.items()}


@memoize
def dtype_to_etype(dtype):
  """Convert from dtype to canonical etype (reading FLAGS.jax_enable_x64)."""
  return _dtype_to_etype[canonicalize_dtype(dtype)]


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
  return {canonicalize_dtype(dtype) for dtype in _etype_to_dtype.values()}


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

  def NumpyArrayConstant(self, value):
    normalized_value = normalize_to_xla_dtypes(value)
    return super(_JaxComputationBuilderBase, self).Constant(normalized_value)

  def ConstantLike(self, example_value, value):
    example_value = onp.asarray(example_value)
    return self.Constant(onp.array(value).astype(example_value.dtype))

  def Constant(self, py_val):
    """Translate constant `py_val` to a constant for this ComputationBuilder.

    Args:
      py_val: a Python value to be translated to a constant.

    Returns:
      A representation of the constant, either a ComputationDataHandle or None
    """
    py_type = type(py_val)
    if py_type in _constant_handlers:
      return _constant_handlers[py_type](self, py_val)
    else:
      raise TypeError("No constant handler for type: {}".format(py_type))


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


def _ndarray_constant_handler(c, val):
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
  if onp.any(onp.equal(0, val.strides)):
    zero_stride_axes, = onp.where(onp.equal(0, val.strides))
    other_axes, = onp.where(onp.not_equal(0, val.strides))
    collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None)
                              for ax in range(val.ndim))]
    xla_val = c.Broadcast(c.NumpyArrayConstant(collapsed_val),
                          onp.take(val.shape, zero_stride_axes))
    permutation = onp.argsort(tuple(zero_stride_axes) + tuple(other_axes))
    return c.Transpose(xla_val, permutation)
  else:
    return c.NumpyArrayConstant(val)
register_constant_handler(onp.ndarray, _ndarray_constant_handler)


for scalar_type in [onp.int8, onp.int16, onp.int32, onp.int64,
                    onp.uint8, onp.uint16, onp.uint32, onp.uint64,
                    onp.float16, onp.float32, onp.float64, onp.float128,
                    float, int, bool, onp.bool_]:
  register_constant_handler(scalar_type,
                            lambda c, val: c.NumpyArrayConstant(val))
