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


from functools import partial
import os
from typing import Callable, Dict
import warnings

from absl import logging
import numpy as np

from ..config import flags
from .. import util
from .. import dtypes
import numpy as onp  # 'onp' rather than 'np' to distinguish from autograd.numpy
import threading

try:
  from . import tpu_client
except ImportError:
  tpu_client = None
from . import version
from . import xla_client

xops = xla_client.ops

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'jax_xla_backend', 'xla',
    'Default is "xla" for the XLA service directly, '
    'or "tpu_driver" for using high-performance access to Cloud TPU hardware.')
flags.DEFINE_string(
    'jax_backend_target', 'local',
    'Either "local" or "rpc:address" to connect to a remote service target.')
flags.DEFINE_string(
    'jax_platform_name',
    os.getenv('JAX_PLATFORM_NAME', ''),
    'Platform name for XLA. The default is to attempt to use a GPU if '
    'available, but fall back to CPU otherwise. To set the platform manually, '
    'pass "cpu" for CPU or "gpu" for GPU.')


def get_compile_options(num_replicas, num_partitions, device_assignment=None):
  """Returns the compile options to use, as derived from flag values.

  Args:
    num_replicas: int indicating the number of replicas for which to compile.
    num_partitions: int indicating the number of partitions for which to compile.
    device_assignment: Optional tuple of integers indicating the assignment of
      logical replicas to physical devices (default inherited from
      xla_client.CompileOptions). Must be consistent with `num_replicas` and
      `num_partitions`.
  """
  compile_options = xla_client.CompileOptions()
  compile_options.num_replicas = num_replicas
  compile_options.num_partitions = num_partitions
  if device_assignment is not None:
    logging.vlog(
        2,
        'get_compile_options: num_replicas=%s num_partitions=%s device_assignment=%s',
        num_replicas, num_partitions, device_assignment)
    device_assignment = onp.array(device_assignment)

    # Allow 1D device assignment if num_partitions is 1.
    if (device_assignment.ndim == 1) and (num_partitions == 1):
      device_assignment = device_assignment[:, None]

    if num_replicas != device_assignment.shape[0]:
      msg = 'device_assignment does not match num_replicas: {} vs {}.'
      raise ValueError(msg.format(device_assignment, num_replicas))

    if num_partitions != device_assignment.shape[1]:
      msg = 'device_assignment does not match num_partitions: {} vs {}.'
      raise ValueError(msg.format(device_assignment, num_partitions))

    device_assignment = xla_client.DeviceAssignment.create(device_assignment)
    assert device_assignment.replica_count() == num_replicas
    assert device_assignment.computation_count() == num_partitions
    compile_options.device_assignment = device_assignment
  return compile_options

_backends = {}

def register_backend(name, factory):
  _backends[name] = factory

def _get_local_backend(platform=None):
  if not platform:
    platform = FLAGS.jax_platform_name

  backend = xla_client.get_local_backend(platform)
  if backend is None:
    raise RuntimeError("No local XLA backends found.")

  if backend.platform == cpu and platform != cpu:
    warnings.warn('No GPU/TPU found, falling back to CPU.')

  return backend


register_backend('xla', _get_local_backend)

# memoize the TPU driver to be consistent with xla_client behavior
_tpu_backend = None

def _get_tpu_driver_backend(platform):
  del platform
  global _tpu_backend
  if _tpu_backend is None:
    backend_target = FLAGS.jax_backend_target
    if backend_target is None:
      raise ValueError('When using TPU Driver as the backend, you must specify '
                       '--jax_backend_target=<hostname>:8470.')
    _tpu_backend = tpu_client.TpuBackend.create(worker=backend_target)
  return _tpu_backend


if tpu_client:
  register_backend('tpu_driver', _get_tpu_driver_backend)


_backend_lock = threading.Lock()

@util.memoize
def get_backend(platform=None):
  # TODO(mattjj,skyewm): remove this input polymorphism after we clean up how
  # 'backend' values are handled
  if isinstance(platform, xla_client.Backend):
    return platform

  if platform == '':
    platform = None

  with _backend_lock:
    backend = _backends.get(FLAGS.jax_xla_backend)
    if backend is None:
      msg = 'Unknown jax_xla_backend value "{}".'
      raise ValueError(msg.format(FLAGS.jax_xla_backend))
    return backend(platform)


def get_device_backend(device=None):
  """Returns the Backend associated with `device`, or the default Backend."""
  platform = device.platform if device else None
  return get_backend(platform)


def device_count(backend=None):
  """Returns the total number of devices.

  On most platforms, this is the same as ``local_device_count()``. However, on
  multi-host platforms, this will return the total number of devices across all
  hosts.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend. 'cpu', 'gpu', or 'tpu'.

  Returns:
    Number of devices.
  """
  return int(get_backend(backend).device_count())


def local_device_count(backend=None):
  """Returns the number of devices on this host."""
  return int(get_backend(backend).local_device_count())


def devices(backend=None):
  """Returns a list of all devices.

  Each device is represented by a subclass of Device (e.g. CpuDevice,
  GpuDevice). The length of the returned list is equal to
  ``device_count()``. Local devices can be identified by comparing
  ``Device.host_id`` to ``host_id()``.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend. 'cpu', 'gpu', or 'tpu'.

  Returns:
    List of Device subclasses.
  """
  return get_backend(backend).devices()


def local_devices(host_id=None, backend=None):
  """Returns a list of devices local to a given host (this host by default)."""
  if host_id is None:
    host_id = get_backend(backend).host_id()
  return [d for d in devices(backend) if d.host_id == host_id]


def host_id(backend=None):
  """Returns the integer host ID of this host.

  On most platforms, this will always be 0. This will vary on multi-host
  platforms though.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend. 'cpu', 'gpu', or 'tpu'.

  Returns:
    Integer host ID.
  """
  return get_backend(backend).host_id()


def host_ids(backend=None):
  """Returns a sorted list of all host IDs."""
  return sorted(list(set(d.host_id for d in devices(backend))))


def host_count(backend=None):
  """Returns the number of hosts."""
  return len(host_ids(backend))


### utility functions

@util.memoize
def dtype_to_etype(dtype):
  """Convert from dtype to canonical etype (reading FLAGS.jax_enable_x64)."""
  return xla_client.dtype_to_etype(dtypes.canonicalize_dtype(dtype))


@util.memoize
def supported_numpy_dtypes():
  return {dtypes.canonicalize_dtype(dtype)
          for dtype in xla_client.XLA_ELEMENT_TYPE_TO_DTYPE.values()}


# TODO(mattjj,frostig): try to remove this function
def normalize_to_xla_dtypes(val):
  """Normalize dtypes in a value."""
  if hasattr(val, '__array__') or onp.isscalar(val):
    return onp.asarray(val,
                       dtype=dtypes.canonicalize_dtype(dtypes.result_type(val)))
  elif isinstance(val, (tuple, list)):
    return tuple(normalize_to_xla_dtypes(x) for x in val)
  raise TypeError('Can\'t convert to XLA: {}'.format(val))

def _numpy_array_constant(builder, value, canonicalize_types=True):
  if canonicalize_types:
    value = normalize_to_xla_dtypes(value)
  return xops.ConstantLiteral(builder, value)

def parameter(builder, num, shape, name=None, replicated=False):
  if name is None:
    name = ''
  if isinstance(replicated, bool):
    replicated = [replicated] * shape.leaf_count()

  return xops.Parameter(builder, num,
                        shape.with_major_to_minor_layout_if_absent(), name,
                        replicated)


def constant(builder, py_val, canonicalize_types=True):
  """Translate constant `py_val` to a constant, canonicalizing its dtype.

  Args:
    py_val: a Python value to be translated to a constant.

  Returns:
    A representation of the constant, either a ComputationDataHandle or None
  """
  py_type = type(py_val)
  if py_type in _constant_handlers:
    return _constant_handlers[py_type](builder, py_val, canonicalize_types)
  else:
    raise TypeError("No constant handler for type: {}".format(py_type))

def make_computation_builder(name):
  return xla_client.XlaBuilder(name)


def register_constant_handler(type_, handler_fun):
  _constant_handlers[type_] = handler_fun
_constant_handlers: Dict[type, Callable] = {}


def _ndarray_constant_handler(c, val, canonicalize_types=True):
  """Constant handler for ndarray literals, handling zero-size strides.

  This function essentially calls _numpy_array_constant(val) except it has
  special handling of arrays with any strides of size zero: for those, it
  generates appropriate calls to NumpyArrayConstant, Broadcast, and Transpose
  to avoid staging in large literals that might arise from np.zeros or np.ones
  or the output of lax.broadcast (which uses onp.broadcast_to which in turn
  uses size-zero strides).

  Args:
    c: an XlaBuilder
    val: an ndarray.

  Returns:
    An XLA ComputationDataHandle / XlaOp representing the constant ndarray
    staged into the XLA Computation.
  """
  # TODO(mattjj): revise this to use xops.BroadcastInDim rather than Transpose
  if onp.any(onp.equal(0, val.strides)) and val.size > 0:
    zero_stride_axes, = onp.where(onp.equal(0, val.strides))
    other_axes, = onp.where(onp.not_equal(0, val.strides))
    collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None)
                              for ax in range(val.ndim))]
    xla_val = xops.Broadcast(
        _numpy_array_constant(c, collapsed_val, canonicalize_types),
        onp.take(val.shape, zero_stride_axes))
    permutation = onp.argsort(tuple(zero_stride_axes) + tuple(other_axes))
    return xops.Transpose(xla_val, permutation)
  else:
    return _numpy_array_constant(c, val, canonicalize_types)
register_constant_handler(onp.ndarray, _ndarray_constant_handler)


def _scalar_constant_handler(c, val, canonicalize_types=True):
  return _numpy_array_constant(c, val, canonicalize_types)

for scalar_type in [onp.int8, onp.int16, onp.int32, onp.int64,
                    onp.uint8, onp.uint16, onp.uint32, onp.uint64,
                    onp.float16, onp.float32, onp.float64, onp.float128,
                    onp.bool_, onp.longlong]:
  register_constant_handler(scalar_type, _scalar_constant_handler)

def _python_scalar_handler(dtype, c, val, canonicalize_dtypes=True):
  return _numpy_array_constant(c, dtype.type(val))

for ptype, dtype in dtypes.python_scalar_dtypes.items():
  register_constant_handler(ptype, partial(_python_scalar_handler, dtype))


# Backwards-compatibility shim that provides a ComputationBuilder-like API to
# older jaxlib releases.
# TODO(phawkins): Remove when the minimum Jaxlib release is 0.1.46 or newer.
class ComputationBuilderShim(object):
  def __init__(self, builder):
    self._builder = builder

  def GetShape(self, op):
    return self._builder.GetShape(op)

  def Constant(self, value):
    return xops.Constant(self._builder, value)

  def ConstantS32Scalar(self, value):
    return xops.Constant(self._builder, np.int32(value))

  def CustomCallWithLayout(self,
                           call_target_name,
                           operands,
                           shape_with_layout,
                           operand_shapes_with_layout,
                           opaque=None):
    return xops.CustomCallWithLayout(self._builder, call_target_name,
                                     list(operands), shape_with_layout,
                                     list(operand_shapes_with_layout),
                                     opaque or b'')

  def GetTupleElement(self, op, i):
    return xops.GetTupleElement(op, i)

  def Complex(self, re, im):
    return xops.Complex(re, im)

  def Transpose(self, op, perm):
    return xops.Transpose(op, perm)

  def Conj(self, op):
    return xops.Conj(op)

  def Slice(self, operand, start_indices, limit_indices, strides=None):
    if strides is None:
      start_indices = list(start_indices)
      strides = [1] * len(start_indices)
    return xops.Slice(operand, start_indices, limit_indices, strides)


def computation_builder_shim(b):
  return b if version > (0, 1, 45) else ComputationBuilderShim(b)
