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

from functools import partial
import os
import warnings

from absl import logging

from ..config import flags
from .. import util
from .. import dtypes
import numpy as onp  # 'onp' rather than 'np' to distinguish from autograd.numpy
import six
import threading

try:
  from . import tpu_client
except ImportError:
  tpu_client = None
from . import version
from . import xla_client

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


def get_compile_options(num_replicas=None, device_assignment=None):
  """Returns the compile options to use, as derived from flag values.

  Args:
    num_replicas: Optional int indicating the number of replicas for which to
      compile (default inherited from xla_client.CompileOptions).
    device_assignment: Optional tuple of integers indicating the assignment of
      logical replicas to physical devices (default inherited from
      xla_client.CompileOptions). Must be consistent with `num_replicas`.
  """
  compile_options = None
  if num_replicas is not None:
    compile_options = compile_options or xla_client.CompileOptions()
    compile_options.num_replicas = num_replicas
  if device_assignment is not None:
    logging.vlog(2, "get_compile_options: num_replicas=%s device_assignment=%s",
                 num_replicas, device_assignment)
    # NOTE(mattjj): xla_client.DeviceAssignment.create expects a 2D ndarray
    # indexed by replica number and computation per replica, respectively, while
    # here we currently assume only one computation per replica, hence the
    # second axis is always trivial.
    if num_replicas is not None and num_replicas != len(device_assignment):
      msg = "device_assignment does not match num_replicas: {} vs {}."
      raise ValueError(msg.format(device_assignment, num_replicas))
    compile_options = compile_options or xla_client.CompileOptions()
    device_assignment = onp.array(device_assignment)[:, None]
    device_assignment = xla_client.DeviceAssignment.create(device_assignment)
    assert num_replicas is None or device_assignment.replica_count() == num_replicas
    compile_options.device_assignment = device_assignment
  return compile_options

_backends = {}

def register_backend(name, factory):
  _backends[name] = factory

def _get_local_backend(platform=None):
  if not platform:
    platform = FLAGS.jax_platform_name

  # Canonicalize platform names.
  cpu = 'cpu'
  gpu = 'gpu'
  if platform == 'Host':
    platform = cpu
  elif platform == 'CUDA':
    platform = gpu
  elif platform == '':
    platform = None

  backend = xla_client.get_local_backend(platform)
  if backend is None:
    raise RuntimeError("No local XLA backends found.")

  if backend.platform == cpu and platform != cpu:
    warnings.warn('No GPU/TPU found, falling back to CPU.')

  return backend


def _get_tpu_driver_backend(platform):
  del platform
  backend_target = FLAGS.jax_backend_target
  if backend_target is None:
    raise ValueError('When using TPU Driver as the backend, you must specify '
                     '--jax_backend_target=<hostname>:8470.')
  return tpu_client.TpuBackend.create(worker=backend_target)


register_backend('xla', _get_local_backend)
if tpu_client:
  register_backend('tpu_driver', _get_tpu_driver_backend)

_backend_lock = threading.Lock()

@util.memoize
def get_backend(platform=None):
  with _backend_lock:
    backend = _backends.get(FLAGS.jax_xla_backend)
    if backend is None:
      msg = 'Unknown jax_xla_backend value "{}".'
      raise ValueError(msg.format(FLAGS.jax_xla_backend))
    return backend(platform)


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
  """Returns a list of all host IDs."""
  return list(set(d.host_id for d in devices(backend)))


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


class _JaxComputationBuilder(xla_client.ComputationBuilder):
  """Base class implementing all of JaxComputationBuilder.

  This class is intended to override and augment the interface of an XLA
  ComputationBuilder to form JaxComputationBuilder
  """

  # Method name case follows that of the XLA ComputationBuilder
  # pylint: disable=invalid-name

  def Build(self, *args, **kwargs):
    return super(_JaxComputationBuilder, self).Build(
        *args, **kwargs)

  def NumpyArrayConstant(self, value, canonicalize_types=True):
    if canonicalize_types:
      value = normalize_to_xla_dtypes(value)
    return super(_JaxComputationBuilder, self).Constant(value)

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

  # TODO(mattjj): remove when CrossReplicaSum is added to XLA:CPU
  def CrossReplicaSum(self, operand, replica_groups):
    """Workaround for CrossReplicaSum not being implemented on some backends."""
    if len(replica_groups[0]) == 1:
      return operand
    else:
      return super(_JaxComputationBuilder, self).CrossReplicaSum(
          operand, replica_groups)

  # TODO(mattjj): remove when AllToAll is added to XLA:CPU
  def AllToAll(self, operand, split_axis, concat_axis, replica_groups):
    """Workaround for AllToAll not being implemented on some backends."""
    if len(replica_groups[0]) == 1:
      return operand
    else:
      return super(_JaxComputationBuilder, self).AllToAll(
          operand, split_axis, concat_axis, replica_groups)


def make_computation_builder(name):
  return _JaxComputationBuilder(name)


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
                    onp.bool_, onp.longlong]:
  register_constant_handler(scalar_type, _scalar_constant_handler)

def _python_scalar_handler(dtype, c, val, canonicalize_dtypes=True):
  return c.NumpyArrayConstant(dtype.type(val))

for ptype, dtype in dtypes.python_scalar_dtypes.items():
  register_constant_handler(ptype, partial(_python_scalar_handler, dtype))
