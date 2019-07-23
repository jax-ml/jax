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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, defaultdict
from distutils.util import strtobool
import itertools as it
import operator as op
import os

import numpy as onp
import six
from six.moves import xrange

from ..config import flags
from .. import core
from .. import ad_util
from .. import tree_util
from .. import linear_util as lu
from ..abstract_arrays import (ConcreteArray, ShapedArray, make_shaped_array,
                               array_types)
from ..core import AbstractTuple, JaxTuple, pack, valid_jaxtype, Literal
from ..util import partial, partialmethod, memoize, concatenate, safe_map, prod
from ..lib import xla_bridge as xb
from . import partial_eval as pe
from . import ad

FLAGS = flags.FLAGS
flags.DEFINE_bool('jax_device_values',
                  strtobool(os.getenv('JAX_DEVICE_VALUES', "True")),
                  'Enable device-persistent values.')
flags.DEFINE_bool('jax_debug_nans',
                  strtobool(os.getenv('JAX_DEBUG_NANS', "False")),
                  'Add nan checks to every operation.')

def apply_primitive(prim, *args, **params):
  """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
  abstract_args = map(abstractify, args)
  compiled_fun = _xla_primitive_callable(prim, *abstract_args, **params)
  return compiled_fun(*args)

@memoize
def _xla_primitive_callable(prim, *abstract_args, **params):
  shapes = tuple(map(xla_shape, abstract_args))
  built_c = primitive_computation(prim, *shapes, **params)
  result_shape = xla_shape_to_result_shape(built_c.GetReturnValueShape())
  handle_result = result_handler(result_shape)
  compiled = built_c.Compile(shapes, xb.get_compile_options(),
                             backend=xb.get_backend())
  return partial(_execute_compiled_primitive, prim.name, compiled, handle_result)

def xla_shape(x):
  try:
    return xb.Shape.array_shape(x.dtype, x.shape)
  except AttributeError:
    if type(x) in (core.AbstractTuple, core.JaxTuple):
      return xb.Shape.tuple_shape(tuple(map(xla_shape, x)))
    else:
      raise TypeError(type(x))

@memoize
def primitive_computation(prim, *shapes, **params):
  """Builds an XLA computation for `prim` with argument `shapes`."""
  c = xb.make_computation_builder("primitive_computation")
  platform = xb.get_backend().platform
  xla_args = map(c.ParameterWithShape, shapes)
  try:
    rule = backend_specific_translations[platform].get(prim) or translations[prim]
  except KeyError:
    raise NotImplementedError("XLA translation rule for {} not found".format(prim))
  rule(c, *xla_args, **params)  # return val set as a side-effect on c
  try:
    return c.Build()
  except RuntimeError as e:
    # try for a better error message by using the abstract_eval checks
    prim.abstract_eval(*map(_aval_from_xla_shape, shapes), **params)
    raise e

def _aval_from_xla_shape(shape):
  if shape.is_tuple():
    return AbstractTuple(map(_aval_from_xla_shape, shape.tuple_shapes()))
  else:
    return ShapedArray(shape.dimensions(), shape.element_type())

def _execute_compiled_primitive(name, compiled, result_handler, *args):
  device_num, = compiled.DeviceOrdinals()
  input_bufs = [device_put(x, device_num) for x in args]
  out_buf = compiled.Execute(input_bufs)
  check_nans(name, out_buf)
  return result_handler(out_buf)

def check_nans(name, buf):
  FLAGS.jax_debug_nans and _check_nans(name, buf.shape(), buf)

def _check_nans(name, xla_shape, buf):
  if xla_shape.is_tuple():
    _map(partial(_check_nans, name), xla_shape.tuple_shapes(), buf.destructure())
  else:
    if onp.issubdtype(xla_shape.element_type(), onp.floating):
      pyval = buf.to_py()
      if onp.any(onp.isnan(pyval)):
        msg = "invalid value (nan) encountered in {}"
        raise FloatingPointError(msg.format(name))

def device_put(x, device_num=0):
  """Place a Python value `x` on device number `device_num`.

  This is a wrapper around jax.lib.xla_bridge.device_put to handle
  additional Python types, namely
    1. the array-like types DeviceArray (which is already backed by device
    memory, though may be on the wrong device) and its subclass DeviceConstant
    (which represents a lazy value to be instantiated), and
    2. the tuple-like types DeviceTuple (which is already backed by device
    memory, though may be on the wrong device) and JaxTuple (which may have some
    elements that are backed by device memory on the correct device).
  In particular, this function avoids transferring data already placed on the
  correct device, and handles instantiating DeviceConstants.

  Args:
    x: a tuplelike-tree with arraylike leaves representing the value to be
      transferred to the device, where tuplelike means a JaxTuple or
      DeviceTuple, and arraylike includes DeviceArray, DeviceConstant, and
      anything that has an '__array__' attr.
    device_num: an int representing the target physical device number.

  Returns:
    A buffer representing the input `x` placed on the appropriate device.
  """
  x = _canonicalize_pyval_dtype(x)
  t = type(x)
  if t is DeviceArray or t is DeviceTuple:
    if x.device_buffer.device() == device_num:
      return x.device_buffer
    else:
      # TODO(phawkins): remove after the minimum Jaxlib version is raised to
      # 0.1.22
      if hasattr(x.device_buffer, 'copy_to_device'):
        return x.device_buffer.copy_to_device(device_num)
      else:
        return device_put(x.device_buffer.to_py(), device_num)
  elif isinstance(x, DeviceConstant):
    return _instantiate_device_constant(x, device_num=device_num)
  elif isinstance(x, (DeviceArray, onp.ndarray)):
    return xb.device_put(x, device_num)  # handle arraylikes
  elif isinstance(x, JaxTuple):
    element_bufs = tuple(map(partial(device_put, device_num=device_num), x))
    return xb.make_tuple(element_bufs, device_num)
  else:
    raise TypeError(t)


# When we execute an XLA computation, we get a raw device buffer back and need
# to package it into a suitable Python object to return to the user. To avoid
# unnecessary device-to-host transfers, we typically return a DeviceValue that
# acts just like a familiar Python type (e.g. an ndarray or JaxTuple) but is
# lazy in that it only copies data back to the host as required. Since the same
# DeviceValue type is formed on every execution of a compiled computation, at
# compile time we set up result handler functions and thus avoid redoing some of
# the Python bookkeeping work on every execution. Since XLA shapes are slower to
# manipulate than simple Python builtins, we store the metadata required for
# forming the DeviceValue result in special _ResultArray / _ResultTuple classes.

# Every JaxType needs to map to an XLA type. However this function's design is
# based on the assumption that XLA types can be mapped uniquely back to a
# JaxType, i.e. that the mapping is bijective. That assumption could be relaxed,
# but it would mean we need to do a bit more bookkeping on the Python side to
# track abstract values of outputs.
def xla_shape_to_result_shape(xla_shape):
  if xla_shape.is_tuple():
    aval = _aval_from_xla_shape(xla_shape)
    result_shapes = tuple(map(xla_shape_to_result_shape, xla_shape.tuple_shapes()))
    return _ResultTuple((aval, result_shapes))
  else:
    shape, dtype = xla_shape.dimensions(), xla_shape.element_type()
    ndim, size = len(shape), prod(shape)
    return _ResultArray((shape, dtype, ndim, size))

class _ResultTuple(tuple): pass
class _ResultArray(tuple): pass

def result_handler(result_shape):
  if FLAGS.jax_device_values:
    return _device_persistent_result_handler(result_shape)
  else:
    return _pyval_result_handler(result_shape)

def _device_persistent_result_handler(result_shape):
  t = type(result_shape)
  if t is _ResultArray:
    return partial(DeviceArray, result_shape)
  elif t is _ResultTuple:
    return partial(DeviceTuple, result_shape)
  else:
    raise TypeError(t)

def _pyval_result_handler(result_shape):
  del result_shape
  def _tuple_to_jaxtuple(v):
    if isinstance(v, tuple):
      return JaxTuple(_tuple_to_jaxtuple(t) for t in v)
    return v
  def f(buf):
    return _tuple_to_jaxtuple(buf.to_py())
  return f

def _compile_jaxpr(jaxpr, device_assignment, axis_env, const_vals, *abstract_args):
  if axis_env.nreps > xb.device_count():
    msg = ("compiling computation that requires {} replicas, but only {} XLA "
           "devices are available")
    raise ValueErrr(msg.format(axis_env.nreps, xb.device_count()))
  arg_shapes = list(map(xla_shape, abstract_args))
  built_c = _jaxpr_computation(jaxpr, axis_env, const_vals, (), *arg_shapes)
  result_shape = xla_shape_to_result_shape(built_c.GetReturnValueShape())
  compile_opts = xb.get_compile_options(num_replicas=axis_env.nreps,
                                        device_assignment=device_assignment)
  compiled_c = built_c.Compile(arg_shapes, compile_opts, backend=xb.get_backend())
  return compiled_c, result_shape

def build_jaxpr(jaxpr, axis_env, const_vals, *abstract_args):
  arg_shapes = list(map(xla_shape, abstract_args))
  built_c = _jaxpr_computation(jaxpr, axis_env, const_vals, (), *arg_shapes)
  return built_c


def _prefetch_jaxpr_literals(jaxpr):
  """Prefetches any DeviceArray values inside a jaxpr to the host."""
  for eqn in jaxpr.eqns:
    for v in eqn.invars:
      if type(v) is core.Literal and isinstance(v.val, DeviceArray):
        v.val.copy_to_host_async()
    if eqn.bound_subjaxprs:
      for subjaxpr, _const_bindings, _freevar_bindings in eqn.bound_subjaxprs:
        _prefetch_jaxpr_literals(subjaxpr)


def jaxpr_computation(jaxpr, const_vals, freevar_shapes, *arg_shapes):
  axis_env = AxisEnv(1, [], [])
  return _jaxpr_computation(jaxpr, axis_env, const_vals, freevar_shapes, *arg_shapes)

def _jaxpr_computation(jaxpr, axis_env, const_vals, freevar_shapes, *arg_shapes):
  assert not any(type(invar) in (tuple, list) for invar in jaxpr.invars)
  c = xb.make_computation_builder("jaxpr_computation")
  platform = xb.get_backend().platform

  def read(v):
    if type(v) is Literal:
      return c.Constant(_canonicalize_pyval_dtype(v.val))
    else:
      return env[v]

  def write(v, node):
    assert node is not None
    env[v] = node

  env = {}
  write(core.unitvar, c.Tuple())
  if const_vals:
    for val in const_vals:
      if isinstance(val, DeviceArray):
        val.copy_to_host_async()
    _map(write, jaxpr.constvars, map(c.Constant, const_vals))
    _map(write, jaxpr.freevars, map(c.ParameterWithShape, freevar_shapes))
  else:
    all_freevars = it.chain(jaxpr.constvars, jaxpr.freevars)
    _map(write, all_freevars, map(c.ParameterWithShape, freevar_shapes))
  _map(write, jaxpr.invars, map(c.ParameterWithShape, arg_shapes))
  _prefetch_jaxpr_literals(jaxpr)
  for eqn in jaxpr.eqns:
    if not eqn.restructure:
      in_nodes = list(map(read, eqn.invars))
    else:
      in_nodes = [xla_pack(c, map(read, invars)) if type(invars) is tuple
                  else read(invars) for invars in eqn.invars]

    if eqn.primitive in backend_specific_translations[platform]:
      rule = backend_specific_translations[platform][eqn.primitive]
      ans = rule(c, *in_nodes, **eqn.params)
    elif eqn.primitive in translations:
      ans = translations[eqn.primitive](c, *in_nodes, **eqn.params)
    elif eqn.primitive in initial_style_translations:
      rule = initial_style_translations[eqn.primitive]
      ans = rule(c, axis_env, *in_nodes, **eqn.params)
    elif eqn.primitive in parallel_translations:
      replica_groups = axis_groups(axis_env, eqn.params['axis_name'])
      new_params = {k: eqn.params[k] for k in eqn.params if k != 'axis_name'}
      rule = parallel_translations[eqn.primitive]
      ans = rule(c, *in_nodes, replica_groups=replica_groups, **new_params)
    elif eqn.primitive in call_translations:
      (subjaxpr, const_bindings, freevar_bindings), = eqn.bound_subjaxprs
      env_nodes = list(map(read, const_bindings + freevar_bindings))
      rule = call_translations[eqn.primitive]
      ans = rule(c, subjaxpr, axis_env, env_nodes, in_nodes, **eqn.params)
    else:
      msg = "XLA translation rule for primitive '{}' not found"
      raise NotImplementedError(msg.format(eqn.primitive.name))

    c.GetShape(ans)  # force xla to do shape error checking
    out_nodes = xla_destructure(c, ans) if eqn.destructure else [ans]
    _map(write, eqn.outvars, out_nodes)
  return c.Build(read(jaxpr.outvar))

def _map(f, *xs):
  return tuple(map(f, *xs))

def xla_destructure(c, ans):
  num_elements = len(c.GetShape(ans).tuple_shapes())
  return [c.GetTupleElement(ans, i) for i in range(num_elements)]

def xla_pack(c, xs):
  return c.Tuple(*xs)

def _tuple_constant(c, val, canonicalize_types=True):
  return c.Tuple(*map(c.Constant, val))
xb.register_constant_handler(JaxTuple, _tuple_constant)

AxisEnv = namedtuple("AxisEnv", ["nreps", "names", "sizes"])

def extend_axis_env(env, name, size):
  return AxisEnv(env.nreps, env.names + [name], env.sizes + [size])

def axis_read(axis_env, axis_name):
  return max(i for i, name in enumerate(axis_env.names) if name == axis_name)

def axis_groups(axis_env, name):
  if isinstance(name, (list, tuple)):
    mesh_axes = tuple(map(partial(axis_read, axis_env), name))
  else:
    mesh_axes = (axis_read(axis_env, name),)
  return _axis_groups(axis_env.nreps, axis_env.sizes, mesh_axes)

def _axis_groups(nrep, mesh_spec, mesh_axes):
  trailing_size, ragged = divmod(nrep, prod(mesh_spec))
  assert not ragged
  full_spec = list(mesh_spec) + [trailing_size]
  iota = onp.arange(prod(full_spec)).reshape(full_spec)
  groups = onp.reshape(
      onp.moveaxis(iota, mesh_axes, onp.arange(len(mesh_axes))),
      (prod(onp.take(full_spec, mesh_axes)), -1))
  return tuple(map(tuple, groups.T))

def jaxpr_replicas(jaxpr):
  nums = (eqn_replicas(eqn) for eqn in jaxpr.eqns if eqn.bound_subjaxprs)
  return max(it.chain([1], nums))  # max(itr, default=1)

def eqn_replicas(eqn):
  (subjaxpr, _, _), = eqn.bound_subjaxprs
  return eqn.params.get('axis_size', 1) * jaxpr_replicas(subjaxpr)


def lower_fun(fun, instantiate=False, initial_style=False):
  """Lowers a traceable function to an XLA function.

  Useful for implementing translation rules for primitives in terms of the
  higher-level lax or NumPy APIs.
  """
  def f(c, *args, **params):
    if initial_style:
      axis_env, xla_args = args[0], args[1:]
    else:
      axis_env, xla_args = AxisEnv(1, [], []), args
    xla_shapes = tuple(map(c.GetShape, xla_args))
    avals = map(_aval_from_xla_shape, xla_shapes)
    pvals = [pe.PartialVal((a, core.unit)) for a in avals]
    jaxpr, _, consts = pe.trace_unwrapped_to_jaxpr(fun, pvals, instantiate,
                                                   **params)
    built_c = _jaxpr_computation(jaxpr, axis_env, consts, (), *xla_shapes)
    return c.Call(built_c, xla_args)
  return f


translations = {}
parallel_translations = {}
initial_style_translations = {}
call_translations = {}
backend_specific_translations = defaultdict(dict)

translations[core.pack_p] = lambda c, *xs: c.Tuple(*xs)
translations[core.call_p] = lambda c, subc_a1, *a2: c.Call(subc_a1[0],
                                                           subc_a1[1] + a2)
translations[core.identity_p] = lambda c, x: x

def _zeros_like_translation_rule(c, x):
  def _zeros_like(shape):
    if shape.is_tuple():
      return c.Tuple(*(_zeros_like(x) for x in shape.tuple_shapes()))
    else:
      return c.Broadcast(c.Constant(onp.array(0, shape.element_type())),
                         shape.dimensions())
  return _zeros_like(c.GetShape(x))

def _add_jaxvals_translation_rule(c, x, y):
  x_shape, y_shape = map(c.GetShape, (x, y))
  if x_shape.is_tuple() and y_shape.is_tuple():
    xs = xla_destructure(c, x)
    ys = xla_destructure(c, y)
    return c.Tuple(*map(partial(_add_jaxvals_translation_rule, c), xs, ys))
  else:
    return c.Add(x, y)

translations[ad_util.zeros_like_p] = _zeros_like_translation_rule
translations[ad_util.add_jaxvals_p] = _add_jaxvals_translation_rule


def _canonicalize_pyval_dtype(x):
  try:
    return canonicalize_dtype_handlers[type(x)](x)
  except KeyError:
    msg = "No canonicalize handler registered for type: {}"
    raise TypeError(msg.format(type(x)))

canonicalize_dtype_handlers = {}

def _canonicalize_tuple_dtype(tup):
  return JaxTuple(map(_canonicalize_pyval_dtype, tup))
canonicalize_dtype_handlers[JaxTuple] = _canonicalize_tuple_dtype

def _canonicalize_ndarray_dtype(x):
  return onp.asarray(x, xb.canonicalize_dtype(onp.result_type(x)))

for _t in array_types:
  canonicalize_dtype_handlers[_t] = _canonicalize_ndarray_dtype

def _identity(x): return x


def abstractify(x):
  try:
    return pytype_aval_mappings[type(x)](x)
  except KeyError:
    raise TypeError("No abstraction handler for type: {}".format(type(x)))

pytype_aval_mappings = {}

def _abstractify_tuple(tup):
  return AbstractTuple(map(abstractify, tup))
pytype_aval_mappings[JaxTuple] = _abstractify_tuple
pytype_aval_mappings[AbstractTuple] = _abstractify_tuple

for _t in array_types:
  pytype_aval_mappings[_t] = make_shaped_array


class DeviceValue(object):
  """A DeviceValue represents a value backed by device memory."""
  __slots__ = ["device_buffer"]
  def __init__(self, device_buffer):
    self.device_buffer = device_buffer

  def _check_if_deleted(self):
    if self.device_buffer is None:
      raise ValueError("DeviceValue has been deleted.")

  def block_until_ready(self):
    """Blocks the caller until the buffer's value has been computed on device.

    This method is mostly useful for timing microbenchmarks that wish to
    time how long a computation takes, without transferring the result back
    to the host.
    """
    self._check_if_deleted()
    self.device_buffer.block_host_until_ready()


class DeviceTuple(DeviceValue):
  """A DeviceTuple is a JaxTuple backed by a single device memory buffer."""
  __slots__ = ["aval", "result_shapes"]

  def __init__(self, result_shape, device_buffer):
    self.device_buffer = device_buffer
    self.aval, self.result_shapes = result_shape

  def __iter__(self):
    bufs = self.device_buffer.destructure()
    handlers = map(_device_persistent_result_handler, self.result_shapes)
    elts = [handler(buf) for handler, buf in zip(handlers, bufs)]
    return iter(elts)

  def __len__(self):
    return len(self.aval)

  def __repr__(self):
    return 'DeviceTuple(len={length})'.format(length=len(self))

  def __eq__(self, other):
    return tuple(self) == tuple(other)


# DeviceValues don't need to be dtype-canonicalized because we assume values on
# the device have already been canonicalized.
core.pytype_aval_mappings[DeviceTuple] = core.pytype_aval_mappings[JaxTuple]
pytype_aval_mappings[DeviceTuple] = op.attrgetter('aval')
canonicalize_dtype_handlers[DeviceTuple] = _identity

def _device_tuple_constant_handler(c, val, canonicalize_types=True):
  const = partial(c.Constant, canonicalize_types=canonicalize_types)
  return c.Tuple(*map(const, val))
xb.register_constant_handler(DeviceTuple, _device_tuple_constant_handler)

# TODO(mattjj): could jit-compile a computation here
ad_util.jaxval_adders[DeviceTuple] = ad_util.add_jaxtuples

# TODO(phawkins): after Jaxlib 0.1.17 has been released, bump the minimum
# jaxlib version and change callers of this function to simply call
# the copy_to_host_async method directly.
def _copy_to_host_async(buffer):
  if hasattr(buffer, "copy_to_host_async"):
    buffer.copy_to_host_async()


def _forward_method(attrname, self, fun, *args):
  return fun(getattr(self, attrname), *args)
_forward_to_value = partial(_forward_method, "_value")

class DeviceArray(DeviceValue):
  """A DeviceArray is an ndarray backed by a single device memory buffer."""
  # We don't subclass ndarray because that would open up a host of issues,
  # but lax_numpy.py overrides isinstance behavior and attaches ndarray methods.
  __slots__ = ["shape", "dtype", "ndim", "size", "_npy_value"]
  __array_priority__ = 100.

  def __init__(self, result_shape, device_buffer):
    self.device_buffer = device_buffer
    self.shape, self.dtype, self.ndim, self.size = result_shape
    self._npy_value = None

  @property
  def _value(self):
    self._check_if_deleted()
    if self._npy_value is None:
      self._npy_value = self.device_buffer.to_py()
      self._npy_value.flags.writeable = False
    return self._npy_value

  def copy(self):
    """Returns an ndarray (backed by host memory, not device memory)."""
    return onp.asarray(self)

  def copy_to_host_async(self):
    """Requests a copy of the buffer to the host."""
    self._check_if_deleted()
    if self._npy_value is None:
      _copy_to_host_async(self.device_buffer)

  def delete(self):
    """Deletes the device array and any cached copy on the host.

    It is an error to access the contents of a `DeviceArray` after it has
    been deleted.

    Use of this method is optional; device buffers will be reclaimed
    automatically by Python when a DeviceArray object is garbage collected.
    However, it is sometimes useful to have more explicit control over the
    time of deletion.
    """
    self.device_buffer.delete()
    self.device_buffer = None
    self._npy_value = None

  def __repr__(self):
    return onp.array_repr(self)

  def item(self):
    if onp.issubdtype(self.dtype, onp.complexfloating):
      return complex(self)
    elif onp.issubdtype(self.dtype, onp.floating):
      return float(self)
    elif onp.issubdtype(self.dtype, onp.integer):
      return int(self)
    elif onp.issubdtype(self.dtype, onp.bool_):
      return bool(self)
    else:
      raise TypeError(self.dtype)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError:
      raise TypeError("len() of unsized object")  # same as numpy error

  def __iter__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")  # same as numpy error
    else:
      return self._value.__iter__()

  def __reversed__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")
    else:
      return reversed(self._value)

  def __format__(self, format_spec):
    # Simulates behavior of https://github.com/numpy/numpy/pull/9883
    if self.ndim == 0:
      return format(self._value[()], format_spec)
    else:
      return format(self._value, format_spec)

  def __array__(self, dtype=None, context=None):
    return onp.asarray(self._value, dtype=dtype)

  __str__ = partialmethod(_forward_to_value, str)
  __bool__ = __nonzero__ = partialmethod(_forward_to_value, bool)
  __float__ = partialmethod(_forward_to_value, float)
  __int__ = partialmethod(_forward_to_value, int)
  if six.PY2:
    __long__ = partialmethod(_forward_to_value, long)  # noqa: F821
  __complex__ = partialmethod(_forward_to_value, complex)
  __hex__ = partialmethod(_forward_to_value, hex)
  __oct__ = partialmethod(_forward_to_value, oct)

  # pickle saves and loads just like an ndarray
  __reduce__ = partialmethod(_forward_to_value, op.methodcaller("__reduce__"))

  # clobbered when jax.numpy is imported, but useful in tests
  def __eq__(self, other): return self._value == other

  def __hash__(self):
    raise TypeError("JAX DeviceArray, like numpy.ndarray, is not hashable.")

core.literalable_types.add(DeviceArray)


# DeviceValues don't need to be canonicalized because we assume values on the
# device have already been canonicalized.
core.pytype_aval_mappings[DeviceArray] = ConcreteArray
pytype_aval_mappings[DeviceArray] = make_shaped_array
canonicalize_dtype_handlers[DeviceArray] = _identity

def _device_array_constant_handler(c, val, canonicalize_types=True):
  return c.Constant(onp.asarray(val), canonicalize_types=canonicalize_types)
xb.register_constant_handler(DeviceArray, _device_array_constant_handler)

pytype_aval_mappings[ConcreteArray] = make_shaped_array
pytype_aval_mappings[ShapedArray] = _identity


class DeviceConstant(DeviceArray):
  def copy_to_host_async(self): pass

  @staticmethod
  def constant_handler(c, constant_instance, canonicalize_types=True):
    assert False

def _instantiate_device_constant(const, cutoff=1e6, device_num=0):
  # dispatch an XLA Computation to build the constant on the device if it's
  # large, or alternatively build it on the host and transfer it if it's small
  # TODO(mattjj): need a way to instantiate on a specific device
  assert isinstance(const, DeviceConstant)
  if const.size > cutoff and device_num == 0:
    c = xb.make_computation_builder("constant_instantiating_computation")
    xla_const = const.constant_handler(c, const)
    compiled = c.Build(xla_const).Compile((), xb.get_compile_options(),
                                          backend=xb.get_backend())
    return compiled.Execute(())
  else:
    return xb.device_put(onp.asarray(const), device_num)


def _xla_call_impl(fun, *args, **params):
  device_values = FLAGS.jax_device_values and params.pop('device_values')
  device_assignment = params.pop('device_assignment')
  compiled_fun = _xla_callable(fun, device_assignment, device_values,
                               *map(abstractify, args))
  try:
    return compiled_fun(*args)
  except FloatingPointError:
    print("Invalid value encountered in the output of a jit function. "
          "Calling the de-optimized version.")
    return fun.call_wrapped(*args)  # probably won't return

@lu.memoize
def _xla_callable(fun, device_assignment, device_values, *abstract_args):
  pvals = [pe.PartialVal((aval, core.unit)) for aval in abstract_args]
  with core.new_master(pe.JaxprTrace, True) as master:
    jaxpr, (pval, consts, env) = pe.trace_to_subjaxpr(fun, master, False).call_wrapped(pvals)
    assert not env  # no subtraces here (though cond might eventually need them)
    axis_env = AxisEnv(jaxpr_replicas(jaxpr), [], [])
    compiled, result_shape = _compile_jaxpr(jaxpr, device_assignment, axis_env, consts,
                                            *abstract_args)
    del master, consts, jaxpr, env
  if device_values:
    handle_result = _device_persistent_result_handler(result_shape)
  else:
    handle_result = _pyval_result_handler(result_shape)
  if axis_env.nreps == 1:
    return partial(_execute_compiled, compiled, pval, handle_result)
  else:
    return partial(_execute_replicated, compiled, pval, handle_result)

def _execute_compiled(compiled, pval, handle_result, *args):
  device_num, = compiled.DeviceOrdinals()
  input_bufs = [device_put(x, device_num) for x in args]
  out_buf = compiled.Execute(input_bufs)
  check_nans("jit-compiled computation", out_buf)
  return pe.merge_pvals(handle_result(out_buf), pval)

def _execute_replicated(compiled, pval, handle_result, *args):
  input_bufs = [[device_put(x, i) for x in args] for i in compiled.DeviceOrdinals()]
  out_buf = compiled.ExecutePerReplica(input_bufs)[0]
  check_nans("jit-compiled computation", out_buf)
  return pe.merge_pvals(handle_result(out_buf), pval)


xla_call_p = core.Primitive('xla_call')
xla_call = partial(core.call_bind, xla_call_p)
xla_call_p.def_custom_bind(xla_call)
xla_call_p.def_impl(_xla_call_impl)

def _xla_call_translation_rule(c, jaxpr, axis_env, env_nodes, in_nodes,
                               device_values, device_assignment):
  del device_values, device_assignment  # Unused.
  subc = _jaxpr_computation(jaxpr, axis_env, (), _map(c.GetShape, env_nodes),
                            *map(c.GetShape, in_nodes))
  return c.Call(subc, env_nodes + in_nodes)
call_translations[xla_call_p] = _xla_call_translation_rule
ad.primitive_transposes[xla_call_p] = partial(ad.call_transpose, xla_call_p)


def _device_put_impl(x, device_num=0):
  try:
    a = abstractify(x)
  except TypeError:
    raise TypeError("Argument '{}' of type {} is not a valid JAX type"
                    .format(x, type(x)))

  result_shape = xla_shape_to_result_shape(xla_shape(a))
  handler = _device_persistent_result_handler(result_shape)
  return handler(device_put(x, device_num))

device_put_p = core.Primitive('device_put')
device_put_p.def_impl(_device_put_impl)
device_put_p.def_abstract_eval(lambda x, **kwargs: x)
translations[device_put_p] = lambda c, x, **kwargs: x
ad.deflinear(device_put_p, lambda cotangent, **kwargs: [cotangent])
