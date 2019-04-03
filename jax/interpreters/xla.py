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
from ..abstract_arrays import ConcreteArray, ShapedArray, make_shaped_array, array_types
from ..core import AbstractTuple, JaxTuple, pack, valid_jaxtype
from ..util import partial, partialmethod, memoize, unzip2, concatenate, safe_map, prod
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

map = safe_map

def apply_primitive(prim, *args, **kwargs):
  abstract_args = map(abstractify, args)
  compiled_fun = xla_primitive_callable(prim, *abstract_args, **kwargs)
  return compiled_fun(*args)

@memoize
def xla_primitive_callable(prim, *abstract_args, **kwargs):
  shapes = map(xla_shape, abstract_args)
  built_c = primitive_computation(prim, *shapes, **kwargs)
  result_shape = xla_shape_to_result_shape(built_c.GetReturnValueShape())
  handle_result = result_handler(result_shape)
  compiled = built_c.Compile(shapes, xb.get_compile_options(),
                             backend=xb.get_backend())
  return partial(execute_compiled_primitive, compiled, handle_result)

@memoize
def primitive_computation(prim, *shapes, **kwargs):
  c = xb.make_computation_builder("primitive_computation")
  xla_args = map(c.ParameterWithShape, shapes)
  xla_result = translation_rule(prim)(c, *xla_args, **kwargs)
  try:
    return c.Build()
  except RuntimeError as e:
    # try for a better error message by using the abstract_eval checks
    prim.abstract_eval(*map(aval_from_xla_shape, shapes), **kwargs)
    raise e

def aval_from_xla_shape(shape):
  if shape.is_tuple():
    return AbstractTuple(map(aval_from_xla_shape, shape.tuple_shapes()))
  else:
    return ShapedArray(shape.dimensions(), shape.element_type())

def execute_compiled_primitive(compiled, result_handler, *args):
  input_bufs = [device_put(x) for x in args]
  return result_handler(compiled.Execute(input_bufs, not core.skip_checks))

def device_put(x, device_num=0):
  x = canonicalize_pyval_dtype(x)
  if type(x) is DeviceArray:
    if x.device_buffer.device() == device_num:
      return x.device_buffer
    else:
      return device_put(x.device_buffer.to_py(), device_num)
  elif isinstance(x, DeviceConstant):
    return instantiate_device_constant(x, device_num=device_num)
  else:
    return xb.device_put(x, device_num)  # round-trips tuple elements

def result_handler(result_shape):
  if type(result_shape) is ResultArray and FLAGS.jax_device_values:
    if FLAGS.jax_debug_nans and onp.issubdtype(result_shape[1], onp.floating):
      def handle_result(device_buffer):
        py_val = device_buffer.to_py()
        if onp.any(onp.isnan(py_val)):
          raise FloatingPointError("invalid value")
        else:
          return DeviceArray(device_buffer, *result_shape)
    else:
      def handle_result(device_buffer):
        return DeviceArray(device_buffer, *result_shape)
  elif type(result_shape) is ResultArray:
    def handle_result(device_buffer):
      return onp.asarray(DeviceArray(device_buffer, *result_shape))
  elif type(result_shape) is ResultTuple:
    handlers = list(map(result_handler, result_shape))
    def handle_result(device_buffer):
      bufs = device_buffer.destructure()
      return JaxTuple(handler(buf) for handler, buf in zip(handlers, bufs))
  else:
    raise TypeError(type(result_shape))
  return handle_result

def xla_shape_to_result_shape(xla_shape):
  if xla_shape.is_tuple():
    return ResultTuple(map(xla_shape_to_result_shape, xla_shape.tuple_shapes()))
  else:
    shape, dtype = xla_shape.dimensions(), xla_shape.element_type()
    ndim, size = len(shape), prod(shape)
    return ResultArray((shape, dtype, ndim, size))
class ResultTuple(tuple): pass
class ResultArray(tuple): pass


def compile_jaxpr(jaxpr, const_vals, *abstract_args):
  arg_shapes = list(map(xla_shape, abstract_args))
  built_c = jaxpr_computation(jaxpr, const_vals, (), *arg_shapes)
  result_shape = xla_shape_to_result_shape(built_c.GetReturnValueShape())
  return built_c.Compile(arg_shapes, xb.get_compile_options(),
                         backend=xb.get_backend()), result_shape

def build_jaxpr(jaxpr, const_vals, *abstract_args):
  arg_shapes = list(map(xla_shape, abstract_args))
  built_c = jaxpr_computation(jaxpr, const_vals, (), *arg_shapes)
  return built_c

def jaxpr_computation(jaxpr, const_vals, freevar_shapes, *arg_shapes):
  c = xb.make_computation_builder("jaxpr_computation")

  def read(v):
    return env[v]

  def write(v, node):
    assert node is not None
    env[v] = node

  env = {}
  consts_env = dict(zip(jaxpr.constvars, const_vals))
  write(core.unitvar, c.Tuple())
  if const_vals:
    map(write, jaxpr.constvars, map(c.Constant, const_vals))
    map(write, jaxpr.freevars, map(c.ParameterWithShape, freevar_shapes))
  else:
    all_freevars = it.chain(jaxpr.constvars, jaxpr.freevars)
    map(write, all_freevars, map(c.ParameterWithShape, freevar_shapes))
  map(write, jaxpr.invars, map(c.ParameterWithShape, arg_shapes))
  for eqn in jaxpr.eqns:
    in_nodes = map(read, eqn.invars)
    in_shapes = map(c.GetShape, in_nodes)
    subcs = [
        jaxpr_computation(
            subjaxpr, (),
            map(c.GetShape, map(read, const_bindings + freevar_bindings)),
            *in_shapes)
        for subjaxpr, const_bindings, freevar_bindings in eqn.bound_subjaxprs]
    subfuns = [(subc, tuple(map(read, const_bindings + freevar_bindings)))
               for subc, (_, const_bindings, freevar_bindings)
               in zip(subcs, eqn.bound_subjaxprs)]
    ans = translation_rule(eqn.primitive)(c, *(subfuns + in_nodes), **eqn.params)
    out_nodes = xla_destructure(c, ans) if eqn.destructure else [ans]
    map(write, eqn.outvars, out_nodes)
  return c.Build(read(jaxpr.outvar))

def xla_destructure(c, ans):
  num_elements = len(c.GetShape(ans).tuple_shapes())
  return [c.GetTupleElement(ans, i) for i in range(num_elements)]

def tuple_constant(c, val, canonicalize_types=True):
  return c.Tuple(*map(c.Constant, val))
xb.register_constant_handler(JaxTuple, tuple_constant)

def translation_rule(p):
  backend = xb.get_backend()
  backend_specific_rule = backend_specific_translations[backend.platform].get(p)
  try:
    return backend_specific_rule or translations[p]
  except KeyError:
    raise NotImplementedError(
        "XLA translation rule for '{}' not implemented".format(p))


def lower_fun(fun, c, *xla_args, **params):
  xla_shapes = map(c.GetShape, xla_args)
  avals = map(aval_from_xla_shape, xla_shapes)
  pvals = [pe.PartialVal((a, core.unit)) for a in avals]
  jaxpr, pvout, consts = pe.trace_unwrapped_to_jaxpr(fun, pvals, **params)
  built_c = jaxpr_computation(jaxpr, consts, (), *xla_shapes)
  return c.Call(built_c, xla_args)


translations = {}
backend_specific_translations = defaultdict(dict)

translations[core.pack_p] = lambda c, *xs: c.Tuple(*xs)
translations[core.call_p] = lambda c, subc_a1, *a2: c.Call(subc_a1[0],
                                                           subc_a1[1] + a2)
translations[core.identity_p] = lambda c, x: x

# TODO(mattjj): add_jaxvals should handle any jaxval
def zeros_like_translation_rule(c, x):
  def _zeros_like(shape):
    if shape.is_tuple():
      return c.Tuple(*(_zeros_like(x) for x in shape.tuple_shapes()))
    else:
      return c.Broadcast(c.Constant(onp.array(0, shape.element_type())),
                         shape.dimensions())
  return _zeros_like(c.GetShape(x))

translations[ad_util.zeros_like_p] = zeros_like_translation_rule
translations[ad_util.add_jaxvals_p] = lambda c, x, y: c.Add(x, y)


def canonicalize_pyval_dtype(x):
  try:
    return canonicalize_dtype_handlers[type(x)](x)
  except KeyError:
    msg = "No canonicalize handler registered for type: {}"
    raise TypeError(msg.format(type(x)))

canonicalize_dtype_handlers = {}

def canonicalize_tuple_dtype(tup):
  return JaxTuple(map(canonicalize_pyval_dtype, tup))
canonicalize_dtype_handlers[JaxTuple] = canonicalize_tuple_dtype

def canonicalize_ndarray_dtype(x):
  return onp.asarray(x, xb.canonicalize_dtype(onp.result_type(x)))

for t in array_types:
  canonicalize_dtype_handlers[t] = canonicalize_ndarray_dtype

def identity(x): return x


def abstractify(x):
  try:
    return pytype_aval_mappings[type(x)](x)
  except KeyError:
    raise TypeError("No abstraction handler for type: {}".format(type(x)))

pytype_aval_mappings = {}

def abstractify_tuple(tup):
  return AbstractTuple(tuple(map(abstractify, tup)))
pytype_aval_mappings[JaxTuple] = abstractify_tuple
pytype_aval_mappings[AbstractTuple] = abstractify_tuple

for t in array_types:
  pytype_aval_mappings[t] = make_shaped_array


class DeviceValue(object):
  __slots__ = ["device_buffer"]
  def __init__(self, device_buffer):
    self.device_buffer = device_buffer

def forward_method(attrname, self, fun, *args):
  return fun(getattr(self, attrname), *args)
forward_to_value = partial(forward_method, "_value")

class DeviceArray(DeviceValue):
  __slots__ = ["shape", "dtype", "ndim", "size", "_npy_value"]
  __array_priority__ = 100.

  def __init__(self, device_buffer, shape, dtype, ndim, size):
    self.device_buffer = device_buffer
    self.shape = shape
    self.dtype = dtype
    self.ndim = ndim
    self.size = size
    self._npy_value = None

  # TODO make device_buffer a property, make the _npy_value writeable, invalidate
  @property
  def _value(self):
    if self._npy_value is None:
      self._npy_value = self.device_buffer.to_py()
      self._npy_value.flags.writeable = False
    return self._npy_value

  def copy(self):
    """Returns an ndarray (backed by host memory, not device memory)."""
    return onp.asarray(self)

  def __repr__(self):
    shape_str = ",".join(map(str, self.shape))
    return "DeviceArray{{{}[{}]}}".format(onp.dtype(self.dtype).name, shape_str)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError:
      raise TypeError("len() of unsized object")  # same as numpy error

  def __iter__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")  # same as numpy error
    else:
      return (self[i] for i in xrange(self.shape[0]))

  def __reversed__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")
    else:
      return (self[i] for i in xrange(self.shape[0] - 1, -1, -1))

  def __format__(self, format_spec):
    # Simulates behavior of https://github.com/numpy/numpy/pull/9883
    if self.ndim == 0:
      return format(self._value[()], format_spec)
    else:
      return format(self._value, format_spec)

  __array__ = partialmethod(forward_to_value, onp.asarray)
  __str__ = partialmethod(forward_to_value, str)
  __bool__ = __nonzero__ = partialmethod(forward_to_value, bool)
  __float__ = partialmethod(forward_to_value, float)
  __int__ = partialmethod(forward_to_value, int)
  if six.PY2:
    __long__ = partialmethod(forward_to_value, long)  # noqa: F821
  __complex__ = partialmethod(forward_to_value, complex)
  __hex__ = partialmethod(forward_to_value, hex)
  __oct__ = partialmethod(forward_to_value, oct)

  # pickle saves and loads just like an ndarray
  __reduce__ = partialmethod(forward_to_value, op.methodcaller("__reduce__"))

  # clobbered when jax.numpy is imported, but useful in tests
  def __eq__(self, other): return self._value == other

  def __hash__(self):
    # TODO(mattjj): this is not semantically correct because it is possible
    # __eq__ is true for values with unequal __hash__ values. However, the
    # main use case at the moment is memoization for which false negatives are
    # fine.
    return id(self)


core.pytype_aval_mappings[DeviceArray] = ConcreteArray
pytype_aval_mappings[DeviceArray] = make_shaped_array
canonicalize_dtype_handlers[DeviceArray] = identity

def _device_array_constant_handler(c, val, canonicalize_types=True):
  return c.Constant(onp.asarray(val), canonicalize_types=canonicalize_types)
xb.register_constant_handler(DeviceArray, _device_array_constant_handler)

pytype_aval_mappings[ConcreteArray] = make_shaped_array
pytype_aval_mappings[ShapedArray] = lambda x: x


class DeviceConstant(DeviceArray):
  @staticmethod
  def constant_handler(c, constant_instance, canonicalize_types=True):
    assert False

def instantiate_device_constant(const, cutoff=1e6, device_num=0):
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


def xla_shape(x):
  try:
    return xb.Shape.array_shape(x.dtype, x.shape)
  except AttributeError:
    if type(x) in (core.AbstractTuple, core.JaxTuple):
      return xb.Shape.tuple_shape(tuple(map(xla_shape, x)))
    else:
      raise TypeError(type(x))


# For callable XLA Computations (as opposed to, e.g., Computations used in the
# body of a While) we flatten functions to take multiple array arguments (no
# tuple arguments) and return either an array output or a flat tuple output that
# is immediately destructured. This flattening avoids the need for the runtime
# to manage multiple references to DeviceValues caused by tuple membership
# (since the XLA runtime depends on single-ownership, rather than e.g.
# refcounting). In particular, we don't have a DeviceTuple representation, and
# instead, for values returned to the user, always destructure tuples.
# The code here is similar to that in tree_util, but is meant to flatten
# JaxTuple trees only.

@lu.transformation_with_aux
def flatten_fun(in_trees, *flat_args):
  jtuple_trees = tuple(map(partial(build_tree, iter(flat_args)), in_trees))
  ans = yield jtuple_trees
  aval = core.get_aval(ans)
  if type(aval) is AbstractTuple:
    ans_flat, out_tree = tree_flatten(ans)
    yield pack(ans_flat), out_tree
  else:
    yield ans, leaf

def tree_flatten(maybe_tree):
  aval = core.get_aval(maybe_tree)
  if type(aval) is AbstractTuple:
    flat_children, child_specs = unzip2(map(tree_flatten, maybe_tree))
    return it.chain.from_iterable(flat_children), JTupleTreeDef(child_specs)
  elif core.skip_checks or valid_jaxtype(maybe_tree):
    return [maybe_tree], leaf
  else:
    raise TypeError(type(maybe_tree))

JTupleTreeDef = namedtuple("JTupleTreeDef", ["child_specs"])

class Leaf(object):
  def __repr__(self):
    return '*'
leaf = Leaf()

def build_tree(xs, tree_spec):
  if tree_spec is leaf:
    return next(xs)
  elif type(tree_spec) is JTupleTreeDef:
    return pack(map(partial(build_tree, xs), tree_spec.child_specs))
  else:
    raise TypeError(type(tree_spec))


def xla_call_impl(fun, *args):
  flat_args, in_trees = unzip2(map(tree_flatten, args))
  flat_args = concatenate(flat_args)
  fun, out_tree = flatten_fun(fun, in_trees)

  compiled_fun = xla_callable(fun, *map(abstractify, flat_args))
  try:
    flat_ans = compiled_fun(*flat_args)
  except FloatingPointError:
    msg = ("Invalid value encountered in the output of a jit function. "
           "Calling the de-optimized version.")
    print(msg)
    return fun.call_wrapped(*args)  # probably won't return

  if out_tree() is leaf:
    return flat_ans
  else:
    return build_tree(iter(flat_ans), out_tree())


@lu.memoize
def xla_callable(fun, *abstract_args):
  pvals = [pe.PartialVal((aval, core.unit)) for aval in abstract_args]
  with core.new_master(pe.JaxprTrace, True) as master:
    jaxpr, (pval, consts, env) = pe.trace_to_subjaxpr(fun, master).call_wrapped(pvals)
    assert not env  # no subtraces here (though cond might eventually need them)
    compiled, result_shape = compile_jaxpr(jaxpr, consts, *abstract_args)
    del master, consts, jaxpr, env
  handle_result = result_handler(result_shape)
  return partial(execute_compiled, compiled, pval, handle_result)

def execute_compiled(compiled, pval, handle_result, *args):
  input_bufs = [device_put(x) for x in args]
  out_buf = compiled.Execute(input_bufs, not core.skip_checks)
  return pe.merge_pvals(handle_result(out_buf), pval)


def xla_call_translation_rule(c, subc_a1, *a2):
  subc, a1 = subc_a1
  return c.Call(subc, a1 + a2)

xla_call_p = core.Primitive('xla_call')
xla_call = partial(core.call_bind, xla_call_p)
xla_call_p.def_custom_bind(xla_call)
xla_call_p.def_impl(xla_call_impl)

translations[xla_call_p] = xla_call_translation_rule
ad.primitive_transposes[xla_call_p] = partial(ad.call_transpose, xla_call_p)
