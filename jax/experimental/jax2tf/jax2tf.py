# Copyright 2020 Google LLC
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
"""Experimental module transforms JAX functions to be executed by TensorFlow."""

import contextlib
import functools
import string
import threading
from typing import Any, Callable, Dict, Iterable, Sequence, Union

import jax
from jax import abstract_arrays
from jax import ad_util
from jax import api
from jax import core
from jax import custom_derivatives
from jax import lax
from jax import linear_util as lu
from jax import numpy as jnp
from jax import random
from jax import tree_util
from jax import util
from jax.api_util import flatten_fun
from jax.lax import lax_control_flow
from jax.lax import lax_fft
from jax import lax_linalg
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import pxla

import numpy as np
import tensorflow as tf  # type: ignore[import]

# These don't have public equivalents.
# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla  # type: ignore[import]
from tensorflow.compiler.xla import xla_data_pb2  # type: ignore[import]

# A value suitable in a TF tracing context: tf.Tensor, tf.Variable,
# or Python scalar or numpy.ndarray. (A tf.EagerTensor is a tf.Tensor.)
TfVal = Any
def _is_tfval(v: TfVal) -> bool:
  if isinstance(v, (tf.Tensor, tf.Variable)):
    return True
  try:
    tf.convert_to_tensor(v)
    return True
  except ValueError:
    return False

# During JAX transformations we sometimes produce a Jaxpr that has arguments
# of abstract value core.abstract_unit and results equal to core.unit.
# These are arguments and results that are not used in the computation.
# Whenever we are in a JAX tracing context we must use `core.unit` values
# in those places. However, when we move to TF we have to turn them into
# some small TFVal; it does not matter which value since it will never be used
# in an actual operation. We will use `tf.constant(np.nan, float32)`.
TfValOrUnit = Union[TfVal, core.Unit]
def _is_tfvalorunit(v: TfValOrUnit) -> bool:
  return v is core.unit or _is_tfval(v)


def _tfval_remove_unit(args: Sequence[TfValOrUnit]) -> Sequence[TfVal]:
  """Replace core.unit with regular TF values."""
  return [tf.constant(np.nan, tf.float32) if a is core.unit else a
          for a in args]

def _tfval_add_unit(vals: Sequence[TfValOrUnit],
                    avals: Sequence[core.AbstractValue]) -> Sequence[TfValOrUnit]:
  """Turn regular TfVals into TfValOrUnit, based on expected abstract values.
  This function is sometimes called with a mix of core.unit and tf.nan in places
  of units.
  """
  def add_unit(v: TfValOrUnit, aval: core.AbstractValue):
    if not core.skip_checks:
      assert ((v is core.unit or tf.math.is_nan(v))
              if aval is core.abstract_unit else _is_tfval(v))
    return core.unit if aval is core.abstract_unit else v
  return util.safe_map(add_unit, vals, avals)

# The implementation rules for primitives. The rule will be called with the
# arguments (TfValOrUnit) and must return TfValOrUnit (or a sequence thereof,
# if primitive.multiple_results). The vast majority of primitives do not need
# to worry about core.unit inputs or results. The exception are primarily the
# control-flow primitives.
tf_impl: Dict[core.Primitive,
              Callable[..., Any]] = {}

# TODO(necula): used for tests, until we handle all control-flow primitives
class _JitState(threading.local):

  def __init__(self):
    super().__init__()
    self.disable_jit = True

_jit_state = _JitState()


@contextlib.contextmanager
def enable_jit():
  """Temporarily allow JAX jit."""
  old_state = _jit_state.disable_jit
  _jit_state.disable_jit = False
  try:
    yield
  finally:
    _jit_state.disable_jit = old_state


def disable_gradient(fun):
  """Prevents the wrapped function from being differentiated."""

  def grad_disabled(*dy, variables=None):
    raise ValueError("jax2tf currently does not support gradients. Please "
                     "reach out to tomhennigan@ if this feature is a blocker "
                     "for you, we are working on it!")

  is_tensor = lambda t: isinstance(t, (tf.Tensor, tf.Variable))

  def wrapper(*args, **kwargs):
    flat_args, treedef = jax.tree_flatten((args, kwargs))
    tensor_idxs = [i for i, t in enumerate(flat_args) if is_tensor(t)]
    tensor_args = [t for t in flat_args if is_tensor(t)]

    @tf.custom_gradient
    def inner_wrapper(*tensor_args):
      flat_copy = list(flat_args)
      for i, t in zip(tensor_idxs, tensor_args):
        flat_copy[i] = t

      args, kwargs = jax.tree_unflatten(treedef, flat_copy)
      out = fun(*args, **kwargs)
      return out, grad_disabled

    return inner_wrapper(*tensor_args)

  return wrapper


def convert(fun):
  """Transforms `fun` to be executed by TensorFlow.

  Args:
    fun: Function to be transformed. Its arguments and return value should be
      JAX arrays, or (nested) standard Python containers (tuple/list/dict)
      thereof.

  Returns:
    A version of `fun` that expects TfVals as arguments (or
    tuple/lists/dicts) thereof, and returns TfVals as outputs.
  """
  api._check_callable(fun)

  @disable_gradient
  def wrapped_fun(*args: TfValOrUnit) -> TfValOrUnit:
    # TODO(necula): remove the jit disabling once we handle all control-flow.
    # Disabling the jit helps to avoid some unsupported jax primitives.
    # E.g. scan will be statically unrolled.
    def doit():
      f = lu.wrap_init(fun)
      args_flat, in_tree = tree_util.tree_flatten((args, {}))
      for a in args_flat:
        if not _is_tfvalorunit(a):
          msg = (f"Argument {a} of type {type(a)} of jax2tf.convert(f) should "
                 "be NumPy array, scalar, tf.Variable, or tf.Tensor")
          raise TypeError(msg)
      flat_fun, out_tree = flatten_fun(f, in_tree)
      out_flat = _interpret_fun(flat_fun, args_flat)
      return tree_util.tree_unflatten(out_tree(), out_flat)

    if _jit_state.disable_jit:
      with jax.disable_jit():
        return doit()
    else:
      return doit()

  return wrapped_fun

# Internals


def _interpret_fun(fun: lu.WrappedFun,
                   in_vals: Sequence[TfValOrUnit]) -> Sequence[TfValOrUnit]:
  with core.new_master(TensorFlowTrace) as master:
    fun = _interpret_subtrace(fun, master)
    out_vals: Sequence[TfValOrUnit] = fun.call_wrapped(*in_vals)
    del master
  return out_vals


@lu.transformation
def _interpret_subtrace(master: core.MasterTrace, *in_vals: TfValOrUnit):
  trace = TensorFlowTrace(master, core.cur_sublevel())
  in_tracers = tuple(TensorFlowTracer(trace, val) for val in in_vals)
  outs = yield in_tracers, {}  # type: Sequence[TfValOrUnit]
  out_tracers: Iterable[TensorFlowTracer] = map(trace.full_raise, outs)  # type: ignore
  out_vals: Sequence[TfValOrUnit] = tuple(t.val for t in out_tracers)
  yield out_vals


def _interpret_jaxpr(jaxpr: core.TypedJaxpr, *args: TfValOrUnit) -> Sequence[TfVal]:
  """Evaluates a Jaxpr with tf.Tensor arguments.

  It is safe to call this function with arguments TfVal or TfValOrUnit, they
  will be replaced with `core.unit` if the `jaxpr` expects units.

  The output is a sequence of TfVal (no `core.unit`), suitable for use with TF.
  """
  fun: lu.WrappedFun = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
  args_jax: Sequence[TfValOrUnit] = _tfval_add_unit(args, jaxpr.in_avals)
  out_vals_jax: Sequence[TfValOrUnit] = _interpret_fun(fun, args_jax)
  return _tfval_remove_unit(out_vals_jax)

### tracer


def abstractify(t: Union[tf.Tensor, tf.Variable]):
  return abstract_arrays.ShapedArray(tuple(t.shape), t.dtype.as_numpy_dtype)

# TODO(b/26854495): pylint doesn't understand slots and inheritance.
# pylint: disable=assigning-non-slot


class TensorFlowTracer(core.Tracer):
  """Tracer class that boxes a `tf.Tensor`."""
  # val: TfValOrUnit
  __slots__ = ["val"]

  def __init__(self, trace: 'TensorFlowTrace', val: TfValOrUnit):
    self._trace = trace
    if val is core.unit:
      self.val = val
    elif isinstance(val, (tf.Tensor, tf.Variable)):
      aval: core.ShapedArray = abstractify(val)
      if np.dtype(aval.dtype) != val.dtype.as_numpy_dtype:  # type: ignore
        # This is expected when JAX 64-bit is not enabled
        self.val = tf.cast(val, dtype=aval.dtype)
      else:
        self.val = val
    else:  # Must be a numeric value
      assert core.skip_checks or _is_tfval(val), f"Non TfVal: {val}"
      aval = xla.abstractify(val)  # type: ignore
      self.val = tf.convert_to_tensor(np.array(val, aval.dtype), dtype=aval.dtype)  # type: ignore
      assert core.skip_checks or aval.strip_weak_type() == self.aval.strip_weak_type(), (
              f"Expected {aval}, got {self.aval}")

  @property
  def aval(self):
    if self.val is core.unit:
      return core.abstract_unit
    else:
      return abstractify(self.val)

  def full_lower(self):
    return self


class TensorFlowTrace(core.Trace):
  """Trace class that underlies the jax2tf transformation."""
  def pure(self, val: TfValOrUnit):
    """Lifts a non-Tracer into the TensorFlowTrace."""
    return TensorFlowTracer(self, val)

  def lift(self, val: core.Tracer):
    """Lifts a core.Tracer from a lower-level master into the TensorFlowTrace."""
    # TODO(necula): this should never be needed
    return TensorFlowTracer(self, val)

  def sublift(self, val: TensorFlowTracer):
    # TODO(necula): this should never be needed
    return TensorFlowTracer(self, val.val)

  def process_primitive(self, primitive: core.Primitive,
                        tracers: Sequence[TensorFlowTracer],
                        params) -> TensorFlowTracer:
    impl = self.get_primitive_impl(primitive)
    args_tf: Sequence[TfValOrUnit] = [t.val for t in tracers]
    # impl takes core.unit and returns core.unit when needed.
    val_out: TfValOrUnit = impl(*args_tf, **params)
    if primitive.multiple_results:
      out = util.safe_map(functools.partial(TensorFlowTracer, self), val_out)  # type: ignore
    else:
      out = TensorFlowTracer(self, val_out)

    # Check that the impl rule returned a value of expected shape and dtype
    if not core.skip_checks:
      expected_out_aval: core.AbstractValue = primitive.abstract_eval(
        *[t.aval for t in tracers], **params)
      if primitive.multiple_results:
        for o, expected_aval in zip(out, expected_out_aval):  # type: ignore
          assert o.aval == expected_aval, (
            f"{primitive}: out.aval = {o.aval}; expected {expected_aval}")
      else:
        assert out.aval == expected_out_aval, (  # type: ignore
          f"{primitive}: out.aval = {out.aval}; expected {expected_out_aval}")  # type: ignore
    return out  # type: ignore

  def process_call(self, call_primitive: core.Primitive, f,
                   tracers: Sequence[TensorFlowTracer], params):
    assert call_primitive.multiple_results
    vals: Sequence[TfValOrUnit] = [t.val for t in tracers]
    f = _interpret_subtrace(f, self.master)
    vals_out: Sequence[TfValOrUnit] = f.call_wrapped(*vals)
    return [TensorFlowTracer(self, v) for v in vals_out]

  def post_process_call(self, call_primitive, out_tracers, params):
    raise NotImplementedError("post_process_call")

  def process_map(self, map_primitive, f, tracers, params):
    raise NotImplementedError("process_map")

  def post_process_map(self, map_primitive, out_tracers, params):
    raise NotImplementedError("post_process_map")

  def get_primitive_impl(self, p):
    try:
      return tf_impl[p]
    except KeyError:
      msg = "TensorFlow interpretation rule for '{}' not implemented"
      raise NotImplementedError(msg.format(p))


def promote_types(*values):
  """Returns values casted to a common type using jnp.promote_types."""
  dtype = tf.dtypes.as_dtype(functools.reduce(
      jnp.promote_types, (v.dtype.as_numpy_dtype for v in values)))
  return tuple(tf.cast(v, dtype) for v in values)


def wrap_binary_op(func):
  def wrapped_func(lhs, rhs, **kwargs):
    return func(*promote_types(lhs, rhs), **kwargs)
  return wrapped_func

def _unexpected_primitive(p: core.Primitive, *args, **kwargs):
  assert False, f"Encountered unexpected primitive {p}"


for unexpected in [
  xla.xla_call_p]:  # Not part of the public API

  tf_impl[unexpected] = functools.partial(_unexpected_primitive, unexpected)

# Primitives that are not yet implemented must be explicitly declared here.
tf_not_yet_impl = [
  ad.custom_lin_p,

  lax.after_all_p, lax.all_to_all_p, lax.create_token_p, lax_fft.fft_p,
  lax.igamma_grad_a_p, lax.infeed_p, lax.linear_solve_p, lax.outfeed_p,
  lax.sort_p, lax.pmax_p, lax.pmin_p, lax.ppermute_p, lax.psum_p,
  lax.population_count_p, lax.reduce_p, lax.reduce_window_p, lax.rng_uniform_p,
  lax.select_and_gather_add_p, lax.select_and_scatter_p,
  lax.top_k_p,

  core.call_p,
  lax_linalg.cholesky_p, lax_linalg.eig_p, lax_linalg.eigh_p,
  lax_linalg.lu_p, lax_linalg.qr_p, lax_linalg.svd_p,
  lax_linalg.triangular_solve_p,

  custom_derivatives.custom_jvp_call_jaxpr_p,
  custom_derivatives.custom_vjp_call_jaxpr_p,

  random.random_gamma_p,
  lax.random_gamma_grad_p,
  pe.remat_call_p,
  pxla.xla_pmap_p, pxla.axis_index_p,
]

tf_impl[lax.tie_in_p] = lambda x, y: y
tf_impl[core.identity_p] = lambda x: x
tf_impl[ad_util.stop_gradient_p] = tf.stop_gradient
tf_impl[ad_util.zeros_like_p] = tf.zeros_like
tf_impl[ad_util.add_jaxvals_p] = wrap_binary_op(tf.math.add)
tf_impl[xla.device_put_p] = lambda x, device=None: x

tf_impl[lax.neg_p] = tf.math.negative
tf_impl[lax.sign_p] = tf.math.sign
tf_impl[lax.floor_p] = tf.math.floor
tf_impl[lax.ceil_p] = tf.math.ceil
tf_impl[lax.round_p] = tf.math.round
tf_impl[lax.nextafter_p] = tf.math.nextafter

tf_impl[lax.is_finite_p] = tf.math.is_finite

tf_impl[lax.abs_p] = tf.math.abs
tf_impl[lax.pow_p] = tf.math.pow
tf_impl[lax.integer_pow_p] = tf.math.pow
tf_impl[lax.exp_p] = tf.math.exp
tf_impl[lax.expm1_p] = tf.math.expm1
tf_impl[lax.log_p] = tf.math.log
tf_impl[lax.log1p_p] = tf.math.log1p
tf_impl[lax.tanh_p] = tf.math.tanh
tf_impl[lax.sin_p] = tf.math.sin
tf_impl[lax.sinh_p] = tf.math.sinh
tf_impl[lax.cos_p] = tf.math.cos
tf_impl[lax.cosh_p] = tf.math.cosh
tf_impl[lax.atan2_p] = wrap_binary_op(tf.math.atan2)
tf_impl[lax.acosh_p] = tf.math.acosh
tf_impl[lax.atanh_p] = tf.math.atanh
tf_impl[lax.asinh_p] = tf.math.asinh

tf_impl[lax.sqrt_p] = tf.math.sqrt
tf_impl[lax.rsqrt_p] = tf.math.rsqrt

tf_impl[lax.lgamma_p] = tf.math.lgamma
tf_impl[lax.digamma_p] = tf.math.digamma
tf_impl[lax.igamma_p] = wrap_binary_op(tf.math.igamma)
tf_impl[lax.igammac_p] = wrap_binary_op(tf.math.igammac)
tf_impl[lax.regularized_incomplete_beta_p] = tf.math.betainc
tf_impl[lax.erf_p] = tf.math.erf
tf_impl[lax.erfc_p] = tf.math.erfc
tf_impl[lax.erf_inv_p] = tf.math.erfinv
tf_impl[lax.bessel_i0e_p] = tf.math.bessel_i0e
tf_impl[lax.bessel_i1e_p] = tf.math.bessel_i1e

tf_impl[lax.complex_p] = tf.complex
tf_impl[lax.conj_p] = tf.math.conj
tf_impl[lax.real_p] = tf.math.real
tf_impl[lax.imag_p] = tf.math.imag

tf_impl[lax.add_p] = wrap_binary_op(tf.math.add)
tf_impl[lax.sub_p] = wrap_binary_op(tf.math.subtract)
tf_impl[lax.mul_p] = wrap_binary_op(tf.math.multiply)


def _div(lhs, rhs):
  if lhs.dtype.is_integer:
    quotient = tf.math.floor_divide(lhs, rhs)
    select = tf.math.logical_and(tf.math.sign(lhs) != tf.math.sign(rhs),
                                 tf.math.floormod(lhs, rhs) != 0)
    return tf.where(select, quotient + 1, quotient)
  else:
    return tf.math.truediv(lhs, rhs)


def _rem(lhs, rhs):
  return tf.math.sign(lhs) * tf.math.floormod(tf.math.abs(lhs),
                                              tf.math.abs(rhs))

tf_impl[lax.div_p] = wrap_binary_op(_div)
tf_impl[lax.rem_p] = wrap_binary_op(_rem)

tf_impl[lax.max_p] = wrap_binary_op(tf.math.maximum)
tf_impl[lax.min_p] = wrap_binary_op(tf.math.minimum)

# Map from TF signed types to TF unsigned types.
_SIGNED_TO_UNSIGNED_TABLE = {
    tf.int8: tf.uint8,
    tf.int16: tf.uint16,
    tf.int32: tf.uint32,
    tf.int64: tf.uint64,
}

# Map from TF unsigned types to TF signed types.
_UNSIGNED_TO_SIGNED_TABLE = {u: s for s, u in _SIGNED_TO_UNSIGNED_TABLE.items()}

# Note: Bitwise operations only yield identical results on unsigned integers!
# pylint: disable=protected-access
def _shift_right_arithmetic(x, y):
  if x.dtype.is_unsigned:
    assert x.dtype == y.dtype
    orig_dtype = x.dtype
    signed_dtype = _UNSIGNED_TO_SIGNED_TABLE[orig_dtype]
    x = tf.cast(x, signed_dtype)
    y = tf.cast(y, signed_dtype)
    res = tf.bitwise.right_shift(x, y)
    return tf.cast(res, orig_dtype)
  else:
    return tf.bitwise.right_shift(x, y)
tf_impl[lax.shift_right_arithmetic_p] = _shift_right_arithmetic

def _shift_right_logical(x, y):
  if x.dtype.is_unsigned:
    return tf.bitwise.right_shift(x, y)
  else:
    assert x.dtype == y.dtype
    orig_dtype = x.dtype
    unsigned_dtype = _SIGNED_TO_UNSIGNED_TABLE[orig_dtype]
    x = tf.cast(x, unsigned_dtype)
    y = tf.cast(y, unsigned_dtype)
    res = tf.bitwise.right_shift(x, y)
    return tf.cast(res, orig_dtype)
tf_impl[lax.shift_right_logical_p] = _shift_right_logical

tf_impl[lax.shift_left_p] = tf.bitwise.left_shift

def _not(x):
  """Computes bitwise not with support for booleans.

  Numpy and JAX support bitwise not for booleans by applying a logical not!
  This means that applying bitwise_not yields an unexected result:
    jnp.bitwise_not(jnp.array([True, False]))
    >> DeviceArray([False,  True], dtype=bool)

  if you assume that booleans are simply casted to integers.
    jnp.bitwise_not(jnp.array([True, False]).astype(np.int32)).astype(bool)
    >> DeviceArray([True,  True], dtype=bool)
  """
  if x.dtype == tf.bool:
    return tf.logical_not(x)
  else:
    return tf.bitwise.invert(x)

tf_impl[lax.not_p] = _not

def bool_to_int8(f, argnums):
  """Computes bool valued functions using int8."""
  argnums = tf.nest.flatten(argnums)
  def wrapper(*args, **kwargs):
    if not any(args[i].dtype == tf.bool for i in argnums):
      return f(*args, **kwargs)
    else:
      args = [(tf.cast(a, tf.int8) if i in argnums else a)
              for i, a in enumerate(args)]
      out = f(*args, **kwargs)
      return tf.nest.map_structure(lambda o: tf.cast(o, tf.bool), out)
  return wrapper

tf_impl[lax.or_p] = bool_to_int8(tf.bitwise.bitwise_or, argnums=(0, 1))
tf_impl[lax.and_p] = bool_to_int8(tf.bitwise.bitwise_and, argnums=(0, 1))
tf_impl[lax.xor_p] = bool_to_int8(tf.bitwise.bitwise_xor, argnums=(0, 1))

tf_impl[lax.eq_p] = wrap_binary_op(tf.math.equal)
tf_impl[lax.ne_p] = wrap_binary_op(tf.math.not_equal)
tf_impl[lax.ge_p] = wrap_binary_op(tf.math.greater_equal)
tf_impl[lax.gt_p] = wrap_binary_op(tf.math.greater)
tf_impl[lax.le_p] = wrap_binary_op(tf.math.less_equal)
tf_impl[lax.lt_p] = wrap_binary_op(tf.math.less)


def _convert_element_type(operand, new_dtype, old_dtype):
  del old_dtype
  return tf.dtypes.cast(operand, new_dtype)
tf_impl[lax.convert_element_type_p] = _convert_element_type


def _bitcast_convert_type(operand, new_dtype):
  return tf.bitcast(operand, new_dtype)
tf_impl[lax.bitcast_convert_type_p] = _bitcast_convert_type


def _clamp(minval, operand, maxval):
  return tf.clip_by_value(operand, minval, maxval)
tf_impl[lax.clamp_p] = _clamp


def _concatenate(*operands, dimension=None):
  return tf.concat(promote_types(*operands), axis=dimension)
tf_impl[lax.concatenate_p] = _concatenate


def _conv_general_proto(dimension_numbers):
  """Converts a ConvDimensionNumbers to an XLA ConvolutionDimensionNumbers."""
  assert isinstance(dimension_numbers, lax.ConvDimensionNumbers)
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  proto = xla_data_pb2.ConvolutionDimensionNumbers()
  proto.input_batch_dimension = lhs_spec[0]
  proto.input_feature_dimension = lhs_spec[1]
  proto.output_batch_dimension = out_spec[0]
  proto.output_feature_dimension = out_spec[1]
  proto.kernel_output_feature_dimension = rhs_spec[0]
  proto.kernel_input_feature_dimension = rhs_spec[1]
  proto.input_spatial_dimensions.extend(lhs_spec[2:])
  proto.kernel_spatial_dimensions.extend(rhs_spec[2:])
  proto.output_spatial_dimensions.extend(out_spec[2:])
  return proto


def _infer_shape_jax(f, *vals, **params):
  avals = map(abstractify, vals)
  return pe.abstract_eval_fun(lambda *a, **k: tree_util.tree_leaves(f(*a, **k)),
                              *avals, **params)


def _conv_general_dilated_shape(lhs, rhs, window_strides, padding, lhs_dilation,
                                rhs_dilation, dimension_numbers,
                                feature_group_count, batch_group_count,
                                lhs_shape, rhs_shape,
                                precision):
  """Shape inference for conv_general_dilated using JAX partial evaluation."""
  del lhs_shape, rhs_shape
  out, = _infer_shape_jax(
      lax.conv_general_dilated, lhs, rhs,
      window_strides=window_strides, padding=padding, lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation, dimension_numbers=dimension_numbers,
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count,
      precision=precision)
  return out.shape


def _conv_general_dilated(lhs, rhs, window_strides, padding, lhs_dilation,
                          rhs_dilation, dimension_numbers, feature_group_count,
                          batch_group_count, lhs_shape, rhs_shape, precision):
  """Implementation of lax.conv_general_dilated_p using XlaConv."""
  out_shape = _conv_general_dilated_shape(
      lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
      dimension_numbers, feature_group_count, batch_group_count, lhs_shape,
      rhs_shape, precision)
  # TODO(phawkins): handle precision
  dnums_proto = _conv_general_proto(dimension_numbers)
  assert batch_group_count == 1  # TODO(phawkins): implement batch_group_count
  out = tfxla.conv(
      lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
      dnums_proto, feature_group_count)
  # TODO(tomhennigan): tf2xla should have a shape inference function.
  out.set_shape(out_shape)
  return out


tf_impl[lax.conv_general_dilated_p] = wrap_binary_op(
    _conv_general_dilated)


def _dot_general(lhs, rhs, dimension_numbers, precision):
  """Implementation of lax.dot_general_p in terms of tf.linalg.einsum."""
  del precision
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  new_id = iter(string.ascii_letters)
  lhs_axis_ids = [next(new_id) for _ in lhs.shape]
  rhs_axis_ids = [next(new_id) for _ in rhs.shape]
  lhs_out_axis_ids = lhs_axis_ids[:]
  rhs_out_axis_ids = rhs_axis_ids[:]

  for lhs_axis, rhs_axis in zip(lhs_contracting, rhs_contracting):
    shared_id = next(new_id)
    lhs_axis_ids[lhs_axis] = shared_id
    rhs_axis_ids[rhs_axis] = shared_id
    lhs_out_axis_ids[lhs_axis] = None
    rhs_out_axis_ids[rhs_axis] = None

  batch_ids = []
  for lhs_axis, rhs_axis in zip(lhs_batch, rhs_batch):
    shared_id = next(new_id)
    lhs_axis_ids[lhs_axis] = shared_id
    rhs_axis_ids[rhs_axis] = shared_id
    lhs_out_axis_ids[lhs_axis] = None
    rhs_out_axis_ids[rhs_axis] = None
    batch_ids.append(shared_id)

  not_none = lambda x: x is not None
  out_axis_ids = list(filter(
      not_none, batch_ids + lhs_out_axis_ids + rhs_out_axis_ids))
  assert lhs.dtype == rhs.dtype
  spec = "{},{}->{}".format("".join(lhs_axis_ids),
                            "".join(rhs_axis_ids),
                            "".join(out_axis_ids))
  return tf.linalg.einsum(spec, lhs, rhs)
tf_impl[lax.dot_general_p] = wrap_binary_op(_dot_general)


def _broadcast(operand, sizes):
  return tf.broadcast_to(operand, sizes + tf.shape(operand))
tf_impl[lax.broadcast_p] = _broadcast


def _broadcast_in_dim(operand, shape, broadcast_dimensions):
  inshape = tuple(1 if i not in broadcast_dimensions else d
                  for i, d in enumerate(shape))
  return tf.broadcast_to(tf.reshape(operand, inshape), shape)
tf_impl[lax.broadcast_in_dim_p] = _broadcast_in_dim


def _reshape(operand, new_sizes, dimensions):
  if dimensions is None:
    dimensions = tf.range(tf.rank(operand))
  return tf.reshape(tf.transpose(operand, dimensions), new_sizes)
tf_impl[lax.reshape_p] = _reshape


def _squeeze(operand, dimensions):
  op_shape = _get_shape_from_tensor_or_array(operand)
  new_shape = tuple(d for i, d in enumerate(op_shape) if i not in dimensions)
  return tf.reshape(operand, new_shape)
tf_impl[lax.squeeze_p] = _squeeze


def _pad_shape(operand, padding_value, padding_config):
  out, = _infer_shape_jax(
      lax.pad, operand, padding_value, padding_config=padding_config)
  return out.shape


def _pad(operand, padding_value, padding_config):
  low, high, interior = util.unzip3(padding_config)
  if all(lo >= 0 and hi >= 0 and i == 0 for lo, hi, i in padding_config):
    return tf.pad(operand, util.safe_zip(low, high),
                  mode="CONSTANT", constant_values=padding_value)
  # TODO(necula): implement shape inference for XlaPad
  out_shape = _pad_shape(operand, padding_value, padding_config)
  out = tfxla.pad(operand, padding_value, low, high, interior)
  out.set_shape(out_shape)
  return out
tf_impl[lax.pad_p] = wrap_binary_op(_pad)


def _rev(operand, dimensions):
  return tf.reverse(operand, dimensions)
tf_impl[lax.rev_p] = _rev

tf_impl[lax.select_p] = tf.where


def _slice(operand, start_indices, limit_indices, strides):
  if strides is None:
    strides = [1] * len(start_indices)
  slices = tuple(map(slice, start_indices, limit_indices, strides))
  return operand[slices]
tf_impl[lax.slice_p] = _slice


def _dynamic_slice(operand, *start_indices, slice_sizes=None):
  return tf.slice(operand, tf.stack(start_indices), slice_sizes)
tf_impl[lax.dynamic_slice_p] = _dynamic_slice


def _dynamic_update_slice(operand, update, *start_indices):
  return tfxla.dynamic_update_slice(*promote_types(operand, update),
                                    tf.stack(start_indices))
tf_impl[lax.dynamic_update_slice_p] = _dynamic_update_slice


def _transpose(operand, permutation):
  return tf.transpose(operand, permutation)
tf_impl[lax.transpose_p] = _transpose

axes_to_axis = lambda func: lambda operand, axes: func(operand, axis=axes)

tf_impl[lax.reduce_sum_p] = (
    bool_to_int8(axes_to_axis(tf.reduce_sum), argnums=0))
tf_impl[lax.reduce_prod_p] = (
    bool_to_int8(axes_to_axis(tf.reduce_prod), argnums=0))
tf_impl[lax.reduce_max_p] = (
    bool_to_int8(axes_to_axis(tf.reduce_max), argnums=0))
tf_impl[lax.reduce_min_p] = (
    bool_to_int8(axes_to_axis(tf.reduce_min), argnums=0))
tf_impl[lax.reduce_or_p] = axes_to_axis(tf.reduce_any)
tf_impl[lax.reduce_and_p] = axes_to_axis(tf.reduce_all)


_add_fn = tf.function(tf.math.add)
_ge_fn = tf.function(tf.math.greater_equal)
_min_fn = tf.function(tf.math.minimum)
_max_fn = tf.function(tf.math.maximum)


tf_impl[lax.cumsum_p] = tf.math.cumsum
tf_impl[lax.cumprod_p] = tf.math.cumprod


def _reduce_window_shape(jax_f, operand, window_dimensions,
                         window_strides, padding, input_shape=None):
  """Shape inference function for reduce_window_{sum,min,max}."""
  params = dict(window_dimensions=window_dimensions,
                window_strides=window_strides,
                padding=padding,
                input_shape=input_shape)
  try:
    out, = _infer_shape_jax(jax_f, operand, **params)
  except TypeError:
    del params["input_shape"]
    out, = _infer_shape_jax(jax_f, operand, **params)
  return out.shape


def _get_shape_from_tensor_or_array(x):
  if isinstance(x.shape, tf.TensorShape):
    return tuple(x.shape.as_list())
  return tuple(x.shape)


def _reduce_window(jax_f, reducer, init_val, operand, window_dimensions,
                   window_strides, padding, input_shape=None):
  """TensorFlow implementation of reduce_window_{sum,min,max}."""
  del input_shape
  # TODO(tomhennigan): tf2xla should have a shape inference function.
  out_shape = _reduce_window_shape(jax_f, operand, window_dimensions,
                                   window_strides, padding)
  padding = lax.padtype_to_pads(_get_shape_from_tensor_or_array(operand),
                                window_dimensions,
                                window_strides, padding)
  a = tf.constant(0, operand.dtype)
  reducer_fn = reducer.get_concrete_function(a, a)
  out = tfxla.reduce_window(operand, tf.constant(init_val, operand.dtype),
                            reducer_fn, window_dimensions,
                            window_strides, padding=padding)
  out.set_shape(out_shape)
  return out
# pylint: disable=protected-access
tf_impl[lax.reduce_window_sum_p] = (
    functools.partial(_reduce_window, lax._reduce_window_sum, _add_fn, 0))
tf_impl[lax.reduce_window_min_p] = (
    functools.partial(_reduce_window, lax._reduce_window_min, _min_fn, np.inf))
tf_impl[lax.reduce_window_max_p] = (
    functools.partial(_reduce_window, lax._reduce_window_max, _max_fn, -np.inf))
# pylint: enable=protected-access


def _select_and_scatter_add(
    operand, source, init_value, select_jaxpr, select_consts, scatter_jaxpr,
    scatter_consts, window_dimensions, window_strides, padding):
  del select_jaxpr, select_consts, scatter_jaxpr, scatter_consts
  # TODO(phawkins): handle the select and scatter jaxprs correctly.
  a = tf.constant(0, operand.dtype)
  select_fn = _ge_fn.get_concrete_function(a, a)
  scatter_fn = _add_fn.get_concrete_function(a, a)
  return tfxla.select_and_scatter(operand, window_dimensions, window_strides,
                                  padding, source, init_value, select_fn,
                                  scatter_fn)
tf_impl[lax.select_and_scatter_add_p] = _select_and_scatter_add


def uadd(a, *b):
  """Workaround to support + with uint32 (not supported in TF)."""
  # Note: Tensorflow's add_n doesn't support broadcasting.
  b = [tf.broadcast_to(b, tf.shape(a)) for b in b]
  return tf.add_n([a] + b)

# TODO(necula): do not repeat the definition of threefry here. Note that on
#  CPU we don't have a direct definition of the primitive; we expand it
#  using xla.lower_fun. Could we do something similar here rather than
#  repeating its definition?
def _threefry2x32(key1, key2, x1, x2):
  """Tensorflow implementation of the jax PRNG."""
  def rotate_left(x, d):
    """Rotate left."""
    return tf.bitwise.bitwise_or(
        tf.bitwise.left_shift(x, np.uint32(d)),
        tf.bitwise.right_shift(x, np.uint32(32 - d)))

  def apply_round(v1, v2, rot):
    v1 = uadd(v1, v2)
    v2 = rotate_left(v2, rot)
    v2 = tf.bitwise.bitwise_xor(v1, v2)
    return v1, v2

  rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  magic_number = tf.constant(np.uint32(0x1BD11BDA), dtype=tf.uint32)

  key3 = tf.bitwise.bitwise_xor(key1,
                                tf.bitwise.bitwise_xor(key2, magic_number))

  x1 = uadd(x1, key1)
  x2 = uadd(x2, key2)

  for r in rotations[0]:
    x1, x2 = apply_round(x1, x2, r)
  x1 = uadd(x1, key2)
  x2 = uadd(x2, key3, np.uint32(1))

  for r in rotations[1]:
    x1, x2 = apply_round(x1, x2, r)
  x1 = uadd(x1, key3)
  x2 = uadd(x2, key1, np.uint32(2))

  for r in rotations[0]:
    x1, x2 = apply_round(x1, x2, r)
  x1 = uadd(x1, key1)
  x2 = uadd(x2, key2, np.uint32(3))

  for r in rotations[1]:
    x1, x2 = apply_round(x1, x2, r)
  x1 = uadd(x1, key2)
  x2 = uadd(x2, key3, np.uint32(4))

  for r in rotations[0]:
    x1, x2 = apply_round(x1, x2, r)
  x1 = uadd(x1, key3)
  x2 = uadd(x2, key1, np.uint32(5))

  return x1, x2

tf_impl[jax.random.threefry2x32_p] = _threefry2x32


def _gather_dimensions_proto(indices_shape, dimension_numbers):
  proto = xla_data_pb2.GatherDimensionNumbers()
  proto.offset_dims.extend(dimension_numbers.offset_dims)
  proto.collapsed_slice_dims.extend(dimension_numbers.collapsed_slice_dims)
  proto.start_index_map.extend(dimension_numbers.start_index_map)
  assert indices_shape
  proto.index_vector_dim = len(indices_shape) - 1
  return proto


def _gather_shape(operand, start_indices, dimension_numbers, slice_sizes):
  out, = _infer_shape_jax(
      lax.gather, operand, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=slice_sizes)
  return out.shape


def _try_tf_gather(operand, start_indices, dimension_numbers, slice_sizes):
  # Handle only the case when batch_dims=0.

  # Find axis to match the tf.gather semantics
  # Let I = len(indices_shape)
  # let O = len(op_shape)
  # slice_sizes == op_shape[:axis] + (1,) + op_shape[axis+1:]
  # collapsed_slice_dims == (axis,)
  # start_index_map == (axis,)
  # offset_dims == (0, 1, ..., axis - 1, axis + I, ..., O + I - 1)
  op_shape = np.shape(operand)
  assert len(op_shape) == len(slice_sizes)
  if not (len(op_shape) >= 1 and
          len(dimension_numbers.start_index_map) == 1 and
          len(dimension_numbers.collapsed_slice_dims) == 1 and
          dimension_numbers.collapsed_slice_dims[0] == dimension_numbers.start_index_map[0] and
          len(dimension_numbers.offset_dims) == len(op_shape) - 1):
    return None
  # We added a trailing dimension of size 1
  if start_indices.shape[-1] != 1:
    return None
  # Guess the axis
  axis = dimension_numbers.collapsed_slice_dims[0]
  index_dims = len(np.shape(start_indices)) - 1
  expected_offset_dims = tuple(
      list(range(axis)) +
      list(range(axis + index_dims, len(op_shape) + index_dims - 1)))
  if dimension_numbers.offset_dims != expected_offset_dims:
    return None
  expected_slice_sizes = op_shape[:axis] + (1,) + op_shape[axis + 1:]
  if slice_sizes != expected_slice_sizes:
    return None
  # TODO: should we allow ourselves to add a reshape, or should we strictly
  #  convert 1:1, or go to TFXLA when not possible?
  start_indices = tf.reshape(start_indices, start_indices.shape[0:-1])
  return tf.gather(operand, start_indices, axis=axis, batch_dims=0)


@functools.partial(bool_to_int8, argnums=0)
def _gather(operand, start_indices, dimension_numbers, slice_sizes):
  """Tensorflow implementation of gather."""
  res = _try_tf_gather(operand, start_indices, dimension_numbers, slice_sizes)
  if res is not None:
    return res
  out_shape = _gather_shape(
      operand, start_indices, dimension_numbers, slice_sizes)
  proto = _gather_dimensions_proto(start_indices.shape, dimension_numbers)
  out, = tf.xla.experimental.compile(
      lambda o, s: tfxla.gather(o, s, proto, slice_sizes, False),
      [operand, start_indices])
  out.set_shape(out_shape)
  return out
tf_impl[lax.gather_p] = _gather


def _scatter_dimensions_proto(indices_shape, dimension_numbers):
  proto = xla_data_pb2.ScatterDimensionNumbers()
  proto.update_window_dims.extend(dimension_numbers.update_window_dims)
  proto.inserted_window_dims.extend(dimension_numbers.inserted_window_dims)
  proto.scatter_dims_to_operand_dims.extend(
      dimension_numbers.scatter_dims_to_operand_dims)
  assert indices_shape
  proto.index_vector_dim = len(indices_shape) - 1
  return proto


def _scatter_shape(operand, scatter_indices, updates, dimension_numbers):
  out, = _infer_shape_jax(
      lax.scatter, operand, scatter_indices, updates,
      dimension_numbers=dimension_numbers)
  return out.shape


@functools.partial(bool_to_int8, argnums=(1, 3))
def _scatter(update_computation, operand, scatter_indices, updates,
             update_jaxpr, update_consts, dimension_numbers):
  """Tensorflow implementation of scatter with an update computation."""
  del update_jaxpr, update_consts
  out_shape = _scatter_shape(operand, scatter_indices, updates,
                             dimension_numbers)
  proto = _scatter_dimensions_proto(scatter_indices.shape, dimension_numbers)
  o_spec = tf.TensorSpec(None, dtype=operand.dtype)
  xla_update_computation = (
      tf.function(update_computation).get_concrete_function(o_spec, o_spec))
  out, = tf.xla.experimental.compile(
      lambda o, s, u: tfxla.scatter(o, s, u, xla_update_computation, proto),
      [operand, scatter_indices, updates])
  out.set_shape(out_shape)
  return out

tf_impl[lax.scatter_p] = functools.partial(_scatter, lambda x, y: y)
tf_impl[lax.scatter_min_p] = functools.partial(_scatter, tf.math.minimum)
tf_impl[lax.scatter_max_p] = functools.partial(_scatter, tf.math.maximum)
tf_impl[lax.scatter_mul_p] = functools.partial(_scatter, tf.math.multiply)
tf_impl[lax.scatter_add_p] = functools.partial(_scatter, tf.math.add)


def _cond(index: TfVal, *operands: TfValOrUnit,
          branches: Sequence[core.TypedJaxpr],
          linear: Sequence[bool]) -> Sequence[TfValOrUnit]:
  del linear
  # tf.cond needs lambdas with no arguments.
  branches_tf = [functools.partial(_interpret_jaxpr, jaxpr, *operands)
                 for jaxpr in branches]
  res_tf: Sequence[TfVal] = tf.switch_case(index, branches_tf)
  return _tfval_add_unit(res_tf, branches[0].out_avals)

tf_impl[lax.cond_p] = _cond


def _while(*args: TfValOrUnit, cond_nconsts: int, cond_jaxpr: core.TypedJaxpr,
           body_nconsts: int, body_jaxpr: core.TypedJaxpr) -> Sequence[TfValOrUnit]:
  cond_consts, body_consts, init_carry = util.split_list(args, [cond_nconsts,
                                                                body_nconsts])
  if cond_jaxpr.out_avals[0].shape:  # type: ignore[attr-defined]
    # The conditional is not a scalar, this must be a batched while
    return _batched_cond_while(*args,
                               cond_nconsts=cond_nconsts, cond_jaxpr=cond_jaxpr,
                               body_nconsts=body_nconsts, body_jaxpr=body_jaxpr)

  # The conditional must return a single value to TF
  def cond_tf_func(*args: TfVal) -> TfVal:
    pred, = _interpret_jaxpr(cond_jaxpr, *cond_consts, *args)
    return pred
  body_tf_func = functools.partial(_interpret_jaxpr, body_jaxpr, *body_consts)
  res_tf = tf.while_loop(cond_tf_func, body_tf_func, _tfval_remove_unit(init_carry))
  return _tfval_add_unit(res_tf, body_jaxpr.out_avals)


def _batched_cond_while(*args: TfValOrUnit,
                        cond_nconsts: int, cond_jaxpr: core.TypedJaxpr,
                        body_nconsts: int, body_jaxpr: core.TypedJaxpr
                        ) -> Sequence[TfValOrUnit]:
  """Interprets a while_loop with a batched condition.

  A batched while has a conditional that returns a tensor of booleans, and
  a body that returns a list of tensors whose leading dimensions match those
  of the conditional tensor.

  We need to turn it into a while with scalar boolean conditional. We will
  expand the loop carry to include a prefix with the current tensor boolean
  condition. We prepend to the loop the first calculation of the tensor boolean
  condition. The loop condition will use a "reduce_any" to calculate a scalar
  boolean from the tensor boolean condition. The end of the loop body will
  compute the new carry using a "tf.where", and we compute the new tensor
  boolean condition.
  """
  cond_consts, body_consts, init_carry = util.split_list(args, [cond_nconsts,
                                                                body_nconsts])
  # Initial computation of batched condition
  init_pred_b, = _interpret_jaxpr(cond_jaxpr, *cond_consts, *init_carry)
  assert init_pred_b is not core.unit

  def new_cond_tf_func(pred_b: TfVal, *carry: TfVal) -> TfVal:
    pred = tf.reduce_any(pred_b, axis=list(range(len(pred_b.shape))))
    return pred

  def new_body_tf_func(pred_b: TfVal, *carry: TfVal) -> Sequence[TfVal]:
    new_carry: Sequence[TfVal] = _interpret_jaxpr(body_jaxpr,
                                                  *body_consts, *carry)

    def select_one_carry(new_c: TfVal, c: TfVal) -> TfVal:
      pred_b_bcast = _broadcast_in_dim(pred_b, new_c.shape,
                                       list(range(len(pred_b.shape))))
      return tf.where(pred_b_bcast, new_c, c)

    selected_carry: Sequence[TfVal] = list(
      util.safe_map(select_one_carry, new_carry, carry))
    next_pred_b, = _interpret_jaxpr(cond_jaxpr, *cond_consts, *selected_carry)
    return (next_pred_b, *selected_carry)

  _, *res_carry = tf.while_loop(new_cond_tf_func, new_body_tf_func,
                                _tfval_remove_unit((init_pred_b, *init_carry)))
  return _tfval_add_unit(res_carry, body_jaxpr.out_avals)

tf_impl[lax.while_p] = _while


def _scan(*tf_args : TfValOrUnit, **kwargs) -> Sequence[TfValOrUnit]:
  # We use the scan impl rule to rewrite in terms of while. We wrap it under
  # _interpret_fun to abstract the TF values from scan_impl.
  def func1(*jax_args):
    return lax_control_flow._scan_impl(*jax_args, **kwargs)

  return _interpret_fun(lu.wrap_init(func1), tf_args)

tf_impl[lax.scan_p] = _scan


def _register_checkpoint_pytrees():
  """Registers TF custom container types as pytrees."""
  m = tf.Module()
  # The types here are automagically changed by TensorFlow's checkpointing
  # infrastructure.
  m.a = (tf.Module(), tf.Module())
  m.b = [tf.Module(), tf.Module()]
  m.c = {"a": tf.Module()}
  tuple_wrapper = type(m.a)
  list_wrapper = type(m.b)
  dict_wrapper = type(m.c)

  # TF AutoTrackable swaps container types out for wrappers.
  assert tuple_wrapper is not tuple
  assert list_wrapper is not list
  assert dict_wrapper is not dict

  jax.tree_util.register_pytree_node(
      tuple_wrapper, lambda xs: (tuple(xs), None), lambda _, xs: tuple(xs))

  jax.tree_util.register_pytree_node(
      list_wrapper, lambda xs: (tuple(xs), None), lambda _, xs: list(xs))

  jax.tree_util.register_pytree_node(
      dict_wrapper,
      lambda s: (tuple(s.values()), tuple(s.keys())),
      lambda k, xs: dict(zip(k, xs)))

_register_checkpoint_pytrees()
