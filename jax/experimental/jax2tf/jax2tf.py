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
import functools
import string
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple, Union

import jax
from jax import abstract_arrays
from jax import ad_util
from jax import api
from jax import core
from jax import custom_derivatives
from jax import dtypes
from jax import lax
from jax import lax_linalg
from jax import linear_util as lu
from jax import numpy as jnp
from jax import random
from jax import tree_util
from jax import util
from jax.api_util import flatten_fun
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import pxla
from jax.interpreters import xla
from jax.lax import lax_control_flow
from jax.lax import lax_fft
import numpy as np
import tensorflow as tf  # type: ignore[import]

# These don't have public equivalents.
# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla  # type: ignore[import]
from tensorflow.compiler.xla import xla_data_pb2  # type: ignore[import]

from jaxlib import xla_client


# A value suitable in a TF tracing context: tf.Tensor, tf.Variable,
# or Python scalar or numpy.ndarray. (A tf.EagerTensor is a tf.Tensor.)
TfVal = Any
def _is_tfval(v: TfVal) -> bool:
  if isinstance(v, (tf.Tensor, tf.Variable)):
    return True
  try:
    # Note: this conversion is overkill and just intended as a type check; this code
    # is in principle only run if core.skip_checks is False.
    _safe_convert_to_tensor(v)
    return True
  except ValueError:
    return False

def _safe_convert_to_tensor(val, dtype=None):
  """Converts val to a Tensor.

  This method wraps TensorFlow's `convert_to_tensor
  <https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor>`_ operator with
  special case handling for when `val` is an instance of `jnp.bfloat16` or has a
  `jnp.bfloat16` dtype. Because this type is not supported in numpy and different
  from `tf.bfloat16.as_numpy_dtype`, `tf.convert_to_tensor` runs into trouble when
  trying to convert it. In such a case, we solve the problem by viewing val as a
  `ndarray` with a `uint16` dtype, for which conversion is properly defined. Then, we
  simply bitcast it back to `bfloat16`.
  """
  dtype = dtype if dtype else (val.dtype if hasattr(val, "dtype") else None)
  if (dtype == jnp.bfloat16 or isinstance(val, jnp.bfloat16)):
    if not isinstance(val, jnp.ndarray):
      val = np.array(val, jnp.bfloat16)

    val = tf.bitcast(tf.convert_to_tensor(val.view(jnp.uint16),
                                          dtype=to_tf_dtype(jnp.uint16)),
                     type=to_tf_dtype(jnp.bfloat16))
  else:
    conversion_type = to_tf_dtype(dtype) if dtype else None
    val = tf.convert_to_tensor(val, dtype=conversion_type)

  return val

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
  When the aval is a unit, the corresponding value is either core.unit,
  or an EagerTensor with the value NaN (we use tf.nan as a concrete TF value
  for units, see _tfval_remove_unit) or may even be a Tensor if we are building
  graphs and the NaN value is abstracted.
  """
  def add_unit(v: TfValOrUnit, aval: core.AbstractValue):
    if not core.skip_checks:
      if aval is core.abstract_unit:
        if v is not core.unit:
          assert isinstance(v, tf.Tensor)
          if v.device:  # Only for EagerTensor
            assert tf.math.is_nan(v)
      else:
        assert _is_tfval(v)
    return core.unit if aval is core.abstract_unit else v
  return util.safe_map(add_unit, vals, avals)

# The implementation rules for primitives. The rule will be called with the
# arguments (TfValOrUnit) and must return TfValOrUnit (or a sequence thereof,
# if primitive.multiple_results). The vast majority of primitives do not need
# to worry about core.unit inputs or results. The exception are primarily the
# control-flow primitives.
tf_impl: Dict[core.Primitive,
              Callable[..., Any]] = {}

def convert(fun, with_gradient=True):
  """Transforms `fun` to be executed by TensorFlow.

  Args:
    fun: Function to be transformed. Its arguments and return value should be
      JAX arrays, or (nested) standard Python containers (tuple/list/dict)
      thereof.
    with_gradient: if set, will add a tf.custom_gradient to the converted
      function, by converting the ``jax.vjp(fun)``. Only first-order
      differentiation is supported for now. If the converted function is
      saved in a SavedModel, the custom gradients are currently lost and
      an error will be raised if a gradient computation is attempted.

  Returns:
    A version of `fun` that expects TfVals as arguments (or
    tuple/lists/dicts) thereof, and returns TfVals as outputs.
  """
  api._check_callable(fun)

  def converted_fun(*args: TfVal) -> TfVal:
    # This function may take pytrees of TfVals. We can only set
    # tf.custom_gradient on functions that take a flat argument list.
    args_flat, in_tree = tree_util.tree_flatten((args, {}))
    for a in args_flat:
      if not _is_tfvalorunit(a):
        msg = (f"Argument {a} of type {type(a)} of jax2tf.convert(f) should "
               "be NumPy array, scalar, tf.Variable, or tf.Tensor")
        raise TypeError(msg)

    f = lu.wrap_init(fun)
    # out_tree_thunk() will be the output tree, after running _interpret_fun.
    flat_fun, out_tree_thunk = flatten_fun(f, in_tree)

    # Prepare the grad_fn for tf.custom_gradient.
    def converted_grad_fn(*out_cts_flat: TfVal, **kwargs):
      # TODO(cl/318778369): change **kwargs with variables=None
      variables = kwargs.get("variables", [])
      if variables:
        raise ValueError("Unexpected variables used in forward pass. "
                         "This should not happen for first-order differentiation. "
                         f"variables={variables}")

      def fun_vjp_jax(args_jax, out_cts_jax):
        # One may think that we can get the pullback while we are converting
        # the main function in the first place. That is problematic, because the
        # pullback may contain captured tracers from the conversion of the
        # main function. Those tracers will confuse the conversion of the
        # pullback. So, we construct the vjp anew.
        _, pullback_jax = jax.vjp(fun, *args_jax)
        return pullback_jax(out_cts_jax)
      out_cts = tree_util.tree_unflatten(out_tree_thunk(), out_cts_flat)
      in_cts = convert(fun_vjp_jax, with_gradient=False)(args, out_cts)
      return in_cts

    if with_gradient:
      @tf.custom_gradient
      def converted_fun_flat_with_custom_gradient(*args_flat: TfVal) -> TfVal:
        return _interpret_fun(flat_fun, args_flat), converted_grad_fn

      out_flat = converted_fun_flat_with_custom_gradient(*args_flat)
    else:
      out_flat_raw = _interpret_fun(flat_fun, args_flat)
      message = ("The jax2tf-converted function does not support gradients. "
                 "Use `with_gradient` parameter to enable gradients")
      # We use PreventGradient, which is propagated through a SavedModel.
      out_flat = [tf.raw_ops.PreventGradient(input=o, message=message)
                  for o in out_flat_raw]

    out = tree_util.tree_unflatten(out_tree_thunk(), out_flat)
    return out

  return converted_fun

# Internals


def _interpret_fun(fun: lu.WrappedFun,
                   in_vals: Sequence[TfValOrUnit]) -> Sequence[TfValOrUnit]:
  with core.new_main(TensorFlowTrace) as main:
    fun = _interpret_subtrace(fun, main)
    out_vals: Sequence[TfValOrUnit] = fun.call_wrapped(*in_vals)
    del main
  return out_vals

def _convert_jax_impl(jax_impl: Callable, multiple_results=True) -> Callable:
  """Convert the JAX implementation of a primitive.

  Args:
    jax_impl: typically the impl-rule for a primitive, with signature
      `(*args: JaxVal, **kwargs) -> Sequence[JaxVal]`. This function implements
      a primitive in terms of other primitives.
    multiple_results: whether `jax_impl` returns a sequence of results.

  Returns:
     a function with signature `(*args: TfValOrUnit, **kwargs) -> Sequence[TfValOrUnit]`.
  """
  def wrapped(*tf_args: TfValOrUnit, **kwargs) -> Union[TfValOrUnit, Sequence[TfValOrUnit]]:

    # We wrap the jax_impl under _interpret_fun to abstract the TF values
    # from jax_impl and turn them into JAX abstract values.
    def jax_impl_jax_args(*jax_args):
      jax_results = jax_impl(*jax_args, **kwargs)
      return jax_results if multiple_results else [jax_results]

    tf_results = _interpret_fun(lu.wrap_init(jax_impl_jax_args), tf_args)
    return tf_results if multiple_results else tf_results[0]
  return wrapped


@lu.transformation
def _interpret_subtrace(main: core.MainTrace, *in_vals: TfValOrUnit):
  trace = TensorFlowTrace(main, core.cur_sublevel())
  in_tracers = tuple(TensorFlowTracer(trace, val) for val in in_vals)
  outs = yield in_tracers, {}  # type: Sequence[TfValOrUnit]
  out_tracers: Iterable[TensorFlowTracer] = map(trace.full_raise, outs)  # type: ignore
  out_vals: Sequence[TfValOrUnit] = tuple(t.val for t in out_tracers)
  yield out_vals


def _interpret_jaxpr(jaxpr: core.ClosedJaxpr, *args: TfValOrUnit) -> Sequence[TfVal]:
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
  return abstract_arrays.ShapedArray(tuple(t.shape), to_jax_dtype(t.dtype))

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
      if np.dtype(aval.dtype) != to_jax_dtype(val.dtype):  # type: ignore
        # This is expected when JAX 64-bit is not enabled
        self.val = tf.cast(val, dtype=aval.dtype)
      else:
        self.val = val
    else:  # Must be a numeric value
      assert core.skip_checks or _is_tfval(val), f"Non TfVal: {val}"
      aval = xla.abstractify(val)  # type: ignore

      self.val = _safe_convert_to_tensor(val, dtype=aval.dtype)

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
    """Lifts a core.Tracer from a lower-level main into the TensorFlowTrace."""
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
    f = _interpret_subtrace(f, self.main)
    vals_out: Sequence[TfValOrUnit] = f.call_wrapped(*vals)
    return [TensorFlowTracer(self, v) for v in vals_out]

  def post_process_call(self, call_primitive: core.Primitive,
                        out_tracers: Sequence[TensorFlowTracer], params):
    # We encountered a call primitive, e.g., remat_call_p, whose result
    # (out_tracers) include TensorFlowTracer that were not passed through
    # its arguments (captured from the environment).
    vals = tuple(t.val for t in out_tracers)
    main = self.main
    def todo(vals: Sequence[TfValOrUnit]):
      trace = TensorFlowTrace(main, core.cur_sublevel())
      return map(functools.partial(TensorFlowTracer, trace), vals)
    return vals, todo

  def process_map(self, map_primitive, f, tracers, params):
    raise NotImplementedError("process_map")

  def post_process_map(self, map_primitive, out_tracers, params):
    raise NotImplementedError("post_process_map")

  def get_primitive_impl(self, p):
    try:
      return tf_impl[p]
    except KeyError as err:
      msg = "TensorFlow interpretation rule for '{}' not implemented"
      raise NotImplementedError(msg.format(p)) from err

def to_tf_dtype(jax_dtype):
  if jax_dtype == jnp.bfloat16:
    return tf.bfloat16
  elif jax_dtype == dtypes.float0:
    return tf.float32
  else:
    return tf.dtypes.as_dtype(jax_dtype)

def to_jax_dtype(tf_dtype):
  return jnp.bfloat16 if tf_dtype == tf.bfloat16 else tf_dtype.as_numpy_dtype

def promote_types(*values):
  """Returns values casted to a common type using jnp.promote_types."""
  dtype = to_tf_dtype(functools.reduce(
      jnp.promote_types, (to_jax_dtype(v.dtype) for v in values)))
  return tuple(tf.cast(v, dtype) for v in values)


def wrap_binary_op(func):
  def wrapped_func(lhs, rhs, **kwargs):
    return func(*promote_types(lhs, rhs), **kwargs)
  return wrapped_func

def _unexpected_primitive(p: core.Primitive, *args, **kwargs):
  assert False, f"Encountered unexpected primitive {p}"


for unexpected in [
    # Call primitives are inlined
    xla.xla_call_p, pe.remat_call_p, core.call_p]:

  tf_impl[unexpected] = functools.partial(_unexpected_primitive, unexpected)

# Primitives that are not yet implemented must be explicitly declared here.
tf_not_yet_impl = [
  lax.reduce_p, lax.rng_uniform_p,

  lax.linear_solve_p,
  lax_linalg.lu_p,
  lax_linalg.triangular_solve_p,

  lax.igamma_grad_a_p,
  lax.random_gamma_grad_p,

  # Not high priority?
  lax.after_all_p, lax.all_to_all_p, lax.create_token_p, lax.cummax_p, lax.cummin_p,
  lax.infeed_p, lax.outfeed_p, lax.pmax_p, lax.pmin_p, lax.ppermute_p, lax.psum_p,
  lax.axis_index_p,

  pxla.xla_pmap_p,
]

try:
  tf_impl[lax.lax.tie_in_p] = lambda x, y: y
except AttributeError:
  pass
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

def _population_count(x):
  orig_dtype = x.dtype
  return tf.cast(tf.raw_ops.PopulationCount(x=x), orig_dtype)

tf_impl[lax.population_count_p] = _population_count
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
    quotient = tf.math.floordiv(lhs, rhs)
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

tf_impl[lax_linalg.cholesky_p] = tf.linalg.cholesky

def _convert_element_type(operand, new_dtype, old_dtype):
  del old_dtype
  return tf.dtypes.cast(operand, to_tf_dtype(new_dtype))
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


def _conv_general_dimension_numbers_proto(dimension_numbers):
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

def _conv_general_precision_config_proto(precision):
  """Convert an integer to an XLA.PrecisionConfig."""
  if precision is None:
    return None

  proto = xla_data_pb2.PrecisionConfig()
  proto.operand_precision.append(int(precision))
  return proto

def _conv_general_dilated(lhs, rhs, window_strides, padding, lhs_dilation,
                          rhs_dilation, dimension_numbers, feature_group_count,
                          batch_group_count, lhs_shape, rhs_shape, precision):
  """Implementation of lax.conv_general_dilated_p using XlaConv."""
  out_shape = _conv_general_dilated_shape(
      lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
      dimension_numbers, feature_group_count, batch_group_count, lhs_shape,
      rhs_shape, precision)
  dnums_proto = _conv_general_dimension_numbers_proto(dimension_numbers)
  precision_config_proto = _conv_general_precision_config_proto(precision)
  assert batch_group_count == 1  # TODO(phawkins): implement batch_group_count
  out = tfxla.conv(
      lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
      dnums_proto, feature_group_count=feature_group_count,
      precision_config=precision_config_proto)
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

def _argminmax(fn, operand, axes, index_dtype):
  axis, = axes
  # TODO(phawkins): handle axes larger than 2^31.
  return fn(operand, axis=axis, output_type=to_tf_dtype(index_dtype))

tf_impl[lax.argmin_p] = functools.partial(_argminmax, tf.math.argmin)
tf_impl[lax.argmax_p] = functools.partial(_argminmax, tf.math.argmax)


_add_fn = tf.function(tf.math.add)
_ge_fn = tf.function(tf.math.greater_equal)

tf_impl[lax.cumsum_p] = tf.math.cumsum
tf_impl[lax.cumprod_p] = tf.math.cumprod

def _select_and_gather_add(tangents: TfVal,
                           operand: TfVal,
                           select_prim: core.Primitive,
                           window_dimensions: Sequence[int],
                           window_strides: Sequence[int],
                           base_dilation: Sequence[int],
                           window_dilation: Sequence[int],
                           padding: Sequence[Tuple[int, int]]):
  # Note: this function follows the pattern in
  # jax.lax._select_and_gather_add_translation.
  dtype = operand.dtype
  nbits = dtypes.finfo(dtype.as_numpy_dtype).bits

  # Specializing the function for 64 bits. Only up to 32 bits are supported on TPU,
  # we thus intend to let the code throw a different exception on this platform.
  max_bits = 64

  assert nbits <= max_bits
  double_word_reduction = nbits * 2 <= max_bits

  const = lambda dtype, x: tf.constant(np.array(x), dtype)

  if double_word_reduction:
    word_dtype = lax.lax._UINT_DTYPES[nbits]
    double_word_dtype = lax.lax._UINT_DTYPES[nbits * 2]

    # Packs two values into a tuple.
    def pack(a, b):
      a = _bitcast_convert_type(a, word_dtype)
      b = _bitcast_convert_type(b, word_dtype)
      a = _convert_element_type(a, double_word_dtype, word_dtype)
      b = _convert_element_type(b, double_word_dtype, word_dtype)
      a = tf.bitwise.left_shift(a, const(double_word_dtype, nbits))
      return tf.bitwise.bitwise_or(a, b)

    # Unpacks the first element of a tuple.
    def fst(t):
      st = _shift_right_logical(t, const(double_word_dtype, nbits))
      return _bitcast_convert_type(
        _convert_element_type(st, word_dtype, double_word_dtype), dtype
      )

    # Unpacks the second element of a tuple.
    def snd(t):
      return _bitcast_convert_type(
        _convert_element_type(t, word_dtype, double_word_dtype), dtype
      )

  else:
    raise NotImplementedError(f"TODO: need to pack {nbits * 2} bits but this platform can only go up to {max_bits} bits.")

  assert select_prim is lax.ge_p or select_prim is lax.le_p, select_prim

  def reducer(x, y):
    which = tf_impl[select_prim]
    return tf_impl[lax.select_p](which(fst(x), fst(y)), x=x, y=y)

  init = -np.inf if select_prim is lax.ge_p else np.inf
  init_identity = lambda x: pack(const(dtype, init), const(dtype, 0))

  out = _specialized_reduce_window(reducer, init_identity,
                                   pack(operand, tangents), window_dimensions,
                                   window_strides, padding, base_dilation,
                                   window_dilation)

  return snd(out)

tf_impl[lax.select_and_gather_add_p] = _select_and_gather_add


def _reduce_window_shape(jax_f, operand, window_dimensions,
                         window_strides, padding, base_dilation,
                         window_dilation, input_shape=None):
  """Shape inference function for reduce_window_{sum,min,max}."""
  params = dict(window_dimensions=window_dimensions,
                window_strides=window_strides,
                padding=padding,
                base_dilation=base_dilation,
                window_dilation=window_dilation,
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

def _common_reduce_window(operand, init_val, reducer, window_dimensions,
                          window_strides, padding, base_dilation,
                          window_dilation):
  # TODO(tomhennigan): tf2xla should have a shape inference function.
  out_shape = _reduce_window_shape(lax._reduce_window_min, operand,
                                   window_dimensions,
                                   window_strides, padding, base_dilation,
                                   window_dilation)

  o_spec = tf.TensorSpec(operand.shape, dtype=operand.dtype)
  reducer_fn = tf.function(reducer).get_concrete_function(o_spec, o_spec)

  if not isinstance(init_val, tf.Tensor):
    assert core.skip_checks or _is_tfval(init_val), f"Non TfVal: {init_val}"
    init_val = tf.constant(init_val, operand.dtype)

  out = tfxla.reduce_window(operand, init_val,
                            reducer_fn, window_dimensions,
                            window_strides, base_dilations=base_dilation,
                            window_dilations=window_dilation, padding=padding)
  out.set_shape(out_shape)
  return out

def _reduce_window(operand, init_value, *, jaxpr, consts, window_dimensions,
                   window_strides, padding, base_dilation, window_dilation):
  """TensorFlow implementation of reduce_window.

  Args:
    operand: N dimensional array containing elements of type T
    init_value: starting value of the reduction
    jaxpr: the jaxpr corresponding to the reduction function
    consts: the constants associated with jaxpr.
    window_dimensions: array of integers for window dimension values
    window_strides: array of integers for window stride values
    padding: array of pairs of integers for padding values
    base_dilation: array of integers for base dilation values
    window_dilation: array of integers for window dilation values

  Returns:
    The reduced operand.
  """
  assert len(consts) == 0, "Reduction computation cannot have constants"

  def reducer(arg1: TfVal, arg2: TfVal) -> TfVal:
    closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
    res, = _interpret_jaxpr(closed_jaxpr, arg1, arg2)
    return res

  return _common_reduce_window(
      operand, init_value, reducer, window_dimensions, window_strides, padding,
      base_dilation, window_dilation
  )

def _specialized_reduce_window(reducer, identity, operand, window_dimensions,
                               window_strides, padding, base_dilation,
                               window_dilation):
  """Wraps the TensorFlow reduce window operation based on a reducer and an identity
  function defining the initial value of the reduction depending on the dtype of the
  operand.

  Args:
    reducer: reduction function of type TfVal -> TfVal -> TfVal
    identity: function that takes a TensorFlow dtype as a parameter and returns the
      starting value of the reduction.
    operand: N dimensional array containing elements of type T
    window_dimensions: array of integers for window dimension values
    window_strides: array of integers for window stride values
    padding: array of pairs of integers for padding values
    base_dilation: array of integers for base dilation values
    window_dilation: array of integers for window dilation values

  Returns:
    The reduced operand.
  """

  return _common_reduce_window(
      operand, identity(operand.dtype), reducer, window_dimensions,
      window_strides, padding, base_dilation, window_dilation
  )

def _get_max_identity(tf_dtype):
  numpy_tf_dtype = tf_dtype.as_numpy_dtype
  if tf_dtype == tf.bfloat16 or dtypes.issubdtype(numpy_tf_dtype, np.inexact):
    return numpy_tf_dtype(-np.inf)
  elif dtypes.issubdtype(numpy_tf_dtype, np.integer):
    return dtypes.iinfo(numpy_tf_dtype).min
  else:
    assert dtypes.issubdtype(numpy_tf_dtype, np.bool_), (
        f"{tf_dtype} has no defined max identity"
    )
    return False

def _get_min_identity(tf_dtype):
  numpy_tf_dtype = tf_dtype.as_numpy_dtype
  if tf_dtype == tf.bfloat16 or dtypes.issubdtype(numpy_tf_dtype, np.inexact):
    return numpy_tf_dtype(np.inf)
  elif dtypes.issubdtype(numpy_tf_dtype, np.integer):
    return dtypes.iinfo(numpy_tf_dtype).max
  else:
    assert dtypes.issubdtype(numpy_tf_dtype, np.bool_), (
        f"{tf_dtype} has no defined min identity"
    )
    return True

# pylint: disable=protected-access
tf_impl[lax.reduce_window_sum_p] = (
    functools.partial(_specialized_reduce_window, tf.math.add, lambda x: 0))
tf_impl[lax.reduce_window_min_p] = (
    functools.partial(_specialized_reduce_window, tf.math.minimum,
                      _get_min_identity))
tf_impl[lax.reduce_window_max_p] = (
    functools.partial(_specialized_reduce_window, tf.math.maximum,
                      _get_max_identity))
tf_impl[lax.reduce_window_p] = _reduce_window
# pylint: enable=protected-access

def _select_and_scatter(
    operand, source, init_value, select_jaxpr, select_consts, scatter_jaxpr,
    scatter_consts, window_dimensions, window_strides, padding):
  raise NotImplementedError("TODO: jax2tf can not convert _select_and_scatter")

tf_impl[lax.select_and_scatter_p] = _select_and_scatter

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

def _threefry2x32_jax_impl(*args: TfValOrUnit):
  # We use the random._threefry2x32_lowering, but since add is not implemented
  # for uint32, we cast to int32 and back.
  args = tuple([tf.cast(a, tf.int32) for a in args])
  res = _convert_jax_impl(
    functools.partial(random._threefry2x32_lowering,
                      use_rolled_loops=False),
    multiple_results=True)(*args)
  res = tuple([tf.cast(r, tf.uint32) for r in res])
  return res
tf_impl[jax.random.threefry2x32_p] = _threefry2x32_jax_impl


# Use the vmap implementation, otherwise on TPU the performance is really bad
# With use_vmap=True on, we get about the same performance for JAX and jax2tf.
tf_impl[random.random_gamma_p] = _convert_jax_impl(
  functools.partial(random._gamma_impl, use_vmap=True),
  multiple_results=False)

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

@functools.partial(bool_to_int8, argnums=0)
def _gather(operand, start_indices, dimension_numbers, slice_sizes):
  """Tensorflow implementation of gather."""
  out_shape = _gather_shape(
      operand, start_indices, dimension_numbers, slice_sizes)
  proto = _gather_dimensions_proto(start_indices.shape, dimension_numbers)
  out = tfxla.gather(operand, start_indices, proto, slice_sizes, False)
  out.set_shape(out_shape)
  return out
tf_impl[lax.gather_p] = _gather

def _slice(operand, start_indices, limit_indices, strides):
  if strides is None:
    strides = [1] * len(start_indices)
  slices = tuple(map(slice, start_indices, limit_indices, strides))
  return operand[slices]
tf_impl[lax.slice_p] = _slice


def _dynamic_slice(operand, *start_indices, slice_sizes):
  # Here we could use tf.slice. Similarly, for lax.gather we can sometimes use
  # tf.gather. But those have different semantics for index-out-of-bounds than
  # JAX (and XLA). We have tried to force compilation, by wrapping into
  # tf.xla.experimental.compile, or tf.function(experimental_compile=True), but
  # those solutions are brittle because they do not work when nested into an
  # outer compilation (see b/162814494 and b/163006262). They also do not
  # survive well being put in a SavedModel. Hence, we now use TFXLA slicing
  # and gather ops.
  res = tfxla.dynamic_slice(operand, tf.stack(start_indices),
                            size_indices=slice_sizes)
  # TODO: implement shape inference for XlaDynamicSlice
  res.set_shape(tuple(slice_sizes))
  return res

tf_impl[lax.dynamic_slice_p] = _dynamic_slice

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

@functools.partial(bool_to_int8, argnums=(0, 2))
def _scatter(operand, scatter_indices, updates, update_jaxpr, update_consts,
             dimension_numbers, indices_are_sorted, unique_indices):
  del unique_indices
  assert len(update_consts) == 0, "Update computation cannot have constants"

  out_shape = _scatter_shape(operand, scatter_indices, updates,
                             dimension_numbers)
  proto = _scatter_dimensions_proto(scatter_indices.shape, dimension_numbers)

  def update_computation(arg1: TfVal, arg2: TfVal) -> TfVal:
    closed_jaxpr = core.ClosedJaxpr(update_jaxpr, update_consts)
    res, = _interpret_jaxpr(closed_jaxpr, arg1, arg2)
    return res

  o_spec = tf.TensorSpec((), dtype=operand.dtype)
  xla_update_computation = (
      tf.function(update_computation).get_concrete_function(o_spec, o_spec))
  out = tfxla.scatter(operand, scatter_indices, updates, xla_update_computation, proto,
                      indices_are_sorted=indices_are_sorted)
  # TODO: implement shape analysis for XlaScatter
  out.set_shape(out_shape)

  return out

tf_impl[lax.scatter_p] = _scatter
tf_impl[lax.scatter_min_p] = _scatter
tf_impl[lax.scatter_max_p] = _scatter
tf_impl[lax.scatter_mul_p] = _scatter
tf_impl[lax.scatter_add_p] = _scatter

def _dynamic_update_slice(operand, update, *start_indices):
  return tfxla.dynamic_update_slice(*promote_types(operand, update),
                                    tf.stack(start_indices))
tf_impl[lax.dynamic_update_slice_p] = _dynamic_update_slice


def _cond(index: TfVal, *operands: TfValOrUnit,
          branches: Sequence[core.ClosedJaxpr],
          linear: Sequence[bool]) -> Sequence[TfValOrUnit]:
  del linear
  # tf.cond needs lambdas with no arguments.
  branches_tf = [functools.partial(_interpret_jaxpr, jaxpr, *operands)
                 for jaxpr in branches]
  res_tf: Sequence[TfVal] = tf.switch_case(index, branches_tf)
  return _tfval_add_unit(res_tf, branches[0].out_avals)

tf_impl[lax.cond_p] = _cond


def _while(*args: TfValOrUnit, cond_nconsts: int, cond_jaxpr: core.ClosedJaxpr,
           body_nconsts: int, body_jaxpr: core.ClosedJaxpr) -> Sequence[TfValOrUnit]:
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
                        cond_nconsts: int, cond_jaxpr: core.ClosedJaxpr,
                        body_nconsts: int, body_jaxpr: core.ClosedJaxpr
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

# We use the scan impl rule to rewrite in terms of while.
tf_impl[lax.scan_p] = _convert_jax_impl(lax_control_flow._scan_impl)

def _top_k(operand: TfVal, k: int) -> Tuple[TfVal, TfVal]:
  # Some types originally incompatible with tf.math.top_k can be promoted
  # to a compatible type without loss of precision.
  def promote_tf_dtype(tf_dtype):
    if tf_dtype in [tf.bool, tf.uint8, tf.uint16]:
      return tf.uint32
    if tf_dtype in [tf.int8, tf.int16]:
      return tf.int32
    if tf_dtype is tf.float16:
      return tf.float32
    return None

  conversion_dtype = promote_tf_dtype(operand.dtype)
  if conversion_dtype:
    values, indices = tf.math.top_k(tf.dtypes.cast(operand, conversion_dtype), k=k, sorted=True)
    return tf.dtypes.cast(values, operand.dtype), indices
  else:
    return tf.math.top_k(operand, k=k, sorted=True)

tf_impl[lax.top_k_p] = _top_k

def _sort(*operand: TfVal, dimension: int, is_stable: bool, num_keys: int) -> Tuple[TfVal, ...]:
  if num_keys != 1:
    raise NotImplementedError("TODO: multiple keys")
  if len(operand) > 2:
    raise NotImplementedError("TODO: handle > 2 tensors")
  if is_stable:
    raise NotImplementedError("TODO: implement stable version of XlaSort")
  if dimension == len(operand[0].shape) - 1:
    if len(operand) == 2:
      return tuple(tfxla.key_value_sort(operand[0], operand[1]))
    else:
      return (tfxla.sort(operand[0]),)
  else:
    raise NotImplementedError("TODO: implement XlaSort for all axes")

tf_impl[lax.sort_p] = _sort

def _fft(x, fft_type, fft_lengths):
  shape = x.shape
  assert len(fft_lengths) <= len(shape)
  if ((fft_type == xla_client.FftType.IRFFT and
       fft_lengths != shape[-len(fft_lengths):-1] + ((shape[-1] - 1) * 2,)) or
      (fft_type != xla_client.FftType.IRFFT and
       fft_lengths != shape[-len(fft_lengths):])):
     raise NotImplementedError(f"Unsupported fft_lengths={fft_lengths} for fft_type={fft_type} of array with shape={shape}.")
  tf_funcs = {xla_client.FftType.FFT: [tf.signal.fft, tf.signal.fft2d,
                                       tf.signal.fft3d],
              xla_client.FftType.IFFT: [tf.signal.ifft, tf.signal.ifft2d,
                                        tf.signal.ifft3d],
              xla_client.FftType.RFFT: [tf.signal.rfft, tf.signal.rfft2d,
                                        tf.signal.rfft3d],
              xla_client.FftType.IRFFT: [tf.signal.irfft, tf.signal.irfft2d,
                                         tf.signal.irfft3d]}

  return tf_funcs[fft_type][len(fft_lengths) - 1](x)

tf_impl[lax_fft.fft_p] = _fft

def _qr(operand, full_matrices):
  return tf.linalg.qr(operand, full_matrices=full_matrices)

tf_impl[lax_linalg.qr_p] = _qr

def _svd(operand, full_matrices, compute_uv):
  result = tf.linalg.svd(operand, full_matrices, compute_uv)
  if not compute_uv:
    return result,
  s, u, v = result
  return s, u, tf.linalg.adjoint(v)

tf_impl[lax_linalg.svd_p] = _svd

def _eig(operand: TfVal, compute_left_eigenvectors: bool,
         compute_right_eigenvectors: bool):
  if compute_left_eigenvectors and compute_right_eigenvectors:
    # TODO(bchetioui): didn't find a 100% reliable, easy and satisfying way to
    # sort the left eigenvectors in the right order. The jax.numpy.linalg API
    # suggests to me that left eigenvectors are anyway seldom used, so I
    # think it is acceptable to leave as unimplemented for now.
    msg = ("Conversion of eig is not implemented when both "
           "compute_left_eigenvectors and compute_right_eigenvectors are set "
           "to True.")
    raise NotImplementedError(msg)
  elif not (compute_left_eigenvectors or compute_right_eigenvectors):
    return tuple([tf.linalg.eigvals(operand)])
  elif compute_right_eigenvectors:
    return tuple(tf.linalg.eig(operand))
  else: # compute_left_eigenvectors == True
    wH, vl = tf.linalg.eig(tf.linalg.adjoint(operand))
    wHH = tf.math.conj(wH)
    return tuple([wHH, vl])

tf_impl[lax_linalg.eig_p] = _eig

def _eigh(operand: TfVal, lower: bool):
  if operand.shape[-1] == 0:
    v, w = operand, tf.reshape(operand, operand.shape[:-1])
  else:
    if not lower:
      operand = tf.linalg.adjoint(operand)
    w, v = tf.linalg.eigh(operand)
  cast_type = { tf.complex64: tf.float32
              , tf.complex128: tf.float64 }.get(operand.dtype)
  if cast_type is not None:
    w = tf.cast(w, cast_type)
  return v, w

tf_impl[lax_linalg.eigh_p] = _eigh

def _custom_jvp_call_jaxpr(*args: TfValOrUnit,
                           fun_jaxpr: core.ClosedJaxpr,
                           jvp_jaxpr_thunk: Callable) -> Sequence[TfValOrUnit]:
  # TODO(necula): ensure that there is no AD transformation in scope
  res = _interpret_jaxpr(fun_jaxpr, *args)
  return _tfval_add_unit(res, fun_jaxpr.out_avals)

tf_impl[custom_derivatives.custom_jvp_call_jaxpr_p] = _custom_jvp_call_jaxpr


def _custom_vjp_call_jaxpr(*args: TfValOrUnit,
                           fun_jaxpr: core.ClosedJaxpr,
                           **_) -> Sequence[TfValOrUnit]:
  # TODO(necula): ensure that there is no AD transformation in scope
  res = _interpret_jaxpr(fun_jaxpr, *args)
  return _tfval_add_unit(res, fun_jaxpr.out_avals)

tf_impl[custom_derivatives.custom_vjp_call_jaxpr_p] = _custom_vjp_call_jaxpr

def _custom_lin(*args: TfValOrUnit, **_) -> Sequence[TfValOrUnit]:
  raise TypeError("can't apply forward-mode autodiff (jvp) to a custom_vjp "
                  "function.")

tf_impl[ad.custom_lin_p] = _custom_lin

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
