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
import re
import string
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import jax
from jax import ad_util, api_util, config
from jax._src import api
from jax import core, custom_derivatives, dtypes
from jax import linear_util as lu
from jax import numpy as jnp
from jax import random, tree_util
from jax._src import util
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lax import fft as lax_fft
from jax._src.lax import lax
from jax._src.lax import linalg as lax_linalg
import jax._src.random
from jax.api_util import flatten_fun
from jax.interpreters import ad
from jax.interpreters import pxla
from jax.interpreters import sharded_jit
from jax.interpreters import xla
from jax.lib import xla_client

from . import shape_poly

import numpy as np
import tensorflow as tf  # type: ignore[import]

# These don't have public equivalents.
# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla  # type: ignore[import]
from tensorflow.compiler.xla import xla_data_pb2  # type: ignore[import]
from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding  # type: ignore[import]
# pylint: enable=g-direct-tensorflow-import


PolyShape = shape_poly.PolyShape

# The scope name need to be a valid TensorFlow name. See
# https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/core/framework/node_def_util.cc#L731
_VALID_SCOPE_REGEX = re.compile("^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$")
_INVALID_SCOPE_CHAR = re.compile("[^A-Za-z0-9_.\\/>-]")


def _sanitize_scope_name(name):
  scope_name = _INVALID_SCOPE_CHAR.sub("_", name)
  if not _VALID_SCOPE_REGEX.match(scope_name):
    scope_name = ".{}".format(scope_name)
  return scope_name

# A value suitable in a TF tracing context: tf.Tensor, tf.Variable,
# or Python scalar or numpy.ndarray. (A tf.EagerTensor is a tf.Tensor.)
TfVal = Any
DType = Any
def _is_tfval(v: TfVal) -> bool:
  if isinstance(v, (tf.Tensor, tf.Variable)):
    return True
  try:
    # Note: this conversion is overkill and just intended as a type check; this
    # code is in principle only run if config.jax_enable_checks is True.
    # TODO: it is not true that this code is run only with jax_enable_checks.
    _safe_convert_to_tensor(v)
    return True
  except ValueError:
    return False

def _safe_convert_to_tensor(val, dtype=None) -> TfVal:
  dtype = dtype if dtype else (val.dtype if hasattr(val, "dtype") else None)
  conversion_type = to_tf_dtype(dtype) if dtype else None
  # The float0 type is not known to TF.
  if dtype and dtype == dtypes.float0:
    val = np.zeros(np.shape(val), conversion_type.as_numpy_dtype)
  return tf.convert_to_tensor(val, dtype=conversion_type)


# The implementation rules for primitives. The rule will be called with the
# arguments (TfVal) and must return TfVal (or a sequence thereof,
# if primitive.multiple_results). The vast majority of primitives do not need
# to worry about core.unit inputs or results. The exception are primarily the
# control-flow primitives.
tf_impl: Dict[core.Primitive,
              Callable[..., Any]] = {}

# Some primitive implementation rules need the abstract values of arguments
# and the results. This is the case for the primitives implemented using
# _convert_jax_impl and those that need to adjust the shape of the outputs
# due to missing TF shape inference rules for TFXLA ops. The rules for these
# primitives should be added to `tf_impl_with_avals`.
# The abstract value are passed to the implementation as two special kwargs
# `_in_avals` (a tuple of core.AbstractValue) and `_out_aval` (a
# core.AbstractValue, or a tuple thereof when primitive.multiple_results).
tf_impl_with_avals: Dict[core.Primitive,
                         Callable[..., Any]] = {}

# XLA is not linked in all environments; when converting a primitive, if this
# variable is disabled, we try harder to use only standard TF ops if they are
# applicable to the concrete use case; if the resulting conversion path ends up
# requiring a TFXLA operation, an exception is thrown instead.
_enable_xla = True

def _xla_path_disabled_error(primitive_name: str) -> Exception:
  assert not _enable_xla
  return NotImplementedError(
    f"Call to {primitive_name} can only be converted through TFXLA, but "
     "XLA is disabled")

@functools.partial(api_util.api_hook, tag="jax2tf_convert")
def convert(fun: Callable, *,
            polymorphic_shapes: Optional[Sequence[Any]]=None,
            in_shapes=None,  # DEPRECATED
            with_gradient=True, enable_xla=True) -> Callable:
  """Transforms `fun` to be executed by TensorFlow.

  See [README](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md)
  for more details about usage and common problems.

  Args:
    fun: Function to be transformed. Its arguments and return value should be
      JAX arrays, or nested standard Python containers (tuple/list/dict)
      thereof (pytrees).

    polymorphic_shapes: Specifies input shapes to be treated polymorphically
      during conversion.

      .. warning::
      The shape-polymorphic conversion is an experimental feature. It is meant
      to be sound, but it is known to reject some JAX programs that are
      shape polymorphic. The details of this feature can change.

      It should be a Python object with the same pytree structure as,
      or a prefix of, the tuple of arguments to the function,
      but with a shape specification corresponding to each argument.
      The default value is `None`, which is a shortcut for a tuple of `None`
      one for each argument, denoting that all shapes are monomorphic.
      See [how optional parameters are matched to arguments](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).

      A shape specification for an array argument
      should be an object `PolyShape(dim0, dim1, ..., dimn)`
      where each `dim` is a dimension specification: a positive integer denoting
      a monomorphic dimension of the given size,
      or a string denoting a dimension variable assumed to range over non-zero
      dimension sizes,
      or the special placeholder string "_" denoting a monomorphic dimension
      whose size is given by the actual argument.
      As a shortcut, an Ellipsis suffix in the
      list of dimension specifications stands for a list of "_" placeholders.
      For convenience, a shape specification can also be given as a string
      representation, e.g.: "batch, ...", "batch, height, width, _", possibly
      with surrounding parentheses: "(batch, ...)".

      The conversion fails if it cannot ensure that the it would produce the same
      sequence of TF ops for any non-zero values of the dimension variables.

      See [the README](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md#shape-polymorphic-conversion)
      for more details.

    in_shapes: DEPRECATED in favor of `polymorphic_shapes`.

    with_gradient: if set, will add a tf.custom_gradient to the converted
      function, by converting the ``jax.vjp(fun)``. Only first-order
      differentiation is supported for now. If the converted function is
      saved in a SavedModel, the custom gradients are currently lost and
      an error will be raised if a gradient computation is attempted.
      This is due to a current bug in TensorFlow.

    enable_xla: if unset, the converter will try harder to use pure TF ops to
      convert the function, and raise an error if it can not be converted
      without resorting to XLA ops (default: True).

  Returns:
    A version of `fun` that expects TfVals as arguments (or
    tuple/lists/dicts) thereof, and returns TfVals as outputs.
  """
  global _enable_xla
  _enable_xla = enable_xla
  api._check_callable(fun)

  def converted_fun(*args: TfVal) -> TfVal:
    # TODO: is there a better way to check if we are inside a transformation?
    if not core.trace_state_clean():
      raise ValueError("convert must be used outside all JAX transformations."
                       + f"Trace state: {core.thread_local_state.trace_state}")

    def check_arg(a):
      if not _is_tfval(a):
        msg = (f"Argument {a} of type {type(a)} of jax2tf.convert(f) should "
               "be NumPy array, scalar, tf.Variable, or tf.Tensor")
        raise TypeError(msg)
    tree_util.tree_map(check_arg, args)

    # Name input tensors
    args = tuple(
        tree_util.tree_map(lambda x, i=i: tf.identity(x, f"jax2tf_arg_{i}"), a)  # type: ignore
        for i, a in enumerate(args))

    # This function may take pytrees of TfVals. We can only set
    # tf.custom_gradient on functions that take a flat argument list.
    args_flat, in_tree = tree_util.tree_flatten((args, {}))

    if polymorphic_shapes is None:
      polymorphic_shapes_ = (None,) * len(args)
    else:
      if not isinstance(polymorphic_shapes, Sequence) or len(args) != len(polymorphic_shapes):
        msg = ("polymorphic_shapes must be a sequence with the same length as the argument list "
               f"({len(args)}). Got polymorphic_shapes={polymorphic_shapes}.")
        raise TypeError(msg)
      polymorphic_shapes_ = tuple(polymorphic_shapes)

    # Expand the in_shapes to match the argument pytree
    polymorphic_shapes_flat = tuple(api_util.flatten_axes("jax2tf.convert polymorphic_shapes",
                                                          in_tree.children()[0],
                                                          polymorphic_shapes_))

    # Construct the abstract values for the flat arguments, possibly based on
    # the input shapes and the in_shapes if given. May create new shape
    # variables.
    args_avals_flat, shapeenv = _args_to_avals_and_env(args_flat,
                                                       polymorphic_shapes_flat)

    f = lu.wrap_init(fun)
    # out_tree_thunk() will be the output tree, after running _interpret_fun.
    flat_fun, out_tree_thunk = flatten_fun(f, in_tree)

    # Prepare the grad_fn for tf.custom_gradient.
    def converted_grad_fn(*out_cts_flat: TfVal,
                          _out_cts_avals: Sequence[core.AbstractValue],
                          variables=None):
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

      if polymorphic_shapes is None:
        vjp_polymorphic_shapes = None
      else:
        args_polymorphic_shapes = tree_util.tree_unflatten(in_tree.children()[0], polymorphic_shapes_flat)
        out_cts_polymorphic_shapes = tree_util.tree_unflatten(
          out_tree_thunk(),
          tuple(str(out_aval.shape) for out_aval in _out_cts_avals))  # type: ignore
        vjp_polymorphic_shapes = [args_polymorphic_shapes, out_cts_polymorphic_shapes]
      out_cts = tree_util.tree_unflatten(out_tree_thunk(), out_cts_flat)
      # TODO: enable higher-order gradients
      with tf.name_scope("jax2tf_vjp"):
        in_cts = convert(fun_vjp_jax, with_gradient=False,
                         polymorphic_shapes=vjp_polymorphic_shapes)(args, out_cts)
      return in_cts

    try:
      global _shape_env
      assert not _shape_env, f"Unexpected shape environment {_shape_env}"
      _shape_env = shapeenv

      if with_gradient:
        @tf.custom_gradient
        def converted_fun_flat_with_custom_gradient(*args_flat: TfVal) -> TfVal:
          out_with_avals = _interpret_fun(flat_fun, args_flat, args_avals_flat)
          outs, out_avals = util.unzip2(out_with_avals)
          return (tuple(outs),
                  functools.partial(converted_grad_fn, _out_cts_avals=tuple(out_avals)))

        out_flat = converted_fun_flat_with_custom_gradient(*args_flat)
      else:
        out_flat_raw = _interpret_fun(flat_fun, args_flat, args_avals_flat)
        message = ("The jax2tf-converted function does not support gradients. "
                   "Use `with_gradient` parameter to enable gradients")
        # We use PreventGradient, which is propagated through a SavedModel.
        out_flat = [tf.raw_ops.PreventGradient(input=o, message=message)
                    for o, _ in out_flat_raw]
    finally:
      _shape_env = {}

    out_flat = [tf.identity(x, "jax2tf_out") for x in out_flat]
    out = tree_util.tree_unflatten(out_tree_thunk(), out_flat)
    return out

  return converted_fun


# Internals


def _interpret_fun(fun: lu.WrappedFun,
                   in_vals: Sequence[TfVal],
                   in_avals: Sequence[core.AbstractValue]
                   ) -> Sequence[Tuple[TfVal, core.AbstractValue]]:
  with core.new_base_main(TensorFlowTrace) as main:  # type: ignore
    fun = _interpret_subtrace(fun, main, in_avals)
    out_vals: Sequence[Tuple[TfVal, core.AbstractValue]] = fun.call_wrapped(*in_vals)
    del main
  return tuple(out_vals)

def _convert_jax_impl(jax_impl: Callable, *, multiple_results=True) -> Callable:
  """Convert the JAX implementation of a primitive.

  Args:
    jax_impl: typically the impl-rule for a primitive, with signature
      `(*args: JaxVal, **kwargs) -> Sequence[JaxVal]`. This function implements
      a primitive in terms of other primitives.
    multiple_results: whether `jax_impl` returns a sequence of results.

  Returns:
     a function with signature `(*args: TfVal, _in_avals, _out_aval, **kwargs) -> Sequence[TfVal]`.
  """
  def wrapped(*tf_args: TfVal,
              _in_avals: Sequence[core.AbstractValue],
              _out_aval: core.AbstractValue, **kwargs) -> Sequence[TfVal]:

    # We wrap the jax_impl under _interpret_fun to abstract the TF values
    # from jax_impl and turn them into JAX abstract values.
    def jax_impl_jax_args(*jax_args):
      jax_results = jax_impl(*jax_args, **kwargs)
      return jax_results if multiple_results else [jax_results]

    tf_results_with_avals = _interpret_fun(lu.wrap_init(jax_impl_jax_args), tf_args, _in_avals)
    tf_results, _ = util.unzip2(tf_results_with_avals)
    return tf_results if multiple_results else tf_results[0]
  return wrapped


@lu.transformation
def _interpret_subtrace(main: core.MainTrace,
                        in_avals: Sequence[core.AbstractValue],
                        *in_vals: TfVal):
  trace = TensorFlowTrace(main, core.cur_sublevel())
  in_tracers = tuple(TensorFlowTracer(trace, val, aval)
                     for val, aval in util.safe_zip(in_vals, in_avals))
  # The outs may be core.unit, see comment in TensorFlowTrace.pure.
  outs = yield in_tracers, {}  # type: Sequence[Union[TfVal, core.Unit]]
  out_tracers: Iterable[TensorFlowTracer] = map(trace.full_raise, outs)  # type: ignore
  out_vals_with_avals: Sequence[Tuple[TfVal, core.AbstractValue]] = (
    tuple((t.val, t.aval) for t in out_tracers))
  yield out_vals_with_avals


def _interpret_jaxpr(jaxpr: core.ClosedJaxpr, *args: TfVal) -> Sequence[TfVal]:
  """Evaluates a Jaxpr with tf.Tensor arguments.

  The output is a sequence of TfVal (no `core.unit`), suitable for use with TF.
  """
  fun: lu.WrappedFun = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
  out_with_avals = _interpret_fun(fun, args, jaxpr.in_avals)
  return tuple(v for v, _ in out_with_avals)

### tracer

def _aval_to_tf_shape(aval: core.AbstractValue) -> Tuple[Optional[int], ...]:
  """Generate a TF shape, possibly containing None for polymorphic dimensions."""
  return tuple(map(lambda d: None if isinstance(d, shape_poly.DimVar) else d,
                   aval.shape))  # type: ignore[attr-defined]


def _tfval_shape_dtype(val: TfVal) -> Tuple[Sequence[Optional[int]], DType]:
  """
  Called for constants that occur in the program, or for input values to the
  converted function. The returned shape may have unknown components, but
  only when called for inputs.
  """
  if isinstance(val, (tf.Tensor, tf.Variable)):
    # May be partially known
    return tuple(val.shape), to_jax_dtype(val.dtype)
  else:  # Must be a numeric value
    assert not config.jax_enable_checks or _is_tfval(val), f"Non TfVal: {val}"
    raw_aval = xla.abstractify(val)
    return raw_aval.shape, raw_aval.dtype  # type: ignore[attr-defined]


# A dimension environment maps dimension variables to TF expressions that
# compute the value of the dimension. These expressions refer to the TF
# function arguments.
_ShapeEnv = Dict[shape_poly.DimVar, TfVal]
def _args_to_avals_and_env(args: Sequence[TfVal],
                           polymorphic_shapes: Sequence[Optional[Union[str, PolyShape]]]) -> \
  Tuple[Sequence[core.AbstractValue], _ShapeEnv]:
  """Computes abstract values and a dimension environment for arguments.

  Args:
    args: the arguments, TF inputs.
    polymorphic_shapes: the polymorphic specifications for the arguments.

  Returns: a tuple of a sequence of abtract values corresponding to the arguments
    and a dimension environment.
  """
  shapeenv: _ShapeEnv = {}
  def input_aval(arg: TfVal, polymorphic_shape: Optional[str]) -> core.AbstractValue:
    """The abstract value for an input."""
    raw_shape, dtype = _tfval_shape_dtype(arg)

    aval_shape = shape_poly.parse_spec(polymorphic_shape, raw_shape)

    for i, d in enumerate(aval_shape):
      if type(d) is int:
        assert d == np.shape(arg)[i]
      elif type(d) is shape_poly.DimVar and d not in shapeenv:
        # Even if the shape of `arg` is known, we still use `tf.shape` for
        # safety, because the promise is that we will convert the function
        # to work for any value of the dimension.
        shapeenv[d] = tf.shape(arg)[i]  # type: ignore[index]
      else:
        # TODO: add an assertion tf.shape(arg)[i] == env[d]
        pass

    return core.ShapedArray(aval_shape, dtype)

  avals = tuple(map(input_aval, args, polymorphic_shapes))  # type: ignore
  return avals, shapeenv

# A shape environment maps shape variables to TfVal.
_shape_env = {}  # type: _ShapeEnv

def _eval_shape(shape: Sequence[shape_poly.DimSize]) -> Sequence[TfVal]:
  assert all(map(lambda x: x is not None, shape)), (
      f"Argument shape should be a valid JAX shape but got {shape}")
  return tuple(_shape_env[d] if type(d) is shape_poly.DimVar else d  # type: ignore[index]
               for d in shape)

def shape_as_value(x):
  """Injects the shape of `x` as an array value.

  **Experimental: please give feedback, and expect changes!**

  This allows the use of a shape expression as array argument to JAX functions.
  A typical example is for implementing a mean operation:

     jnp.sum(x) / np.prod(jax2tf.shape_as_value(x))
  """
  # return shape_as_value_p.bind(x)
  return NotImplementedError("shape_as_value is deprecated")


# # TODO: move this to masking or to some common library, if approved
# shape_as_value_p = core.Primitive("shape_as_value")
# shape_as_value_p.multiple_results = True
# def _shape_as_value_impl(x):
#   x_shape = np.shape(x)
#   def dim_to_int(dim: shape_poly.DimSize) -> int:
#     dim_int = _poly_dim_to_tf_dim(dim)
#     if dim_int is None:
#       msg = ("shape_as_value is not implemented for non-constant shapes "
#              "except for masking and jax2tf. "
#              f"Has shape: {x_shape}")
#       raise TypeError(msg)
#     else:
#       return dim_int
#   return tuple(map(dim_to_int, x_shape))
#
# shape_as_value_p.def_impl(_shape_as_value_impl)
#
# def _shape_as_value_abstract(x_aval: core.AbstractValue) -> Sequence[core.AbstractValue]:
#   rank = len(x_aval.shape)  # type: ignore[attr-defined]
#   return (core.ShapedArray((), dtypes.canonicalize_dtype(np.int_), weak_type=True),) * rank
#
# shape_as_value_p.def_abstract_eval(_shape_as_value_abstract)
#
# def _shape_as_value_translation(comp, x):
#   return xla_client._xla.ops.Tuple(comp,
#                                    tuple(xb.constant(comp, d)
#                                          for d in comp.GetShape(x).dimensions()))
#
# xla.translations[shape_as_value_p] = _shape_as_value_translation
#
# def _shape_as_value_jvp_rule(primals, tangents):
#   # The shape does not depend on the contents of the input
#   x, = primals
#   zero = ad.Zero.from_value(0.)
#   return shape_as_value(x), (zero,) * len(x.shape)
#
# ad.primitive_jvps[shape_as_value_p] = _shape_as_value_jvp_rule
#
# def _shape_as_value__batching_rule(batched_args, batch_dims):
#   xv, = batched_args
#   batch_dim, = batch_dims
#   batch_size = xv.shape[batch_dim]
#   batched_shape = shape_as_value(xv)
#   one_shape = batched_shape[0:batch_dim] + batched_shape[batch_dim+1:]
#   res = tuple(jnp.broadcast_to(d, (batch_size, 1)) for d in one_shape)
#   return res, (0,) * len(one_shape)
#
# batching.primitive_batchers[shape_as_value_p] = _shape_as_value__batching_rule
#
# def _shape_as_value_masking_rule(operands, operands_logical_shapes):
#   x_logical_shape, = operands_logical_shapes
#   return tuple(x_logical_shape)
#
# masking.masking_rules[shape_as_value_p] = _shape_as_value_masking_rule
#
# def _shape_as_value_tf(x: TfVal,
#                        _in_avals: Sequence[core.AbstractValue],
#                        _out_aval: core.AbstractValue) -> TfVal:
#   x_aval = _in_avals[0]
#   def dim_to_tfval(dim: shape_poly.DimSize, dim_idx: int) -> TfVal:
#     dim_int = _poly_dim_to_tf_dim(dim)
#     if dim_int is not None:
#       return tf.convert_to_tensor(dim_int)
#     else:
#       return tf.shape(x)[dim_idx]
#   return tuple(dim_to_tfval(dim, dim_idx)
#                for dim_idx, dim in enumerate(x_aval.shape))  # type: ignore[attr-defined]
#
# tf_impl_with_avals[shape_as_value_p] = _shape_as_value_tf

# TODO(b/26854495): pylint doesn't understand slots and inheritance.
# pylint: disable=assigning-non-slot


class TensorFlowTracer(core.Tracer):
  """Tracer class that boxes a TF value and a JAX abstract value.

  In addition to the TF value we carry the JAX abstract value because there are
  two cases when it cannot be recovered from the value: (a) when the abstract
  value is core.abstract_unit, in which case the value is tf.nan; (b) when we
  are converting with polymorphic shapes, in which case the shape of the value
  may have dimensions set to `None`, which the JAX abstract value may contain
  more precise information.

  When the value has a partially-known shape, the dimensions marked as `None`
  must correspond to non-constant dimensions in the abstract value.

  See README.md for details.
  """
  # val: TfVal
  # _aval: core.AbstractValue
  __slots__ = ["val", "_aval"]

  def __init__(self, trace: 'TensorFlowTrace', val: TfVal,
               aval: core.AbstractValue):
    self._trace = trace
    self._aval = aval
    if aval is core.abstract_unit:
      self.val = val
    elif isinstance(val, (tf.Tensor, tf.Variable)):
      val_shape, val_dtype = _tfval_shape_dtype(val)
      aval_dtype = np.dtype(self._aval.dtype)  # type: ignore[attr-defined]
      if val_dtype != aval_dtype and (val_dtype == tf.int32 and aval_dtype == jnp.int64 or
                                      val_dtype == tf.int64 and aval_dtype == jnp.int32 or
                                      val_dtype == tf.float32 and aval_dtype == jnp.float64 or
                                      val_dtype == tf.float64 and aval_dtype == jnp.float32):
        # We expect that x64 values are turned into x32
        val = tf.cast(val, dtype=aval_dtype)
        val_dtype = aval_dtype

      if config.jax_enable_checks:
        assert aval_dtype == val_dtype, f"expected {aval_dtype} == {val_dtype}"
        for aval_dim, val_dim in util.safe_zip(self._aval.shape, val_shape):  # type: ignore[attr-defined]
          if val_dim is None:
            assert isinstance(aval_dim, shape_poly.DimVar), f"expected {self._aval.shape} == {val_shape}"  # type: ignore[attr-defined]
          elif not isinstance(aval_dim, shape_poly.DimVar):
            assert aval_dim == val_dim, f"expected {self._aval.shape} == {val_shape}"  # type: ignore[attr-defined]
          else:
            # We have a TF value with known shape, and the abstract shape is a shape variable.
            try:
             aval_int = int(_eval_shape([aval_dim]))  # type: ignore
            except TypeError:
             continue
            assert aval_int == val_dim, f"expected {self._aval.shape} == {val_shape}. Found {aval_int} != {val_dim}."  # type: ignore

      self.val = val
    else:  # Must be a numeric value
      self.val = _safe_convert_to_tensor(val, dtype=self._aval.dtype)  # type: ignore[attr-defined]

  @property
  def aval(self):
    return self._aval

  def full_lower(self):
    return self


class TensorFlowTrace(core.Trace):
  """Trace class that underlies the jax2tf transformation.

  We are going to ensure that jax2tf.convert is never nested inside other
  transformations. This is sufficient for intended use cases (converting
  fully-transformed JAX code). It also simplifies our job because we do not have
  to handle situations where we apply primitives on a mix of TF values and
  JAX tracers from an outer transformation. E.g., for addition both the TF values
  and the JAX tracers have an override and they get confused if they see values
  from the other world.

  Hence a TFT trace does not interact with non-TFT traces at lower-level. For
  higher-order control-flow primitives we invoke recursively
  _interpret_fun on the body of the conditional, which will create a nested TFT.

  We do want to allow transformations nested inside a TensorFlowTrace (TFT), but
  those will introduce their own MainTrace, and any operations involving those
  will be done on those traces, i.e., not a concern for TFT.
  """
  def pure(self, val: Union[TfVal, core.Unit]) -> TensorFlowTracer:
    """Lifts a non-Tracer into the TensorFlowTracer.

    This function may be called by way of trace.full_raise.

    The value may be a core.unit. During JAX transformations we sometimes
    produce a Jaxpr that has arguments of abstract value core.abstract_unit
    and results equal to core.unit. These are arguments and results that are
    not used in the computation.

    In TF world, we represent core.unit as NaN. This is safe, as these values
    should never be used.
    """
    if val is core.unit:
      return TensorFlowTracer(self, tf.constant(np.nan, tf.float32), core.abstract_unit)
    else:
      shape, dtype = _tfval_shape_dtype(val)
      return TensorFlowTracer(self, val, core.ShapedArray(shape, dtype))

  def lift(self, val: core.Tracer) -> TensorFlowTracer:
    # This would be called when we need to raise a tracer from a lower-level
    # main into the TensorFlowTrace. Since the TensorFlowTrace is never nested
    # inside another transform, there are no lower-level main traces.
    assert False

  def sublift(self, val: TensorFlowTracer) -> TensorFlowTracer:
    # This is called when we need to raise a tracer from the same master,
    # but a lower sublevel. This could come from a nested jit.
    return TensorFlowTracer(self, val.val, val._aval)

  def process_primitive(self, primitive: core.Primitive,
                        tracers: Sequence[TensorFlowTracer],
                        params) -> TensorFlowTracer:
    impl, impl_needs_avals = self.get_primitive_impl(primitive)
    args_avals: Sequence[core.AbstractValue] = tuple(t.aval for t in tracers)
    out_aval = primitive.abstract_eval(*args_avals, **params)
    args_tf: Sequence[TfVal] = [t.val for t in tracers]
    if impl_needs_avals:
      val_out: TfVal = impl(*args_tf, _in_avals=args_avals,  # type: ignore
                            _out_aval=out_aval, **params)
    else:
      val_out = impl(*args_tf, **params)

    if primitive.multiple_results:
      out = [TensorFlowTracer(self, v, a)
             for v, a in util.safe_zip(val_out, out_aval)]  # type: ignore
    else:
      out = TensorFlowTracer(self, val_out, out_aval)  # type: ignore

    # Check that the impl rule returned a value of expected shape and dtype
    # TODO: adapt this to match polymorphic shapes
    if config.jax_enable_checks:
      if primitive.multiple_results:
        for o, expected_aval in zip(out, out_aval):  # type: ignore
          assert o.aval.strip_weak_type() == expected_aval.strip_weak_type(), (
            f"{primitive}: out.aval = {o.aval}; expected {expected_aval}")
      else:
        assert out.aval == out_aval, (  # type: ignore
          f"{primitive}: out.aval = {out.aval}; expected {out_aval}")  # type: ignore
    return out  # type: ignore

  def process_call(self, call_primitive: core.Primitive, f: lu.WrappedFun,
                   tracers: Sequence[TensorFlowTracer], params):
    assert call_primitive.multiple_results
    vals: Sequence[TfVal] = [t.val for t in tracers]
    f = _interpret_subtrace(f, self.main, tuple(t.aval for t in tracers))
    if call_primitive == core.named_call_p:
      with tf.name_scope(_sanitize_scope_name(params["name"])):
        vals_out: Sequence[Tuple[TfVal,
                                 core.AbstractValue]] = f.call_wrapped(*vals)
    elif call_primitive == sharded_jit.sharded_call_p:
      vals_out = _sharded_call(f, vals, **params)
    else:
      vals_out = f.call_wrapped(*vals)
    return [TensorFlowTracer(self, v, a) for v, a in vals_out]

  def post_process_call(self, call_primitive: core.Primitive,
                        out_tracers: Sequence[TensorFlowTracer], params):
    # We encountered a call primitive, e.g., remat_call_p, whose result
    # (out_tracers) include TensorFlowTracer that were not passed through
    # its arguments (captured from the environment).
    vals = tuple(t.val for t in out_tracers)
    main = self.main
    def todo(vals: Sequence[TfVal]):
      trace = TensorFlowTrace(main, core.cur_sublevel())
      return [TensorFlowTracer(trace, v, out_tracer.aval)
              for v, out_tracer in util.safe_zip(vals, out_tracers)]
    return vals, todo

  def process_map(self, map_primitive, f, tracers, params):
    raise NotImplementedError("process_map")

  def post_process_map(self, map_primitive, out_tracers, params):
    raise NotImplementedError("post_process_map")

  def process_custom_jvp_call(self, prim, fun, jvp, tracers):
    # Drop the custom differentiation rule and act like a call primitive. This
    # behavior is desirable because jax2tf stages code out of the JAX system, so
    # there are no more JAX differentiation transformations to be applied.
    del jvp  # Unused.
    return self.process_call(core.call_p, fun, tracers, {})

  def post_process_custom_jvp_call(self, out_tracers, params):
    assert False  # unreachable assuming jax2tf runs with clean trace state

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees):
    # Drop the custom differentiation rule and act like a call primitive. This
    # behavior is desirable because jax2tf stages code out of the JAX system, so
    # there are no more JAX differentiation transformations to be applied.
    del fwd, bwd, out_trees  # Unused.
    return self.process_call(core.call_p, fun, tracers, {})

  def post_process_custom_vjp_call(self, out_tracers, params):
    assert False  # unreachable assuming jax2tf runs with clean trace state

  def get_primitive_impl(self, p: core.Primitive) -> Tuple[Callable, bool]:
    # Returns the primitive implementation and whether the implementation
    # takes abstract values (see definition of tf_impl_with_avals)
    try:
      return tf_impl[p], False
    except KeyError:
      try:
        return tf_impl_with_avals[p], True
      except KeyError as err:
        msg = "TensorFlow interpretation rule for '{}' not implemented"
        raise NotImplementedError(msg.format(p)) from err

def to_tf_dtype(jax_dtype):
  if jax_dtype == dtypes.float0:
    jax_dtype = dtypes.bfloat16
  return tf.dtypes.as_dtype(jax_dtype)

def to_jax_dtype(tf_dtype):
  return tf_dtype.as_numpy_dtype

def _unexpected_primitive(p: core.Primitive, *args, **kwargs):
  assert False, f"Encountered unexpected primitive {p}"


for unexpected in xla.call_translations: # Call primitives are inlined
  tf_impl[unexpected] = functools.partial(_unexpected_primitive, unexpected)

# Primitives that are not yet implemented must be explicitly declared here.
tf_not_yet_impl = [
  "reduce", "rng_uniform", "clz",

  "igamma_grad_a",
  "random_gamma_grad",
  "reduce_precision",

  # Not high priority?
  "after_all", "all_to_all", "create_token",
  "infeed", "outfeed", "pmax_p",
  "pmin", "ppermute", "psum", "pmax", "pgather",
  "axis_index", "pdot", "all_gather",
  "lu_pivots_to_permutation",
  "rng_bit_generator",

  "xla_pmap",
  "call_tf",
]

tf_impl[ad_util.stop_gradient_p] = tf.stop_gradient
tf_impl[ad_util.zeros_like_p] = tf.zeros_like

def _add(x: TfVal, y: TfVal) -> TfVal:
  return tf.raw_ops.AddV2(x=x, y=y)

tf_impl[ad_util.add_jaxvals_p] = _add
tf_impl[xla.device_put_p] = lambda x, device=None: x

tf_impl[lax.neg_p] = tf.math.negative
tf_impl[lax.sign_p] = tf.math.sign
tf_impl[lax.floor_p] = tf.math.floor
tf_impl[lax.ceil_p] = tf.math.ceil

def _round(operand, *, rounding_method):
  if rounding_method is lax.RoundingMethod.AWAY_FROM_ZERO:
    sign = tf.math.sign(operand)
    operand *= sign
    floor = tf.math.floor(operand)
    operand -= floor
    cond = tf.math.equal(operand, tf.constant(np.array(0.5), operand.dtype))
    return sign * (tf.where(cond, tf.constant(np.array(1), operand.dtype),
                            tf.math.round(operand)) + floor)
  else:
    return tf.math.round(operand)

tf_impl[lax.round_p] = _round
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
tf_impl[lax.tan_p] = tf.math.tan
tf_impl[lax.tanh_p] = tf.math.tanh
tf_impl[lax.sin_p] = tf.math.sin
tf_impl[lax.sinh_p] = tf.math.sinh
tf_impl[lax.cos_p] = tf.math.cos
tf_impl[lax.cosh_p] = tf.math.cosh
tf_impl[lax.acos_p] = tf.math.acos
tf_impl[lax.asin_p] = tf.math.asin
tf_impl[lax.atan_p] = tf.math.atan
tf_impl[lax.atan2_p] = tf.math.atan2
tf_impl[lax.acosh_p] = tf.math.acosh
tf_impl[lax.atanh_p] = tf.math.atanh
tf_impl[lax.asinh_p] = tf.math.asinh

tf_impl[lax.sqrt_p] = tf.math.sqrt
tf_impl[lax.rsqrt_p] = tf.math.rsqrt

tf_impl[lax.lgamma_p] = tf.math.lgamma
tf_impl[lax.digamma_p] = tf.math.digamma
tf_impl[lax.igamma_p] = tf.math.igamma
tf_impl[lax.igammac_p] = tf.math.igammac
tf_impl[lax.regularized_incomplete_beta_p] = tf.math.betainc
tf_impl[lax.erf_p] = tf.math.erf
tf_impl[lax.erfc_p] = tf.math.erfc
tf_impl[lax.erf_inv_p] = tf.math.erfinv
tf_impl[lax.bessel_i0e_p] = tf.math.bessel_i0e
tf_impl[lax.bessel_i1e_p] = tf.math.bessel_i1e

tf_impl[lax.complex_p] = tf.complex

def _conj(x, **kwargs):
  # The only dtypes that are allowed are: float32, float64, complex64, and
  # complex128.
  if x.dtype == tf.float32:
    return tf.cast(x, tf.complex64)
  elif x.dtype == tf.float64:
    return tf.cast(x, tf.complex128)
  else:
    return tf.math.conj(x)

tf_impl[lax.conj_p] = _conj
tf_impl[lax.real_p] = tf.math.real
tf_impl[lax.imag_p] = tf.math.imag

tf_impl[lax.add_p] = _add
tf_impl[lax.sub_p] = tf.math.subtract
tf_impl[lax.mul_p] = tf.math.multiply


def _iota(*, dtype, shape, dimension):
  dtype = to_tf_dtype(dtype)
  # Some dtypes are unsupported, like uint32, so we just fall back to int32.
  # TODO(mattjj, necula): improve tf.range dtype handling
  shape_tf = _eval_shape(shape)
  vec = tf.range(tf.cast(shape_tf[dimension], tf.int32), dtype=tf.int32)
  vec_shape = [-1 if i == dimension else 1 for i in range(len(shape))]
  return tf.cast(tf.broadcast_to(tf.reshape(vec, vec_shape), shape_tf), dtype)

tf_impl[lax.iota_p] = _iota


def _div(lhs, rhs):
  if lhs.dtype.is_integer:
    quotient = tf.math.floordiv(lhs, rhs)
    select = tf.math.logical_and(
        tf.not_equal(tf.math.sign(lhs), tf.math.sign(rhs)),
        tf.not_equal(tf.math.floormod(lhs, rhs), 0))
    return tf.where(select, quotient + 1, quotient)
  else:
    return tf.math.truediv(lhs, rhs)


def _rem(lhs, rhs):
  return tf.math.sign(lhs) * tf.math.floormod(tf.math.abs(lhs),
                                              tf.math.abs(rhs))

tf_impl[lax.div_p] = _div
tf_impl[lax.rem_p] = _rem

tf_impl[lax.max_p] = tf.math.maximum
tf_impl[lax.min_p] = tf.math.minimum

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
def _shift_right_arithmetic_raw(x, y):
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

def _shift_right_arithmetic(x, y):
  # TF shift is "implementation defined" if the shift amount is negative
  # or larger or equal to the size of the value. We implement the XLA
  # semantics to return the shift by the max value (x_bits - 1).
  # TODO: it is likely better to add XlaOps for shifts
  x_bits = 8 * x.dtype.size
  clamp_y = tf.where(_shift_in_bounds(x, y), y, x_bits - 1)
  return _shift_right_arithmetic_raw(x, clamp_y)

tf_impl[lax.shift_right_arithmetic_p] = _shift_right_arithmetic

def _shift_right_logical_raw(x, y):
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

def _shift_right_logical(x, y):
  # TF shift is "implementation defined" if the shift amount is negative
  # or larger or equal to the size of the value. We implement the XLA semantics
  # to return 0.
  # TODO: it is likely better to add XlaOps for shifts
  return tf.where(_shift_in_bounds(x, y),
                  _shift_right_logical_raw(x, y),
                  tf.zeros_like(x))

tf_impl[lax.shift_right_logical_p] = _shift_right_logical

def _shift_left(x, y):
  # TF shift is "implementation defined" if the shift amount is negative
  # or larger or equal to the size of the value. We implement the XLA semantics
  # to return 0.
  # TODO: it is likely better to add XlaOps for shifts
  return tf.where(_shift_in_bounds(x, y),
                  tf.bitwise.left_shift(x, y),
                  tf.zeros_like(x))

tf_impl[lax.shift_left_p] = _shift_left

def _shift_in_bounds(x: TfVal, y: TfVal) -> TfVal:
  # Return the TF expression for when y is within bounds (0 <= y < |x|)
  x_bits = 8 * x.dtype.size
  # TF does not have comparisons for uint16 and uint32 (despite what the
  # documentation says)
  y_comp = tf.cast(y, _UNSIGNED_TO_SIGNED_TABLE[y.dtype]) if y.dtype.is_unsigned else y
  y_lt_x_bits = tf.math.less(y_comp, x_bits)
  y_ge_0 = tf.math.greater_equal(y_comp, 0)
  return tf.logical_and(y_lt_x_bits, y_ge_0)

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
      args_cast = [(tf.cast(a, tf.int8) if i in argnums else a)
                   for i, a in enumerate(args)]
      if "_in_avals" in kwargs:
        def cast_aval(aval):
          return core.ShapedArray(aval.shape, np.int8)
        _in_avals_cast = [cast_aval(aval) if i in argnums else aval
                          for i, aval in enumerate(kwargs["_in_avals"])]
        _out_aval_cast = tf.nest.map_structure(cast_aval, kwargs["_out_aval"])
        kwargs = dict(kwargs, _in_avals=_in_avals_cast, _out_aval=_out_aval_cast)
      out = f(*args_cast, **kwargs)
      return tf.nest.map_structure(lambda o: tf.cast(o, tf.bool), out)
  return wrapper

tf_impl[lax.or_p] = bool_to_int8(tf.bitwise.bitwise_or, argnums=(0, 1))
tf_impl[lax.and_p] = bool_to_int8(tf.bitwise.bitwise_and, argnums=(0, 1))
tf_impl[lax.xor_p] = bool_to_int8(tf.bitwise.bitwise_xor, argnums=(0, 1))

tf_impl[lax.eq_p] = tf.math.equal
tf_impl[lax.ne_p] = tf.math.not_equal
tf_impl[lax.ge_p] = tf.math.greater_equal
tf_impl[lax.gt_p] = tf.math.greater
tf_impl[lax.le_p] = tf.math.less_equal
tf_impl[lax.lt_p] = tf.math.less

tf_impl[lax_linalg.cholesky_p] = tf.linalg.cholesky

def _convert_element_type(operand, *, new_dtype, weak_type=False):
  old_dtype = operand.dtype.as_numpy_dtype
  if (dtypes.issubdtype(old_dtype, np.complexfloating) and
      not dtypes.issubdtype(new_dtype, np.complexfloating)):
    operand = tf.math.real(operand)
  if (dtypes.issubdtype(old_dtype, np.floating) and
      not (dtypes.issubdtype(new_dtype, np.floating) or
           dtypes.issubdtype(new_dtype, np.complexfloating) or
           new_dtype == np.bool_)):
    sign = tf.math.sign(operand)
    operand = sign * tf.math.floor(sign * operand)
  return tf.dtypes.cast(operand, to_tf_dtype(new_dtype))
tf_impl[lax.convert_element_type_p] = _convert_element_type


def _bitcast_convert_type(operand, new_dtype):
  return tf.bitcast(operand, to_tf_dtype(new_dtype))
tf_impl[lax.bitcast_convert_type_p] = _bitcast_convert_type


def _clamp(minval, operand, maxval, *, _in_avals, _out_aval):
  # The below permits mirroring the behavior of JAX when maxval < minval
  op_shape_tf_val = _eval_shape(_in_avals[1].shape)
  maxval = tf.broadcast_to(maxval, op_shape_tf_val)
  minval = tf.math.minimum(tf.broadcast_to(minval, op_shape_tf_val), maxval)
  return tf.clip_by_value(operand, minval, maxval)
tf_impl_with_avals[lax.clamp_p] = _clamp


def _concatenate(*operands, dimension):
  return tf.concat(operands, axis=dimension)
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


def _conv_general_precision_config_proto(precision):
  """Convert an integer to an XLA.PrecisionConfig."""
  if precision is None:
    return None

  proto = xla_data_pb2.PrecisionConfig()
  proto.operand_precision.append(int(precision))
  return proto

# _try_tf_conv returns a Tensor when it succeeds, or a string describing why
# it did not succeed otherwise.
def _try_tf_conv(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
                 dimension_numbers, feature_group_count, batch_group_count,
                 out_shape) -> Union[str, TfVal]:
  # TODO(bchetioui): this function is not exhaustive wrt which convolution cases
  # can be translated into TF primitives. Further investigation is needed to
  # fully flesh it out.
  if not lhs.dtype in [tf.float16, tf.float32, tf.float64]:
    return f"tf.nn.convolution is not supported for dtype {lhs.dtype}"
  if feature_group_count != 1:
    return "tf.nn.convolution does not support grouped convolutions"
  # TODO(bchetioui): is there something to do with batch_group_count?
  if batch_group_count != 1:
    return "Unimplemented support for batch_group_count != 1"
  nb_spatial_dimensions = len(lhs.shape) - 2
  # TF can only deal with 1D, 2D and 3D convolution
  if nb_spatial_dimensions < 1 or nb_spatial_dimensions > 3:
    return ("TensorFlow can only handle convolutions with 1, 2, or 3 "
            "spatial dimensions")
  # TODO(bchetioui): handle different stride cases
  if list(window_strides) != [1] * nb_spatial_dimensions:
    return ("Unimplemented support for window_strides != "
            f"{tuple([1] * nb_spatial_dimensions)}")

  success = lambda res: (res, None)
  failure = lambda msg: (None, msg)

  def convert_padding():
    # TODO(bchetioui): in this instance, we can not use padtype_to_pads as
    # string padding is not implemented for transposed convolution.
    if list(lhs_dilation) != [1] * nb_spatial_dimensions:
      return failure("Padding conversion is not supported for transposed "
                     "convolution.")
    lhs_perm, rhs_perm, _ = dimension_numbers
    effective_rhs_shape = [(k-1) * r + 1 for k, r in
                           zip(np.take(rhs.shape, rhs_perm)[2:], rhs_dilation)]
    lhs_shape = np.take(lhs.shape, lhs_perm)[2:]
    # TF only allows 'VALID' and 'SAME' padding
    for pad_str in ['VALID', 'SAME']:
      gen_padding = lax.padtype_to_pads(
          lhs_shape, effective_rhs_shape, window_strides, pad_str)
      if list(gen_padding) == list(padding):
        return success(pad_str)
    return failure("Input padding not supported in TensorFlow.")

  def convert_dim_nums():
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    # TF only allows filters with shape:
    # spatial_filter_shape + [in_channels, out_channels]. In JAX however,
    # rhs_spec is represented as a tuple containing the following:
    # [out_channels, in_channels] + spatial_filter_shape.
    supported_rhs_shape = ([nb_spatial_dimensions + 1, nb_spatial_dimensions] +
                           list(range(nb_spatial_dimensions)))
    if list(rhs_spec) != supported_rhs_shape:
      return failure("Input filter (RHS) shape format not supported in "
                     "TensorFlow")
    # TF only supports same LHS and output data format
    if lhs_spec != out_spec:
      return failure("TensorFlow requires the same data format for LHS and "
                     "output.")
    # Alphabet extracted from the documentation of tf.conv{1,2,3}d
    spatial_dim_alphabet = 'DHW'[-nb_spatial_dimensions:]
    # TF only supports the following data formats:
    # - [batch_size, in_channels] + input_spatial_shape

    # TODO(bchetioui): TF currently does not support the above on CPU. To avoid
    # failing on this platform, this path is commented out for now.
    #if list(lhs_spec) == list(range(len(lhs_spec))):
    #  return "NC" + spatial_dim_alphabet

    # - [batch_size] + input_spatial_shape + [in_channels]
    if list(lhs_spec) == ([0, len(lhs_spec) - 1] +
                          list(range(1, len(lhs_spec) - 1))):
      return success("N" + spatial_dim_alphabet + "C")
    return failure("Data format is unsupported by TensorFlow")

  def convert_dilation_and_compute_result(tf_padding, tf_dim_nums):
    no_dilation = [1] * nb_spatial_dimensions
    # TODO(bchetioui): is there a generic way to do a transposed atrous
    # convolution in TensorFlow?
    if not (list(lhs_dilation) == no_dilation or
            list(rhs_dilation) == no_dilation):
      return "Both LHS and RHS dilations are set"
    # This is a non-dilated or atrous convolution
    if list(lhs_dilation) == no_dilation:
      return tf.nn.convolution(
          lhs, rhs, strides=window_strides, padding=tf_padding,
          data_format=tf_dim_nums, dilations=rhs_dilation)
    # TODO(bchetioui): the below path is unreachable for now, as passing a lhs
    # dilation to this function will result in convert_padding returning None
    # systematically. This must be investigated further.
    # Dilation of the LHS is transposed convolution
    return tf.nn.conv_transpose(
        lhs, rhs, out_shape, window_strides, padding=tf_padding,
        data_format=tf_dim_nums, dilations=lhs_dilation)

  tf_padding, error = convert_padding()
  if tf_padding is None:
    return error
  tf_dim_nums, error = convert_dim_nums()
  if tf_dim_nums is None:
    return error
  return convert_dilation_and_compute_result(tf_padding, tf_dim_nums)

def _conv_general_dilated(lhs, rhs, window_strides, padding, lhs_dilation,
                          rhs_dilation, dimension_numbers, feature_group_count,
                          batch_group_count, lhs_shape, rhs_shape, precision,
                          preferred_element_type, _in_avals, _out_aval):
  """Implementation of lax.conv_general_dilated_p using XlaConv."""
  if not _enable_xla:
    info_or_result = _try_tf_conv(
        lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
        dimension_numbers, feature_group_count, batch_group_count, _aval_to_tf_shape(_out_aval)
    )
    if not isinstance(info_or_result, str):
      return info_or_result
    else:
      raise _xla_path_disabled_error("conv_general_dilated")

  dnums_proto = _conv_general_dimension_numbers_proto(dimension_numbers)
  precision_config_proto = _conv_general_precision_config_proto(precision)
  assert batch_group_count == 1  # TODO(phawkins): implement batch_group_count
  out = tfxla.conv(
      lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
      dnums_proto, feature_group_count=feature_group_count,
      precision_config=precision_config_proto)
  # TODO: implement shape inference for XlaConv
  out.set_shape(_aval_to_tf_shape(_out_aval))
  return out


tf_impl_with_avals[lax.conv_general_dilated_p] = _conv_general_dilated


def _dot_general(lhs, rhs, dimension_numbers, precision, preferred_element_type):
  """Implementation of lax.dot_general_p in terms of tf.linalg.einsum."""
  del precision
  del preferred_element_type
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_ndim, rhs_ndim = len(lhs.shape), len(rhs.shape)
  # This condition ensures that:
  # 1) the considered dtype is not tf.bfloat16/tf.int32, which are supported by
  #    tf.linalg.einsum but not by tf.linalg.matmul;
  # 2) the batch dimensions are ordered in the same way in lhs and rhs (this is
  #    not strictly necessary, but we would have to reshape the array if that
  #    were not the case;
  # 3) lhs and rhs have the same number of dimensions +/- 1
  # 4) the number of non-batch dimensions in both tensors is either 1 or 2
  # 5) the contracting dimensions are consistent with those of a classic
  #    matrix/matrix, vector/matrix or matrix/vector multiplication.
  if (not lhs.dtype in [tf.bfloat16, tf.int32]
      and lhs_batch == rhs_batch == tuple(range(len(lhs_batch)))
      and lhs_ndim - rhs_ndim in [-1, 0, 1]
      and 1 <= lhs_ndim - len(lhs_batch) <= 2
      and 1 <= rhs_ndim - len(rhs_batch) <= 2
      and lhs_contracting == (len(lhs.shape) - 1,)
      and rhs_contracting == (len(lhs_batch),)):
    # All the inputs to tf.linalg.matmul must have 2 inner dimensions,
    # after their batch dimensions, so we need to expand the dimensions
    # appropriately. We can get to this branch with three combinations of
    # inner shapes:
    # - lhs.inner_shape == [a, b], rhs.inner_shape == [b, c]
    #   - in this case, the resulting inner shape is [a, c];
    # - lhs.inner_shape == [b]   , rhs.inner_shape == [b, c]
    #   - in this case, we need to expand lhs to [1, b], and the resulting
    #     shape is [c]. We need to squeeze the result of tf.linalg.matmul
    #     as it will have shape [1, c];
    # - lhs.shape == [batch] + [a, b], rhs.shape == [batch] + [b]
    #   - in this case, we need to expand rhs to [b, 1], and the resulting
    #     shape is [a]. We need to squeeze the result of tf.linalg.matmul
    #     as it will have shape [a, 1];
    # - lhs.shape == [batch] + [b]   , rhs.shape == [batch] + [b]
    #   - in this case, we need to expand lhs to [1, b] and rhs to [b, 1],
    #     and the resulting shape is (). We need to squeeze the result of
    #     tf.linalg.matmul as it will have shape [1, 1].
    squeeze_idxs = []
    if lhs_ndim - len(lhs_batch) == 1:
      lhs = tf.expand_dims(lhs, lhs_ndim - 1)
      squeeze_idxs.append(len(lhs.shape) - 2)
    if rhs_ndim - len(rhs_batch) == 1:
      rhs = tf.expand_dims(rhs, rhs_ndim)
      squeeze_idxs.append(len(rhs.shape) - 1)
    result = tf.linalg.matmul(lhs, rhs)
    if len(squeeze_idxs) != 0:
      assert all([result.shape[i] == 1 for i in squeeze_idxs])
      result = tf.squeeze(result, squeeze_idxs)
    return result

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
tf_impl[lax.dot_general_p] = _dot_general


def _broadcast(operand, *, sizes):
  result_shape = tf.TensorShape(sizes).concatenate(operand.shape)
  return tf.broadcast_to(operand, result_shape)
tf_impl[lax.broadcast_p] = _broadcast


def _broadcast_in_dim(operand, *, shape, broadcast_dimensions):
  inshape = [1] * len(shape)
  for orig_shape_i, broadcast_dim_i in zip(operand.shape, broadcast_dimensions):
    if orig_shape_i != 1: inshape[broadcast_dim_i] = shape[broadcast_dim_i]
  inshape_tf = _eval_shape(inshape)
  shape_tf = _eval_shape(shape)
  return tf.broadcast_to(tf.reshape(operand, inshape_tf), shape_tf)
tf_impl[lax.broadcast_in_dim_p] = _broadcast_in_dim


def _reshape(operand, *, new_sizes, dimensions):
  if dimensions is None:
    dimensions = tf.range(tf.rank(operand))
  new_sizes_tf = _eval_shape(new_sizes)
  return tf.reshape(tf.transpose(operand, dimensions), new_sizes_tf)
tf_impl[lax.reshape_p] = _reshape


def _squeeze(operand, *, dimensions, _in_avals, _out_aval):
  op_shape = _in_avals[0].shape
  new_shape = tuple(d for i, d in enumerate(op_shape) if i not in dimensions)
  new_shape_tf = _eval_shape(new_shape)
  return tf.reshape(operand, new_shape_tf)
tf_impl_with_avals[lax.squeeze_p] = _squeeze


def _pad(operand, padding_value, *, padding_config,
         _in_avals: Sequence[core.AbstractValue],
         _out_aval: core.AbstractValue):
  del _in_avals
  low, high, interior = util.unzip3(padding_config)
  if all(lo >= 0 and hi >= 0 and i == 0 for lo, hi, i in padding_config):
    return tf.pad(operand, util.safe_zip(low, high),
                  mode="CONSTANT", constant_values=padding_value)
  if not _enable_xla:
    raise _xla_path_disabled_error("pad")
  out = tfxla.pad(operand, padding_value, low, high, interior)
  # TODO(b/184499027): improve shape inference for XlaPad
  out.set_shape(_aval_to_tf_shape(_out_aval))
  return out
tf_impl_with_avals[lax.pad_p] = _pad


def _rev(operand, *, dimensions):
  return tf.reverse(operand, dimensions)
tf_impl[lax.rev_p] = _rev

tf_impl[lax.select_p] = tf.where

def _transpose(operand, *, permutation):
  return tf.transpose(operand, perm=permutation)
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
  output_type = tf.int32
  if dtypes.iinfo(index_dtype).bits > 32:
    output_type = tf.int64
  # TODO(phawkins): handle axes larger than 2^31.
  result = fn(operand, axis=axis, output_type=output_type)
  return tf.cast(result, to_tf_dtype(index_dtype))

tf_impl[lax.argmin_p] = functools.partial(_argminmax, tf.math.argmin)
tf_impl[lax.argmax_p] = functools.partial(_argminmax, tf.math.argmax)


_add_fn = tf.function(_add, autograph=False)
_ge_fn = tf.function(tf.math.greater_equal, autograph=False)

def _select_and_gather_add(tangents: TfVal,
                           operand: TfVal,
                           select_prim: core.Primitive,
                           window_dimensions: Sequence[int],
                           window_strides: Sequence[int],
                           base_dilation: Sequence[int],
                           window_dilation: Sequence[int],
                           padding: Sequence[Tuple[int, int]],
                           _in_avals: Sequence[core.AbstractValue],
                           _out_aval: core.AbstractValue):
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
    word_dtype = lax._UINT_DTYPES[nbits]
    double_word_dtype = lax._UINT_DTYPES[nbits * 2]

    # Packs two values into a tuple.
    def pack(a, b):
      a = _bitcast_convert_type(a, word_dtype)
      b = _bitcast_convert_type(b, word_dtype)
      a = _convert_element_type(a, new_dtype=double_word_dtype)
      b = _convert_element_type(b, new_dtype=double_word_dtype)
      a = tf.bitwise.left_shift(a, const(double_word_dtype, nbits))
      return tf.bitwise.bitwise_or(a, b)

    # Unpacks the first element of a tuple.
    def fst(t):
      assert t.dtype == double_word_dtype
      st = _shift_right_logical(t, const(double_word_dtype, nbits))
      return _bitcast_convert_type(
        _convert_element_type(st, new_dtype=word_dtype), dtype
      )

    # Unpacks the second element of a tuple.
    def snd(t):
      return _bitcast_convert_type(
        _convert_element_type(t, new_dtype=word_dtype), dtype
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
                                   pack(operand, tangents),
                                   window_dimensions=window_dimensions,
                                   window_strides=window_strides,
                                   padding=padding, base_dilation=base_dilation,
                                   window_dilation=window_dilation,
                                   _in_avals=_in_avals, _out_aval=_out_aval)

  return snd(out)

tf_impl_with_avals[lax.select_and_gather_add_p] = _select_and_gather_add


def _get_shape_from_tensor_or_array(x):
  if isinstance(x.shape, tf.TensorShape):
    return tuple(x.shape.as_list())
  return tuple(x.shape)

def _common_reduce_window(operand, init_val, reducer, window_dimensions,
                          window_strides, padding, base_dilation,
                          window_dilation, _in_avals, _out_aval):
  if not _enable_xla:
    raise _xla_path_disabled_error("reduce_window")
  o_spec = tf.TensorSpec((), dtype=operand.dtype)
  reducer_fn = tf.function(reducer, autograph=False).get_concrete_function(o_spec, o_spec)

  if not isinstance(init_val, tf.Tensor):
    assert not config.jax_enable_checks or _is_tfval(init_val), f"Non TfVal: {init_val}"
    init_val = tf.constant(init_val, operand.dtype)
  out = tfxla.reduce_window(operand, init_val,
                            reducer_fn, window_dimensions,
                            window_strides, base_dilations=base_dilation,
                            window_dilations=window_dilation, padding=padding)
  # TODO: implement shape inference for XlaReduceWindow
  out.set_shape(_aval_to_tf_shape(_out_aval))
  return out

def _reduce_window(operand, init_value, *, jaxpr, consts, window_dimensions,
                   window_strides, padding, base_dilation, window_dilation,
                   _in_avals, _out_aval):
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
      base_dilation, window_dilation, _in_avals, _out_aval
  )


# _try_tf_pool returns a Tensor when it succeeds, or a string describing why
# it did not succeed otherwise. It currently only supports reduce_window_max
# and reduce_window_sum.
# TODO(bchetioui): this function is not exhaustive wrt which
# reduce_window_max or reduce_window_sum cases can be translated into a call to
# max_pool or avg_pool. Further investigation is needed to fully flesh it out.
def _try_tf_pool(op_name, operand, window_dimensions, window_strides, padding,
                 base_dilation, window_dilation) -> Union[str, TfVal]:
  # Contrarily to the main path, tf.int8 is actually a valid type for
  # tf.nn.max_pool.
  if op_name == "reduce_window_max" and operand.dtype in [
      tf.bool, tf.uint32, tf.uint64, tf.complex64, tf.complex128
  ]:
    return f"tf.nn.max_pool does not support operands of type {operand.dtype}"
  if op_name == "reduce_window_sum" and operand.dtype not in [
      tf.float16, tf.float32, tf.float64
  ]:
    return f"tf.nn.avg_pool does not support operands of type {operand.dtype}"
  has_batch_dim = window_dimensions[0] == 1
  has_channel_dim = window_dimensions[-1] == 1
  nb_spatial_dimensions = len(operand.shape) - has_batch_dim - has_channel_dim
  if nb_spatial_dimensions < 1 or nb_spatial_dimensions > 3:
    return ("TensorFlow can only handle pooling for arrays with 1, 2, or "
            "3 spatial dimensions")
  # TODO(bchetioui): does a simple conversion with another base dilation exist?
  if list(base_dilation) != [1] * len(operand.shape):
    return "Unimplemented support for base dilation"
  # TODO(bchetioui): does a simple conversion with another window_dilation
  # exist? The whole story seems similar to convolution.
  if list(window_dilation) != [1] * len(operand.shape):
    return "Unimplemented support for window dilation"
  if list(padding) != [(0, 0)] * len(operand.shape):
    return "Unimplemented support for padding"
  # ReduceWindow in XLA takes an array of rank N as a parameter, but
  # tf.nn.max_pool / tf.nn.avg_pool take an array of rank N+2, with a default
  # shape of the form [batch_size] + input_spatial_shape + [num_channels]
  tf_operand = operand
  tf_window_dimensions = list(window_dimensions)
  tf_window_strides = list(window_strides)
  if not has_batch_dim:
    tf_operand = tf.expand_dims(tf_operand, 0)
    tf_window_dimensions = [1] + tf_window_dimensions
    tf_window_strides = [1] + tf_window_strides
  if not has_channel_dim:
    tf_operand = tf.expand_dims(tf_operand, -1)
    tf_window_dimensions.append(1)
    tf_window_strides.append(1)
  tf_data_format = "N" + "DHW"[-nb_spatial_dimensions:] + "C"
  tf_padding = "VALID"
  if op_name == "reduce_window_max":
    result = tf.nn.max_pool(tf_operand, tf_window_dimensions, tf_window_strides,
                            tf_padding, tf_data_format)
  elif op_name == "reduce_window_sum":
    avg = tf.nn.avg_pool(tf_operand, tf_window_dimensions, tf_window_strides,
                         tf_padding, tf_data_format)
    result = avg * np.prod(tf_window_dimensions)
  else:
    return f"Unimplemented support for {op_name}"

  if not has_batch_dim:
    result = tf.squeeze(result, 0)
  if not has_channel_dim:
    result = tf.squeeze(result, -1)
  return result


def _specialized_reduce_window(reducer, identity, operand, *, window_dimensions,
                               window_strides, padding, base_dilation,
                               window_dilation, _in_avals, _out_aval,
                               name=None):
  """Wraps the TensorFlow reduce window operation based on a reducer and an
  identity function defining the initial value of the reduction depending on
  the dtype of the operand.

  Args:
    reducer: reduction function of type TfVal -> TfVal -> TfVal
    identity: function that takes a TensorFlow dtype as a parameter and returns
      the starting value of the reduction.
    operand: N dimensional array containing elements of type T
    window_dimensions: array of integers for window dimension values
    window_strides: array of integers for window stride values
    padding: array of pairs of integers for padding values
    base_dilation: array of integers for base dilation values
    window_dilation: array of integers for window dilation values
    name: the name of the specialized reduce window primitive for which this
      conversion function is called. This information may help to choose a
      different conversion path (optional)

  Returns:
    The reduced operand.
  """
  if name in ["reduce_window_max", "reduce_window_sum"]:
    res = _try_tf_pool(name, operand, window_dimensions, window_strides,
                       padding, base_dilation, window_dilation)
    if not isinstance(res, str):
      return res

  return _common_reduce_window(
      operand, identity(operand.dtype), reducer, window_dimensions,
      window_strides, padding, base_dilation, window_dilation, _in_avals,
      _out_aval
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
tf_impl_with_avals[lax.reduce_window_sum_p] = (
    functools.partial(_specialized_reduce_window, _add, lambda x: 0,
                      name="reduce_window_sum"))
tf_impl_with_avals[lax.reduce_window_min_p] = (
    functools.partial(_specialized_reduce_window, tf.math.minimum,
                      _get_min_identity, name="reduce_window_min"))
tf_impl_with_avals[lax.reduce_window_max_p] = (
    functools.partial(_specialized_reduce_window, tf.math.maximum,
                      _get_max_identity, name="reduce_window_max"))
tf_impl_with_avals[lax.reduce_window_p] = _reduce_window
# pylint: enable=protected-access

# We use lax_control_flow._cumred_tpu_translation_rule to convert cummax,
# cummin, cumsum and cumprod. This is efficient on TPU, but the complexity is
# O(n^2) on other backends. This may be implemented using associative_scan
# instead to favor different backends.
tf_impl_with_avals[lax_control_flow.cummin_p] = _convert_jax_impl(
    functools.partial(lax_control_flow._cumred_tpu_translation_rule,
                      lax._reduce_window_min), multiple_results=False)
tf_impl_with_avals[lax_control_flow.cummax_p] = _convert_jax_impl(
    functools.partial(lax_control_flow._cumred_tpu_translation_rule,
                      lax._reduce_window_max), multiple_results=False)
# TODO(bchetioui): cumsum and cumprod can be converted using pure TF ops for
# certain dtypes: bfloat16, float16, float32, float64, and int32. Other dtypes
# will fail when running in compiled mode, but are otherwise compatible with
# the operation. A non-XLA path can thus be defined for all dtypes, though the
# tests will crash.
tf_impl_with_avals[lax_control_flow.cumsum_p] = _convert_jax_impl(
    functools.partial(lax_control_flow._cumred_tpu_translation_rule,
                      lax._reduce_window_sum), multiple_results=False)
tf_impl_with_avals[lax_control_flow.cumprod_p] = _convert_jax_impl(
    functools.partial(lax_control_flow._cumred_tpu_translation_rule,
                      lax._reduce_window_prod), multiple_results=False)

def _select_and_scatter(
    operand, source, init_value, select_jaxpr, select_consts, scatter_jaxpr,
    scatter_consts, window_dimensions, window_strides, padding):
  raise NotImplementedError("TODO: jax2tf can not convert _select_and_scatter")

tf_impl[lax.select_and_scatter_p] = _select_and_scatter

@functools.partial(bool_to_int8, argnums=(0, 1))
def _select_and_scatter_add(source, operand, *, select_prim, window_dimensions,
                            window_strides, padding, _in_avals, _out_aval):
  if not _enable_xla:
    raise _xla_path_disabled_error("select_and_scatter_add")
  init_value = tf.zeros((), operand.dtype)
  select_fn = (tf.function(tf_impl[select_prim], autograph=False)
                 .get_concrete_function(init_value, init_value))
  scatter_fn = _add_fn.get_concrete_function(init_value, init_value)
  out = tfxla.select_and_scatter(operand, window_dimensions, window_strides,
                                 padding, source, init_value, select_fn,
                                 scatter_fn)
  out.set_shape(_aval_to_tf_shape(_out_aval))
  return out

tf_impl_with_avals[lax.select_and_scatter_add_p] = _select_and_scatter_add

def _threefry2x32_jax_impl(*args: TfVal, _in_avals, _out_aval):
  res = _convert_jax_impl(
    functools.partial(jax._src.random._threefry2x32_lowering,
                      use_rolled_loops=False),
    multiple_results=True)(*args, _in_avals=_in_avals, _out_aval=_out_aval)
  return res
tf_impl_with_avals[jax.random.threefry2x32_p] = _threefry2x32_jax_impl


# Use the vmap implementation, otherwise on TPU the performance is really bad
# With use_vmap=True on, we get about the same performance for JAX and jax2tf.
tf_impl_with_avals[random.random_gamma_p] = _convert_jax_impl(
  functools.partial(jax._src.random._gamma_impl, use_vmap=True),
  multiple_results=False)

def _gather_dimensions_proto(indices_shape, dimension_numbers):
  proto = xla_data_pb2.GatherDimensionNumbers()
  proto.offset_dims.extend(dimension_numbers.offset_dims)
  proto.collapsed_slice_dims.extend(dimension_numbers.collapsed_slice_dims)
  proto.start_index_map.extend(dimension_numbers.start_index_map)
  assert indices_shape
  proto.index_vector_dim = len(indices_shape) - 1
  return proto

@functools.partial(bool_to_int8, argnums=0)
def _gather(operand, start_indices, *, dimension_numbers, slice_sizes,
            _in_avals, _out_aval):
  """Tensorflow implementation of gather."""
  del _in_avals
  if not _enable_xla:
    raise _xla_path_disabled_error("gather")
  proto = _gather_dimensions_proto(start_indices.shape, dimension_numbers)
  slice_sizes_tf = _eval_shape(slice_sizes)
  out = tfxla.gather(operand, start_indices, proto, slice_sizes_tf, False)
  out.set_shape(_aval_to_tf_shape(_out_aval))
  return out
tf_impl_with_avals[lax.gather_p] = _gather

def _slice(operand, start_indices, limit_indices, strides,
           _in_avals, _out_aval):
  if strides is None:
    strides = [1] * len(start_indices)
  slices = tuple(map(slice,
                     _eval_shape(start_indices),
                     _eval_shape(limit_indices),
                     _eval_shape(strides)))
  out = operand[slices]
  # TODO(b/184503314): improve shape inference for __getitem__
  out.set_shape(_aval_to_tf_shape(_out_aval))
  return out

tf_impl_with_avals[lax.slice_p] = _slice


def _dynamic_slice(operand, *start_indices, slice_sizes,
                   _in_avals: Sequence[core.ShapedArray],
                   _out_aval: core.ShapedArray):
  # Here we could use tf.slice. Similarly, for lax.gather we can sometimes use
  # tf.gather. But those have different semantics for index-out-of-bounds than
  # JAX (and XLA). We have tried to force compilation, by wrapping into
  # tf.xla.experimental.compile, or tf.function(jit_compile=True), but
  # those solutions are brittle because they do not work when nested into an
  # outer compilation (see b/162814494 and b/163006262). They also do not
  # survive well being put in a SavedModel. Hence, we now use TFXLA slicing
  # and gather ops.
  if not _enable_xla:
    raise _xla_path_disabled_error("dynamic_slice")
  res = tfxla.dynamic_slice(operand, tf.stack(start_indices),
                            size_indices=_eval_shape(slice_sizes))
  # TODO: implement shape inference for XlaDynamicSlice
  res.set_shape(_aval_to_tf_shape(_out_aval))
  return res

tf_impl_with_avals[lax.dynamic_slice_p] = _dynamic_slice

def _scatter_dimensions_proto(indices_shape, dimension_numbers):
  proto = xla_data_pb2.ScatterDimensionNumbers()
  proto.update_window_dims.extend(dimension_numbers.update_window_dims)
  proto.inserted_window_dims.extend(dimension_numbers.inserted_window_dims)
  proto.scatter_dims_to_operand_dims.extend(
      dimension_numbers.scatter_dims_to_operand_dims)
  assert indices_shape
  proto.index_vector_dim = len(indices_shape) - 1
  return proto

def _scatter(operand, scatter_indices, updates, *,
             update_jaxpr, update_consts,
             dimension_numbers, indices_are_sorted, unique_indices,
             _in_avals: Sequence[core.AbstractValue],
             _out_aval: core.AbstractValue):
  del unique_indices, _in_avals
  assert len(update_consts) == 0, "Update computation cannot have constants"

  if not _enable_xla:
    raise _xla_path_disabled_error("scatter")

  proto = _scatter_dimensions_proto(scatter_indices.shape, dimension_numbers)

  def update_computation(arg1: TfVal, arg2: TfVal) -> TfVal:
    closed_jaxpr = core.ClosedJaxpr(update_jaxpr, update_consts)
    res, = _interpret_jaxpr(closed_jaxpr, arg1, arg2)
    return res

  o_spec = tf.TensorSpec((), dtype=operand.dtype)
  xla_update_computation = (
      tf.function(update_computation, autograph=False).get_concrete_function(o_spec, o_spec))
  out = tfxla.scatter(operand, scatter_indices, updates, xla_update_computation, proto,
                      indices_are_sorted=indices_are_sorted)
  # TODO: implement shape analysis for XlaScatter
  out.set_shape(_aval_to_tf_shape(_out_aval))
  return out

tf_impl_with_avals[lax.scatter_p] = _scatter
tf_impl_with_avals[lax.scatter_min_p] = _scatter
tf_impl_with_avals[lax.scatter_max_p] = _scatter
tf_impl_with_avals[lax.scatter_mul_p] = _scatter
tf_impl_with_avals[lax.scatter_add_p] = _scatter

def _dynamic_update_slice(operand, update, *start_indices):
  if not _enable_xla:
    raise _xla_path_disabled_error("dynamic_update_slice")
  return tfxla.dynamic_update_slice(operand, update, tf.stack(start_indices))
tf_impl[lax.dynamic_update_slice_p] = _dynamic_update_slice


def _cond(index: TfVal, *operands: TfVal,
          branches: Sequence[core.ClosedJaxpr],
          linear: Sequence[bool]) -> Sequence[TfVal]:
  del linear
  # tf.cond needs lambdas with no arguments.
  branches_tf = [functools.partial(_interpret_jaxpr, jaxpr, *operands)
                 for jaxpr in branches]
  return tf.switch_case(index, branches_tf)

tf_impl[lax_control_flow.cond_p] = _cond


def _while(*args: TfVal, cond_nconsts: int, cond_jaxpr: core.ClosedJaxpr,
           body_nconsts: int, body_jaxpr: core.ClosedJaxpr) -> Sequence[TfVal]:
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
  return tf.while_loop(cond_tf_func, body_tf_func, init_carry)


def _batched_cond_while(*args: TfVal,
                        cond_nconsts: int, cond_jaxpr: core.ClosedJaxpr,
                        body_nconsts: int, body_jaxpr: core.ClosedJaxpr
                        ) -> Sequence[TfVal]:
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
      pred_b_bcast = _broadcast_in_dim(pred_b,
                                       shape=new_c.shape,
                                       broadcast_dimensions=list(range(len(pred_b.shape))))
      return tf.where(pred_b_bcast, new_c, c)

    selected_carry: Sequence[TfVal] = list(
      util.safe_map(select_one_carry, new_carry, carry))
    next_pred_b, = _interpret_jaxpr(cond_jaxpr, *cond_consts, *selected_carry)
    return (next_pred_b, *selected_carry)

  _, *res_carry = tf.while_loop(new_cond_tf_func, new_body_tf_func,
                                (init_pred_b, *init_carry))
  return res_carry

tf_impl[lax_control_flow.while_p] = _while

# We use the scan impl rule to rewrite in terms of while.
tf_impl_with_avals[lax_control_flow.scan_p] = _convert_jax_impl(lax_control_flow._scan_impl)

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
    values, indices = tf.math.top_k(tf.dtypes.cast(operand, conversion_dtype),
                                    k=k, sorted=True)
    return tf.dtypes.cast(values, operand.dtype), indices
  else:
    return tf.math.top_k(operand, k=k, sorted=True)

tf_impl[lax.top_k_p] = _top_k


def _sort(*operands: TfVal, dimension: int, is_stable: bool,
          num_keys: int) -> Tuple[TfVal, ...]:
  if not _enable_xla:
    raise _xla_path_disabled_error("sort")
  assert 1 <= num_keys <= len(operands)
  assert 0 <= dimension < len(
      operands[0].shape
  ), f"Invalid {dimension} for ndim {len(operands[0].shape)}"

  # The comparator is a 2N-argument TF function, with arguments [2k] and [2k +1]
  # corresponding to two scalars from operand[k].
  def lexicographic_comparator_old(*tf_args: TfVal) -> TfVal:
    assert len(tf_args) == 2 * len(operands)
    # We build a comparison:
    #     arg[0] < arg[1] or (arg[0] == arg[1] and (arg[2] < arg[3] or ...))
    # all the way to arg[2 * num_keys - 2] < arg[2 * num_keys - 1]
    inside_comparison = None
    for key_idx in range(num_keys - 1, -1, -1):
      a = tf_args[2 * key_idx]
      b = tf_args[2 * key_idx + 1]
      a_lt_b = tf.math.less(a, b)
      if inside_comparison is None:
        inside_comparison = a_lt_b
      else:
        inside_comparison = tf.math.logical_or(
            a_lt_b, tf.math.logical_and(tf.math.equal(a, b), inside_comparison))
    return inside_comparison

  comparator_spec: List[tf.TensorSpec] = []
  comparator_jax_in_avals: List[core.AbstractValue] = []
  for op in operands:
    o_spec = tf.TensorSpec((), dtype=op.dtype)
    comparator_spec.extend([o_spec, o_spec])
    o_aval = core.ShapedArray((), to_jax_dtype(op.dtype))
    comparator_jax_in_avals.extend([o_aval, o_aval])

  # Use the same comparator that JAX uses when compiling to XLA, to get the
  # proper NaN/Inf total order, and the lexicographic ordering.
  # The comparator is a 2N-argument TF function, with arguments [2k] and [2k +1]
  # corresponding to two scalars from operand[k].
  def lexicographic_comparator(*tf_args: TfVal) -> TfVal:
    return _convert_jax_impl(
        lax._sort_lt_comparator, multiple_results=False)(
            *tf_args,
            _in_avals=comparator_jax_in_avals,
            _out_aval=core.ShapedArray((), np.bool_),
            num_keys=num_keys)

  xla_comparator_computation = (
      tf.function(lexicographic_comparator,
                  autograph=False).get_concrete_function(*comparator_spec))
  results = tfxla.variadic_sort(operands, dimension=dimension,
                                is_stable=is_stable,
                                comparator=xla_comparator_computation)
  return results


tf_impl[lax.sort_p] = _sort

def _fft(x, fft_type, fft_lengths):
  FFT, IFFT, RFFT, IRFFT = list(map(xla_client.FftType, [0, 1, 2, 3]))
  if fft_type == IRFFT:
    expected_lengths = x.shape[-len(fft_lengths):-1] + ((x.shape[-1] - 1) * 2,)
  else:
    expected_lengths = x.shape[-len(fft_lengths):]
  if expected_lengths != fft_lengths:
    raise NotImplementedError(
      f"Unsupported fft_lengths={fft_lengths} for fft_type={fft_type} of "
      f"array with shape={x.shape}.")
  tf_funcs = {FFT: [tf.signal.fft, tf.signal.fft2d, tf.signal.fft3d],
              IFFT: [tf.signal.ifft, tf.signal.ifft2d, tf.signal.ifft3d],
              RFFT: [tf.signal.rfft, tf.signal.rfft2d, tf.signal.rfft3d],
              IRFFT: [tf.signal.irfft, tf.signal.irfft2d, tf.signal.irfft3d]}
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

def _eigh(operand: TfVal, lower: bool, _in_avals, _out_aval):
  if operand.shape[-1] == 0:
    v, w = operand, tf.reshape(operand, _eval_shape(_in_avals[0].shape[:-1]))
  else:
    if not lower:
      operand = tf.linalg.adjoint(operand)
    w, v = tf.linalg.eigh(operand)
  cast_type = { tf.complex64: tf.float32,
                tf.complex128: tf.float64 }.get(operand.dtype)
  if cast_type is not None:
    w = tf.cast(w, cast_type)
  return v, w

tf_impl_with_avals[lax_linalg.eigh_p] = _eigh

def _lu(operand: TfVal, _in_avals, _out_aval):
  return _convert_jax_impl(lax_linalg._lu_python)(operand, _in_avals=_in_avals,
                                                  _out_aval=_out_aval)

tf_impl_with_avals[lax_linalg.lu_p] = _lu

def _triangular_solve(a: TfVal, b: TfVal, *, left_side: bool, lower: bool,
                      transpose_a: bool, conjugate_a: bool,
                      unit_diagonal: bool,
                      _in_avals: Sequence[core.ShapedArray],
                      _out_aval: core.ShapedArray):
  if unit_diagonal:
    a_aval, _ = _in_avals
    a_shape = _eval_shape(a_aval.shape)
    a = tf.linalg.set_diag(a, tf.ones(a_shape[:-1], dtype=a.dtype))
  if not left_side:
    rank = len(a.shape)
    transpose_dimensions = list(range(rank - 2)) + [rank - 1, rank - 2]
    a = tf.transpose(a, transpose_dimensions)
    b = tf.transpose(b, transpose_dimensions)
    lower = not lower
  # adjoint == transpose for real dtypes, so special care need only be taken
  # for complex types.
  if a.dtype in [tf.complex64, tf.complex128]:
    if (transpose_a and not conjugate_a) or (not transpose_a and conjugate_a):
      a = tf.math.conj(a)
  result = tf.linalg.triangular_solve(a, b, lower=lower, adjoint=transpose_a)
  if not left_side:
    result = tf.transpose(result, transpose_dimensions)
  return result

tf_impl_with_avals[lax_linalg.triangular_solve_p] = _triangular_solve

def _linear_solve(*args: TfVal, const_lengths, jaxprs, _in_avals, _out_aval):
  return _convert_jax_impl(lax_control_flow._custom_linear_solve_impl)(
    *args, const_lengths=const_lengths, jaxprs=jaxprs, _in_avals=_in_avals, _out_aval=_out_aval)

tf_impl_with_avals[lax_control_flow.linear_solve_p] = _linear_solve

def _custom_jvp_call_jaxpr(*args: TfVal,
                           fun_jaxpr: core.ClosedJaxpr,
                           jvp_jaxpr_thunk: Callable,
                           num_consts: int) -> Sequence[TfVal]:
  # TODO(necula): ensure that there is no AD transformation in scope
  return _interpret_jaxpr(fun_jaxpr, *args)

tf_impl[custom_derivatives.custom_jvp_call_jaxpr_p] = _custom_jvp_call_jaxpr


def _custom_vjp_call_jaxpr(*args: TfVal,
                           fun_jaxpr: core.ClosedJaxpr,
                           **_) -> Sequence[TfVal]:
  # TODO(necula): ensure that there is no AD transformation in scope
  return _interpret_jaxpr(fun_jaxpr, *args)

tf_impl[custom_derivatives.custom_vjp_call_jaxpr_p] = _custom_vjp_call_jaxpr

def _custom_lin(*args: TfVal, **_) -> Sequence[TfVal]:
  raise TypeError("can't apply forward-mode autodiff (jvp) to a custom_vjp "
                  "function.")

tf_impl[ad.custom_lin_p] = _custom_lin


def split_to_logical_devices(
    tensor: TfVal,
    partition_dimensions: pxla.PartitionsOrReplicated):
  """Like TPUMPStrategy.experimental_split_to_logical_devices.

  For jax2tf purposes we want to avoid needing to thread the `strategy` object
  through the generated computation. It seems that the original function needs
  the strategy object only for error checking, which we assume is done upstream
  by JAX.

  Args:
    tensor: Input tensor to annotate.
    partition_dimensions: A list of integers, with one integer per tensor
      dimension, specifying in how many parts the dimension should be split. The
      product of integers must equal the number of devices per replica.
    use_sharding_op: whether to use a sharding op, or not.

  Returns:
    an annotated tensor.
  """
  # This corresponds to the sharding annotations in
  # xla_bridge._sharding_to_proto.
  if partition_dimensions is None:
    return xla_sharding.replicate(tensor, use_sharding_op=True)
  num_partition_splits = np.prod(partition_dimensions)
  tile_assignment = np.arange(num_partition_splits).reshape(
      partition_dimensions)
  return xla_sharding.tile(tensor, tile_assignment, use_sharding_op=True)


def _sharded_call(f: lu.WrappedFun, vals: Sequence[TfVal],
                  in_parts: Sequence[pxla.PartitionsOrReplicated],
                  out_parts_thunk,
                  **_) -> Sequence[Tuple[TfVal, core.AbstractValue]]:
  sharded_vals = util.safe_map(split_to_logical_devices, vals, in_parts)
  vals_out = f.call_wrapped(*sharded_vals)
  out_parts_flat = out_parts_thunk()
  assert len(out_parts_flat) == len(vals_out), f"expected {len(out_parts_flat)} == {len(vals_out)}"
  sharded_vals_out = [
      (split_to_logical_devices(val, val_part), val_aval)
      for (val, val_aval), val_part in util.safe_zip(vals_out, out_parts_flat)
  ]
  return sharded_vals_out


def _sharding_constraint(arg: TfVal, *,
                         partitions: pxla.PartitionsOrReplicated):
  return split_to_logical_devices(arg, partitions)


tf_impl[sharded_jit.sharding_constraint_p] = _sharding_constraint


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
