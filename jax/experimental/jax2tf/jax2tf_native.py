# Copyright 2020 The JAX Authors.
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
from functools import partial
import contextlib
import os
import re
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast

from absl import logging

import jax
from jax import lax
from jax import config
from jax import core, custom_derivatives
from jax import linear_util as lu
from jax import random, tree_util
from jax import numpy as jnp
from jax.experimental import maps
from jax.experimental import pjit
from jax._src import sharding
from jax.interpreters import ad
from jax.interpreters import mlir
from jax.interpreters import pxla
from jax.interpreters import xla

from jax._src import ad_checkpoint
from jax._src import ad_util
from jax._src import api
from jax._src import api_util
from jax._src import dispatch
from jax._src import dtypes
from jax._src import prng
from jax._src import random as random_internal
from jax._src import source_info_util
from jax._src import util
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lax import lax as lax_internal
from jax._src.lax import linalg as lax_linalg
from jax._src.lax import slicing as lax_slicing
from jax._src.lax import windowed_reductions as lax_windowed_reductions
from jax._src.lib import xla_client
from jax._src.numpy.ufuncs import logaddexp

from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.jax2tf import shape_poly
from jax.experimental.jax2tf import impl_no_xla


import numpy as np
import tensorflow as tf  # type: ignore[import]

# These don't have public equivalents.
# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla  # type: ignore[import]
from tensorflow.compiler.xla import xla_data_pb2  # type: ignore[import]
from tensorflow.core.framework import attr_value_pb2  # type: ignore[import]
try:
  from tensorflow.python.compiler.xla.experimental import xla_sharding  # type: ignore[import]
except ModuleNotFoundError:
  # This can be removed when TF 2.10 support is no longer needed.
  from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding  # type: ignore[import]
from tensorflow.python.framework import ops as tf_ops  # type: ignore[import]
from tensorflow.python.eager import context as tf_context  # type: ignore[import]
# pylint: enable=g-direct-tensorflow-import

NameStack = source_info_util.NameStack
PolyShape = shape_poly.PolyShape

# A temporary internal flag, to enable the wrapping of jax.jit functions
# with tf.function(jit_compile=True). See #7389. This change has triggered a
# number of failures in TF. We keep this until we are confident that it does
# not create problems.
# TODO(b/207464757): figure out why this change breaks test
_WRAP_JAX_JIT_WITH_TF_FUNCTION = False

# The scope name need to be a valid TensorFlow name. See
# https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/core/framework/node_def_util.cc#L731
_VALID_SCOPE_REGEX = re.compile("^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$")
_INVALID_SCOPE_CHAR = re.compile("[^A-Za-z0-9_.\\/-]")

map = util.safe_map
zip = util.safe_zip


def _sanitize_scope_name(name):
  scope_name = _INVALID_SCOPE_CHAR.sub("_", name)
  if not _VALID_SCOPE_REGEX.match(scope_name):
    scope_name = f".{scope_name}"
  return scope_name


# A value suitable in a TF tracing context: tf.Tensor, tf.Variable,
# or Python scalar or numpy.ndarray. (A tf.EagerTensor is a tf.Tensor.)
TfVal = Any
DType = Any
PrecisionType = int  # Enum xla_data.PrecisionConfig.Precision

def _is_tfval(v: TfVal) -> bool:
  if isinstance(v, (tf.Tensor, tf.Variable)):
    return True
  try:
    # Include all convertible types, even if not supported on accelerators.
    with tf.device("CPU"):
      tf.constant(v)
    return True
  except:
    return False

# In order to ensure that JAX picks up the proper user-frame for source
# locations we will register the TensorFlow source path as an internal
# path with source_info_util. The typical stack when a JAX primitive
# conversion happens is:
#    jax2tf.process_primitive  (top of stack)
#    jax tracing machinery ...
#    tf.custom_gradient machinery ...
#    jax2tf.converted_fun
#    tf function machinery ...
#    user code invokes the converted function on TF tensors
#
# We need to skip over not only JAX internal frames, but TF internal frames
# also.
# We register the TensorFlow source path lazily
_has_registered_tf_source_path = False

class _ThreadLocalState(threading.local):
  def __init__(self):
    self.name_stack = ""
    # XLA is not linked in all environments; when converting a primitive, if this
    # variable is disabled, we try harder to use only standard TF ops if they are
    # applicable to the concrete use case; if the resulting conversion path ends up
    # requiring a TFXLA operation, an exception is thrown instead.
    self.enable_xla = True

    # Keep track if we are inside a call_tf. In that context we disable the
    # safety check that we are not inside JAX transformations.
    self.inside_call_tf = False

    # Maps dimension variables to TF expressions
    self.shape_env: Sequence[Tuple[str, TfVal]] = ()

    # Whether to actually include XLA op metadata in the generated TF ops
    # TODO(b/189306134): implement support for XLA metadata
    self.include_xla_op_metadata = False

    # A cache for the tf.convert_to_tensor for constants. We try to preserve
    # sharing for constants, to enable tf.Graph to take advantage of it.
    # See https://github.com/google/jax/issues/7992.
    self.constant_cache = None  # None means that we don't use a cache. We
    # may be outside a conversion scope.

_thread_local_state = _ThreadLocalState()

def _get_current_name_stack() -> Union[NameStack, str]:
  return source_info_util.current_name_stack()

@contextlib.contextmanager
def inside_call_tf():
  # Set the inside_call_tf flag for a context.
  prev = _thread_local_state.inside_call_tf
  _thread_local_state.inside_call_tf = True
  try:
    yield
  finally:
    _thread_local_state.inside_call_tf = prev


def convert(fun_jax: Callable,
            *,
            polymorphic_shapes=None,
            with_gradient=True,
            enable_xla=True) -> Callable:
  """jax2tf_default_experimental_native_lowering."""
  if not enable_xla:
    raise ValueError(
        "experimental_native_lowering is not supported with enable_xla=False")
  api._check_callable(fun_jax)
  fun_name = getattr(fun_jax, "__name__", "unknown")
  name_stack = util.wrap_name(fun_name, "jax2tf")
  def converted_fun_tf(*args_tf: TfVal, **kwargs_tf: TfVal) -> TfVal:
    # TODO: is there a better way to check if we are inside a transformation?
    if not core.trace_state_clean() and not _thread_local_state.inside_call_tf:
      # It is Ok to nest convert when we are inside a call_tf
      raise ValueError("convert must be used outside all JAX transformations." +
                       f"Trace state: {core.thread_local_state.trace_state.trace_stack}")

    fun_flat_jax, args_flat_tf, in_tree, out_tree_thunk = flatten_fun_jax(fun_jax, args_tf, kwargs_tf)
    # out_tree_thunk will be ready after we call fun_flat_jax below.

    # Expand the polymorphic_shapes to match the args_flat_tf. The polymorphic_shapes
    # argument refers to positional arguments only.
    if polymorphic_shapes is None or isinstance(polymorphic_shapes, (PolyShape, str)):
      polymorphic_shapes_ = (polymorphic_shapes,) * len(args_tf)
    else:
      if not (isinstance(polymorphic_shapes, Sequence) and len(polymorphic_shapes) == len(args_tf)):
        msg = ("polymorphic_shapes must be a sequence with the same length as the positional argument list "
               f"({len(args_tf)}). Got polymorphic_shapes={repr(polymorphic_shapes)}.")
        raise TypeError(msg)
      polymorphic_shapes_ = tuple(polymorphic_shapes)

    polymorphic_shapes_flat = tuple(
        api_util.flatten_axes("jax2tf.convert polymorphic_shapes",
                              in_tree,
                              (polymorphic_shapes_, {k: None for k in kwargs_tf.keys()})))

    args_and_avals = tuple(map(preprocess_arg_tf,
                               range(len(args_flat_tf)), args_flat_tf, polymorphic_shapes_flat))
    args_flat_tf, args_avals_flat = util.unzip2(args_and_avals)

    dim_vars, get_dim_values_jax = shape_poly.prepare_dim_var_env(args_avals_flat)
    dim_values, _ = _interpret_fun_jax(get_dim_values_jax, args_flat_tf,
                                       args_avals_flat, name_stack)
    shape_env = zip(dim_vars, dim_values)

    try:
      assert not _thread_local_state.shape_env, f"Unexpected shape environment {_thread_local_state.shape_env}"

      prev_enable_xla = _thread_local_state.enable_xla
      _thread_local_state.enable_xla = enable_xla

      prev_include_xla_op_metadata = _thread_local_state.include_xla_op_metadata
      # TODO(b/189306134): implement support for XLA metadata
      _thread_local_state.include_xla_op_metadata = False

      _thread_local_state.shape_env = shape_env
      global _has_registered_tf_source_path
      if not _has_registered_tf_source_path:
        source_info_util.register_exclusion(os.path.dirname(tf.__file__))
        _has_registered_tf_source_path = True

      if with_gradient:

        @tf.custom_gradient
        def converted_fun_flat_with_custom_gradient_tf(
            *args_flat_tf: TfVal) -> TfVal:
          outs_tf, out_avals = _interpret_fun_jax(
              fun_flat_jax,
              args_flat_tf,
              args_avals_flat,
              name_stack,
              fresh_constant_cache=True)
          return (tuple(outs_tf),
                  make_custom_gradient_fn_tf(
                      fun_flat_jax=fun_flat_jax,
                      args_flat_tf=args_flat_tf,
                      args_avals_flat=args_avals_flat,
                      polymorphic_shapes_flat=polymorphic_shapes_flat,
                      out_avals=out_avals))

        out_flat_tf = converted_fun_flat_with_custom_gradient_tf(*args_flat_tf)
      else:
        outs_tf, out_avals = _interpret_fun_jax(
            fun_flat_jax,
            args_flat_tf,
            args_avals_flat,
            name_stack,
            fresh_constant_cache=True)
        message = ("The jax2tf-converted function does not support gradients. "
                   "Use `with_gradient` parameter to enable gradients")
        # We use PreventGradient, which is propagated through a SavedModel.
        out_flat_tf = [
            tf.raw_ops.PreventGradient(input=o, message=message)
            for o in outs_tf
        ]
    finally:
      _thread_local_state.shape_env = ()
      _thread_local_state.enable_xla = prev_enable_xla
      _thread_local_state.include_xla_op_metadata = prev_include_xla_op_metadata

    out_flat_tf = [tf.identity(x, "jax2tf_out") for x in out_flat_tf]
    out_tf = tree_util.tree_unflatten(out_tree_thunk(), out_flat_tf)
    return out_tf

  return converted_fun_tf


# Internals

def flatten_fun_jax(fun_jax: Callable, args_tf: Sequence[TfVal],
                    kwargs_tf: Dict[str, TfVal]
                    ) -> Tuple[Callable, Sequence[TfVal], Any, Callable]:
  """Wraps the function to take a (flat) list of positional args.

  jax2tf works better and is simpler when the JAX function takes and returns
  just a tuple of values (no pytrees, no kwargs). This is in part because
  jax.vjp does not support kwargs and we can only set
  tf.custom_gradient on functions with flat arguments and results

  Returns:
     * the wrapped JAX function taking and returning a flat list of arguments
     * the flat list of TF arguments
     * the in_tree corresponding to the tuple (args_tf, kwargs_tf)
     * a thunk that can be called after the wrapped function has been called
       to return the output pytree.
  """
  # TODO(necula): technically we should use TF's flattening and unflattening
  # because we are working with TF values.
  args_flat_tf, in_tree = tree_util.tree_flatten((args_tf, kwargs_tf))

  out_tree_ref = None
  def fun_flat_jax(*args_flat_jax):
    tree_args, tree_kwargs = tree_util.tree_unflatten(in_tree, args_flat_jax)
    tree_res = fun_jax(*tree_args, **tree_kwargs)
    res_flat_jax, out_tree = tree_util.tree_flatten(tree_res)
    nonlocal out_tree_ref
    assert out_tree_ref is None or out_tree_ref == out_tree
    out_tree_ref = out_tree
    return res_flat_jax

  if hasattr(fun_jax, "lower"):
    # If the fun_jax is already a jit(f) or pjit(f), we must
    # preserve the lowering function. This will be used in the _lower_native_and_run.
    # We rely on the fact that the lowering is the same for the function
    # taking pytrees, and the one taking flat args.
    def fun_flat_jax_lower(*args_flat_jax):
      tree_args, tree_kwargs = tree_util.tree_unflatten(in_tree, args_flat_jax)
      lowered = fun_jax.lower(*tree_args, **tree_kwargs)
      out_tree = lowered.out_tree
      nonlocal out_tree_ref
      assert out_tree_ref is None or out_tree_ref == out_tree
      out_tree_ref = out_tree
      return lowered
    setattr(fun_flat_jax, "lower", fun_flat_jax_lower)

  return fun_flat_jax, args_flat_tf, in_tree, lambda: out_tree_ref

def preprocess_arg_tf(arg_idx: int,
                      arg_tf: TfVal,
                      polymorphic_shape: Optional[str]
                      ) -> Tuple[TfVal, core.ShapedArray]:
  if not _is_tfval(arg_tf):
    msg = (f"Argument {arg_tf} of type {type(arg_tf)} of jax2tf.convert(f) should "
           "be NumPy array, scalar, tf.Variable, or tf.Tensor")
    raise TypeError(msg)

  # May cast the args_flat to JAX types, using JAX's interpretation
  # of types of constants.
  arg_tf, arg_jax_dtype = _tfval_to_tensor_jax_dtype(arg_tf)
  # Name input tensors; do this after we have cast the arguments
  arg_tf = tf.identity(arg_tf, f"jax2tf_arg_{arg_idx}")

  # Fix the shape for TF1
  tf_arg_shape = np.shape(arg_tf)
  arg_shape = tuple(d.value if isinstance(d, tf.compat.v1.Dimension) else d for d in tf_arg_shape)

  arg_aval = shape_poly.arg_aval(arg_shape, arg_jax_dtype, polymorphic_shape)
  return arg_tf, arg_aval


# Prepare the grad_fn for tf.custom_gradient.
def make_custom_gradient_fn_tf(
    fun_flat_jax: Callable,
    args_flat_tf: Sequence[TfVal],
    polymorphic_shapes_flat: Sequence[str],
    args_avals_flat: Sequence[core.ShapedArray],
    out_avals: Sequence[core.ShapedArray]):

  def grad_fn_tf(*out_cts_flat_tf: TfVal,
                 variables=None):
    if variables:
      raise ValueError(
          "Unexpected variables used in forward pass. "
          "This should not happen for first-order differentiation. "
          f"variables={variables}")

    out_cts_flat_polymorphic_shapes = tuple(str(out_aval.shape)  # Note: may be polynomials, not just DimVar
                                            for out_aval in out_avals)  # type: ignore
    vjp_polymorphic_shapes = [
        polymorphic_shapes_flat, out_cts_flat_polymorphic_shapes
    ]

    def fun_vjp_jax(args_flat_jax, out_cts_flat_jax):
      # One may think that we can get the pullback while we are converting
      # the main function in the first place. That is problematic, because the
      # pullback may contain captured tracers from the conversion of the
      # main function. Those tracers will confuse the conversion of the
      # pullback. So, we construct the vjp anew and we convert it separately.
      _, pullback_jax = jax.vjp(fun_flat_jax, *args_flat_jax)

      def fix_out_ct(out_ct_jax, out_ct_aval: core.ShapedArray):
        # If the primal function has outputs of integer or bool types, and if we are
        # under a tf.function context, then TF will pass None in _out_cts_flat
        # in place of these values. We should change these to float0 or
        # else JAX gets unhappy. See issue #6975.
        if out_ct_jax is not None:
          return out_ct_jax
        assert core.primal_dtype_to_tangent_dtype(out_ct_aval.dtype) == dtypes.float0, f"out_ct_jax={out_ct_jax}"
        # Note that out_ct_aval.shape contains dimension variable from the
        # primal function scope. It is Ok to use them here because we
        # use the same shape variables for the VJP function.
        return jnp.zeros(out_ct_aval.shape, dtype=_tf_np_dtype_for_float0)

      out_cts_fixed_flat = list(map(fix_out_ct, out_cts_flat_jax, out_avals))
      in_cts_flat_jax = pullback_jax(out_cts_fixed_flat)

      def fix_in_ct(in_ct_jax, arg_aval: core.ShapedArray):
        if jnp.issubdtype(arg_aval.dtype, jnp.inexact):
          return in_ct_jax
        else:
          assert in_ct_jax.dtype == dtypes.float0
          return jnp.zeros(arg_aval.shape, _tf_np_dtype_for_float0)

      in_cts_fixed_flat_jax = tuple(map(fix_in_ct, in_cts_flat_jax, args_avals_flat))
      return in_cts_fixed_flat_jax

    # TODO: enable higher-order gradients
    with tf.name_scope("jax2tf_vjp"):
      in_cts_flat = convert(
          fun_vjp_jax,
          with_gradient=False,
          polymorphic_shapes=vjp_polymorphic_shapes)(args_flat_tf, out_cts_flat_tf)
    return in_cts_flat

  return grad_fn_tf


def _interpret_fun_jax(
    fun_jax: Callable,
    args_tf: Sequence[TfVal],
    args_avals: Sequence[core.ShapedArray],
    extra_name_stack: Optional[str],
    fresh_constant_cache: bool = False
) -> Tuple[Tuple[TfVal, ...], Tuple[core.ShapedArray, ...]]:
  del extra_name_stack
  return _lower_native_and_run(fun_jax, args_avals, args_tf)


def _lower_native_and_run(fun_jax: Callable,
                          args_avals: Sequence[core.ShapedArray],
                          args_tf: Sequence[TfVal],
                          ) -> Tuple[Tuple[TfVal, ...], Tuple[core.ShapedArray, ...]]:
  """Lowers the function using native lowering and then invokes it.

  Work-in-progress.

  Uses JAX native lowering to MHLO, and then wraps the result in a
  XlaCallModule TF op. This op does not have backward-compatibility yet.

  Special care must be taken in presence of shape polymorphism.
  """
  # Look for shape polymorphism
  # For each arg, map axis idx to dimension variable name
  abstracted_axes: Sequence[Dict[int, str]] = []
  # For each dimension variable, encode how to compute its value from the
  # shape of the explicit arguments. E.g., "2.1" denotes args_tf[2].shape[1].
  # Note: We assume that lowering will introduce dim args in the order in which
  # dim variables are first seen when scanning the explicit arguments
  # in order and then scanning their shapes for dim variables.
  dim_args_spec: List[str] = []
  dim_vars_seen: Set[str] = set()
  for arg_idx, aval in enumerate(args_avals):
    one_abstract_axes = {}
    for axis_idx, d in enumerate(aval.shape):
      if not core.is_constant_dim(d):
        d_var = d.to_var()
        if d_var is None:
          raise ValueError(f"Only simple dimension variables supported: {aval.shape}")
        if not d_var in dim_vars_seen:
          dim_args_spec.append(f"{arg_idx}.{axis_idx}")
          dim_vars_seen.add(d_var)
        one_abstract_axes[axis_idx] = d_var
    abstracted_axes.append(one_abstract_axes)

  if any(abstracted_axes):
    if not config.jax_dynamic_shapes:
      raise ValueError(
          "Found shape polymorphism but --jax_dynamic_shapes is not set")
    abstracted_axes = tuple(abstracted_axes)
  else:
    abstracted_axes = None  # type: ignore

  arg_specs_jax = [
    jax.ShapeDtypeStruct(aval.shape, aval.dtype, named_shape=aval.named_shape)
    for aval in args_avals
  ]
  # TODO: specify the backend for experimental_native_lowering
  backend = jax.default_backend()
  if not hasattr(fun_jax, "lower") or abstracted_axes:
    # We support convert(pjit(f_jax, ...)) and convert(jit(f_jax)) but also
    # convert(f_jax), in which case a "jit" is implied. We also add a jit when
    # we need to pass the abstracted axes.
    fun_jax_lower = jax.jit(fun_jax, backend=backend,
                            keep_unused=True,  # TODO: allow dropping unused
                            abstracted_axes=abstracted_axes).lower
  else:
    fun_jax_lower = fun_jax.lower
  lowered = fun_jax_lower(*arg_specs_jax)._lowering
  mhlo_module = lowered.mhlo()
  if logging.vlog_is_on(3):
    mhlo_module_text = mlir.module_to_string(mhlo_module)
    logging.vlog(3, "XlaCallModule %s", mhlo_module_text)

  mhlo_serialized_module = mlir.module_to_bytecode(mhlo_module)
  # Figure out the result types and shapes
  if "global_out_avals" in lowered.compile_args:
    # This is currently the case for pjit
    out_avals = lowered.compile_args["global_out_avals"]
  else:
    out_avals = lowered.compile_args["out_avals"]
  if lowered.compile_args["host_callbacks"]:
    raise NotImplementedError("host_callbacks are not yet implemented for the jax2tf native lowering")

  # TODO(necula): handle d being InDBIdx
  out_shapes = tuple(
      tuple(d if type(d) is int else None
            for d in out_aval.shape)
      for out_aval in out_avals)

  def _out_type(jax_type):
    if jax_type == dtypes.float0:
      return dtypes.bool_
    return jax_type
  out_types = tuple(_out_type(out_aval.dtype) for out_aval in out_avals)

  # Apply the shardings on arguments and results for pjit. This is redundant
  # because the mhlo_module_text will already contain the shardings, but it
  # makes it easier for tools like the TPU inference converter to see the
  # sharding without digging into the `module` attribute of the `XlaCallModule`
  # op, in the same way as it is done for the legacy jax2tf conversion.
  if "in_shardings" in lowered.compile_args:
    args_tf = tuple(
      map(_shard_value, args_tf, args_avals, lowered.compile_args["in_shardings"]))
  res = tfxla.call_module(
      args_tf,
      module=mhlo_serialized_module,
      Tout=out_types,
      Sout=out_shapes,
      dim_args_spec=dim_args_spec)
  if "out_shardings" in lowered.compile_args:
    res = list(map(_shard_value, res, out_avals, lowered.compile_args["out_shardings"]))

  # Convert the results to the needed TF types
  def _convert_res(res_val, res_jax_type):
    conversion_dtype = _to_tf_dtype(res_jax_type)
    if conversion_dtype != res_jax_type:
      return tf.cast(res_val, conversion_dtype)
    else:
      return res_val

  res = tuple(
      _convert_res(res_val, out_aval.dtype)
      for res_val, out_aval in zip(res, out_avals))
  return res, out_avals


def _jax_physical_aval(aval: core.ShapedArray) -> core.ShapedArray:
  """Converts JAX avals from logical to physical, if relevant.

  JAX might have avals whose logical vs physical shape/dtype may
  differ, and only the physical view is expected to possibly
  relate to TF. TF impl rules should operate on the physical form.

  A JAX logical aval might even correspond, in principle, to several
  physical avals, but we don't support those here. Instead we assert
  there is only one and return it.
  """
  if core.is_opaque_dtype(aval.dtype):
    aval, = aval.dtype._rules.physical_avals(aval)
    return aval
  return aval

def _jax_physical_dtype(dtype):
  # assuming () is a fine stand-in shape
  return _jax_physical_aval(core.ShapedArray((), dtype)).dtype


# In the TF world, we represent float0 as zeros of this type.
_tf_np_dtype_for_float0 = np.int32

def _to_tf_dtype(jax_dtype):
  # Note that converting _to_tf_dtype and _to_jax_dtype are not inverses,
  # due to float0 and 64-bit behavior.
  try:
    jax_dtype = _jax_physical_dtype(jax_dtype)
  except TypeError:
    # `jax_dtype` isn't actually a valid jax dtype (e.g. it is
    # tf.float32), so there is no physical dtype anyway
    pass
  if jax_dtype == dtypes.float0:
    jax_dtype = _tf_np_dtype_for_float0
  return tf.dtypes.as_dtype(jax_dtype)


def _to_jax_dtype(tf_dtype):
  # Note that converting _to_tf_dtype and _to_jax_dtype are not inverses,
  # due to float0 and 64-bit behavior.
  return dtypes.canonicalize_dtype(tf_dtype.as_numpy_dtype)


def _maybe_decode_gda(gda_or_py_object: Any):
  """Convert GlobalDeviceArray into numpy object."""
  if isinstance(gda_or_py_object, GlobalDeviceArray):
    if jax.process_count() != 1:
      raise RuntimeError("GlobalDeviceArray does not support multi-process"
                         f" currently. Process num = {jax.process_count()}")
    return gda_or_py_object._value
  return gda_or_py_object


def _tfval_to_tensor_jax_dtype(val: TfVal,
                               jax_dtype: Optional[DType] = None,
                               memoize_constants=False) -> Tuple[TfVal, DType]:
  """Converts a scalar, ndarray, or tf.Tensor to a tf.Tensor with proper type.

  If `jax_dtype` is missing, uses JAX typing rules.
  See README.md for details regarding 64-bit values.

  Args:
    val: a scalar, ndarray, tf.Tensor, or tf.Variable
    jax_dtype: an optional dtype to use. If missing, uses JAX type inference
      rules for constants.
    memoize_constants: whether to memoize TF constants. We can't do this
      everywhere, we may be outside of a conversion scope.

  Returns:
    a tuple with a tf.Tensor with the type as needed by JAX, and the JAX type.
  """
  if isinstance(val, (tf.Tensor, tf.Variable)):
    jax_dtype = jax_dtype or _to_jax_dtype(val.dtype)  # Give JAX a chance to pick the type
    conversion_dtype = _to_tf_dtype(jax_dtype)
    if conversion_dtype != val.dtype:  # May need to cast for 64-bit values
      return tf.cast(val, conversion_dtype), jax_dtype
    else:
      return val, jax_dtype
  else:  # A constant
    jax_dtype = jax_dtype or xla.abstractify(val).dtype
    # TODO(document): We assume that the value of a constant does not
    # change through the scope of the function. But it may be an ndarray, ...
    # JAX has the same problem when generating HLO.
    const_key = (id(val), jax_dtype)
    # Since we use id(val) as a cache key, we have to make sure that we keep
    # the previous `val` alive. Otherwise, for an ndarray, it can get garbage
    # collected and reused for a different value, which would create correctness
    # issues. We keep the `val` alive by storing in the cache the pair
    # `(val, tf_val)`.
    do_memoize = (memoize_constants and np.shape(val) and _thread_local_state.constant_cache is not None)
    if do_memoize:
      _, tf_val = _thread_local_state.constant_cache.get(const_key, (None, None))
    else:
      tf_val = None
    if tf_val is None:
      conversion_dtype = _to_tf_dtype(jax_dtype)
      # The float0 type is not known to TF.
      if jax_dtype == dtypes.float0:
        val = np.zeros(np.shape(val), conversion_dtype.as_numpy_dtype)
      tf_val = tf.convert_to_tensor(
          _maybe_decode_gda(val), dtype=conversion_dtype)
      if do_memoize:
        _thread_local_state.constant_cache[const_key] = (val, tf_val)
    return tf_val, jax_dtype


def _shard_value(val: TfVal,
                 aval: core.ShapedArray,
                 sd: sharding.XLACompatibleSharding) -> TfVal:
  """Apply sharding to a TfVal."""
  if pxla._is_unspecified(sd):
    return val

  sharding_proto: xla_client.OpSharding = cast(
      xla_client.OpSharding, sd._to_xla_op_sharding(aval.ndim))
  # Do not apply XlaSharding for REPLICATED. This is an agreed convention, and
  # also improves usability under TF eager. See b/255511660.
  if pxla.is_op_sharding_replicated(sharding_proto):
    return val

  # To use xla_sharding.py, we must have a xla_data_pb2.OpSharding.
  xla_sharding_proto: xla_data_pb2.OpSharding = (
      xla_data_pb2.OpSharding(
          type=int(sharding_proto.type),
          tile_assignment_dimensions=sharding_proto.tile_assignment_dimensions,
          tile_assignment_devices=sharding_proto.tile_assignment_devices,
          replicate_on_last_tile_dim=sharding_proto.replicate_on_last_tile_dim,
          last_tile_dims=sharding_proto.last_tile_dims))
  if tf_context.executing_eagerly():
    raise ValueError(
        "A jit function with sharded (not replicated) arguments or results must be used under a `tf.function` context. "
        "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#support-for-partitioning for a discussion")

  return xla_sharding.Sharding(proto=xla_sharding_proto).apply_to_tensor(
      val, use_sharding_op=True)
