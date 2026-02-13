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
"""Provides JAX and TensorFlow interoperation APIs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
import contextlib
import math
import os
import threading
from typing import Any, Union
import warnings

from absl import logging
import numpy as np

import jax
from jax import tree_util
from jax import export

from jax._src import api
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import op_shardings
from jax._src import source_info_util
from jax._src import util
from jax._src.export import _export
from jax._src.export import shape_poly
from jax._src.lib import xla_client

import tensorflow as tf

# These don't have public equivalents.
# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla
from tensorflow.compiler.xla import xla_data_pb2
try:
  from tensorflow.python.compiler.xla.experimental import xla_sharding
except ModuleNotFoundError:
  # This can be removed when TF 2.10 support is no longer needed.
  from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
from tensorflow.python.eager import context as tf_context
# pylint: enable=g-direct-tensorflow-import

NameStack = source_info_util.NameStack
PolyShape = shape_poly.PolyShape  # TODO: deprecate
DType = Any

DisabledSafetyCheck = export.DisabledSafetyCheck


map = util.safe_map
zip = util.safe_zip

# A value suitable in a TF tracing context: tf.Tensor, tf.Variable,
# or Python scalar or numpy.ndarray. (A tf.EagerTensor is a tf.Tensor.)
TfVal = Any
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

class _DefaultNativeSerialization:
  pass
DEFAULT_NATIVE_SERIALIZATION = _DefaultNativeSerialization()


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
    # Keep track if we are inside a call_tf. In that context we disable the
    # safety check that we are not inside JAX transformations.
    self.inside_call_tf = False

    # Maps dimension variables to TF expressions, for non-native lowering
    self.shape_env: Sequence[tuple[str, TfVal]] = ()

    # A dict collecting all tf concrete_functions called by stablehlo.custom_call
    # This is used only by native serialization (unlike all the other
    # thread-local state).
    self.call_tf_concrete_function_list: list[Any] | None = None

_thread_local_state = _ThreadLocalState()

@contextlib.contextmanager
def inside_call_tf():
  # Set the inside_call_tf flag for a context.
  prev = _thread_local_state.inside_call_tf
  _thread_local_state.inside_call_tf = True
  try:
    yield
  finally:
    _thread_local_state.inside_call_tf = prev


def get_thread_local_state_call_tf_concrete_function_list() -> (
    list[Any] | None
):
  return _thread_local_state.call_tf_concrete_function_list


@partial(api_util.api_hook, tag="jax2tf_convert")
def convert(fun_jax: Callable,
            *,
            polymorphic_shapes: str | PolyShape | None | Sequence[str | PolyShape | None] = None,
            polymorphic_constraints: Sequence[str] = (),
            with_gradient: bool = True,
            enable_xla: bool = DEFAULT_NATIVE_SERIALIZATION,  # type: ignore
            native_serialization: bool | _DefaultNativeSerialization = DEFAULT_NATIVE_SERIALIZATION,  # type: ignore
            native_serialization_platforms: Sequence[str] | None = None,
            native_serialization_disabled_checks: Sequence[DisabledSafetyCheck] = (),
            ) -> Callable:
  """Allows calling a JAX function from a TensorFlow program.

  See
  [README](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md)
  for more details about usage and common problems.

  Args:
    fun_jax: target JAX function to be called. Its arguments and return value
      should be JAX arrays, or nested standard Python containers
      (tuple/list/dict) thereof (pytrees).
    polymorphic_shapes: Specifies input shapes to be treated polymorphically
      during lowering.

      .. warning:: The shape-polymorphic lowering is an experimental feature.
        It is meant to be sound, but it is known to reject some JAX programs
        that are shape polymorphic. The details of this feature can change.

      It should be `None` (all arguments are monomorphic), a single PolyShape
      or string (applies to all arguments), or a tuple/list of the same length
      as the function arguments. For each argument the shape specification
      should be `None` (monomorphic argument), or a Python object with the
      same pytree structure as the argument.
      See [how optional parameters are matched to
      arguments](https://docs.jax.dev/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).

      A shape specification for an array argument should be an object
      `PolyShape(dim0, dim1, ..., dimn)`
      where each `dim` is a dimension specification: a positive integer denoting
      a monomorphic dimension of the given size, or a string denoting a
      dimension variable assumed to range over non-zero dimension sizes, or
      the special placeholder string "_" denoting a monomorphic dimension
      whose size is given by the actual argument. As a shortcut, an Ellipsis
      suffix in the list of dimension specifications stands for a list of "_"
      placeholders.

      For convenience, a shape specification can also be given as a string
      representation, e.g.: "batch, ...", "batch, height, width, _", possibly
      with surrounding parentheses: "(batch, ...)".

      The lowering fails if it cannot ensure that the it would produce the same
      sequence of TF ops for any non-zero values of the dimension variables.

      polymorphic_shapes are only supported for positional arguments; shape
      polymorphism is not supported for keyword arguments.

      See [the README](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#shape-polymorphic-conversion)
      for more details.

    polymorphic_constraints: a sequence of constraints on symbolic dimension
      expressions, of the form `e1 >= e2` or `e1 <= e2`.
      See more details at https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#user-specified-symbolic-constraints.
    with_gradient: if set (default), add a tf.custom_gradient to the lowered
      function, by converting the ``jax.vjp(fun)``. This means that reverse-mode
      TensorFlow AD is supported for the output TensorFlow function, and the
      value of the gradient will be JAX-accurate.
    native_serialization_platforms: Specifies the platform(s)
      for which to lower the code. Must be a tuple of
      strings, including a subset of: 'cpu', 'cuda', 'rocm', 'tpu'.
      The default (`None``), specifies the JAX default
      backend on the machine where the lowering is done.
    native_serialization_disabled_checks: Disables the specified safety checks.
      See docstring of `DisabledSafetyCheck`.

  Returns:
    A version of `fun_jax` that expects TfVals as arguments (or
    tuple/lists/dicts thereof), and returns TfVals as outputs, and uses
    only TensorFlow ops and thus can be called from a TensorFlow program.
  """
  if native_serialization is not DEFAULT_NATIVE_SERIALIZATION:
    warnings.warn(
        "The `native_serialization` parameter is deprecated and "
        "will be removed in a future version of JAX.",
        DeprecationWarning, stacklevel=2)
  del native_serialization
  if enable_xla is not DEFAULT_NATIVE_SERIALIZATION:
    warnings.warn(
        "The `enable_xla` parameter is deprecated and "
        "will be removed in a future version of JAX.",
        DeprecationWarning, stacklevel=2)
  del enable_xla

  if native_serialization_platforms:
    if (not isinstance(native_serialization_platforms, (list, tuple)) or
        not all(p in ["cpu", "cuda", "rocm", "tpu"]
                for p in native_serialization_platforms)):
      raise ValueError(
          "native_serialization_platforms must be a sequence "
          "containing a subset of {'cpu', 'cuda', 'rocm', 'tpu'}. "
          f"Got: {native_serialization_platforms}")
    native_serialization_platforms = tuple(native_serialization_platforms)

  api.check_callable(fun_jax)

  def converted_fun_tf(*args_tf: TfVal, **kwargs_tf: TfVal) -> TfVal:

    # TODO: is there a better way to check if we are inside a transformation?
    if not core.trace_state_clean() and not _thread_local_state.inside_call_tf:
      # It is Ok to nest convert when we are inside a call_tf
      raise ValueError(
          "convert must be used outside all JAX transformations." +
          f"Trace state: {core.trace_ctx}")

    global _has_registered_tf_source_path
    if not _has_registered_tf_source_path:
      source_info_util.register_exclusion(os.path.dirname(tf.__file__))
      _has_registered_tf_source_path = True

    def jax_arg_spec_from_tf(a: TfVal) -> jax.ShapeDtypeStruct:
      # The shape and JAX dtype for a TF argument
      tf_arg_shape = np.shape(a)
      # Fix the shape for TF1
      tf_arg_shape = tuple(d.value
                           if isinstance(d, tf.compat.v1.Dimension) else d
                           for d in tf_arg_shape)
      _, a_jax_dtype = _tfval_to_tensor_jax_dtype(a)
      # We count on the fact that jax.ShapeDtypeStruct allows shapes that
      # contain None.
      return jax.ShapeDtypeStruct(tf_arg_shape, a_jax_dtype)

    args_jax_specs = tree_util.tree_map(jax_arg_spec_from_tf, args_tf)
    args_specs = export.symbolic_args_specs(
        args_jax_specs, polymorphic_shapes,
        constraints=polymorphic_constraints)
    # The polymorphic_shapes argument refers to positional arguments only.
    # We assume None for the kwargs.
    kwargs_jax_specs = tree_util.tree_map(jax_arg_spec_from_tf, kwargs_tf)
    kwargs_specs = export.symbolic_args_specs(
        kwargs_jax_specs, None)
    combined_args_tf = (args_tf, kwargs_tf)
    args_flat_tf: Sequence[TfVal]
    args_flat_tf, args_kwargs_tree = tree_util.tree_flatten(combined_args_tf)

    args_flat_tf = tuple(
        map(preprocess_arg_tf, range(len(args_flat_tf)), args_flat_tf))

    impl = NativeSerializationImpl(
        fun_jax,
        args_specs=args_specs, kwargs_specs=kwargs_specs,
        native_serialization_platforms=native_serialization_platforms,
        native_serialization_disabled_checks=native_serialization_disabled_checks)
    try:
      impl.before_conversion()

      outs_tree: tree_util.PyTreeDef = None  # type: ignore
      if with_gradient:
        @tf.custom_gradient
        def converted_fun_flat_with_custom_gradient_tf(*args_flat_tf: TfVal) -> TfVal:
          nonlocal outs_tree
          outs_tf, outs_avals, outs_tree = impl.run_fun_tf(args_flat_tf)
          return (tuple(outs_tf),
                  _make_custom_gradient_fn_tf(
                      fun_jax,
                      impl=impl,
                      with_gradient=with_gradient,
                      args_specs=args_specs, kwargs_specs=kwargs_specs,
                      args_tf=args_flat_tf,
                      outs_avals=outs_avals,
                      outs_tf=outs_tf))

        outs_flat_tf = converted_fun_flat_with_custom_gradient_tf(*args_flat_tf)
      else:
        outs_tf, _, outs_tree = impl.run_fun_tf(args_flat_tf)
        message = ("The jax2tf-converted function does not support gradients. "
                   "Use `with_gradient` parameter to enable gradients")
        # We use PreventGradient, which is propagated through a SavedModel.
        outs_flat_tf = [
            tf.raw_ops.PreventGradient(input=o, message=message)
            for o in outs_tf
        ]
    finally:
      impl.after_conversion()

    outs_flat_tf = [tf.identity(x, "jax2tf_out") for x in outs_flat_tf]
    out_tf = tree_util.tree_unflatten(outs_tree, outs_flat_tf)
    return out_tf

  return converted_fun_tf


class NativeSerializationImpl:
  def __init__(self, fun_jax, *,
               args_specs, kwargs_specs,
               native_serialization_platforms: Sequence[str] | None,
               native_serialization_disabled_checks: Sequence[DisabledSafetyCheck]):
    self.convert_kwargs = dict(native_serialization_platforms=native_serialization_platforms,
                               native_serialization_disabled_checks=native_serialization_disabled_checks)
    if hasattr(fun_jax, "trace"):
      # If we have a pjit or pmap already we do not wrap with another, and we
      # allow shardings.
      fun_jit = fun_jax
    else:
      # We support convert(pjit(f_jax)) and convert(jit(f_jax)) but also
      # convert(f_jax), in which case a "jit" is implied. In that case we raise
      # an error if the lowered function contains non-replicated sharding annotations.
      fun_jit = jax.jit(fun_jax)
    self.fun_jax = fun_jit
    self.args_specs = args_specs
    self.kwargs_specs = kwargs_specs
    self.native_serialization_disabled_checks = native_serialization_disabled_checks
    self.native_serialization_platforms = native_serialization_platforms

  def before_conversion(self):
    _prev_func_list = _thread_local_state.call_tf_concrete_function_list
    _thread_local_state.call_tf_concrete_function_list = []

    def _restore_context():
      _thread_local_state.call_tf_concrete_function_list = _prev_func_list

    self._restore_context = _restore_context
    _exported_device_assignment = [None]
    self.exported = _export._export_internal(
        self.fun_jax,
        platforms=self.native_serialization_platforms,
        disabled_checks=self.native_serialization_disabled_checks,
        _device_assignment_for_internal_jax2tf_use_only=_exported_device_assignment,
    )(*self.args_specs, **self.kwargs_specs)
    assert(_exported_device_assignment[0] is not None)
    self.device_assignment = _exported_device_assignment[0]

  def after_conversion(self):
    self._restore_context()

  def run_fun_tf(self,
                 args_flat_tf: Sequence[TfVal]
                 ) -> tuple[Sequence[TfVal], Sequence[core.ShapedArray], tree_util.PyTreeDef]:
    results = _run_exported_as_tf(args_flat_tf, self.exported)
    return results, tuple(self.exported.out_avals), self.exported.out_tree

  def get_vjp_fun(self) -> tuple[Callable,
                                 Sequence[core.AbstractValue]]:
    return _export._get_vjp_fun(self.fun_jax,
                                in_tree=self.exported.in_tree,
                                in_avals=self.exported.in_avals,
                                has_named_shardings=self.exported._has_named_shardings,
                                in_named_shardings=self.exported._in_named_shardings,
                                out_named_shardings=self.exported._out_named_shardings,
                                in_shardings_hlo=self.exported.in_shardings_hlo,
                                out_avals=self.exported.out_avals,
                                out_shardings_hlo=self.exported.out_shardings_hlo,
                                device_assignment=self.device_assignment,
                                apply_jit=True)


def dtype_of_val(val: TfVal) -> DType:
  """Computes the TensorFlow dtype using JAX's typing rules.

  If the value is a tf.Tensor, it starts with its dtype. If the value is a
  constant it uses JAX to infer its dtype. The resulting dtype follows the
  JAX type inference rules, and depends on the value of the
  JAX_ENABLE_X64 flag.

  See README.md for how 64-bit values are treated.
  """
  tval, _ = _tfval_to_tensor_jax_dtype(val)
  return tval.dtype

@partial(api_util.api_hook, tag="jax2tf_eval_polymorphic_shapes")
def eval_polymorphic_shape(fun_jax: Callable,
               *,
               polymorphic_shapes=None) -> Callable:
  """Evaluates the output shape in presence of shape polymorphism.

  This is done without lowering or executing the function, same as for
  `jax.eval_shape`.

  Args:
    fun_jax: target JAX function to be called. Its arguments and return value
      should be JAX arrays, or nested standard Python containers
      (tuple/list/dict) thereof (pytrees).
    polymorphic_shapes: Specifies input shapes to be treated polymorphically
      during shape evaluation. See discussion for `jax2tf.convert`.

      .. warning:: The shape-polymorphic lowering is an experimental feature.

  Returns: a function that takes `jax.ShapeDtypeStruct`s (or any values
    with `.shape` and `.dtype` attributes) corresponding to the inputs for
    `fun_jax`, and returns a tuple with:

      * the jax.ShapeDtypeStruct corresponding to the result, as for
       `jax.eval_shape`. The shape may contain symbolic dimension expressions.
      * the value that can be passed to `polymorphic_shapes` for a subsequent
        call to `jax2tf.eval_polymorphic_shape`, or `jax2tf.convert`.

  For example:

  >>> import jax
  >>> from jax.experimental import jax2tf
  >>> from jax import numpy as jnp
  >>>
  >>> f = lambda A, x: jnp.sin(jnp.dot(A, x))
  >>> A = jax.ShapeDtypeStruct((2000, 3000), jnp.float32)
  >>> x = jax.ShapeDtypeStruct((3000, 1000), jnp.float32)
  >>> out_spec, out_poly_shape = jax2tf.eval_polymorphic_shape(f, polymorphic_shapes=["a, b", "b, c"])(A, x)
  >>> print(out_spec.shape)
  ("a", "c")
  >>> print(out_poly_shape)
  (a, c)
  >>> res_spec, res_poly_shape = jax2tf.eval_polymorphic_shape(lambda x: x.T, polymorphic_shapes=[out_poly_shape])(out_spec)
  >>> print(res_poly_shape)
  (c, a)
  """
  def do_eval_polymorphic_shape(*args_specs) -> Any:
    args_poly_specs = export.symbolic_args_specs(
        args_specs, polymorphic_shapes)
    res_poly_spec = jax.eval_shape(fun_jax, *args_poly_specs)
    # TODO(necula): For now we export the polymorphic shapes using `str`.
    res_polymorphic_shape = tree_util.tree_map(lambda r: str(r.shape), res_poly_spec)
    return res_poly_spec, res_polymorphic_shape

  return do_eval_polymorphic_shape


# Internals

def preprocess_arg_tf(arg_idx: int,
                      arg_tf: TfVal) -> TfVal:
  """Pre-processes the TF args.

  Returns: a tuple with the pre-processed TF arg, the TF shape, and the
      JAX dtype.
  """
  if not _is_tfval(arg_tf):
    msg = (f"Argument {arg_tf} of type {type(arg_tf)} of jax2tf.convert(f) should "
           "be NumPy array, scalar, tf.Variable, or tf.Tensor")
    raise TypeError(msg)

  # May cast the args_flat to JAX types, using JAX's interpretation
  # of types of constants.
  arg_tf, _ = _tfval_to_tensor_jax_dtype(arg_tf)
  # Name input tensors; do this after we have cast the arguments
  arg_tf = tf.identity(arg_tf, f"jax2tf_arg_{arg_idx}")
  return arg_tf


def _make_custom_gradient_fn_tf(fun_jax,
                                *,
                                impl: NativeSerializationImpl,
                                with_gradient: bool,
                                args_specs, kwargs_specs,
                                args_tf: Sequence[TfVal],
                                outs_avals: Sequence[core.ShapedArray],
                                outs_tf: Sequence[TfVal]):
  """Prepares the TF function to be used with tf.custom_gradient.

  Args:
    impl: the serialization implementation details
    with_gradient: whether to include a tf.custom_gradient
    args_specs, kwargs_specs: the jax.ShapeDtypeArrays for the args and kwargs
    args_tf: the flattened TF arguments of the primal function
    outs_avals: the flattened output JAX abstract values of the primal function
    outs_tf: the flattened TF outputs of the primal function
  """
  def grad_fn_tf(*out_cts_flat_tf: TfVal,
                 variables=None):
    if variables:
      raise ValueError(
          "Unexpected variables used in forward pass. "
          "This should not happen for first-order differentiation. "
          f"{variables=}")

    # TODO: enable higher-order gradients
    with tf.name_scope("jax2tf_vjp"):
      def fix_out_ct(out_ct_tf, out_ct_aval: core.ShapedArray, out_tf: TfVal):
        # If the primal function has outputs of integer or bool types, and if we are
        # under a tf.function context, then TF will pass None in _out_cts_flat
        # in place of these values. We should change these to float0 or
        # else JAX gets unhappy. See issue #6975.
        if out_ct_tf is not None:
          return out_ct_tf
        assert core.primal_dtype_to_tangent_dtype(out_ct_aval.dtype) == dtypes.float0, f"{out_ct_tf=}"
        # Note that out_ct_aval.shape contains dimension variable from the
        # primal function scope. We use tf.zeros_like to make a 0 of the right shape.
        return tf.zeros_like(out_tf, dtype=_tf_np_dtype_for_float0)

      out_cts_fixed_flat_tf = tuple(map(fix_out_ct, out_cts_flat_tf, outs_avals, outs_tf))
      vjp_args_flat_tf = tuple(args_tf) + out_cts_fixed_flat_tf

      fun_vjp_jax, vjp_in_avals = impl.get_vjp_fun()

      vjp_polymorphic_shapes = tuple(
        str(a.shape)  # Note: may be _DimExpr, not just DimVar
        for a in vjp_in_avals)
      in_cts_flat = convert(
        fun_vjp_jax,
        with_gradient=with_gradient,
        polymorphic_shapes=vjp_polymorphic_shapes,
        **impl.convert_kwargs)(*vjp_args_flat_tf)

    # We do not need to fix the in_cts because the TF gradient machinery
    # will adjust the unconnected gradients and those for integer types.
    return in_cts_flat

  return grad_fn_tf


def _run_exported_as_tf(args_flat_tf: Sequence[TfVal],
                        exported: export.Exported,
                        ) -> Sequence[TfVal]:
  """Runs the `exported` as an XlaCallModule TF op.

  Returns: the flattened tuple of results.
  """
  args_avals = exported.in_avals

  # TF values may be integer types for float0
  def _convert_value(val, aval):
    # Check the shape
    assert all(d_aval == d_val
               for d_aval, d_val in zip(aval.shape, val.shape)
               if core.is_constant_dim(d_aval)), (aval, val)
    conversion_dtype = _to_tf_dtype(aval.dtype)
    if conversion_dtype != aval.dtype:
      return tf.cast(val, conversion_dtype)
    else:
      return val

  args_flat_tf = tuple(map(_convert_value, args_flat_tf, args_avals))

  out_shapes_tf = tuple(
      tuple(d if core.is_constant_dim(d) else None
            for d in out_aval.shape)
      for out_aval in exported.out_avals)

  out_types = tuple(_to_tf_dtype(out_aval.dtype) for out_aval in exported.out_avals)

  kept_args_flat_tf = [atf for i, atf in enumerate(args_flat_tf) if i in exported.module_kept_var_idx]

  version = exported.calling_convention_version

  try:
    get_max_supported_version = tfxla.call_module_maximum_supported_version
  except AttributeError:
    get_max_supported_version = None

  if get_max_supported_version:
    max_supported_version = get_max_supported_version()
  else:
    max_supported_version = 6

  if version > max_supported_version:
    raise NotImplementedError(
      "XlaCallModule from your TensorFlow installation supports up to "
      f"serialization version {max_supported_version} but the serialized "
      f"module needs version {version}. "
      "You should upgrade TensorFlow, e.g., to tf_nightly."
    )

  call_module_attrs = dict(
      version=version,
      Tout=out_types,
      Sout=out_shapes_tf,
      function_list=[
          concrete_fn.function_def.signature.name
          for concrete_fn in _thread_local_state.call_tf_concrete_function_list
      ] if _thread_local_state.call_tf_concrete_function_list is not None else [],
      # We always set has_token_input_output because it requires real tokens
      # for versions less than 9 and is not used starting with version 9.
      has_token_input_output=False
  )

  call_module_attrs["platforms"] = tuple(p.upper() for p in exported.platforms)
  if version >= 6:
    call_module_attrs["disabled_checks"] = tuple(
        str(dc)
        for dc in exported.disabled_safety_checks)
  else:
    if version >= 3:
      if DisabledSafetyCheck.platform() in exported.disabled_safety_checks:
        call_module_attrs["platforms"] = ()  # No platform checking

  if version >= 10:
    call_module_attrs["use_shardy_partitioner"] = (
        config.use_shardy_partitioner.value
    )

  if logging.vlog_is_on(3):
    # We already logged the MLIR module when we exported it.
    logging.vlog(3, "XlaCallModule %s", str(call_module_attrs))

  call_module_attrs["module"] = exported.mlir_module_serialized

  # Apply the shardings on arguments and results for pjit. This is redundant
  # because the mlir_module_text will already contain the shardings, but it
  # makes it easier for tools like the TPU inference converter to see the
  # sharding without digging into the `module` attribute of the `XlaCallModule`
  # op, in the same way as it is done for the legacy jax2tf conversion.
  # Do not apply XlaSharding for REPLICATED, on inputs and outputs.
  # This is an agreed convention, and also improves usability under TF eager.
  # See b/255511660.
  kept_in_shardings = []
  for i in exported.module_kept_var_idx:
    kept_in_shardings.append(exported.in_shardings_hlo[i])
  args_flat_tf = tuple(
    map(partial(_shard_value,
                skip_replicated_sharding=tf.executing_eagerly()),
        kept_args_flat_tf, kept_in_shardings))
  res = tfxla.call_module(args_flat_tf, **call_module_attrs)
  # TODO(b/278940799): Replace the TF v1 API with public TF2 API.
  # Add the custom call tf.function into the default graph, so those functions
  # will be available during tf.SavedModel.save.
  if _thread_local_state.call_tf_concrete_function_list is not None:
    for concrete_fn in _thread_local_state.call_tf_concrete_function_list:
      tf.compat.v1.get_default_graph()._add_function_recursive(
          concrete_fn._inference_function
      )

  res = list(map(partial(_shard_value,
                         skip_replicated_sharding=tf.executing_eagerly()),
                 res, exported.out_shardings_hlo))
  res = tuple(map(_convert_value, res, exported.out_avals))
  return res


def _jax_physical_aval(aval: core.ShapedArray) -> core.ShapedArray:
  """Converts JAX avals from logical to physical, if relevant.

  JAX might have avals whose logical vs physical shape/dtype may
  differ, and only the physical view is expected to possibly
  relate to TF. TF impl rules should operate on the physical form.

  A JAX logical aval might even correspond, in principle, to several
  physical avals, but we don't support those here. Instead we assert
  there is only one and return it.
  """
  physical_aval = core.physical_aval(aval)
  assert (len(physical_aval.shape) >= len(aval.shape) and
          physical_aval.shape[:len(aval.shape)] == aval.shape), (physical_aval, aval)
  return physical_aval

def _jax_physical_dtype(dtype):
  # assuming () is a fine stand-in shape
  return _jax_physical_aval(core.ShapedArray((), dtype)).dtype


def _aval_to_tf_shape(aval: core.ShapedArray) -> tuple[int | None, ...]:

  """Generate a TF shape, possibly containing None for polymorphic dimensions."""
  aval = _jax_physical_aval(aval)
  return tuple(map(lambda d: None if export.is_symbolic_dim(d) else d,
                   aval.shape))

# In the TF world, we represent float0 as zeros of this type.
# We pick bool because this is what JAX uses when it lowers float0 to HLO.
_tf_np_dtype_for_float0 = np.bool_

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
  dt = dtypes.canonicalize_dtype(tf_dtype.as_numpy_dtype)
  if dt not in dtypes._jax_dtype_set:
    raise TypeError(f"dtype {dt} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")
  return dt


def _tfval_to_tensor_jax_dtype(val: TfVal,
                               jax_dtype: DType | None = None,
                               memoize_constants=False) -> tuple[TfVal, DType]:
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
    jax_dtype = jax_dtype or core.abstractify(val).dtype
    # TODO(document): We assume that the value of a constant does not
    # change through the scope of the function. But it may be an ndarray, ...
    # JAX has the same problem when generating HLO.
    const_key = (id(val), jax_dtype)
    # Since we use id(val) as a cache key, we have to make sure that we keep
    # the previous `val` alive. Otherwise, for a ndarray, it can get garbage
    # collected and reused for a different value, which would create correctness
    # issues. We keep the `val` alive by storing in the cache the pair
    # `(val, tf_val)`.
    # Only memoize non-scalars. JAX will lift all non-scalar constants as
    # Jaxpr consts, to the top level of the Jaxpr. This ensures that we see them
    # early, when entering the Jaxpr, so we create the tf.const early and its
    # scope is the entire Jaxpr.
    do_memoize = (memoize_constants and np.size(val) > 1 and _thread_local_state.constant_cache is not None)
    if do_memoize:
      _, tf_val = _thread_local_state.constant_cache.get(const_key, (None, None))
    else:
      tf_val = None
    if tf_val is None:
      conversion_dtype = _to_tf_dtype(jax_dtype)
      # The float0 type is not known to TF.
      if jax_dtype == dtypes.float0:
        val = np.zeros(np.shape(val), conversion_dtype.as_numpy_dtype)
      if hasattr(val, 'dtype') and dtypes.issubdtype(val.dtype, dtypes.extended):
        val = val.dtype._rules.physical_const(val)
      tf_val = tf.convert_to_tensor(val, dtype=conversion_dtype)
      if do_memoize:
        _thread_local_state.constant_cache[const_key] = (val, tf_val)
    return tf_val, jax_dtype


PartitionsOrReplicated = Union[tuple[int, ...], None]

def split_to_logical_devices(tensor: TfVal,
                             partition_dimensions: PartitionsOrReplicated):
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
  # TODO: this is only for sharded_jit. Either remove, or implement in terms
  # of _shard_values.
  if partition_dimensions is None:
    return xla_sharding.replicate(tensor, use_sharding_op=True)
  num_partition_splits = math.prod(partition_dimensions)
  tile_assignment = np.arange(num_partition_splits).reshape(
      partition_dimensions)
  return xla_sharding.tile(tensor, tile_assignment, use_sharding_op=True)


def _shard_value(val: TfVal,
                 sd: xla_client.HloSharding | None, *,
                 skip_replicated_sharding: bool) -> TfVal:
  """Apply sharding to a TfVal."""
  if sd is None:
    return val

  sharding_proto = sd.to_proto()
  if (skip_replicated_sharding and
      op_shardings.is_hlo_sharding_replicated(sd)):
    return val

  # Tensorflow heavily relies on tile_assignment_devices proto fields specific
  # to V1 sharding format, falling back to this format.
  if (
      not sharding_proto.tile_assignment_devices
      and sharding_proto.iota_reshape_dims
  ):
    tad = list(
        np.arange(math.prod(sharding_proto.tile_assignment_dimensions))
        .reshape(sharding_proto.iota_reshape_dims)
        .transpose(sharding_proto.iota_transpose_perm)
        .flat
    )
  else:
    tad = sharding_proto.tile_assignment_devices  # type: ignore

  # To use xla_sharding.py, we must have a xla_data_pb2.OpSharding.
  xla_sharding_v1_proto: xla_data_pb2.OpSharding = xla_data_pb2.OpSharding(
      type=int(sharding_proto.type),
      tile_assignment_dimensions=sharding_proto.tile_assignment_dimensions,
      tile_assignment_devices=tad,
      replicate_on_last_tile_dim=sharding_proto.replicate_on_last_tile_dim,
      last_tile_dims=sharding_proto.last_tile_dims,
  )
  # Shardy requires V2 sharding format.
  if config.use_shardy_partitioner.value:
    xla_sharding_v2_proto: xla_data_pb2.OpSharding = xla_data_pb2.OpSharding(
        type=int(sharding_proto.type),
        tile_assignment_dimensions=sharding_proto.tile_assignment_dimensions,
        tile_assignment_devices=sharding_proto.tile_assignment_devices,
        iota_reshape_dims=sharding_proto.iota_reshape_dims,
        iota_transpose_perm=sharding_proto.iota_transpose_perm,
        replicate_on_last_tile_dim=sharding_proto.replicate_on_last_tile_dim,
        last_tile_dims=sharding_proto.last_tile_dims,
    )
  else:
    xla_sharding_v2_proto = None
  if tf_context.executing_eagerly():
    raise ValueError(
        "A jit function with sharded arguments or results must be used under a `tf.function` context. "
        "See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#support-for-partitioning for a discussion")

  tf_version = tuple(int(v) for v in tf.__version__.split(".")[:2])
  # apply_to_tensor comes from a tensorflow package, check the tensorflow
  # version to make sure that it has the sharding_v2_proto parameter.
  if tf_version < (2, 20):
    return xla_sharding.Sharding(proto=xla_sharding_v1_proto).apply_to_tensor(
        val, use_sharding_op=True
    )
  return xla_sharding.Sharding(proto=xla_sharding_v1_proto).apply_to_tensor(
      val, use_sharding_op=True, sharding_v2_proto=xla_sharding_v2_proto
  )


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

  jax.tree_util.register_pytree_node(tuple_wrapper, lambda xs:
                                     (tuple(xs), None), lambda _, xs: tuple(xs))

  jax.tree_util.register_pytree_node(list_wrapper, lambda xs: (tuple(xs), None),
                                     lambda _, xs: list(xs))

  jax.tree_util.register_pytree_node(
      dict_wrapper,
      lambda s: (tuple(s.values()), tuple(s.keys())),
      lambda k, xs: dict_wrapper(zip(k, xs)))


_register_checkpoint_pytrees()
