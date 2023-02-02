# Copyright 2021 The JAX Authors.
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
"""Allows JAX to call TensorFlow functions with support for autodiff.

**Experimental: please give feedback, and expect changes.**

This module introduces the function :func:`call_tf` that allows JAX to call
TensorFlow functions.

For examples and details, see
https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax.

"""
import enum
import functools
from typing import Any, Callable, Optional, Sequence, Tuple

from absl import logging

import jax
from jax import dlpack
from jax import dtypes
from jax import numpy as jnp
from jax import tree_util
from jax._src import core
from jax._src import ad_checkpoint
from jax._src import custom_derivatives
from jax._src import ad_util
from jax._src import effects
from jax._src import util
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo
from jax._src.lib import xla_client
from jax.experimental.jax2tf import jax2tf as jax2tf_internal
from jax.interpreters import mlir
from jax.interpreters import xla

import numpy as np
import tensorflow as tf  # type: ignore[import]

map = util.safe_map
zip = util.safe_zip

TfConcreteFunction = Any

# The platforms for which to use DLPack to avoid copying (only works on GPU
# and CPU at the moment, and only for DeviceArray). For CPU we don't need
# DLPack, if we are careful.
_DLPACK_PLATFORMS = ("gpu",)

def call_tf(callable_tf: Callable, has_side_effects=True) -> Callable:
  """Calls a TensorFlow function from JAX, with support for reverse autodiff.

  The ``callable_tf`` will be called with TensorFlow-compatible arguments (
  numpy.ndarray, ``tf.Tensor`` or ``tf.Variable``) or pytrees thereof. The
  function must return the same type of results.

  If ``call_tf`` appears in a JAX staging context (:func:`jax.jit`,
  or :func:`jax.pmap`, or :func:`jax.xmap`, or a control-flow primitive) then
  ``callable_tf`` will be compiled with ``tf.function(callable_tf, jit_compile=True)``
  and the resulting XLA computation will be embedded in JAX's XLA computation.

  If ``call_tf`` appears outside a JAX staging context, it will be called inline
  using TensorFlow eager mode.

  The ``call_tf`` supports JAX's reverse-mode autodiff, in which case the
  ``callable_tf`` will be differentiated using ``tf.GradientTape``. This means
  that the gradient will be TensorFlow-accurate, e.g., will respect the
  custom gradients that may be defined for the code in ``callable_tf``.

  For an example and more details see the
  `README <https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax>`_.

  Args:
    callable_tf: a TensorFlow Callable that can take a pytree of TensorFlow
      arguments.
    has_side_effects: if True then it ensures that instances of this primitive
      are not removed or replicated by JAX optimizations such as dead-code
      elimination.

  Returns: a JAX callable that can be invoked with JAX pytree arguments, in
    op-by-op mode or in a staged context. This callable can be used with
    JAX's reverse-mode autodiff (:func:`jax.grad`).
  """

  @jax.custom_vjp
  def make_call(*args_jax):
    """We wrap it all in `make_call` so that we can attach custom VJP."""

    args_flat_jax, args_treedef = tree_util.tree_flatten(args_jax)
    # Canonicalize the arguments; e.g., makes them x32 if JAX is in 32-bit mode
    def canonical_arg(v):
      v = v if getattr(v, "dtype", None) else np.asarray(v)
      dtype = dtypes.canonicalize_dtype(v.dtype)
      if dtype != v.dtype:
        v = v.astype(dtype)
      return v

    args_flat_jax = tuple(map(canonical_arg, args_flat_jax))
    def make_tensorspec(a_jax):
      a_tf_dtype = jax2tf_internal._to_tf_dtype(a_jax.dtype)
      a_tf_shape = [
          d if core.is_constant_dim(d) else None for d in a_jax.shape
      ]
      return tf.TensorSpec(a_tf_shape, a_tf_dtype)
    args_flat_sig_tf = tuple(map(make_tensorspec, args_flat_jax))

    def check_tf_result(r_tf):
      # Check that the TF function returns values of expected types. This
      # improves error reporting, preventing hard-to-diagnose errors downstream
      try:
        jax2tf_internal._tfval_to_tensor_jax_dtype(r_tf)
      except Exception as e:
        msg = ("The called TF function returns a result that is not "
               f"convertible to JAX: {r_tf}.")
        raise ValueError(msg) from e

    res_treedef = None  # We'll store here the result treedef
    res_tf_flat = None  # For error reporting
    # The function below will be called at least once, either in eager
    # or in graph mode.
    def callable_flat_tf(*args_tf_flat: TfVal) -> Sequence[TfVal]:
      args_tf = args_treedef.unflatten(args_tf_flat)
      res_tf = callable_tf(*args_tf)
      nonlocal res_treedef, res_tf_flat
      res_tf_flat, res_treedef_now = tree_util.tree_flatten(res_tf)
      for r_tf in res_tf_flat:
        check_tf_result(r_tf)
      assert res_treedef is None or res_treedef == res_treedef_now, f"Subsequent calls had different results. Previous {res_treedef} and now {res_treedef_now}"
      res_treedef = res_treedef_now
      return res_tf_flat

    # Prepare a tf.function ahead of time, to cache the concrete functions. This
    # won't be used in op-by-op execution mode.
    function_flat_tf = tf.function(callable_flat_tf, autograph=False, jit_compile=True)

    input_shapes_tf = [s.shape for s in args_flat_sig_tf]
    output_shapes_tf = _get_concrete_function_tf(
        function_flat_tf, args_flat_sig_tf
    ).output_shapes

    if not all(s.is_fully_defined() for s in input_shapes_tf) and not all(
        s.is_fully_defined() for s in output_shapes_tf
    ):
      for a_jax, a_tf_shape in zip(args_flat_jax, input_shapes_tf):
        if not a_tf_shape.is_fully_defined():
          msg = (
              "call_tf cannot be applied to shape-polymorphic arguments unless"
              " all the output shapes are static. Found argument shape:"
              f" {a_jax.shape}. See"
              " https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf"
              " for a discussion."
          )
          raise ValueError(msg)

    res_jax_flat = call_tf_p.bind(
        *args_flat_jax,
        # Carry the actual function such that op-by-op call can call in TF eager mode.
        callable_flat_tf=callable_flat_tf,
        function_flat_tf=function_flat_tf,
        args_flat_sig_tf=args_flat_sig_tf,
        has_side_effects=has_side_effects)

    assert res_treedef is not None
    # Sometimes, in compiled mode, we get a different number of results than we
    # got when tracing the TF function (and building the res_treedef). This
    # can happen, e.g., when returning tf.TensorArray, which appears as one
    # leaf when tracing but after compilation we get a tuple. See
    # call_tf_test.test_error_bad_result_tensorarray.
    if res_treedef.num_leaves != len(res_jax_flat):
      # It is not clear if this error can happen once we have check_tf_result
      # in callable_flat_tf, but we keep it for safety.
      msg = (f"Incorrect number of results ({len(res_jax_flat)}) from the "
             "called TF function after compilation. "
             f"Expected {res_treedef.num_leaves} leaves based on observed "
             f"results during tracing: {res_tf_flat}.")
      raise ValueError(msg)
    return res_treedef.unflatten(res_jax_flat)

  # Define the fwd and bwd custom_vjp functions
  def make_call_vjp_fwd(*args_jax):
    # Return the primal arguments as the residual
    return make_call(*args_jax), args_jax

  def make_call_vjp_bwd(residual_jax, ct_res_jax):
    args_jax = residual_jax  # residual is the primal argument

    def tf_vjp_fun(args_tf, ct_res_tf):
      """Invoke TF gradient."""

      # TF does not like us to watch non-float vars
      def replace_non_float(arg_tf):
        if arg_tf.dtype.is_floating or arg_tf.dtype.is_complex:
          return arg_tf
        else:
          # When watched, this will be ignored. When use in results it will
          # result in a floating 0. gradient, which JAX will ignore (and
          # replace it with a float0)
          return tf.zeros((), dtype=tf.float32)

      watched_args_tf = tf.nest.map_structure(replace_non_float, args_tf)
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(watched_args_tf)
        res = callable_tf(*args_tf)

      tf.nest.assert_same_structure(res, ct_res_tf)
      dres_darg = tape.gradient(
          tf.nest.map_structure(replace_non_float, res),
          sources=watched_args_tf,
          output_gradients=ct_res_tf,
          unconnected_gradients=tf.UnconnectedGradients.ZERO)

      dres_darg = tree_util.tree_map(
          lambda x: x if x is None else tf.convert_to_tensor(x),
          dres_darg,
      )
      tf.nest.assert_same_structure(dres_darg, args_tf)
      return dres_darg

    # Use call_tf to call the VJP function
    ct_args_jax = call_tf(tf_vjp_fun)(args_jax, ct_res_jax)
    # We must make the float0s that JAX expects
    def fix_float0(arg_jax, ct_arg_jax):
      arg_dtype = dtypes.result_type(arg_jax)  # May be scalar
      ct_arg_dtype = core.primal_dtype_to_tangent_dtype(arg_dtype)
      if ct_arg_dtype != ct_arg_jax.dtype:
        return ad_util.zeros_like_aval(core.ShapedArray(np.shape(arg_jax),
                                                        ct_arg_dtype))
      return ct_arg_jax

    ct_args_jax_fixed = tree_util.tree_map(fix_float0, args_jax, ct_args_jax)
    return ct_args_jax_fixed

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return util.wraps(callable_tf)(make_call)


call_tf_p = core.Primitive("call_tf")
call_tf_p.multiple_results = True

# The impl will be used in op-by-op mode and calls callable_tf in TF eager mode.
def _call_tf_impl(*args_jax_flat, callable_flat_tf, **_):
  # On GPU we use dlpack to avoid copies of data to the host.
  def _arg_jax_to_tf(arg_jax):
    if (isinstance(arg_jax, jax.Array) and
        arg_jax.device().platform in _DLPACK_PLATFORMS and
        arg_jax.dtype in dlpack.SUPPORTED_DTYPES):
      arg_dlpack = jax.dlpack.to_dlpack(arg_jax, take_ownership=False)
      return tf.experimental.dlpack.from_dlpack(arg_dlpack)
    # The following avoids copies to the host on CPU, always for DeviceArray
    # and even for ndarray if they are sufficiently aligned.
    # TODO(necula): on TPU this copies to the host!
    return tf.constant(np.asarray(arg_jax))

  args_tf_flat = tuple(map(_arg_jax_to_tf, args_jax_flat))
  with jax2tf_internal.inside_call_tf():
    # Call in TF eager mode
    res_tf_flat = callable_flat_tf(*args_tf_flat)

  def _res_tf_to_jax(res_tf: TfVal):
    res_tf, _ = jax2tf_internal._tfval_to_tensor_jax_dtype(res_tf)
    if isinstance(res_tf, tf.Tensor) and res_tf.dtype in dlpack.SUPPORTED_DTYPES:
      res_tf_platform = tf.DeviceSpec.from_string(res_tf.backing_device).device_type
      res_jax_platform = res_tf_platform.lower()
      if res_jax_platform in _DLPACK_PLATFORMS:
        res_dlpack = tf.experimental.dlpack.to_dlpack(res_tf)
        return jax.dlpack.from_dlpack(res_dlpack)

    # When working with a bfloat16 scalar tf.Tensor,np.asarray() can fail.
    # To handle this special case, we create a numpy copy.
    if res_tf.shape == tf.TensorShape([]) and res_tf.dtype == tf.bfloat16:
      return jax.device_put(jnp.array(res_tf.numpy()))
    else:
      return jax.device_put(np.asarray(res_tf))

  return list(map(_res_tf_to_jax, res_tf_flat))


call_tf_p.def_impl(_call_tf_impl)

@functools.lru_cache(maxsize=128)
def _get_concrete_function_tf(function_flat_tf, args_flat_sig_tf):  # -> tf.ConcreteFunction
  with jax2tf_internal.inside_call_tf():
    return function_flat_tf.get_concrete_function(*args_flat_sig_tf)


# Mark the effectful instances of call_tf
class CallTfEffect(effects.Effect):
  __str__ = lambda _: "CallTfEffect"

call_tf_effect = CallTfEffect()

effects.lowerable_effects.add_type(CallTfEffect)
effects.control_flow_allowed_effects.add_type(CallTfEffect)
effects.remat_allowed_effects.add_type(CallTfEffect)
effects.custom_derivatives_allowed_effects.add_type(CallTfEffect)


def _call_tf_abstract_eval(*_,
                           function_flat_tf,
                           args_flat_sig_tf,
                           has_side_effects, **__):
  # Called only when we form a Jaxpr, i.e., under jit, scan, etc.

  concrete_function_flat_tf = _get_concrete_function_tf(function_flat_tf,
                                                        args_flat_sig_tf)

  def is_fully_known_shape(s):
    return s.rank is not None and all([d is not None for d in s])
  effects = {call_tf_effect} if has_side_effects else set()
  if all([is_fully_known_shape(s)
          for s in concrete_function_flat_tf.output_shapes]):
    return (
        tuple([
            # We convert to JAX type, and canonicalize to 32-bit if necessary
            core.ShapedArray(shape, jax2tf_internal._to_jax_dtype(dtype))
            for dtype, shape in zip(concrete_function_flat_tf.output_dtypes,
                                    concrete_function_flat_tf.output_shapes)
        ]),
        effects)

  # There are some cases when TF shape inference is not powerful enough to
  # figure out the output shapes (e.g., b/128924522), even in situations where
  # XLA can compile the code, from which we can get the shapes.

  # We use the "cpu" as the platform, since JAX abstract eval is not platform
  # specific; the "cpu" backend is always available and for abstract evaluation
  # it should not matter which platform we use.
  _, result_avals = _code_generator_and_avals(function_flat_tf, args_flat_sig_tf,
                                              "CPU")
  return tuple(result_avals), effects

call_tf_p.def_effectful_abstract_eval(_call_tf_abstract_eval)


def _call_tf_lowering(ctx, *args_op, platform,
                      function_flat_tf, args_flat_sig_tf, **_):
  # This will most likely hit the cache, because we used it for abstract_eval
  # We use the same TF lowering device as for the embedding JAX computation.
  # One example when this is needed is when the code refers to variables on one
  # device. Or, for sharding annotations (only supported on TPU).
  if platform in ["cpu", "tpu"]:
    tf_platform = platform.upper()
  elif platform == "cuda":
    tf_platform = "GPU"
  else:
    raise ValueError("platform {platform} not supported")
  code_gen, _ = _code_generator_and_avals(function_flat_tf, args_flat_sig_tf,  # type: ignore
                                          tf_platform)
  assert code_gen is not None
  return code_gen(ctx.module_context, args_op)


@functools.lru_cache(maxsize=128)
def _code_generator_and_avals(
    function_flat_tf,
    args_flat_sig_tf,
    tf_platform,
) -> Tuple[Optional[Callable[[mlir.ModuleContext, Sequence[ir.Value]],
                             Sequence[ir.Value]]],
           Sequence[core.ShapedArray]]:
  # Returns and caches a code generator (taking a builder and the
  # XlaOps for the arguments) and a sequence of result abstract shapes.

  concrete_function_flat_tf = _get_concrete_function_tf(function_flat_tf, args_flat_sig_tf)

  captured_inputs = []
  if concrete_function_flat_tf.captured_inputs:
    # The function uses either captured variables or tensors.
    msg = (
        "call_tf works best with a TensorFlow function that does not capture "
        "variables or tensors from the context. "
        "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion. "
        f"The following captures were found {concrete_function_flat_tf.captured_inputs}")
    logging.warning(msg)
    for inp in concrete_function_flat_tf.captured_inputs:
      if inp.dtype == tf.resource:  # A variable; lookup by handle
        inp_vars = [v for v in concrete_function_flat_tf.variables if inp is v.handle]
        assert len(inp_vars) == 1, f"Found {inp_vars}"
        captured_inputs.append(inp_vars[0])
      else:
        captured_inputs.append(inp)

  def convert_to_spec(x):
    if isinstance(x, tf.TensorSpec):
      return x
    else:
      return tf.TensorSpec.from_tensor(x)

  args_tf_flat = [convert_to_spec(a) for a in args_flat_sig_tf]

  with jax2tf_internal.inside_call_tf():
    # When the TF computation uses variables on a particular device, we must
    # get_compiler_ir for that exact device.
    tf_device_name = f"/device:{tf_platform}:0"
    try:
      func_tf_hlo = function_flat_tf.experimental_get_compiler_ir(*args_tf_flat)(
        stage="hlo_serialized", device_name=tf_device_name)
    except Exception as e:
      msg = ("Error compiling TensorFlow function. call_tf can used " +
              "in a staged context (under jax.jit, lax.scan, etc.) only with " +
              "compileable functions with static output shapes. " +
              "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion.")
      raise ValueError(msg) from e

  xla_comp = xla_client.XlaComputation(func_tf_hlo)

  # Canonicalize the results; e.g., makes them x32 if JAX is in 32-bit mode
  def canonical_res_aval(res_shape: xla_client.Shape) -> core.ShapedArray:
    if not res_shape.is_static():
      msg = ("Compiled TensorFlow function has dynamic output shape " +
             f"{res_shape}. call_tf can used " +
             "in a staged context (under jax.jit, lax.scan, etc.) only with " +
             "compileable functions with static output shapes. " +
             "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion.")
      raise ValueError(msg)

    res_dtype = res_shape.numpy_dtype()
    jax_res_dtype = dtypes.canonicalize_dtype(res_dtype)
    return core.ShapedArray(res_shape.dimensions(), jax_res_dtype)

  result_shape = xla_comp.program_shape().result_shape()
  if not result_shape.is_tuple():
    # TF does not wrap singletons as tuples, but JAX expects tuples because
    # call_tf is a multiple_results primitive.
    result_shapes = (result_shape,)
  else:
    result_shapes = result_shape.tuple_shapes()  # type: ignore

  result_avals = tuple(map(canonical_res_aval, result_shapes))  # type: ignore

  def code_gen(ctx: mlir.ModuleContext, args_op: Sequence[ir.Value]
              ) -> Sequence[ir.Value]:
    captured_ops = tuple(mlir.ir_constant(np.asarray(inp),
                                          canonicalize_types=False)
                         for inp in captured_inputs)
    submodule = mlir.xla_computation_to_mlir_module(xla_comp)
    symtab = ir.SymbolTable(submodule.operation)
    callee_result_types = symtab["main"].type.results
    fn = mlir.merge_mlir_modules(ctx.module, f"call_tf_{function_flat_tf.name}",
                                 submodule)
    call = func_dialect.CallOp(callee_result_types,
                               ir.FlatSymbolRefAttr.get(fn),
                               tuple(args_op) + captured_ops)
    if result_shape.is_tuple():
      flat_results = [hlo.GetTupleElementOp(call, mlir.i32_attr(i)).result
                      for i in range(len(result_shapes))]
    else:
      flat_results = call.results

    outputs = []
    for op, res_aval, res_shape in zip(flat_results, result_avals,
                                       result_shapes):
      if res_aval.dtype != res_shape.numpy_dtype():
        op = hlo.ConvertOp(mlir.aval_to_ir_type(res_aval), op).result
      outputs.append(op)
    return outputs

  return code_gen, result_avals

def _register_call_lowering(platform):
  mlir.register_lowering(call_tf_p, functools.partial(_call_tf_lowering,
                                                      platform=platform),
                         platform=platform)
for platform in ("cpu", "cuda", "tpu"):
  _register_call_lowering(platform)

# Support the call_tf under jax2tf.convert
TfVal = jax2tf_internal.TfVal
def _jax2tf_call_tf(*args: TfVal,
                    callable_flat_tf: Callable,
                    **_) -> TfVal:
  with jax2tf_internal.inside_call_tf():
    res_tf_flat = callable_flat_tf(*args)
  return res_tf_flat

jax2tf_internal.tf_impl[call_tf_p] = _jax2tf_call_tf
