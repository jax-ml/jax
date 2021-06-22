# Copyright 2021 Google LLC
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
import logging
from typing import Callable, Sequence

import jax
from jax import core
from jax import dlpack
from jax import dtypes
from jax import numpy as jnp
from jax import tree_util
from jax._src import util
from jax.interpreters import xla
from jax.lib import xla_client
from . import jax2tf as jax2tf_internal

import numpy as np
import tensorflow as tf  # type: ignore[import]

map = util.safe_map
zip = util.safe_zip
xops = xla_client._xla.ops  # type: ignore

# The platforms for which to use DLPack to avoid copying (only works on GPU
# and CPU at the moment, and only for DeviceArray). For CPU we don't need
# DLPack, if we are careful.
_DLPACK_PLATFORMS = ("gpu",)

def call_tf(func_tf: Callable) -> Callable:
  """Calls a TensorFlow function from JAX, with support for reverse autodiff.

  The ``func_tf`` will be called with TensorFlow-compatible arguments (
  numpy.ndarray, ``tf.Tensor`` or ``tf.Variable``) or pytrees thereof. The
  function must return the same type of results.

  If ``call_tf`` appears in a JAX staging context (:func:`jax.jit`,
  or :func:`jax.pmap`, or :func:`jax.xmap`, or a control-flow primitive) then
  ``func_tf`` will be compiled with ``tf.function(func_tf, jit_compile=True)``
  and the resulting XLA computation will be embedded in JAX's XLA computation.

  If ``call_tf`` appears outside a JAX staging context, it will be called inline
  using TensorFlow eager mode.

  The ``call_tf`` supports JAX's reverse-mode autodiff, in which case the
  ``func_tf`` will be differentiated using ``tf.GradientTape``. This means
  that the gradient will be TensorFlow-accurate, e.g., will respect the
  custom gradients that may be defined for the code in ``func_tf``.

  For an example and more details see the
  `README <https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax>`_.

  Args:
    func_tf: a TensorFlow Callable that can take a pytree of TensorFlow
      arguments.
  Returns: a JAX callable that can be invoked with JAX pytree arguments, in
    op-by-op mode or in a staged context. This callable can be used with
    JAX's reverse-mode autodiff (:func:`jax.grad`).
  """

  @jax.custom_vjp
  def make_call(*args_jax):
    """We wrap it all in `make_call` so that we can attach custom VJP."""

    args_jax_flat, args_jax_treedef = tree_util.tree_flatten(args_jax)
    # Canonicalize the arguments; e.g., makes them x32 if JAX is in 32-bit mode
    def canonical_arg(v):
      v = v if getattr(v, "dtype", None) else np.asarray(v)
      dtype = dtypes.canonicalize_dtype(v.dtype)
      if dtype != v.dtype:
        v = v.astype(dtype)
      return v

    args_jax_flat = tuple(map(canonical_arg, args_jax_flat))
    args_tf_sig_flat = [
        tf.TensorSpec(a_jax.shape, jax2tf_internal._to_tf_dtype(a_jax.dtype))
        for a_jax in args_jax_flat
    ]
    args_tf_sig = args_jax_treedef.unflatten(args_tf_sig_flat)

    # Trace once through the function to get the result shape
    with jax2tf_internal.inside_call_tf():
      func_tf_concrete = tf.function(func_tf).get_concrete_function(*args_tf_sig)

    res_tf_sig_flat, res_treedef = tree_util.tree_flatten(
        func_tf_concrete.structured_outputs)

    # Canonicalize the result signature; e.g., makes them x32 if JAX is in 32-bit mode
    def res_sig_to_aval(res_sig: tf.TensorSpec) -> core.AbstractValue:
      return core.ShapedArray(res_sig.shape, jax2tf_internal._to_jax_dtype(res_sig.dtype))

    out_avals = tuple(map(res_sig_to_aval, res_tf_sig_flat))
    res_jax_flat = call_tf_p.bind(
        *args_jax_flat,
        # Carry the actual function such that op-by-op call can call in TF eager mode.
        func_tf=func_tf,
        func_tf_concrete=func_tf_concrete,
        args_treedef=args_jax_treedef,
        args_tf_sig_flat=args_tf_sig_flat,
        res_treedef=res_treedef,
        out_avals=out_avals)
    # TODO(necula): check the expected result signature
    assert len(res_jax_flat) == len(out_avals)
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
      def replace_non_float(arg):
        if np.issubdtype(arg.dtype.as_numpy_dtype, np.inexact):
          return arg
        else:
          # When watched, this will be ignored. When use in results it will
          # result in a floating 0. gradient, which JAX will ignore (and
          # replace it with a float0)
          return tf.zeros((), dtype=tf.float32)

      watched_args_tf = tf.nest.map_structure(replace_non_float, args_tf)
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(watched_args_tf)
        res = func_tf(*args_tf)

      tf.nest.assert_same_structure(res, ct_res_tf)
      dres_darg = tape.gradient(
          tf.nest.map_structure(replace_non_float, res),
          sources=watched_args_tf,
          output_gradients=ct_res_tf,
          unconnected_gradients=tf.UnconnectedGradients.ZERO)

      tf.nest.assert_same_structure(dres_darg, args_tf)
      return dres_darg

    # Use call_tf to call the VJP function
    return call_tf(tf_vjp_fun)(args_jax, ct_res_jax)

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return util.wraps(func_tf)(make_call)


call_tf_p = core.Primitive("call_tf")
call_tf_p.multiple_results = True


# The impl will be used in op-by-op mode and calls func_tf in TF eager mode.
def _call_tf_impl(*args_jax_flat, args_treedef, func_tf, out_avals, **_):
  # On GPU we use dlpack to avoid copies of data to the host.
  def _arg_jax_to_tf(arg_jax):
    if (isinstance(arg_jax, xla.DeviceArray) and
        arg_jax.device_buffer.client.platform in _DLPACK_PLATFORMS and
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
    res_tf = func_tf(*args_treedef.unflatten(args_tf_flat))
  res_tf_flat, _ = tree_util.tree_flatten(res_tf)
  # TODO(necula): check the result for tree and aval

  def _res_tf_to_jax(res_tf: TfVal, out_aval: core.AbstractValue):
    res_tf, _ = jax2tf_internal._tfval_to_tensor_jax_dtype(res_tf, jax_dtype=out_aval.dtype)
    if isinstance(res_tf, tf.Tensor) and res_tf.dtype in dlpack.SUPPORTED_DTYPES:
      res_tf_platform = tf.DeviceSpec.from_string(res_tf.backing_device).device_type
      res_jax_platform = res_tf_platform.lower()
      if res_jax_platform in _DLPACK_PLATFORMS:
        res_dlpack = tf.experimental.dlpack.to_dlpack(res_tf)
        return jax.dlpack.from_dlpack(res_dlpack)

    return jnp.asarray(np.asarray(res_tf))

  return list(map(_res_tf_to_jax, res_tf_flat, out_avals))


call_tf_p.def_impl(_call_tf_impl)


def _call_tf_abstract_eval(*_, out_avals, **__):
  return out_avals


call_tf_p.def_abstract_eval(_call_tf_abstract_eval)


def _call_tf_translation_rule(builder, *args_op, func_tf, func_tf_concrete,
                              args_treedef, args_tf_sig_flat, out_avals,
                              **_):
  # TODO(necula): It seems that we need concrete tensors for get_compiler_ir?
  args_tf_flat = [
      tf.constant((0 if a.dtype != tf.bool else False),
                  shape=a.shape,
                  dtype=a.dtype) for a in args_tf_sig_flat
  ]
  args_tf = args_treedef.unflatten(args_tf_flat)
  func_tf = tf.function(func_tf, jit_compile=True)
  #func_tf_concrete = func_tf.get_concrete_function(*args_tf)
  captured_ops = []  # Same order as captured_inputs
  if func_tf_concrete.captured_inputs:
    # The function uses either captured variables or tensors.
    msg = (
      "call_tf works best with a TensorFlow function that does not capture "
      "variables or tensors from the context. "
      "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax for a discussion. "
      f"The following captures were found {func_tf_concrete.captured_inputs}")
    logging.warning(msg)

    next_var_idx = 0
    for inp in func_tf_concrete.captured_inputs:
      if inp.dtype == tf.resource:  # A variable; assume the next variable
        assert next_var_idx < len(func_tf_concrete.variables)
        # TODO(necula): better checking that we are picking the right variable
        var = func_tf_concrete.variables[next_var_idx]
        next_var_idx += 1
        inp_const = np.asarray(var)
      else:
        inp_const = np.asarray(inp)
      captured_ops.append(xops.ConstantLiteral(builder, np.asarray(inp_const)))

  # TODO(necula): For unoptimized HLO, does it make a difference which device we use?
  tf_device_name = "/device:CPU:0"
  func_tf_hlo = func_tf.experimental_get_compiler_ir(*args_tf)(
      stage="hlo_serialized", device_name=tf_device_name)
  callee_xla_comp = xla_client.XlaComputation(func_tf_hlo)
  res_tf = xops.Call(builder, callee_xla_comp, args_op + tuple(captured_ops))
  if len(out_avals) == 1:
    # TF does not wrap singletons as tuples, but JAX expects tuples because
    # call_tf is a multiple_results primitive.
    res_untupled = (res_tf,)
  else:
    res_untupled = tuple(xops.GetTupleElement(res_tf, i)
                         for i in range(len(out_avals)))
  # We may have to cast the results to x32 for JAX
  def canonicalize_res(res, out_aval: core.AbstractValue):
    res_dtype = builder.get_shape(res).numpy_dtype()
    if res_dtype != out_aval.dtype:
      new_etype = xla_client.dtype_to_etype(out_aval.dtype)
      return xops.ConvertElementType(res, new_element_type=new_etype)
    else:
      return res

  canonical_res_untupled = tuple(map(canonicalize_res,
                                     res_untupled,
                                     out_avals))
  return xops.Tuple(builder, canonical_res_untupled)




xla.translations[call_tf_p] = _call_tf_translation_rule

TfVal = jax2tf_internal.TfVal
def _jax2tf_call_tf(*args: TfVal,
                    _in_avals: Sequence[core.ShapedArray],
                    _out_aval: core.ShapedArray,
                    func_tf: Callable,
                    **kwargs) -> TfVal:
  res_tf = func_tf(*args)
  res_tf_flat = tf.nest.flatten(res_tf)
  # TODO: check that the return values have the right signature
  return res_tf_flat

jax2tf_internal.tf_impl_with_avals[call_tf_p] = _jax2tf_call_tf
