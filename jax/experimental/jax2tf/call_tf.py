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
https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax.

"""
import logging
from typing import Callable

import jax
from jax import core
from jax import dlpack
from jax import dtypes
from jax import numpy as jnp
from jax import tree_util
from jax._src import util
from jax.interpreters import xla
from jax.lib import xla_bridge
from jax.lib import xla_client

import numpy as np
import tensorflow as tf  # type: ignore[import]

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
  `README <https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax>`_.

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

    def _dtype(x):
      return (getattr(x, "dtype", None) or np.asarray(x).dtype)

    args_jax_flat, args_jax_treedef = tree_util.tree_flatten(args_jax)
    args_tf_sig_flat = [
        tf.TensorSpec(np.shape(a_jax), _to_tf_dtype(_dtype(a_jax)))
        for a_jax in args_jax_flat
    ]
    args_tf_sig = tf.nest.map_structure(
        lambda a_jax: tf.TensorSpec(
            np.shape(a_jax), _to_tf_dtype(_dtype(a_jax))), args_jax)
    func_tf_concrete = tf.function(func_tf).get_concrete_function(*args_tf_sig)
    res_tf_sig_flat, res_treedef = tree_util.tree_flatten(
        func_tf_concrete.structured_outputs)

    res_jax_flat = call_tf_p.bind(
        *args_jax_flat,
        func_tf=func_tf,
        args_treedef=args_jax_treedef,
        args_tf_sig_flat=args_tf_sig_flat,
        res_treedef=res_treedef,
        res_tf_sig_flat=res_tf_sig_flat)
    # TODO(necula): check the expected result signature
    assert len(res_jax_flat) == len(res_tf_sig_flat)
    return res_treedef.unflatten(res_jax_flat)

  # Define the fwd and bwd custom_vjp functions
  def make_call_vjp_fwd(*args_jax):
    # Return the primal argument as the residual
    return make_call(*args_jax), args_jax

  def make_call_vjp_bwd(residual, ct_res):
    args_jax = residual  # residual is the primal argument

    def tf_vjp_fun(args, ct_res):
      """Invoke TF gradient."""
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(args)
        res = func_tf(*args)

      tf.nest.assert_same_structure(res, ct_res)
      # If the result is not a scalar, we must accumulate arguments cotangents.
      accumulator = None  # Same structure as "arg"

      def acc_ct(res_, ct_res_):
        dres_darg = tape.gradient(
            res_,
            sources=args,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        tf.nest.assert_same_structure(dres_darg, args)
        scaled_dres_darg = tf.nest.map_structure(lambda d: d * ct_res_,
                                                 dres_darg)
        nonlocal accumulator
        accumulator = (
            scaled_dres_darg if accumulator is None
            else tf.nest.map_structure(
                lambda x, y: x + y, accumulator, scaled_dres_darg))

      tf.nest.map_structure(acc_ct, res, ct_res)
      return accumulator
    # Use call_tf to call the VJP function
    return call_tf(tf_vjp_fun)(args_jax, ct_res)

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return util.wraps(func_tf)(make_call)


call_tf_p = core.Primitive("call_tf")
call_tf_p.multiple_results = True


# The impl will be used in op-by-op mode and calls func_tf in TF eager mode.
def _call_tf_impl(*args_jax_flat, args_treedef, func_tf, **_):
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
  res_tf = func_tf(*args_treedef.unflatten(args_tf_flat))
  res_tf_flat, _ = tree_util.tree_flatten(res_tf)
  # TODO(necula): check the result for tree and aval

  def _res_tf_to_jax(res_tf):
    if isinstance(res_tf, tf.Tensor) and res_tf.dtype in dlpack.SUPPORTED_DTYPES:
      res_tf_platform = tf.DeviceSpec.from_string(res_tf.backing_device).device_type
      res_jax_platform = res_tf_platform.lower()
      if res_jax_platform in _DLPACK_PLATFORMS:
        res_dlpack = tf.experimental.dlpack.to_dlpack(res_tf)
        return jax.dlpack.from_dlpack(
            res_dlpack, backend=xla_bridge.get_backend(res_jax_platform))

    return jnp.asarray(np.asarray(res_tf))

  return list(map(_res_tf_to_jax, res_tf_flat))


call_tf_p.def_impl(_call_tf_impl)


def _call_tf_abstract_eval(*_, res_tf_sig_flat, **__):
  return tuple([
      core.ShapedArray(np.shape(r), _to_jax_dtype(r.dtype))
      for r in res_tf_sig_flat
  ])


call_tf_p.def_abstract_eval(_call_tf_abstract_eval)


def _call_tf_translation_rule(builder, *args_op, func_tf,
                              args_treedef, args_tf_sig_flat, res_tf_sig_flat,
                              **_):
  # TODO(necula): It seems that we need concrete tensors for get_compiler_ir?
  args_tf_flat = [
      tf.constant((0 if a.dtype != tf.bool else False),
                  shape=a.shape,
                  dtype=a.dtype) for a in args_tf_sig_flat
  ]
  args_tf = args_treedef.unflatten(args_tf_flat)
  func_tf = tf.function(func_tf, jit_compile=True)
  func_tf_concrete = func_tf.get_concrete_function(*args_tf)
  captured_ops = []  # Same order as captured_inputs
  if func_tf_concrete.captured_inputs:
    # The function uses either captured variables or tensors.
    msg = (
      "call_tf works best with a TensorFlow function that does not capture "
      "variables or tensors from the context. "
      "See https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax for a discussion. "
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
  if len(res_tf_sig_flat) == 1:
    # TF does not wrap singletons as tuples, but JAX expects tuples because
    # call_tf is a multiple_results primitive.
    return xops.Tuple(builder, [res_tf])
  else:
    return res_tf


xla.translations[call_tf_p] = _call_tf_translation_rule


def _to_tf_dtype(jax_dtype):
  if jax_dtype == dtypes.float0:
    return tf.float32
  else:
    return tf.dtypes.as_dtype(jax_dtype)


def _to_jax_dtype(tf_dtype):
  return tf_dtype.as_numpy_dtype
