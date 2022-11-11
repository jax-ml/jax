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
import functools

from jax import config
from jax._src import api_util

from jax.experimental.jax2tf import jax2tf_common  # pylint: disable=unused-import
from jax.experimental.jax2tf import jax2tf_native
from jax.experimental.jax2tf import jax2tf_old


@functools.partial(api_util.api_hook, tag="jax2tf_convert")
def convert(fun_jax,
            *,
            polymorphic_shapes=None,
            with_gradient=True,
            enable_xla=True,
            experimental_native_lowering="default"):
  """Lowers `fun_jax` into a function that uses only TensorFlow ops.

  See
  [README](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md)
  for more details about usage and common problems.

  Args:
    fun_jax: JAX function to be lowered. Its arguments and return value should be
      JAX arrays, or nested standard Python containers (tuple/list/dict) thereof
      (pytrees).
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
      arguments](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).

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

      See [the README](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#shape-polymorphic-conversion)
      for more details.

    with_gradient: if set (default), add a tf.custom_gradient to the lowered
      function, by converting the ``jax.vjp(fun)``. This means that reverse-mode
      TensorFlow AD is supported for the output TensorFlow function, and the
      value of the gradient will be JAX-accurate.
    enable_xla: if set (default), use the simplest conversion
      and use XLA TF ops when necessary. These ops are known to create issues
      for the TFLite and TFjs converters. For those cases, unset this parameter
      so the the lowering tries harder to use non-XLA TF ops to lower the
      function and aborts if this is not possible.
    experimental_native_lowering: DO NOT USE, for experimental purposes only.
      The value "default" defers to --jax2tf_default_experimental_native_lowering.

  Returns:
    A version of `fun_jax` that expects TfVals as arguments (or
    tuple/lists/dicts thereof), and returns TfVals as outputs, and uses
    only TensorFlow ops.
  """
  if experimental_native_lowering == "default":
    experimental_native_lowering = config.jax2tf_default_experimental_native_lowering

  if experimental_native_lowering:
    return jax2tf_native.convert(
        fun_jax,
        polymorphic_shapes=polymorphic_shapes,
        with_gradient=with_gradient,
        enable_xla=enable_xla)
  else:
    return jax2tf_old.convert(
        fun_jax,
        polymorphic_shapes=polymorphic_shapes,
        with_gradient=with_gradient,
        enable_xla=enable_xla)


TfVal = jax2tf_old.TfVal
tf_impl = jax2tf_old.tf_impl
inside_call_tf = jax2tf_old.inside_call_tf
_to_tf_dtype = jax2tf_old._to_tf_dtype
_tfval_to_tensor_jax_dtype = jax2tf_old._tfval_to_tensor_jax_dtype
_to_jax_dtype = jax2tf_old._to_jax_dtype
