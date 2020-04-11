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

"""Implementation of an experimental primitive for printing, including
   from transformed and compiled code.

See documentation for `id_print` below.
For usage example, see tests/host_callback_test.py.

Implementation plan:
  * Write the API for the `id_print` primitive, using data-dependence as
    explained in `id_print` documentation.
  * Implement the transformations. DONE (except pmap)
  * Implement the JIT for CPU using CustomCall in C++.
    DONE (need unit tests, google3 build rules)
  * Implement the JIT for GPU using also CustomCall in C++. STARTED.
  * Explore how to pipe the printed data back to the Colab cell,
    when running in Colab. ?
  * Explore implementation using outfeed, hoping that it works for all
    platforms, and can pipe data more easily. STARTED.
  * Explore feeding the data back to the Python program (the `id_tap` primitive). ?
  * Explore a simpler API that uses Python program-order, instead of
    data dependency-order. Need to add support to JAX for stateful primitives.

"""

import copy
from functools import partial
import io
import itertools
import numpy as onp
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import jax
from jax import abstract_arrays
from jax import lax, core
from jax.lax import lax_control_flow
from jax import tree_util
from jax.lib import pytree, xla_bridge
from jax import numpy as jnp
from jax.interpreters import partial_eval as pe
from jax.interpreters import ad, xla, batching, pxla
from jax.lax import lax_control_flow
from jax.util import unzip2, safe_map

from jaxlib import xla_client, xla_extension
import msgpack
import numpy as np

XlaOp = xla_extension.XlaOp
XlaShape = xla_client.Shape
XlaComputationBuilder = xla_bridge._JaxComputationBuilder

try:
  from jaxlib import host_callback

  for _name, _value in host_callback.customcall_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")
except ImportError as e:
  pass

id_print_p = core.Primitive("id_print")


def id_print(*args, **kwargs):
  """Behaves like identify function for positional arguments, but prints all arguments.

     The return value is a tuple with the value of `args` or the value of the keyword
     parameter `result` if present. If there is a single positional argument, returns
     just that argument without packing it in a tuple.

     The positional arguments must be JAX values. The keyword arguments are serialized
     to a string and printed along with the positional arguments. The exception is the
     `result` keyword argument, which must be a JAX value.

     Usage:
     >>> y = id_print(x * 2)  # prints and returns 2x
     >>> y, z = id_print(x * 2, x * 3)  # prints and returns 2x and 3x
     >>> y = id_print(x * 2, result=y)  # prints 2x and returns y
     >>> y = id_print(x * 2, what='x')  # prints what=x followed by 2x

     The order of execution is by data dependency: after all the arguments are computed
     and before the result is used. At least one of the returned values must be used
     in the rest of the computation, or else this operation has no effect.

     Upon JAX transformations, the transformed values are wrapped with `id_print`, and
     a special `transforms` tuple keyword argument is added with the sequence of
     transformations applied:

        - For `vmap` the arguments are batched, and transforms=('vmap')
        - For `jvp` there will be an id_print for the primal values, and a separate
          `id_print` for the tangents with `transforms=('jvp')`.
        - For `grad` there will be an `id_print` for the primal values (if needed
          in the computation of `grad` and an `id_print` with the adjoints of the
          results, with transforms=('vjp').
  """
  if "tie_in" in kwargs:
    params = dict(kwargs)
    tie_into = params["tie_in"]
    try:
      core.get_aval(tie_into)
    except Exception as e:
      raise ValueError(f"The 'tie_in' parameter for print must be a JAX value: {tie_into}")
    del params["tie_in"]
  else:
    params = kwargs
    tie_into = None
  flat_args, args_treedef = pytree.flatten(args)
  if tie_into is not None:
    return lax.tie_in(id_print_p.bind(*flat_args, **params)[0], tie_into)
  else:
    flat_outs = id_print_p.bind(*flat_args, **kwargs)  # id_print returns multiple results
    res = args_treedef.unflatten(flat_outs)
    return res if len(args) > 1 else res[0]


def _expand_params_transform(params: Dict, transform: str) -> Dict:
  return dict(params, transforms=params.get("transforms", ()) + (transform,))


def _id_print_impl(*args, **params):
  if "output_stream" in params:
    output_stream = params["output_stream"]
    print_params = dict(params)
    del print_params["output_stream"]
  else:
    import sys
    output_stream = sys.stdout
    print_params = params

  # TODO: maybe have a way to format?
  to_print = f"{args}  {print_params}"
  output_stream.write(to_print)

  return args


def _id_print_abstract_eval(*args_a: Sequence[pe.AbstractValue], **params) \
    -> Sequence[pe.AbstractValue]:
  return args_a


id_print_p.multiple_results = True
id_print_p.def_impl(_id_print_impl)
id_print_p.def_abstract_eval(_id_print_abstract_eval)

# We must pass the printing library some metadata. See host_callback.cc.
_lib_print_metadata_version = None
_MINIMUM_LIB_PRINT_METADATA_VERSION = 1
_CURRENT_PRINT_METADATA_VERSION = 1


def _make_id_print_metadata(args_xla_shape: Sequence[XlaShape],
                            params: Dict) -> bytes:
  global _lib_print_metadata_version
  if _lib_print_metadata_version is None:
    _lib_print_metadata_version = host_callback.get_print_metadata_version()
    if _lib_print_metadata_version < _MINIMUM_LIB_PRINT_METADATA_VERSION:
      # TODO(necula): this should really be handled by minimum version
      raise NotImplementedError(f"id_print requires minimum metadata version "
                                f"{_MINIMUM_LIB_PRINT_METADATA_VERSION}. "
                                f"Found version {_lib_print_metadata_version}."
                                f"Update jaxlib.")

  def _one_arg_descriptor(arg_xla_shape: XlaShape) -> Tuple[str, Tuple[int, ...]]:
    """For each argument, a type descriptor and the shape."""
    dtype = np.dtype(arg_xla_shape.element_type())
    return (f"{dtype.kind}{dtype.itemsize}", arg_xla_shape.dimensions())

  preamble = ", ".join([f"{k}: {v}" for k, v in sorted(params.items())
                        if k != "output_stream"])
  iobuff = io.BytesIO()
  iobuff.write(msgpack.packb(preamble))
  iobuff.write(msgpack.packb(", \n"))  # the separator
  iobuff.write(msgpack.packb(tuple(_one_arg_descriptor(arg_shape)
                                   for arg_shape in args_xla_shape)))
  iobuff.seek(0)
  return iobuff.read()


def _id_print_translation_rule(platform: str,
                               comp: XlaComputationBuilder,
                               *args_op: XlaOp, **params):
  args_xla_shape = tuple([comp.GetShape(a) for a in args_op])

  descriptor: bytes = _make_id_print_metadata(args_xla_shape, params)
  if platform == "cpu":
    # Prepend the length of the descriptor and the descriptor.
    additional_ops = (
      comp.ConstantS32Scalar(len(descriptor)),
      comp.Constant(onp.array(list(descriptor), onp.uint8)))
    opaque = b""
  elif platform == "gpu":
    additional_ops = ()
    opaque = descriptor
  else:
    raise NotImplementedError(f"id_print not implemented for {platform}")

  def _shape_with_default_layout(xla_shape: XlaShape):
    return xla_client.Shape.array_shape(
      xla_shape.element_type(), xla_shape.dimensions(),
      tuple(range(xla_shape.rank() - 1, -1, -1)))

  # The CustomCall will return the value True. It does not return the input
  # arrays because that involves a copy in C++. We use the result:
  #   lax.cond(custom_call_res, args_op, lambda x: x,
  #            args_op, lambda x: zeros_like(x))
  #
  result_shape_with_layout = _shape_with_default_layout(
    xla.aval_to_xla_shape(pe.get_aval(True)))

  operand_shapes_with_layout = tuple(
    [_shape_with_default_layout(comp.GetShape(arg_op))
     for arg_op in additional_ops + args_op])

  print_out = comp.CustomCallWithLayout(
    b"jax_print_cpu",
    operands=additional_ops + args_op,
    shape_with_layout=result_shape_with_layout,
    operand_shapes_with_layout=operand_shapes_with_layout,
    opaque=opaque)
  args_tuple = comp.Tuple(*args_op)

  # The true branch: lambda args: args
  true_comp_builder = xla_bridge.make_computation_builder('print_true')
  true_parameter = true_comp_builder.ParameterWithShape(comp.GetShape(args_tuple))
  true_comp = true_comp_builder.Build(true_parameter)

  # The false branch: lambda args: zeros_like(args)
  false_comp_builder = xla_bridge.make_computation_builder('print_false')
  _ = false_comp_builder.ParameterWithShape(xla_client.Shape.token_shape())

  def make_one_zero(arg_xla_shape: XlaShape):
    zero = false_comp_builder.Constant(np.array(0, dtype=arg_xla_shape.element_type()))
    return lax.standard_translate("broadcast_in_dim", false_comp_builder,
                                  zero, shape=arg_xla_shape.dimensions(),
                                  broadcast_dimensions=())

  false_comp = false_comp_builder.Build(
    false_comp_builder.Tuple(*map(make_one_zero, args_xla_shape)))

  return comp.Conditional(print_out, args_tuple, true_comp,
                          comp.CreateToken(), false_comp)


xla.backend_specific_translations["cpu"][id_print_p] = \
  partial(_id_print_translation_rule, "cpu")
xla.backend_specific_translations["gpu"][id_print_p] = \
  partial(_id_print_translation_rule, "gpu")


def _id_print_jvp_rule(primals, tangents, **params):
  primals_out = id_print(primals, **params)
  tangents_out = id_print(tangents, **_expand_params_transform(params, "jvp"))
  return primals_out, tangents_out


ad.primitive_jvps[id_print_p] = _id_print_jvp_rule


def _id_print_transpose_rule(cts, *args, **params):
  assert all([ad.is_undefined_primal(x) for x in args])
  assert len(cts) == len(args)
  ct_args = id_print_p.bind(*cts, **_expand_params_transform(params, "transpose"))
  return ct_args


ad.primitive_transposes[id_print_p] = _id_print_transpose_rule


def _id_print_batching_rule(batched_args, batch_dims, **params):
  res = id_print_p.bind(*batched_args, **_expand_params_transform(params, "batch"))
  return res, batch_dims


batching.primitive_batchers[id_print_p] = _id_print_batching_rule
