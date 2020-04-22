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
    explained in `id_print` documentation (DONE).
  * Implement the transformations. DONE (except pmap)
  * Implement the JIT for CPU using CustomCall in C++. DONE (except unit tests
    do not run in OSS; also missing float16 and bfloat16).
  * Implement the JIT for GPU using also CustomCall in C++. DONE.
  * Explore how to pipe the printed data back to the Colab cell,
    when running in Colab. ?
  * Explore implementation using outfeed, hoping that it works for all
    platforms, and can pipe data more easily. STARTED.
  * Explore feeding the data back to the Python program (the `id_tap`
    primitive). ?
  * Explore a simpler API that uses Python program-order, instead of
    data dependency-order. Need to add support to JAX for stateful primitives.
"""
from contextlib import contextmanager
from functools import partial
import io
import itertools

from jax import abstract_arrays
from jax import core
from jax import dtypes
from jax import lax
from jax.lib import pytree, xla_bridge
from jax.interpreters import ad, xla, batching
from jax.interpreters import partial_eval as pe
from jax import util
from jaxlib import xla_client
from jaxlib import xla_extension

import logging
import numpy as onp
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

# TODO(necula): fix mypy errors if I define the type aliases below
XlaOp = Any  # xla_extension.XlaOp
XlaShape = Any # xla_client.Shape
XlaComputationBuilder = Any  # xla_bridge._JaxComputationBuilder

id_print_p = core.Primitive("id_print")
id_print_p.multiple_results = True

xops = xla_client._xla.ops

def id_print(*args, result=None, **kwargs):
  """Behaves like the identify function for positional arguments, but prints all
     arguments on the host, even from transformed or compiled code.

     The return value is a tuple with the value of `args` or the value of the
     keyword parameter `result` if present. If there is a single positional
     argument, it returns just that argument without packing it in a tuple.

     The positional arguments must be JAX values. The keyword arguments are
     serialized to a string and printed along with the positional arguments.
     There are a few special keywork arguments that are not printed:

      * `result`: is the result of `id_print`, must be a JAX value or a
        pytree of values.
      * `output_stream`: is the output stream where the values should be
      printed. (Note: does not yet work from under JIT).

     Usage:
     >>> y = id_print(x * 2)  # prints and returns 2x
     >>> y, z = id_print(x * 2, x * 3)  # prints and returns 2x and 3x
     >>> y = id_print(x * 2, result=y)  # prints 2x and returns y
     >>> y = id_print(x * 2, what='x')  # prints what=x followed by 2x

     The order of execution is by data dependency: after all the arguments are
     computed and before the result is used. At least one of the returned values
     must be used in the rest of the computation, or else this operation has
     no effect.

     Upon JAX transformations, the transformed values are wrapped with
     `id_print`, and a special `transforms` tuple keyword argument is added with
     the sequence of transformations applied:

        - For `vmap` the arguments are batched, and transforms=('vmap')
        - For `jvp` there will be an id_print for the primal values, and a
        separate `id_print` for the tangents with `transforms=('jvp')`.
        - For `grad` there will be an `id_print` for the primal values (if
        needed in the computation of `grad` and an `id_print` with the
        adjoints of the results, with transforms=('vjp').
  """
  flat_args, args_treedef = pytree.flatten(args)

  params = dict(kwargs)  #  copy
  if result is not None:
    flat_results, results_treedef = pytree.flatten(result)
    params["nr_results"] = len(flat_results)
    all_args = flat_args + flat_results
  else:
    all_args = flat_args
  flat_outs = id_print_p.bind(*all_args, **params)  # Always returns a tuple of all args
  if result is not None:
    return results_treedef.unflatten(flat_outs[-params["nr_results"]:])
  else:
    res = args_treedef.unflatten(flat_outs)
    return res if len(args) > 1 else res[0]


def _expand_params_transform(params: Dict, transform: str) -> Dict:
  """Adds the `transform` to the params["transforms"]."""
  return dict(params, transforms=params.get("transforms", ()) + (transform,))


def _id_print_impl(*args, **params):
  output_stream = params.get("output_stream")
  if output_stream is not None:
    print_params = dict(params)
    del print_params["output_stream"]
  else:
    import sys
    output_stream = sys.stdout
    print_params = params

  # TODO: use the JITed version to do the actual printing.
  to_print = f"{args}  {print_params}"
  output_stream.write(to_print)

  return args


id_print_p.def_impl(_id_print_impl)

def _id_print_abstract_eval(*args_a: pe.AbstractValue, **params) \
    -> Sequence[pe.AbstractValue]:
  return args_a


id_print_p.def_abstract_eval(_id_print_abstract_eval)


# Each array sent to the outfeed is preceeded by a descriptor, which
# is an array s32[32], with the format:
#  [0]: special header value
#  [1]: an encoding of the element_type()
#  [2]: the number of dimensions
#  [3:...]: the size of the dimensions
#  padded with 0s
#
_OUTFEED_DESCRIPTOR_PRINT_HEADER = 13579
_OUTFEED_DESCRIPTOR_LENGTH = 32
_CODE_TO_DTYPE = {
  0: onp.dtype(onp.int8),
  1: onp.dtype(onp.int16),
  2: onp.dtype(onp.int32),
  3: onp.dtype(onp.int64),
  4: onp.dtype(onp.uint8),
  5: onp.dtype(onp.uint16),
  6: onp.dtype(onp.uint32),
  7: onp.dtype(onp.uint64),
  8: onp.dtype(onp.float16),
  9: onp.dtype(onp.float32),
  10: onp.dtype(onp.float64),
  11: onp.dtype(dtypes.bfloat16),
}
_DTYPE_STR_TO_CODE = dict([(str(d), c) for c, d in _CODE_TO_DTYPE.items()])



def _id_print_translation_rule_outfeed(
    comp: XlaComputationBuilder,
    *args_op: XlaOp, **params):

  # TODO: outfeed a whole tuple at once?
  def _outfeed_one_array(a: XlaOp, token: XlaOp) -> XlaOp:
    a_shape = comp.GetShape(a)
    dimensions = a_shape.dimensions()
    descriptor = [_OUTFEED_DESCRIPTOR_PRINT_HEADER,
                  _DTYPE_STR_TO_CODE[str(onp.dtype(a_shape.element_type()))],
                  len(dimensions)] + list(dimensions)
    if len(descriptor) > _OUTFEED_DESCRIPTOR_LENGTH:
      raise ValueError(f"Too many dimensions in array to print: {a_shape}")
    descriptor += [0] * (_OUTFEED_DESCRIPTOR_LENGTH - len(descriptor))

    data = xops.ConstantLiteral(comp, onp.array(descriptor, dtype=onp.int32))
    token = xops.OutfeedWithToken(data, token, comp.GetShape(data))
    token = xops.OutfeedWithToken(a, token, a_shape)
    return token

  prev_token = xla.computation_state_carry.current_token(comp)
  nr_args_to_emit = len(args_op) - params.get("nr_results", 0)
  for i, a_op in enumerate(args_op):
    if i < nr_args_to_emit:
      prev_token = _outfeed_one_array(a_op, prev_token)
  xla.computation_state_carry.set_current_token(comp, prev_token)
  return xops.Tuple(comp, args_op)

xla.translations[id_print_p] = _id_print_translation_rule_outfeed


# TODO: find a better way to signal the end of printing
_END_PRINTING = onp.int32(12345678)
def end_printing(res):
  return id_print(_END_PRINTING, result=res)

@contextmanager
def print_receiver(output_stream=None,
                   receiver_name="",
                   timeout_sec=10):
  # TODO: better timeout management
  """Starts a receiver for the id_print outfeed.

  Usage:
    with print_receiver():
      jax.jit(func)(args)

  """
  # TODO: start receivers for each device
  platform = xla_client.get_local_backend(None).platform
  def _get_data(data_shape: XlaShape) -> XlaShape:
    if platform == "gpu":
      return xla_client.transfer_from_outfeed(data_shape)
    else:
      return xla_client.transfer_from_outfeed(
          xla_client.Shape.tuple_shape((data_shape,)))[0]

  def _consume_one_array_from_outfeed():
    descriptor_shape = xla_client.Shape.array_shape(onp.dtype(onp.int32),
                                                    (_OUTFEED_DESCRIPTOR_LENGTH,))
    descriptor = _get_data(descriptor_shape)

    logging.info(f"[{receiver_name}] Read descriptor: {descriptor}")
    if descriptor[0] != _OUTFEED_DESCRIPTOR_PRINT_HEADER:
      raise ValueError(f"Read unexpected print descriptor {descriptor} [{receiver_name}]")
    data_dimensions = tuple(descriptor[3:3+descriptor[2]])
    data_shape = xla_client.Shape.array_shape(_CODE_TO_DTYPE[descriptor[1]],
                                              data_dimensions)
    data = _get_data(data_shape)
    logging.info(f"[{receiver_name}] Read data of shape {data.dtype}{data.shape}")
    return data

  def receiver_loop():
    i = 0
    while (True):
      got = _consume_one_array_from_outfeed()
      if not got.shape and got == _END_PRINTING:
        logging.info(f"[{receiver_name}] Received END_PRINTING")
        return
      got_str = onp.array2string(got, threshold=1024)
      logging.info(f"[{receiver_name}] Received {i} ({got.dtype}{got.shape}): {got_str}")
      if output_stream is not None:
        output_stream.write(got_str)
      i += 1

  receiver = threading.Thread(target=receiver_loop)
  receiver.start()
  try:
    yield receiver
  finally:
    # TODO: proper termination
    receiver.join(timeout=timeout_sec)
    if receiver.is_alive():
      logging.error(f"[{receiver_name}] Receiver still alive")
    else:
      logging.info(f"[{receiver_name}] Receiver finished")


def _id_print_jvp_rule(primals, tangents, **params):
  primals_out = id_print(primals, **params)
  tangents_out = id_print(tangents, **_expand_params_transform(params, "jvp"))
  return primals_out, tangents_out


ad.primitive_jvps[id_print_p] = _id_print_jvp_rule


def _id_print_transpose_rule(cts, *args, **params):
  assert all([ad.is_undefined_primal(x) for x in args])
  assert len(cts) == len(args)
  ct_args = id_print_p.bind(*cts,
                            **_expand_params_transform(params, "transpose"))
  return ct_args


ad.primitive_transposes[id_print_p] = _id_print_transpose_rule


def _id_print_batching_rule(batched_args, batch_dims, **params):
  res = id_print_p.bind(*batched_args,
                        **_expand_params_transform(params, "batch"))
  return res, batch_dims


batching.primitive_batchers[id_print_p] = _id_print_batching_rule
