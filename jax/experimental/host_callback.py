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
from concurrent import futures
from contextlib import contextmanager
from functools import partial
import io
import itertools

from jax import abstract_arrays
from jax import ad_util
from jax import api
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
import msgpack  # type: ignore
import numpy as onp
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# TODO(necula): fix mypy errors if I define the type aliases below
XlaOp = Any  # xla_extension.XlaOp
XlaShape = Any # xla_client.Shape
XlaComputationBuilder = Any  # xla_bridge._JaxComputationBuilder
XlaDevice = Any  # xla_client.Device

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

# The data on the outfeed follows a protocol that allows multiplexing the
# outfeed among multiple consumers, and communicates in-stream shape and
# type of the data.
# Each batch of array data is preceeded by one or more header messages.
# A header message is of type uint32[_OUTFEED_HEADER_LENGTH // 4], with the
# uint32 being the big-endian encoding of the following array of bytes:
#  [0], [1]: special header values 21 and 78
#  [2]: a consumer id (e.g., _OUTFEED_CONSUMER_ID_PRINT)
#  [3], [4]: big-endian encoding of metadata length (up to 2**16). The
#     metadata is a msgpack-encoded value of type:
#     ([ (type_code, (d0, d1, ...)), ...],  # for each array, element type code
#                                           # and the dimensions.
#      { ... })                             # kwargs to be passed to the consumer
#  padded with 0s to _OUTFEED_HEADER_LENGTH
#
#  If the metadata is too long to fit in one header array, several more
#  header arrays will follow, with identical content except for the metadata
#  bytes.
#
_OUTFEED_HEADER_LENGTH = 64  # In bytes
_OUTFEED_HEADER_START0 = 21
_OUTFEED_HEADER_START1 = 78
_OUTFEED_HEADER_METADATA_LENGTH = _OUTFEED_HEADER_LENGTH - 3 - 2
_OUTFEED_CONSUMER_ID_PRINT = 31

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


def _emit_outfeed(comp: XlaComputationBuilder, token: XlaOp,
                  consumer_id: int, arrays: Sequence[XlaOp], kwargs: Dict) -> XlaOp:
  """Emits the arrays to the outfeed for the current device.

  The consumer_id, arrays, and kwargs will be passed to the receiver.
  """
  arrays_shape = [comp.GetShape(a) for a in arrays]
  def _array_shape_to_tuple(a_shape: XlaShape):
    # (element_type_code, (d0, d1, ..., dn))
    return (_DTYPE_STR_TO_CODE[str(onp.dtype(a_shape.element_type()))],
            a_shape.dimensions())
  metadata = msgpack.dumps((tuple(map(_array_shape_to_tuple, arrays_shape)),
                            kwargs))
  metadata_len = len(metadata)
  if len(metadata) > 0xffff:
    raise ValueError("Outfeed metadata too long")
  metadatas = [metadata[i:i + _OUTFEED_HEADER_METADATA_LENGTH]
               for i in range(0, metadata_len, _OUTFEED_HEADER_METADATA_LENGTH)]
  for meta in metadatas:
    header = ((_OUTFEED_HEADER_START0, _OUTFEED_HEADER_START1,
               consumer_id,
              (metadata_len >> 8) & 0xff, metadata_len & 0xff) +
              tuple(meta))
    header += (0,) * (_OUTFEED_HEADER_LENGTH - len(header))
    # Encode as uint32
    header_uint32 = [int.from_bytes(header[i:i+4], byteorder="big")
                     for i in range(0, _OUTFEED_HEADER_LENGTH, 4)]
    data = xops.ConstantLiteral(comp, onp.array(header_uint32, dtype=onp.uint32))
    token = xops.OutfeedWithToken(data, token, comp.GetShape(data))

  # Now send the arrays
  for a, a_shape in zip(arrays, arrays_shape):
    token = xops.OutfeedWithToken(a, token, a_shape)
  return token

def _receive_outfeed(device: XlaDevice, receiver_name: str
                     ) -> Tuple[int, List, Dict]:
  """Receives a set of arrays on the outfeed for the specificied device.
  Args:
    receiver_name: a name used for debugging and logging
  Returns: a tuple with the consumer_id, the arrays received, and
    a kwargs dictionary that was passed to _emit_outfeed.
  """
  platform = xla_client.get_local_backend(None).platform
  header_shape = xla_client.Shape.array_shape(onp.dtype(onp.uint32),
                                              (_OUTFEED_HEADER_LENGTH // 4,))

  def _get_data(data_shape: XlaShape, device: XlaDevice) -> XlaShape:
    if platform in ("gpu", "cpu"):
      return xla_client.transfer_from_outfeed(data_shape, device)
    else:
      return xla_client.transfer_from_outfeed(
          xla_client.Shape.tuple_shape((data_shape,)), device)[0]

  metadatas: List[bytes] = []
  remaining_metadata_length = 0
  while(True):
    header_uint32 = _get_data(header_shape, device)
    header = [b for h in header_uint32
                for b in int(h).to_bytes(4, byteorder="big")]
    if header[0] != _OUTFEED_HEADER_START0 or header[1] != _OUTFEED_HEADER_START1:
      raise ValueError(f"Read unexpected outfeed header {header[0:2]} [{receiver_name}]")
    logging.info(f"[{receiver_name}:{device}] Outfeed read header: {header}")
    consumer_id = header[2]
    metadata_length = (header[3] << 8) + header[4]
    if not metadatas:  # First header packet
      remaining_metadata_length = metadata_length
    if remaining_metadata_length <= _OUTFEED_HEADER_METADATA_LENGTH:  # All here
      metadatas.append(bytes(header[5:5 + remaining_metadata_length]))
      break
    else:
      metadatas.append(bytes(header[5:5 + _OUTFEED_HEADER_METADATA_LENGTH]))
      remaining_metadata_length -= _OUTFEED_HEADER_METADATA_LENGTH

  array_descriptors, kwargs = msgpack.unpackb(b"".join(metadatas))
  arrays = []
  for a_descr in array_descriptors:
    a_shape = xla_client.Shape.array_shape(_CODE_TO_DTYPE[a_descr[0]],
                                           a_descr[1])
    data = _get_data(a_shape, device)
    logging.info(f"[{receiver_name}:{device}] Outfeed read data of shape "
                 f"{data.dtype}{data.shape}")
    arrays.append(data)
  return (consumer_id, arrays, kwargs)

def _id_print_translation_rule_outfeed(
    comp: XlaComputationBuilder,
    *args_op: XlaOp, **params):

  prev_token = xla.computation_state_carry.current_token(comp)
  nr_args_to_emit = len(args_op) - params.get("nr_results", 0)
  next_token = _emit_outfeed(comp, prev_token,
                             _OUTFEED_CONSUMER_ID_PRINT,
                             args_op[0:nr_args_to_emit], {})
  xla.computation_state_carry.set_current_token(comp, next_token)
  return xops.Tuple(comp, args_op)

xla.translations[id_print_p] = _id_print_translation_rule_outfeed

_END_PRINTING = onp.int32(987654321)
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
  # TODO: pass the backend?
  devices = api.devices()
  executor = futures.ThreadPoolExecutor(thread_name_prefix="outfeed",
                                        max_workers=len(devices))

  def device_receiver_loop(device: XlaDevice) -> XlaDevice:
    i = 0
    while (True):
      consumer_id, arrays, kwargs = _receive_outfeed(device, receiver_name)
      if consumer_id != _OUTFEED_CONSUMER_ID_PRINT:
        raise NotImplementedError(f"Encountered unexpected consumer {consumer_id}")
      for a in arrays:
        if not a.shape and a == _END_PRINTING:
          logging.info(f"[{receiver_name}:{device}] Outfeed received END_PRINTING")
          return device
        a_str = onp.array2string(a, threshold=1024)
        logging.info(f"[{receiver_name}:{device}] Outfeed received {i} "
                     f"({a.dtype}{a.shape}): {a_str}")
        if output_stream is not None:
          output_stream.write(a_str)
        i += 1

  receiver_futures = [executor.submit(device_receiver_loop, d) for d in devices]
  try:
    yield
  finally:
    for d in devices:
      api.jit(lambda x: id_print(_END_PRINTING, result=x), device=d)(0)
    for f in futures.as_completed(receiver_futures, timeout=timeout_sec):
      finished_device = f.result()
      logging.info(f"[{receiver_name}:{finished_device} Outfeed receiver finished")


def _id_print_jvp_rule(primals, tangents, **params):
  primals_out = id_print(primals, **params)
  tangents_out = id_print(tangents, **_expand_params_transform(params, "jvp"))
  return primals_out, tangents_out


ad.primitive_jvps[id_print_p] = _id_print_jvp_rule


def _id_print_transpose_rule(cts, *args, **params):
  assert all([ad.is_undefined_primal(x) for x in args])
  assert len(cts) == len(args)
  cts_zeros = [ad.instantiate_zeros_aval(a.aval, ct)
               for a, ct in zip(args, cts)]
  ct_args = id_print_p.bind(*cts_zeros,
                            **_expand_params_transform(params, "transpose"))
  return ct_args


ad.primitive_transposes[id_print_p] = _id_print_transpose_rule


def _id_print_batching_rule(batched_args, batch_dims, **params):
  res = id_print_p.bind(*batched_args,
                        **_expand_params_transform(params, "batch"))
  return res, batch_dims


batching.primitive_batchers[id_print_p] = _id_print_batching_rule
