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
"""Primitives for calling user-defined Python functions
on the host, even from compiled and transformed code.

**Experimental: please give feedback, and expect changes.**

Tapping works even for code executed on accelerators and
even for code under JAX transformations. A few examples::

  # calls func(2x) and returns 2x
  y = id_tap(func, x * 2)
  # calls func((2x, 3x)) and returns (2x, 3x)
  y, z = id_tap(func, (x * 2, x * 3))  # The argument can be a pytree
  # calls func(2x) and returns y
  y = id_tap(func, x * 2, result=y)
  # calls func(2x, what='x') and returns 2x
  y = id_tap(func, x * 2, what='x')
  # calls func(dict(x=x, y=y), what='foo') and returns dict(x=x, y=y)
  x, y = id_tap(func, dict(x=x, y=y), what='a dict')

The order of execution is by data dependency: after all the arguments are
computed and before the result is used. At least one of the returned values
must be used in the rest of the computation, or else this operation has no effect.

*At the moment*, in order to use the callback primitives in compiled code,
one must wrap any invocation with an :func:`outfeed_receiver` (an exception is
raised otherwise)::

  with outfeed_receiver():
     ...calls to compiled code that may invoke callback primitives...


The printing and the tap functions execute in separate threads that are started
by :func:`outfeed_receiver`. Exceptions from the primitives are printed along with
the traceback, but the execution does not stop until the body of the
:func:`outfeed_receiver` terminates and all the outfeeds are received.
At that point, a ``TapFunctionException`` is raised.

We describe the behaviour under transformations in the context of the
following function definition::

  def power3(x):
     y = x * x
     _, y = id_print(x, y, what="x,x^2")
     return y * x


For :func:`jax.vmap` the arguments are batched, ``vmap`` is appended
to ``transforms``, and `batch_dims` is added to specify the tuple of
batched dimensions::

  jax.vmap(power3)(np.arange(3.))
  # what=x,x^2 transforms=vmap batch_dims=(0, 0): ([0, 1, 2], [0, 1, 4])


For :func:`jax.jvp` there will be two callbacks, one with the values of the primals and
one with the tangents::

  jax.jvp(power3, (3.,), (0.1,))
  # what=x,x^2: (3., 9.)
  # what=x,x^2 transforms=jvp: (0.1, 0.6)

For :func:`jax.vjp` or :func:`jax.grad` there will be one callback with the values of
the adjoints for the arguments. You may also see a callback with the values of
the primals from the forward pass, if those values are needed for the
backward pass::

  jax.grad(power3)(3.)
  # what=x,x^2: (3., 9.)  # from forward pass, since y is needed in backward pass
  # what=x,x^2 transforms=(jvp, transpose): (0., 3.)  # from backward pass, adjoints of _, y

See documentation for :func:`id_tap` and :func:`id_print`.
For usage example, see tests/host_callback_test.py.

Still to do:
  * Performance tests.
  * Add flags for logging.
  * Add unit tests with mocks.
  * Improve the XLA compilation code.
  * Improve the ergonomics of starting the consumer loop. Currently, when
    invoking jit-ed code, one must start a consumer loop. This is not needed
    when invoking code that does not involve jit. There is an error when
    attempting to start a compiled computation without starting the outfeed
    receiver. Perhaps we can put the receiver threads in the runtime.
  * Explore a simpler API that uses Python program-order, instead of
    data dependency-order. Need to add support to JAX for stateful primitives.
  * Explore implementation with outside compilation.


"""
from collections import defaultdict, namedtuple
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
from jax.interpreters import ad, xla, batching, masking
from jax.interpreters import partial_eval as pe
from jax import pprint_util as ppu
from jax import util
from jaxlib import xla_client
from jaxlib import xla_extension

import logging
import msgpack  # type: ignore
import numpy as onp
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, NamedTuple, Sequence, Tuple

xops = xla_client._xla.ops

# TODO(necula): fix mypy errors if I define the type aliases below
XlaOp = Any  # xla_extension.XlaOp
XlaShape = Any # xla_client.Shape
XlaComputationBuilder = Any  # xla_bridge._JaxComputationBuilder
XlaDevice = Any  # xla_client.Device

# TODO: add a flag
_LOGGING = True

def id_tap(func: Callable, arg, *,
           result=None,
           **kwargs):
  """Host-callback tap primitive, like identity function with a call to ``func``.

**Experimental: please give feedback, and expect changes!**

``id_tap`` behaves semantically like the identity function but has the side-effect
that a user-defined Python function is called with the runtime values of the
argument.

Args:
  * arg: the argument passed to the tap function, can be a pytree of JAX
    types.
  * result: if given, then specifies the return value of ``id_tap``. By default,
    the return type is ``arg``.
  * kwargs: will be passed directly to the tap function. Can be anything,
    these are kept in the host Python process.

Returns:
  * the value of ``result`` or otherwise ``arg``

Tapping works even for code executed on accelerators and
even for code under JAX transformations.

For more details see the `module documentation <https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html>`_.
  """
  if func not in (_end_consumer, _unknown_consumer):
    api._check_callable(func)
  flat_args, arg_treedef = pytree.flatten(arg)
  api._check_args(flat_args)
  params = dict(kwargs)  #  we pass a copy of params to the primitive
  # See definition of id_tap_p for what parameters it takes
  params["func"] = func
  params["arg_treedef"] = arg_treedef
  if result is not None:
    flat_results, result_treedef = pytree.flatten(result)
    api._check_args(flat_results)
    params["nr_untapped"] = len(flat_results)
    all_args = flat_args + flat_results
  else:
    all_args = flat_args
  flat_outs = id_tap_p.bind(*all_args, **params)  # Always a tuple of all args
  if result is not None:
    return result_treedef.unflatten(flat_outs[-params["nr_untapped"]:])  # type: ignore[unsupported-operands]
  else:
    return arg_treedef.unflatten(flat_outs)


def id_print(arg, *, result=None, output_stream=None, threshold=None,
             **kwargs):
  """Like :func:`id_tap` with a printing tap function.

     **Experimental: please give feedback, and expect changes!**

     On each invocation of the printing tap, the ``kwargs`` if present
     will be printed first (sorted by keys). Then arg will be printed,
     with the arrays stringified with ``numpy.array2string``.

     See the :func:`id_tap` documentation.

     Additional keyword arguments:

     * ``output_stream`` if given then it will be used instead of the
       built-in ``print``. The string will be passed as ``output_stream.write(s)``.
     * ``threshold`` is passed to ``numpy.array2string``.
  """
  return id_tap(_print_consumer, arg,
                result=result, output_stream=output_stream,
                threshold=threshold, **kwargs)


# A registry of outfeed consumers
class _ConsumerCallable(NamedTuple):
  """Host-side information for a outfeed consumer."""
  func: Callable
  kwargs: Tuple[Tuple[str, Any], ...]
  arg_treedef: Any

_consumer_registry: Dict[_ConsumerCallable, int] = dict()
_consumer_registry_by_id: Dict[int, _ConsumerCallable] = dict()


def _register_consumer(cons: _ConsumerCallable) -> int:
  """Registers a tap function, cache by function identity"""
  cons_id = _consumer_registry.get(cons)
  if cons_id is not None:
    return cons_id
  cons_id = id(cons)
  _consumer_registry[cons] = cons_id
  _consumer_registry_by_id[cons_id] = cons
  return cons_id

def _print_consumer(arg, *, output_stream=None,
                    threshold=1024, **kwargs):
  """The consumer for id_print"""
  def emit_str(s: str):
    if output_stream is not None:
      output_stream.write(s + "\n")
    else:
      print(s)
  kv_pairs = " ".join([f"{k}: {v}"
                      for k, v in sorted(kwargs.items())
                      if k not in ("consumer_id", "nr_untapped")])
  if kv_pairs:
    emit_str(kv_pairs)

  def pp_val(arg) -> ppu.PrettyPrint:
    if isinstance(arg, (tuple, list)):
      return (ppu.pp('[ ') >>
              ppu.vcat([pp_val(e) for e in arg]) >> ppu.pp(' ]'))
    elif isinstance(arg, dict):
      return (ppu.pp('{ ') >>
              ppu.vcat(
                [ppu.pp(f"{k}=") >> pp_val(v)
                for k, v in sorted(arg.items())]) >>
              ppu.pp(' }'))
    elif isinstance(arg, onp.ndarray):
      return ppu.pp(onp.array2string(arg, threshold=threshold))
    else:
      return ppu.pp(str(arg))

  emit_str(str(pp_val(arg)))



"""The id_tap primitive acts like the identity function. It has a number of
positional arguments and parameters:
  * func: the actual (Python) function to invoke with the positional arguments
  and the parameters.
  * nr_untapped: how many positional arguments (from the tail) should not be
  passed to the tap function.
  * arg_treedef: the treedef of the tapped positional arguments
  * transforms: a tuple of the transformations that have been applied.
  * batch_dims: a tuple of the dims that have been batched, for vmap
  * logical_shapes: a tuple of evaluated logical shapes, for mask

  * the remaining parameters are passed to the tap function.
"""
# TODO: handle multiple vmap and mask
id_tap_p = core.Primitive("id_tap")
id_tap_p.multiple_results = True
xla.outfeed_primitives.add(id_tap_p)


def _add_transform_name(params: Dict, transform: str) -> Dict:
  """Adds the `transform` to the params["transforms"]."""
  return dict(params, transforms=params.get("transforms", ()) + (transform,))


def _id_tap_impl(*arrays, func=None, nr_untapped=0, arg_treedef=None,
                 **params):
  assert isinstance(func, Callable)
  func_params = dict(params)
  # TODO: consolidate logic with the outfeed receiver
  try:
    assert nr_untapped <= len(arrays)
    func_arrays = arrays[:-nr_untapped] if nr_untapped > 0 else arrays
    arg = api.tree_unflatten(arg_treedef, func_arrays)
    func(arg, **func_params)
  except Exception as e:
    raise TapFunctionException from e
    # We continue for now, we need to keep reading the outfeed
  return arrays  # return all

id_tap_p.def_impl(_id_tap_impl)

def _id_tap_abstract_eval(*args_a: pe.AbstractValue, **params) \
    -> Sequence[pe.AbstractValue]:
  return args_a


id_tap_p.def_abstract_eval(_id_tap_abstract_eval)

def _instantiate_zeros(tan, arg):
  """Turn special ad.zero tangents into arrays of 0s."""
  if tan is not ad.zero:
    return tan
  elif isinstance(arg, core.Tracer):
    # TODO: why do I have to do this to get a zero?
    try:
      aval = arg.aval
      return ad.instantiate_zeros_aval(aval, tan)
    except:
      # It seems that we get here for ConcreteArray
      return ad.instantiate_zeros(arg, tan)

def _id_tap_jvp_rule(primals, tangents, *, func, nr_untapped=0, **params):
  # Put primals through id_tap separately, so that partial evaluation
  # can do its job for grad
  out_primals = id_tap_p.bind(*primals, func=func, nr_untapped=nr_untapped, **params)
  # Add one primal output as untapped, to create dependency.
  tangent_zeros = tuple(map(_instantiate_zeros, tangents, primals))
  out_tangents_extra = id_tap_p.bind(*tangent_zeros, out_primals[0],
                                     func=func, nr_untapped=nr_untapped + 1,
                                     **_add_transform_name(params, "jvp"))
  return tuple(out_primals), tuple(out_tangents_extra[:-1])


ad.primitive_jvps[id_tap_p] = _id_tap_jvp_rule


def _id_tap_transpose_rule(cts, *args, func=None, nr_untapped=0, **params):
  assert len(cts) == len(args)
  cts_zeros = tuple(map(_instantiate_zeros, cts, args))
  ct_args = id_tap_p.bind(*cts_zeros, func=func, nr_untapped=nr_untapped,
                          **_add_transform_name(params, "transpose"))
  return ct_args


ad.primitive_transposes[id_tap_p] = _id_tap_transpose_rule


def _id_tap_batching_rule(batched_args, batch_dims, **params):
  new_params = _add_transform_name(params, "batch")
  new_params["batch_dims"] = batch_dims
  res = id_tap_p.bind(*batched_args, **new_params)
  return res, batch_dims


batching.primitive_batchers[id_tap_p] = _id_tap_batching_rule

# def _id_tap_shape_rule(*operands, **params):
#  return tuple([op.shape for op in operands])

# TODO: these disappeared
# masking.shape_rules[id_tap_p] = _id_tap_shape_rule  # type: ignore[module-attr]

def _id_tap_masking_rule(operands, operands_logical_shapes, **params):
  new_params = _add_transform_name(params, "mask")
  new_params["logical_shapes"] = operands_logical_shapes
  return id_tap_p.bind(*operands, **new_params)


masking.masking_rules[id_tap_p] = _id_tap_masking_rule

#### XLA compilation ####
# Special consumer to mark the end of outfeed stream for a device
_end_consumer = 0
_unknown_consumer = 1  # for testing error cases


def _id_print_translation_rule_outfeed(comp: XlaComputationBuilder,
                                       *args_op: XlaOp, func=None,
                                       nr_untapped=0, arg_treedef=None,
                                       **params):
  params = dict(params)
  if func is _end_consumer:
    params["consumer_id"] = _end_consumer
  elif func is _unknown_consumer:
    params["consumer_id"] = _unknown_consumer  # Will trigger an error, for testing
  else:
    params["consumer_id"] = _register_consumer(
      _ConsumerCallable(func, tuple(params.items()), arg_treedef))

  prev_token = xla.state_carry.current_token(comp)
  nr_args_to_emit = len(args_op) - nr_untapped
  next_token = _emit_outfeed(comp, prev_token,
                             args_op[0:nr_args_to_emit], params["consumer_id"])
  xla.state_carry.set_current_token(comp, next_token)
  return xops.Tuple(comp, args_op)

xla.translations[id_tap_p] = _id_print_translation_rule_outfeed


# The data on the outfeed follows a protocol that allows multiplexing the
# outfeed among multiple consumers, and communicates in-stream shape and
# type of the data.
# Each batch of array data is preceeded by a header message, of type
# uint32[_OUTFEED_HEADER_LENGTH]:
#  [0]: special header value 2178
#  [1, 2]: a consumer id (64-bits, big-endian encoding as uint32[2]). The
#       consumer id encodes the tap function (by id), the
#       descriptor of the arrays to be outfed, and the kwargs (a sorted tuple
#       of keys and values).
#  [3]: the metadata length in bytes. The metadata is a msgpack-encoded value of type:
#     [ (type_code, (d0, d1, ...)), ...]  # for each array, element type code
#                                         # and the dimensions.
#       padded with 0s to _OUTFEED_HEADER_LENGTH
#
#
_OUTFEED_HEADER_LENGTH = 32  # In uint32 words
_OUTFEED_HEADER_START  = 2178   # [0]
# consumer_id                     [1, 2]
# metadata_length in bytes        [3]
_OUTFEED_HEADER_METADATA_LENGTH = 4 * (_OUTFEED_HEADER_LENGTH - 4)

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
                  arrays: Sequence[XlaOp], consumer_id: int) -> XlaOp:
  """Emits the arrays to the outfeed for the current device.

  The kwargs must have at least "consumer_id" key.
  """
  arrays_shape = [comp.GetShape(a) for a in arrays]
  def _array_shape_to_tuple(a_shape: XlaShape):
    # (element_type_code, (d0, d1, ..., dn))
    return (_DTYPE_STR_TO_CODE[str(onp.dtype(a_shape.element_type()))],
            a_shape.dimensions())
  metadata = msgpack.dumps(tuple(map(_array_shape_to_tuple, arrays_shape)))
  metadata_len = len(metadata)
  if metadata_len > _OUTFEED_HEADER_METADATA_LENGTH:
    # TODO: configurable
    raise ValueError("Outfeed metadata too long")
  metadata += b" " * (((metadata_len + 3) // 4) * 4 - metadata_len)  # pad
  header = ((_OUTFEED_HEADER_START,
             (consumer_id >> 32) & 0xffffffff, (consumer_id & 0xffffffff),
             metadata_len) +
             tuple([int.from_bytes(metadata[i:i+4], byteorder="big")
                    for i in range(0, _OUTFEED_HEADER_METADATA_LENGTH, 4)]))
  header += (0,) * (_OUTFEED_HEADER_LENGTH - len(header))
  data = xops.ConstantLiteral(comp, onp.array(header, dtype=onp.uint32))
  token = xops.OutfeedWithToken(data, token, comp.GetShape(data))

  # Now send the arrays
  entire_shape = xla_client.Shape.tuple_shape(arrays_shape)
  token = xops.OutfeedWithToken(xops.Tuple(comp, arrays), token, entire_shape)
  return token

def _receive_outfeed(device: XlaDevice, receiver_name: str
                     ) -> Tuple[int, List]:
  """Receives a set of arrays on the outfeed for the specificied device.
  Args:
    receiver_name: a name used for debugging and logging
  Returns: a tuple with the consumer_id, the arrays received, and
    a kwargs dictionary that was passed to _emit_outfeed.
  """
  platform = xla_client.get_local_backend(None).platform
  header_shape = xla_client.Shape.array_shape(onp.dtype(onp.uint32),
                                              (_OUTFEED_HEADER_LENGTH,))

  def _get_data(data_shape: XlaShape, device: XlaDevice) -> XlaShape:
    return xla_client.transfer_from_outfeed(data_shape, device)

  header = _get_data(header_shape, device)
  if header[0] != _OUTFEED_HEADER_START:
    raise ValueError(f"Read unexpected outfeed header {header[0]} [{receiver_name}]")
  logging.info(f"[{receiver_name}:{device}] Outfeed read header: {header}")
  consumer_id = (header[1] << 32) + header[2]
  metadata_length = header[3]
  assert metadata_length <= _OUTFEED_HEADER_METADATA_LENGTH
  metadatas = [int(header[i]).to_bytes(4, byteorder="big")
               for i in range(4, 4 + (metadata_length + 3) // 4)]
  metadata = b"".join(metadatas)[:metadata_length]
  array_descriptors = msgpack.unpackb(metadata)
  arrays_shape = [xla_client.Shape.array_shape(_CODE_TO_DTYPE[a_descr[0]],
                                               a_descr[1])
                  for a_descr in array_descriptors]
  entire_shape = xla_client.Shape.tuple_shape(arrays_shape)
  arrays = _get_data(entire_shape, device)
  logging.info(f"[{receiver_name}:{device}] Outfeed read data of shape "
               ",".join([f"{data.dtype}{data.shape}" for data in arrays]))
  return (consumer_id, arrays)


class TapFunctionException(Exception):
  """Signals that some tap function had exceptions.

  Raised by :func:`outfeed_receiver`.
  """
  pass

@contextmanager
def outfeed_receiver(*,
                     timeout_sec=10,
                     backends: Optional[Sequence[str]] = None,
                     devices: Optional[Sequence[XlaDevice]] = None,
                     receiver_name=""):
  # TODO: better timeout management.
  # TODO: prevent multiple consumers.
  """Starts receivers for the :func:`id_tap` outfeed from several devices.

  The receivers will run in a threadpool. The tapped functions will be invoked
  in those threads. If a tap function raises an exception, an error is
  printed, but the receving continues until the body of the context manager
  terminates and all outfeeds from all devices have been received. Only then
  will a :exc:`TapFunctionException` be raised.

  Args:
    backends: (optional) sequence of backend names for which to listen.
      Will listen to all devices on those backends. By default, listed to
      all devices on all known backends.
    devices: (optional) sequence of devices to listed to. At most one
      of `backends` or `devices` must be given.
    receiver_name: (optional) a name to use with debug logging
  Usage::

    with outfeed_receiver():
      jax.jit(func)(args)
      ...
      jax.pmap(another_func)(args)

  The ``outfeed_receiver`` must be started outside any jitted computation.

  """
  if not devices:
    backends = backends or xla_client._get_local_backends().keys()
    devices = tuple(itertools.chain(*[api.devices(backend)
                                      for backend in backends]))
  else:
    if backends:
      raise ValueError("At most one of `devices` or `backends` must be given.")
  executor = futures.ThreadPoolExecutor(
    thread_name_prefix=f"outfeed_receiver_{receiver_name}",
    max_workers=len(devices))

  count_tap_exceptions = 0
  def device_receiver_loop(device: XlaDevice) -> XlaDevice:
    """Polls the outfeed for a device in a loop."""
    nonlocal count_tap_exceptions
    while (True):
      consumer_id, arrays = _receive_outfeed(device, receiver_name)
      if _LOGGING:
        logging.info(f"[{receiver_name}:{device}] Outfeed received for consumer {consumer_id} " +
                     (" ".join([f"({a.dtype}{a.shape})" for a in arrays])))
      if consumer_id == _end_consumer:
        assert not arrays
        if _LOGGING:
          logging.info(f"[{receiver_name}:{device}] Outfeed received END_OUTFEED")
        return device
      consumer = _consumer_registry_by_id.get(consumer_id)
      if consumer is None:
        logging.error(f"Ignoring received outfeed for unknown tap consumer")
        count_tap_exceptions += 1
        continue  # We need to read the entire outfeed
      try:
        arg = api.tree_unflatten(consumer.arg_treedef, arrays)
        consumer.func(arg, **dict(consumer.kwargs))  # type: ignore[attribute-error]
      except Exception as e:
        logging.error(f"Postponing exception raised in tap function: {str(e)}\n{traceback.format_exc()}")
        count_tap_exceptions += 1
        # We continue for now, we need to keep reading the outfeed

  receiver_futures = [executor.submit(device_receiver_loop, d) for d in devices]
  # Register a callback to raise errors if any. These exception come from
  # bugs in our code, not from the tap functions.
  for rf in receiver_futures:
    rf.add_done_callback(lambda rf: rf.result())
  xla.set_outfeed_allowed(True)
  try:
    yield
  finally:
    for d in devices:  # Signal the end of printing
      api.jit(lambda x: id_tap(_end_consumer, None, result=x), device=d)(0)  # type: ignore[arg-type]
    xla.set_outfeed_allowed(False)
    for f in futures.as_completed(receiver_futures, timeout=timeout_sec):
      finished_device = f.result()  # Throw exceptions here
      if _LOGGING:
        logging.info(f"[{receiver_name}:{finished_device} Outfeed receiver finished")
    if count_tap_exceptions > 0:
      raise TapFunctionException

