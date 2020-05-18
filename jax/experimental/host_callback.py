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

Host callbacks work even for code executed on accelerators and
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

**At the moment**, in order to use the callback primitives one must wrap any
code that uses them with an :func:`outfeed_receiver` (an error is
raised otherwise)::

  with outfeed_receiver():
     id_print(x)
     jax.jit(func_with_taps)(x)

The printing and the tap functions execute in separate threads that are started
by :func:`outfeed_receiver`. There is one thread per device. This ensures
that the outfeed from a certain device will come in order. Exceptions from
the user-define tap functions are printed along with the traceback, but the
outfeed listening does not stop until the body of the
:func:`outfeed_receiver` terminates and all the outfeeds are received.
At that point, a ``TapFunctionException`` is raised if there was an exception
in one of the tap functions.

**We intend to implement an alternative outfeed receiving mechanism that will
not require the user to start the `outfeed_receiver`.**

The current implementation uses the outfeed mechanism provided by XLA. The
mechanism itself is quite primitive in the sense that a receiver must know
exactly the shape of each incoming packet, and how many packets are expected.
This makes it hard to use for multiple kinds of data in the same computation,
and it is practically impossible to use it under conditionals.

The tapping API introduced here makes it easy to share the outfeed mechanism
for multiple purposes. Instead of using it directly, just use :func:`id_tap`.

We describe the behaviour under transformations in the context of the
following function definition::

  def power3(x):
     y = x * x
     _, y = id_print(x, y, what="x,x^2")
     return y * x


For :func:`jax.vmap` the arguments are batched, ``('batch', batch_dims)`` is
appended to ``transforms``, where ``batch_dims`` specifies the tuple of
batched dimensions::

  jax.vmap(power3)(np.arange(3.))
  # what=x,x^2 transforms=(batch, (0, 0)): ([0, 1, 2], [0, 1, 4])


For :func:`jax.jvp` there will be two callbacks, one with the values of
the primals and one with the tangents::

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
  * Improve the ergonomics of starting the consumer loop. Currently, when
    invoking jit-ed code, one must start a consumer loop. This is not needed
    when invoking code that does not involve jit. There is an error when
    attempting to start a compiled computation without starting the outfeed
    receiver. Perhaps we can put the receiver threads in the runtime.
  * Explore a simpler API that uses Python program-order, instead of
    data dependency-order.
  * Explore implementation with outside compilation.

"""
from concurrent import futures
from contextlib import contextmanager
import itertools

from jax import api
from jax import core
from jax import dtypes
from jax import lax
from jax.lib import pytree
from jax.interpreters import ad, xla, batching, masking
from jax.interpreters import partial_eval as pe
from jax import pprint_util as ppu
from jax import util
from jaxlib import xla_client
from jaxlib import xla_extension
from jaxlib import version as jaxlib_version

import logging
import msgpack  # type: ignore
import numpy as np
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, NamedTuple, Sequence, Tuple

xops = xla_client._xla.ops

# TODO(necula): fix mypy errors if I define the type aliases below
XlaOp = Any  # xla_extension.XlaOp
XlaShape = Any # xla_client.Shape
XlaComputationBuilder = Any  # xla_bridge._JaxComputationBuilder
XlaDevice = Any  # xla_client.Device

# TODO: add a flag
_LOGGING = True

# Starting with 0.1.46 the outfeed is now on the Device
# TODO(necula): remove once we fix XLA
_jaxlib_version = tuple(int(x)for x in jaxlib_version.__version__.split('.'))
if _jaxlib_version == (0, 1, 45):
  _OUTFEED_MECHANISM = "xla_client"
elif _jaxlib_version >= (0, 1, 47):
  _OUTFEED_MECHANISM = "device"
else:
  _OUTFEED_MECHANISM = "none"


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
    * result: if given, specifies the return value of ``id_tap``. By default,
      the return type is ``arg``.
    * kwargs: will be passed directly to the tap function. Can be anything that
      is hashable, these are kept in the host Python process until outfeeds are
      received.

  Returns:
    * ``arg``, or ``result`` if given.

  Tapping works even for code executed on accelerators and even for code under
  JAX transformations. Code that uses taps must be run embedded in
  :func:`outfeed_receiver`.

  For more details see the
  `module documentation <https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html>`_.
  """
  if _OUTFEED_MECHANISM == "none":
    raise NotImplementedError("id_tap works only with jaxlib 0.1.47 and higher")
  if func not in (_end_consumer, _unknown_testing_consumer):
    api._check_callable(func)
  flat_args, arg_treedef = pytree.flatten(arg)
  for arg in flat_args: api._check_arg(arg)
  params = dict(kwargs)  #  we pass a copy of params to the primitive
  # See definition of id_tap_p for what parameters it takes
  params["func"] = func
  params["arg_treedef"] = arg_treedef
  if result is not None:
    flat_results, result_treedef = pytree.flatten(result)
    for result in flat_results: api._check_arg(result)
    all_args = flat_args + flat_results
    params["nr_untapped"] = len(flat_results)
  else:
    all_args = flat_args
  flat_outs = id_tap_p.bind(*all_args, **params)  # Returns all_args
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


# A registry of outfeed consumers, used upon receiving outfeeds
class _ConsumerCallable(NamedTuple):
  """Host-side information for an outfeed consumer."""
  func: Callable
  kwargs: Tuple[Tuple[str, Any], ...]
  arg_treedef: Any

_consumer_registry: Dict[_ConsumerCallable, int] = dict()
_consumer_registry_by_id: Dict[int, _ConsumerCallable] = dict()


def _register_consumer(cons: _ConsumerCallable) -> int:
  """Registers a tap function, cache by hash of cons."""
  cons_id = _consumer_registry.get(cons)
  if cons_id is not None:
    return cons_id
  cons_id = hash(cons)
  _consumer_registry[cons] = cons_id
  _consumer_registry_by_id[cons_id] = cons
  return cons_id

def _print_consumer(arg, *, output_stream=None,
                    threshold=1024, **kwargs):
  """The consumer for id_print.

  We provide this as a simple tapping function for printing.
  This is **experimental**
  and may not want to add many features to it; it should be easy for the user
  to roll their own printing function.

  Args:
    output_stream: a function whose `write` method is called with the strings
      to be output.
    threshold: the value of numpy.array2string threshold parameter.
    **kwargs: all other keyword args are printed before printing `arg`.
  """
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
    elif isinstance(arg, np.ndarray):
      return ppu.pp(np.array2string(arg, threshold=threshold))
    else:
      return ppu.pp(str(arg))

  emit_str(str(pp_val(arg)))


"""The id_tap primitive acts like the identity function. It has a number of
positional arguments and parameters:
  * func: the actual (Python) function to invoke with the positional arguments
  and the parameters.
  * nr_untapped: how many positional arguments (from the tail) should not be
  passed to the tap function.
  * arg_treedef: the treedef of the tapped positional arguments.
  * transforms: a tuple of the transformations that have been applied. Most
    elements of this tuple are a string naming the transformations. For some
    transformation, the elements are a tuple of the name of the transformation
    and the parameters. For example, for vmap, the parameters are the 
    dimensions that have been batched. Also, for mask, the parameter is the 
    logical shapes. 

  * the remaining parameters are passed to the tap function.
"""
id_tap_p = core.Primitive("id_tap")
id_tap_p.multiple_results = True
xla.outfeed_primitives.add(id_tap_p)

def _add_transform_name(params: Dict, transform: str,
                        transform_params: Any = None) -> Dict:
  """Adds the `transform` to the params["transforms"]."""
  if transform_params is None:
    new_transform = transform
  else:
    new_transform = (transform, transform_params)
  return dict(params, transforms=(
      params.get("transforms", ()) + (new_transform,)))


def _id_tap_impl(*arrays, **params):
  # We use the jitted-version of the primitive even for eager execution, both
  # so that we do not duplicate logic, but also so that all outfeed is received
  # by the outfeed_listeners, in the same thread from a given device. If we were
  # to process the tap here, it would be coming from the main thread. Also,
  # even in eager execution some primitives, such as while, are compiled.
  # It would be confusing to process a sequence "id_tap; while" in two
  # different threads.
  return xla.apply_primitive(id_tap_p, *arrays, **params)

id_tap_p.def_impl(_id_tap_impl)

def _id_tap_abstract_eval(*args_a: pe.AbstractValue, **params) \
    -> Sequence[pe.AbstractValue]:
  return args_a


id_tap_p.def_abstract_eval(_id_tap_abstract_eval)

# TODO(necula): there must be a better way to do this.
# The AttributeError is for regular values, the KeyError is for ConcreteArray
def _instantiate_zeros(arg, tan):
  """Turn special ad.zero tangents into arrays of 0s."""
  if tan is not ad.zero:
    return tan

  try:
    aval = arg.aval
    return ad.instantiate_zeros_aval(aval, tan)
  except (AttributeError, KeyError):
    # We get here for regular Python values
    return ad.zeros_like_jaxval(arg)


def _id_tap_jvp_rule(primals, tangents, *, func, nr_untapped=0, **params):
  # Put primals through id_tap separately, so that partial evaluation
  # can do its job when they are known (for grad)
  out_primals = id_tap_p.bind(*primals, func=func, nr_untapped=nr_untapped, **params)
  # Add one primal output as untapped, to create data dependency.
  tangent_zeros = tuple(map(_instantiate_zeros, primals, tangents))
  out_tangents_extra = id_tap_p.bind(*tangent_zeros, out_primals[0],
                                     func=func, nr_untapped=nr_untapped + 1,
                                     **_add_transform_name(params, "jvp"))
  return tuple(out_primals), tuple(out_tangents_extra[:-1])

ad.primitive_jvps[id_tap_p] = _id_tap_jvp_rule


def _id_tap_transpose_rule(cts, *args, func=None, nr_untapped=0, **params):
  assert len(cts) == len(args)
  cts_zeros = tuple(map(_instantiate_zeros, args, cts))
  ct_args = id_tap_p.bind(*cts_zeros, func=func, nr_untapped=nr_untapped,
                          **_add_transform_name(params, "transpose"))
  return ct_args

ad.primitive_transposes[id_tap_p] = _id_tap_transpose_rule


def _id_tap_batching_rule(batched_args, batch_dims, **params):
  new_params = _add_transform_name(params, "batch", batch_dims)
  res = id_tap_p.bind(*batched_args, **new_params)
  return res, batch_dims

batching.primitive_batchers[id_tap_p] = _id_tap_batching_rule

# def _id_tap_shape_rule(*operands, **params):
#  return tuple([op.shape for op in operands])

# TODO(necula): these disappeared
# masking.shape_rules[id_tap_p] = _id_tap_shape_rule  # type: ignore[module-attr]

def _id_tap_masking_rule(operands, operands_logical_shapes, **params):
  new_params = _add_transform_name(params, "mask",
                                   operands_logical_shapes)
  return id_tap_p.bind(*operands, **new_params)

masking.masking_rules[id_tap_p] = _id_tap_masking_rule

####
#### XLA compilation ####
####
# Special consumer to mark the end of outfeed stream for a device
_end_consumer = 0
_unknown_testing_consumer = 1  # for testing error cases


def _id_tap_translation_rule_outfeed(comp: XlaComputationBuilder,
                                     *args_op: XlaOp, func=None,
                                     nr_untapped=0, arg_treedef=None,
                                     **params):
  params = dict(params)
  if func in (_end_consumer, _unknown_testing_consumer):
    params["consumer_id"] = func
  else:
    params["consumer_id"] = _register_consumer(
      _ConsumerCallable(func, tuple(sorted(params.items())), arg_treedef))

  # We expect the current token at the end, inserted by _rewrite_jaxpr.
  current_token = args_op[-1]
  assert not comp.get_shape(current_token).is_array(), "The last argument must be a token"

  nr_args_to_emit = len(args_op) - nr_untapped - 1
  next_token = _emit_outfeed(comp, current_token,
                             args_op[0:nr_args_to_emit], params["consumer_id"])
  results = (args_op[:-1] + (next_token,))
  return xops.Tuple(comp, results)

xla.translations[id_tap_p] = _id_tap_translation_rule_outfeed

####
#### Jaxpr rewriting logic to thread the tokens through stateful primitives.
####
def _jaxpr_var_defs(jaxpr: core.Jaxpr) -> Iterable[int]:
  """Iterates over all the vars defined at the top-level of a Jaxpr"""
  for iv in jaxpr.invars:
    yield iv.count
  for cv in jaxpr.constvars:
    yield cv.count
  for eqn in jaxpr.eqns:
    for ov in eqn.outvars:
      yield ov.count

def _rewrite_typed_jaxpr(tjaxpr: core.TypedJaxpr,
                         has_input_token: bool,
                         has_output_token: bool) -> Tuple[core.TypedJaxpr, bool]:
  """Rewrites a TypedJaxpr to thread the token, if needed.

  Returns the rewritten Jaxpr, and whether it uses outfeed."""
  new_jaxpr, uses_outfeed = _rewrite_jaxpr(tjaxpr.jaxpr, has_input_token,
                                           has_output_token)
  return (core.TypedJaxpr(new_jaxpr, tjaxpr.literals,
                         tuple(map(lambda v: v.aval, new_jaxpr.invars)),
                         tuple(map(lambda v: v.aval, new_jaxpr.outvars))),
          uses_outfeed)


def _rewrite_jaxpr(jaxpr: core.Jaxpr,
                   has_input_token: bool,
                   has_output_token: bool) -> Tuple[core.Jaxpr, bool]:
  """Rewrite a Jaxpr to thread the token, if needed."""
  assert has_input_token or not has_output_token

  if not has_input_token and not xla.jaxpr_uses_outfeed(jaxpr):
    return (jaxpr, False)
  max_var_count = max(_jaxpr_var_defs(jaxpr))
  mk_new_id = itertools.count(start=max_var_count + 1)

  def mk_new_var(aval: core.AbstractValue) -> core.Var:
    return core.Var(next(mk_new_id), '', aval)

  eqns: List[core.JaxprEqn] = []
  last_token_var = mk_new_var(core.abstract_token)  # store the incoming token
  if has_input_token:
    invars = jaxpr.invars + [last_token_var]
  else:
    invars = jaxpr.invars
    eqns.append(core.new_jaxpr_eqn([jaxpr.invars[0]], [last_token_var],
                                   lax.create_token_p, {}))

  for eqn in jaxpr.eqns:
    if not xla.primitive_uses_outfeed(eqn.primitive, eqn.params):
      eqns.append(eqn)
    else:
      output_token_var = mk_new_var(core.abstract_token)
      _rewrite_eqn(eqn, eqns, last_token_var, output_token_var)
      last_token_var = output_token_var

  outvars = jaxpr.outvars + ([last_token_var] if has_output_token else [])
  return (core.Jaxpr(jaxpr.constvars, invars, outvars, eqns), True)

def _rewrite_eqn(eqn: core.JaxprEqn,
                 eqns: List[core.JaxprEqn],
                 input_token_var: core.Var,
                 output_token_var: core.Var):
  """Rewrite an `eqn` and append equations to `eqns`. Assume that the
  current token is in `input_token_var` and the resulting token must end in
  `output_token_var`."""
  if eqn.primitive is id_tap_p:
    eqns.append(core.new_jaxpr_eqn(eqn.invars + [input_token_var],
                                   eqn.outvars + [output_token_var],
                                   eqn.primitive, eqn.params))
  elif eqn.primitive is lax.while_p:
    cond_jaxpr, cond_nconsts, body_jaxpr, body_nconsts = util.split_dict(
      eqn.params, ["cond_jaxpr", "cond_nconsts", "body_jaxpr", "body_nconsts"])
    if xla.jaxpr_uses_outfeed(cond_jaxpr.jaxpr):
      # TODO(necula): implement tapping from the conditional of a while
      raise NotImplementedError("outfeed not supported in the conditional of a while")

    eqns.append(core.new_jaxpr_eqn(
      eqn.invars + [input_token_var],
      eqn.outvars + [output_token_var],
      eqn.primitive,
      dict(eqn.params,
           body_jaxpr=_rewrite_typed_jaxpr(body_jaxpr, True, True)[0],
           cond_jaxpr=_rewrite_typed_jaxpr(cond_jaxpr, True, False)[0])))
  elif eqn.primitive is lax.cond_p:
    true_jaxpr, false_jaxpr, linear = util.split_dict(
      eqn.params, ["true_jaxpr", "false_jaxpr", "linear"])
    nr_operands = len(true_jaxpr.jaxpr.invars)
    pred, *operands = eqn.invars
    new_invars = (pred, *operands, input_token_var)
    eqns.append(core.new_jaxpr_eqn(
      new_invars, eqn.outvars + [output_token_var],
      eqn.primitive,
      dict(eqn.params,
           true_jaxpr=_rewrite_typed_jaxpr(true_jaxpr, True, True)[0],
           false_jaxpr=_rewrite_typed_jaxpr(false_jaxpr, True, True)[0],
           linear=(*linear, False))))
  elif eqn.primitive is lax.scan_p:
    num_consts, num_carry, carry_jaxpr, linear, _, _ = util.split_dict(
      eqn.params, ["num_consts", "num_carry", "jaxpr", "linear",
                   "reverse", "length"])
    # We add the token right at the end of carry
    nr_const_and_carry = num_consts + num_carry
    new_invars = eqn.invars[0:nr_const_and_carry] + [input_token_var] + eqn.invars[nr_const_and_carry:]
    new_jaxpr = _rewrite_typed_jaxpr(carry_jaxpr, True, True)[0]
    # The rewrite has put the token at end, it has to be at end of carry
    new_jaxpr_invars = new_jaxpr.jaxpr.invars
    new_jaxpr_invars = (new_jaxpr_invars[0:nr_const_and_carry] +
                        [new_jaxpr_invars[-1]] +
                        new_jaxpr_invars[nr_const_and_carry:-1])
    new_jaxpr.jaxpr.invars = new_jaxpr_invars
    new_jaxpr.in_avals = [v.aval for v in new_jaxpr_invars]

    new_jaxpr_outvars = new_jaxpr.jaxpr.outvars
    new_jaxpr_outvars = (new_jaxpr_outvars[0:num_carry] +
                         [new_jaxpr_outvars[-1]] +
                         new_jaxpr_outvars[num_carry:-1])
    new_jaxpr.jaxpr.outvars = new_jaxpr_outvars
    new_jaxpr.out_avals = [v.aval for v in new_jaxpr_outvars]
    eqns.append(core.new_jaxpr_eqn(
      new_invars,
      # Output token is at the end of carry result
      eqn.outvars[0:num_carry] + [output_token_var] + eqn.outvars[num_carry:],
      eqn.primitive,
      dict(eqn.params,
           jaxpr=new_jaxpr,
           num_carry=num_carry + 1,
           linear=linear + (False,))))
  elif eqn.primitive is xla.xla_call_p:
    call_jaxpr = eqn.params["call_jaxpr"]
    eqns.append(core.new_jaxpr_eqn(
      eqn.invars + [input_token_var],
      eqn.outvars + [output_token_var],
      eqn.primitive,
      dict(eqn.params, call_jaxpr=_rewrite_jaxpr(call_jaxpr, True, True)[0])))
  else:
    raise NotImplementedError(f"outfeed rewrite {eqn.primitive}")

xla.outfeed_rewriter = lambda j: _rewrite_jaxpr(j, False, False)

# The data on the outfeed follows a protocol that allows multiplexing the
# outfeed among multiple consumers, and communicates in-stream shape and
# type of the data.
# Each batch of array data is preceeded by a header message, of type
# uint32[_OUTFEED_HEADER_LENGTH]:
#  [0]: special header value (271828)
#  [1, 2]: a consumer id (64-bits, big-endian encoding as uint32[2]). The
#       consumer id encodes the tap function (by hash), the
#       descriptor of the arrays to be outfed, and the kwargs (a sorted tuple
#       of keys and values).
#  [3]: the metadata length in bytes. The metadata is a msgpack-encoded value of type:
#     [ (type_code, (d0, d1, ...)), ...]  # for each array, element type code
#                                         # and the dimensions.
#       padded with 0s to _OUTFEED_HEADER_LENGTH
#
#
_OUTFEED_HEADER_LENGTH = 32  # In uint32 words
_OUTFEED_HEADER_START  = 271828   # [0]
# consumer_id                       [1, 2]
# metadata_length in bytes          [3]
_OUTFEED_HEADER_METADATA_LENGTH = 4 * (_OUTFEED_HEADER_LENGTH - 4)

_CODE_TO_DTYPE = {
  0: np.dtype(np.int8),
  1: np.dtype(np.int16),
  2: np.dtype(np.int32),
  3: np.dtype(np.int64),
  4: np.dtype(np.uint8),
  5: np.dtype(np.uint16),
  6: np.dtype(np.uint32),
  7: np.dtype(np.uint64),
  8: np.dtype(np.float16),
  9: np.dtype(np.float32),
  10: np.dtype(np.float64),
  11: np.dtype(dtypes.bfloat16),
}
_DTYPE_STR_TO_CODE = dict([(str(d), c) for c, d in _CODE_TO_DTYPE.items()])


def _emit_outfeed(comp: XlaComputationBuilder, token: XlaOp,
                  arrays: Sequence[XlaOp], consumer_id: int) -> XlaOp:
  """Emits the arrays to the outfeed for the current device."""
  arrays_shape = [comp.get_shape(a) for a in arrays]
  def _array_shape_to_tuple(a_shape: XlaShape):
    # (element_type_code, (d0, d1, ..., dn))
    return (_DTYPE_STR_TO_CODE[str(np.dtype(a_shape.element_type()))],
            a_shape.dimensions())
  metadata = msgpack.dumps(tuple(map(_array_shape_to_tuple, arrays_shape)))
  metadata_len = len(metadata)
  if metadata_len > _OUTFEED_HEADER_METADATA_LENGTH:
    # TODO(necula): configurable metadata length
    raise ValueError("Outfeed metadata too long")
  metadata += b" " * (((metadata_len + 3) // 4) * 4 - metadata_len)  # pad
  header = ((_OUTFEED_HEADER_START,
             (consumer_id >> 32) & 0xffffffff, (consumer_id & 0xffffffff),
             metadata_len) +
             tuple([int.from_bytes(metadata[i:i+4], byteorder="big")
                    for i in range(0, _OUTFEED_HEADER_METADATA_LENGTH, 4)]))
  header += (0,) * (_OUTFEED_HEADER_LENGTH - len(header))
  data = xops.ConstantLiteral(comp, np.array(header, dtype=np.uint32))
  token = xops.OutfeedWithToken(data, token, comp.get_shape(data))

  # Now send the arrays, all at once
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
  header_shape = xla_client.Shape.array_shape(np.dtype(np.uint32),
                                              (_OUTFEED_HEADER_LENGTH,))

  def _get_data(data_shape: XlaShape, device: XlaDevice) -> XlaShape:
    if _OUTFEED_MECHANISM == "device":
      return device.transfer_from_outfeed(data_shape)
    else:
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

_outfeed_receiver_started = False
@contextmanager
def outfeed_receiver(*,
                     timeout_sec=10,
                     backends: Optional[Sequence[str]] = None,
                     devices: Optional[Sequence[XlaDevice]] = None,
                     receiver_name=""):
  # TODO: better timeout management.
  """Starts receivers for the :func:`id_tap` outfeed from several devices.

  The receivers will run in a threadpool. The tapped functions will be invoked
  in those threads. If a tap function raises an exception, an error is
  printed, but the receiving continues until the body of the context manager
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
  global _outfeed_receiver_started
  if _outfeed_receiver_started:
    raise ValueError("At most one outfeed_receiver can be running at once.")
  _outfeed_receiver_started = True
  xla.can_execute_outfeed_computations = True
  try:
    yield
  finally:
    for d in devices:  # Signal the end of printing
      api.jit(lambda x: id_tap(_end_consumer, None, result=x), device=d)(0)  # type: ignore[arg-type]
    xla.can_execute_outfeed_computations = False
    _outfeed_receiver_started = False
    for f in futures.as_completed(receiver_futures, timeout=timeout_sec):
      finished_device = f.result()  # Throw exceptions here
      if _LOGGING:
        logging.info(f"[{receiver_name}:{finished_device} Outfeed receiver finished")
    if count_tap_exceptions > 0:
      raise TapFunctionException
