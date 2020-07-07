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
"""Primitives for calling from accelerators to Python functions on the host.

**Experimental: please give feedback, and expect changes.**

This module introduces the host callback functions :func:`id_tap` and
:func:`id_print`, which behave like the identity function but have the
side-effect of sending the arguments from the accelerator to the host and
invoking a user-specified Python function (for :func:`id_tap`) or printing the
arguments on the host (for :func:`id_print`). A few examples::

  # call func(2x) on host and return 2x
  y = id_tap(func, 2 * x)
  # call func((2x, 3x)) and return (2x, 3x)
  y, z = id_tap(func, (2 * x, 3 * x))  # The argument can be a pytree
  # call func(2x) and return y
  y = id_tap(func, 2 * x, result=y)
  # call func(2x, what='activation') and return 2x
  y = id_tap(func, 2 * x, what='activation')
  # call func(dict(x=x, y=y), what='data') and return dict(x=x, y=y)
  x, y = id_tap(func, dict(x=x, y=y), what='data')

The order of execution of the tap functions is constrained by data dependency:
the arguments are sent after all the arguments are computed and before the
result of the call is used. **At least one of the returned values must be
used in the rest of the computation, or else this operation has no effect.**
The host tap functions will be executed for each device in the order in which
the send operations were performed on the device.

The data from the devices is received by separate threads managed by the JAX
runtime (one thread per device). The runtime maintains a buffer of
configurable size. When the buffer is full, all the receiving threads are paused
which eventually pauses the computation on devices. The runtime has one
additional thread that invokes the Python user functions with the received data.
If the processing of the callbacks is slow, it may actually lead to the runtime
buffer filling up, and eventually pausing the computation on the devices
when they need to send something. For more details on the runtime mechanism see
`runtime code
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/outfeed_receiver.cc>`_.

Exceptions from the user-defined tap functions are logged along with their
stack traces, but the receiving threads are not stopped.

In order to pause the execution until all data from computations already
started on devices has arrived and has been processed, use :func:`barrier_wait`.
This will also raise :exc:`TapFunctionException` if any exception had occurred
in one of the tap functions.

The current implementation uses the outfeed mechanism provided by XLA. The
mechanism itself is quite primitive in the sense that a receiver must know
exactly the shape of each incoming packet, and how many packets are expected.
This makes it hard to use for multiple kinds of data in the same computation,
and it is practically impossible to use it under conditionals or in loops
of non-constant iteration count. Furthermore, code that uses the outfeed
mechanism directly cannot be transformed by JAX. All these limitations are
addressed by the host callback functions. The tapping API introduced here
makes it easy to share the outfeed mechanism for multiple purposes, while
supporting all transformations.

**Note that after you have used the host callback functions, you cannot
use lax.outfeed directly**. You may want to :func:`stop_outfeed_receiver`
if you later need to use lax.outfeed.

We describe the behaviour under transformations in the context of the
following function definition::

  def power3(x):
     y = x * x
     _, y = id_print(x, y, what="x,x^2")
     return y * x

During JAX transformations the special parameter ``transforms`` is added to
contain a list of transformation descriptors. Each descriptor is a dictionary
containing the key ``name`` holding the name of the transformation and
additional keys holding transformation parameters, if applicable. This
parameter is passed to the tap function (or printed), in addition to
user-defined parameters.

For :func:`jax.vmap` the arguments are batched, and ``transforms`` is extended
with transformation name ``batch`` and ``batch_dims`` set to the the tuple of
batched dimensions (one entry per argument, ``None`` denotes an argument that
was broadcast)::

  jax.vmap(power3)(np.arange(3.))
  # what=x,x^2 transforms=({name=batch, batch_dims=(0, 0)}): ([0, 1, 2], [0, 1,
  4])

For :func:`jax.jvp` there will be two callbacks, one with the values of
the primals and one with the tangents::

  jax.jvp(power3, (3.,), (0.1,))
  # what=x,x^2: (3., 9.)
  # what=x,x^2 transforms={name=jvp}: (0.1, 0.6)

For :func:`jax.vjp` or :func:`jax.grad` there will be one callback with the
values of the adjoints for the arguments. You may also see a callback with
the values of the primals from the forward pass, if those values are needed for
the backward pass::

  jax.grad(power3)(3.)
  # what=x,x^2: (3., 9.)  # from forward pass, since y is needed in backward
  pass
  # what=x,x^2 transforms=({name=jvp}, {name=transpose}): (0., 3.)  # from
  backward pass, adjoints of _, y

See documentation for :func:`id_tap` and :func:`id_print`.
For usage example, see tests/host_callback_test.py.

Still to do:
  * Performance tests.
  * Add flags for logging.
  * Add unit tests with mocks.
  * Explore a simpler API that uses Python program-order, instead of
    data dependency-order.
  * Explore implementation with outside compilation.
  * Explore an extended API that allows the host function to return
    values to the accelerator computation.

"""
from absl import logging
import atexit
import contextlib
import itertools

from jax import api
from jax import core
from jax import lax
from jax.lib import pytree
from jax.interpreters import ad, xla, batching, masking
from jax.interpreters import partial_eval as pe
from jax import pprint_util as ppu
from jax import source_info_util
from jax import util
from jaxlib import xla_client
from jaxlib import xla_extension


import numpy as np
import threading
import traceback
from typing import (Any, Callable, Dict, List, Optional, NamedTuple, Sequence,
                    Tuple, cast)
import warnings

xops = xla_client._xla.ops

# TODO(necula): fix mypy errors if I define the type aliases below
XlaOp = Any  # xla_extension.XlaOp
XlaShape = Any  # xla_client.Shape
XlaComputationBuilder = Any  # xla_bridge._JaxComputationBuilder
XlaDevice = Any  # xla_client.Device
XlaLocalClient = Any  # xla_extension.LocalClient


def id_tap(tap_func: Callable, arg, *, result=None, **kwargs):
  """Host-callback tap primitive, like identity function with a call to ``tap_func``.

  **Experimental: please give feedback, and expect changes!**

  ``id_tap`` behaves semantically like the identity function but has the
  side-effect
  that a user-defined Python function is called with the runtime values of the
  argument.

  Args:
    * tap_func: the tap function to call.
    * arg: the argument passed to the tap function, can be a pytree of JAX
      types.
    * result: if given, specifies the return value of ``id_tap``. This value is
      not passed to the tap function, and in fact is not sent from the device to
      the host. If the ``result`` parameter is not specified then the return
      value of ``id_tap`` is ``arg``.
    * kwargs: will be passed directly to the tap function. Can be anything that
      is hashable, these are kept in the host Python process until outfeeds are
      received.

  Returns:
    * ``arg``, or ``result`` if given.

  The order of execution is by data dependency: after all the arguments and
  the value of ``result`` if present, are computed and before the returned
  value is used. At least one of the returned values of ``id_tap`` must be
  used in the rest of the computation, or else this operation has no effect.

  If you want to tap a constant value, you should use the ``result`` parameter
  to control when it is tapped, otherwise it will be tapped during tracing
  of the function::

    x = id_tap(42, result=x)

  Tapping works even for code executed on accelerators and even for code under
  JAX transformations. Code that uses taps must be run embedded in
  :func:`outfeed_receiver`.

  For more details see the
  `module documentation
  <https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html>`_.
  """
  _initialize_outfeed_receiver()  # Lazy initialization
  api._check_callable(tap_func)
  flat_args, arg_treedef = pytree.flatten(arg)
  for arg in flat_args:
    api._check_arg(arg)
  params = dict(kwargs)  #  we pass a copy of params to the primitive
  # See definition of id_tap_p for what parameters it takes
  params["tap_func_"] = tap_func
  params["arg_treedef_"] = arg_treedef
  params["nr_tapped_args_"] = len(flat_args)
  if result is not None:
    flat_results, result_treedef = pytree.flatten(result)
    for result in flat_results:
      api._check_arg(result)
    all_args = flat_args + flat_results
    nr_results = len(flat_results)
  else:
    all_args = flat_args
    nr_results = 0
  flat_outs = id_tap_p.bind(*all_args, **params)  # Returns all_args
  if result is not None:
    flat_results = flat_outs[-nr_results:]  # type: ignore[unsupported-operands]
    return result_treedef.unflatten(flat_results)
  else:
    return arg_treedef.unflatten(flat_outs)


def id_print(arg, *, result=None, output_stream=None, threshold=None, **kwargs):
  """Like :func:`id_tap` with a printing tap function.

   **Experimental: please give feedback, and expect changes!**

   On each invocation of the printing tap, the ``kwargs`` if present
   will be printed first (sorted by keys). Then arg will be printed,
   with the arrays stringified with ``numpy.array2string``.

   See the :func:`id_tap` documentation.

   Additional keyword arguments:

   * ``output_stream`` if given then it will be used instead of the
     built-in ``print``. The string will be passed as
     ``output_stream.write(s)``.
   * ``threshold`` is passed to ``numpy.array2string``.
  """
  return id_tap(
      _print_consumer,
      arg,
      result=result,
      output_stream=output_stream,
      threshold=threshold,
      **kwargs)


# A registry of outfeed consumers, used upon receiving outfeeds
class _ConsumerCallable(NamedTuple):
  """Host-side information for an outfeed consumer."""
  func: Callable
  kwargs: Tuple[Tuple[str, Any], ...]
  arg_treedef: Any
  arg_shape: XlaShape  # XlaShape implements __hash__.

  def unpack_kwargs(self):
    kwargs = dict(self.kwargs)
    transforms = kwargs.get("transforms")
    if transforms is None:
      return kwargs
    else:

      def unpack_transform(name, *params):
        if name == "batch":
          return dict(name=name, batch_dims=params[0])
        elif name == "mask":
          return dict(name=name, logical_shapes=params[0])
        else:
          assert not params, f"{name}, {params}"
          return dict(name=name)

      return dict(
          kwargs, transforms=tuple([unpack_transform(*t) for t in transforms]))


def _register_consumer(cons: _ConsumerCallable) -> int:
  """Registers a tap function, cache by hash of cons."""
  cons_id = _outfeed_receiver.consumer_registry.get(cons)
  if cons_id is not None:
    return cons_id
  cons_id = hash(cons) & 0xFFFFFFFC  # pybind11 has trouble here with large ints
  cons_id += 1  # Reserve the consumer ID 0
  assert cons_id not in _outfeed_receiver.consumer_registry, (
      "consumer id collision")
  _outfeed_receiver.consumer_registry[cons] = cons_id
  _outfeed_receiver.consumer_registry_by_id[cons_id] = cons
  return cons_id


def _print_consumer(arg, *, output_stream=None, threshold=1024, **kwargs):
  """The consumer for id_print.

  We provide this as a simple tapping function for printing.
  This is **experimental** and may not want to add many features to it;
  it should be easy for the user to roll their own printing function.

  Args:
    output_stream: a function whose `write` method is called with the strings to
      be output.
    threshold: the value of numpy.array2string threshold parameter.
    **kwargs: all other keyword args are printed before printing `arg`.
  """

  def emit_str(s: str):
    if output_stream is not None:
      output_stream.write(s + "\n")
    else:
      print(s)

  kv_pairs = " ".join([
      f"{k}: {v}" for k, v in sorted(kwargs.items())
  ])
  if kv_pairs:
    emit_str(kv_pairs)

  def pp_val(arg) -> ppu.PrettyPrint:
    if isinstance(arg, (tuple, list)):
      return (
          ppu.pp("[ ") >> ppu.vcat([pp_val(e) for e in arg]) >> ppu.pp(" ]"))
    elif isinstance(arg, dict):
      return (ppu.pp("{ ") >> ppu.vcat([
          ppu.pp(f"{k}=") >> pp_val(v) for k, v in sorted(arg.items())
      ]) >> ppu.pp(" }"))
    elif isinstance(arg, np.ndarray):
      return ppu.pp(np.array2string(arg, threshold=threshold))
    else:
      return ppu.pp(str(arg))

  emit_str(str(pp_val(arg)))


"""The id_tap_p primitive acts like the identity function.

It has a number of positional arguments. The result of the primitive are
the positional arguments.

The primitive has the following parameters:
  * has_token_: a boolean, when True it means that the last positional argument
    is the current token. In this case, the result of the primitive is
    going to be the non-token positional arguments, along with the updated
    token. The tokens and this parameter are added after all the JAX
    transformations, just before staging XLA.
  * nr_tapped_args_: how many positional arguments from the head should be
    passed to the tap function. The remaining positional arguments are there
    for data dependency, for implementing the "result" feature, and for
    the current token.
  * tapped_args_treedef_: the treedef of the tapped positional arguments.
  * tap_func_: the actual (Python) function to invoke with the tapped positional
    arguments (unflatted according to tapped_args_treedef_) and
    the parameters that were passed to the id_tap function.
  * transforms: a tuple of the transformations that have been applied. Each
    element of the tuple is itself a tuple with the first element the name
    of the transform. The remaining elements depend on the transform. For
    example, for `batch`, the parameters are the dimensions that have been
    batched, and for `mask` the logical shapes. These are unpacked by
    _ConsumerCallable before passing to the user function.
  * the remaining parameters are from the user's invocation of the id_tap
    API function and are passed to the tap function.
"""
id_tap_p = core.Primitive("id_tap")
id_tap_p.multiple_results = True
xla.outfeed_primitives.add(id_tap_p)


def _add_transform(params: Dict, name: str, *transform_params) -> Dict:
  """Adds the `transform` to the params["transforms"].

  Uses a tuple representation internally, will be unpacked before the
  callback by _ConsumerCallable.
  """
  new_transform = (name, *transform_params)
  return dict(
      params, transforms=(params.get("transforms", ()) + (new_transform,)))


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
  if type(tan) is not ad.Zero:
    return tan

  try:
    aval = arg.aval
    return ad.instantiate_zeros_aval(aval, tan)
  except (AttributeError, KeyError):
    # We get here for regular Python values
    return ad.zeros_like_jaxval(arg)


def _id_tap_jvp_rule(primals, tangents, **params):
  # Put primals through id_tap separately, so that partial evaluation
  # can do its job when they are known (for grad)
  out_primals = id_tap_p.bind(
      *primals, **params)
  # Add one primal output as untapped, to create data dependency.
  tangent_zeros = tuple(map(_instantiate_zeros, primals, tangents))
  out_tangents_extra = id_tap_p.bind(
      *tangent_zeros,
      out_primals[0],
      **_add_transform(params, "jvp"))
  return tuple(out_primals), tuple(out_tangents_extra[:-1])


ad.primitive_jvps[id_tap_p] = _id_tap_jvp_rule


def _id_tap_transpose_rule(cts, *args, **params):
  assert len(cts) == len(args)
  cts_zeros = tuple(map(_instantiate_zeros, args, cts))
  ct_args = id_tap_p.bind(
      *cts_zeros,
      **_add_transform(params, "transpose"))
  return ct_args


ad.primitive_transposes[id_tap_p] = _id_tap_transpose_rule


def _id_tap_batching_rule(batched_args, batch_dims, **params):
  new_params = _add_transform(params, "batch", batch_dims)
  res = id_tap_p.bind(*batched_args, **new_params)
  return res, batch_dims


batching.primitive_batchers[id_tap_p] = _id_tap_batching_rule

# def _id_tap_shape_rule(*operands, **params):
#  return tuple([op.shape for op in operands])


def _id_tap_masking_rule(operands, operands_logical_shapes, **params):
  new_params = _add_transform(params, "mask", operands_logical_shapes)
  return id_tap_p.bind(*operands, **new_params)


masking.masking_rules[id_tap_p] = _id_tap_masking_rule

####
#### XLA compilation ####
####


def _id_tap_translation_rule(comp: XlaComputationBuilder,
                             *args_op: XlaOp,
                             tap_func_=None,
                             nr_tapped_args_,
                             arg_treedef_=None,
                             has_token_=False,
                             **params):

  # We expect the current token at the end, inserted by _rewrite_jaxpr.
  assert has_token_
  current_token = args_op[-1]
  assert not comp.get_shape(current_token).is_array(), (
      "The last argument must be a token")

  args_to_outfeed = args_op[0:nr_tapped_args_]
  consumer_id = _register_consumer(
      _ConsumerCallable(tap_func_, tuple(sorted(params.items())), arg_treedef_,
                        comp.get_shape(xops.Tuple(comp, args_to_outfeed))))
  next_token = _outfeed_receiver.receiver.add_outfeed(comp, current_token,
                                                      consumer_id,
                                                      args_to_outfeed)
  results = (args_op[:-1] + (next_token,))
  return xops.Tuple(comp, results)


xla.translations[id_tap_p] = _id_tap_translation_rule

####
#### Jaxpr rewriting logic to thread the tokens through stateful primitives.
####


# TODO: we should really get rid of TypedJaxpr
def _mk_typed_jaxpr(jaxpr: core.Jaxpr, literals: Sequence) -> core.TypedJaxpr:
  return core.TypedJaxpr(jaxpr, literals,
                         tuple(map(lambda v: v.aval, jaxpr.invars)),
                         tuple(map(lambda v: v.aval, jaxpr.outvars)))


def _rewrite_typed_jaxpr(
    tjaxpr: core.TypedJaxpr, has_input_token: bool,
    has_output_token: bool) -> core.TypedJaxpr:
  """Rewrites a TypedJaxpr to thread the token, if needed."""
  new_jaxpr = _rewrite_jaxpr(tjaxpr.jaxpr, has_input_token, has_output_token)
  return _mk_typed_jaxpr(new_jaxpr, tjaxpr.literals)


def _rewrite_jaxpr(jaxpr: core.Jaxpr, has_input_token: bool,
                   has_output_token: bool) -> core.Jaxpr:
  """Rewrite a Jaxpr to thread the token, if needed."""
  assert has_input_token or not has_output_token

  if not has_input_token and not xla.jaxpr_uses_outfeed(jaxpr):
    return jaxpr

  mk_new_var = core.gensym([jaxpr])

  eqns: List[core.JaxprEqn] = []
  last_token_var = mk_new_var(core.abstract_token)  # store the incoming token
  if has_input_token:
    invars = jaxpr.invars + [last_token_var]
  else:
    invars = jaxpr.invars
    eqns.append(
        core.new_jaxpr_eqn([jaxpr.invars[0]], [last_token_var],
                           lax.create_token_p, {}, source_info_util.current()))

  for eqn in jaxpr.eqns:
    if not xla.primitive_uses_outfeed(eqn.primitive, eqn.params):
      eqns.append(eqn)
    else:
      output_token_var = mk_new_var(core.abstract_token)
      _rewrite_eqn(eqn, eqns, last_token_var, output_token_var, mk_new_var)
      last_token_var = output_token_var

  outvars = jaxpr.outvars + ([last_token_var] if has_output_token else [])
  new_jaxpr = core.Jaxpr(jaxpr.constvars, invars, outvars, eqns)
  return new_jaxpr


def _rewrite_eqn(eqn: core.JaxprEqn, eqns: List[core.JaxprEqn],
                 input_token_var: core.Var, output_token_var: core.Var,
                 mk_new_var: Callable[[core.AbstractValue], core.Var]):
  """Rewrite an `eqn` and append equations to `eqns`.

  Assume that the current token is in `input_token_var` and the resulting
  token must end in `output_token_var`.
  """
  if eqn.primitive is id_tap_p:
    assert "has_token_" not in eqn.params
    eqns.append(
        core.new_jaxpr_eqn(eqn.invars + [input_token_var],
                           eqn.outvars + [output_token_var], eqn.primitive,
                           dict(eqn.params, has_token_=True),
                           eqn.source_info))
  elif eqn.primitive is lax.while_p:
    cond_jaxpr, _, body_jaxpr, _ = util.split_dict(
        eqn.params,
        ["cond_jaxpr", "cond_nconsts", "body_jaxpr", "body_nconsts"])
    if xla.jaxpr_uses_outfeed(cond_jaxpr.jaxpr):
      _rewrite_while_outfeed_cond(eqn, eqns, input_token_var, output_token_var,
                                  mk_new_var)
      return

    eqns.append(
        core.new_jaxpr_eqn(
            eqn.invars + [input_token_var], eqn.outvars + [output_token_var],
            eqn.primitive,
            dict(
                eqn.params,
                body_jaxpr=_rewrite_typed_jaxpr(body_jaxpr, True, True),
                cond_jaxpr=_rewrite_typed_jaxpr(cond_jaxpr, True,
                                                False)), eqn.source_info))
  elif eqn.primitive is lax.cond_p:
    branches, linear = util.split_dict(eqn.params, ["branches", "linear"])
    index, *operands = eqn.invars
    new_invars = [index, *operands, input_token_var]
    eqns.append(
        core.new_jaxpr_eqn(
            new_invars, eqn.outvars + [output_token_var], eqn.primitive,
            dict(
                eqn.params,
                branches=tuple(
                    _rewrite_typed_jaxpr(jaxpr, True, True)
                    for jaxpr in branches),
                linear=(*linear, False)), eqn.source_info))
  elif eqn.primitive is lax.scan_p:
    num_consts, num_carry, carry_jaxpr, linear, _, _ = util.split_dict(
        eqn.params,
        ["num_consts", "num_carry", "jaxpr", "linear", "reverse", "length"])
    # We add the token right at the end of carry
    nr_const_and_carry = num_consts + num_carry
    new_invars = eqn.invars[0:nr_const_and_carry] + [
        input_token_var
    ] + eqn.invars[nr_const_and_carry:]
    new_jaxpr = _rewrite_typed_jaxpr(carry_jaxpr, True, True)
    # The rewrite has put the token at end, it has to be at end of carry
    new_jaxpr_invars = new_jaxpr.jaxpr.invars
    new_jaxpr_invars = (
        new_jaxpr_invars[0:nr_const_and_carry] + [new_jaxpr_invars[-1]] +
        new_jaxpr_invars[nr_const_and_carry:-1])
    new_jaxpr.jaxpr.invars = new_jaxpr_invars
    new_jaxpr.in_avals = [v.aval for v in new_jaxpr_invars]

    new_jaxpr_outvars = new_jaxpr.jaxpr.outvars
    new_jaxpr_outvars = (
        new_jaxpr_outvars[0:num_carry] + [new_jaxpr_outvars[-1]] +
        new_jaxpr_outvars[num_carry:-1])
    new_jaxpr.jaxpr.outvars = new_jaxpr_outvars
    new_jaxpr.out_avals = [v.aval for v in new_jaxpr_outvars]
    eqns.append(
        core.new_jaxpr_eqn(
            new_invars,
            # Output token is at the end of carry result
            eqn.outvars[0:num_carry] + [output_token_var] +
            eqn.outvars[num_carry:],
            eqn.primitive,
            dict(
                eqn.params,
                jaxpr=new_jaxpr,
                num_carry=num_carry + 1,
                linear=linear + (False,)),
            eqn.source_info))
  elif eqn.primitive is xla.xla_call_p:
    call_jaxpr = cast(core.Jaxpr, eqn.params["call_jaxpr"])
    eqns.append(
        core.new_jaxpr_eqn(
            eqn.invars + [input_token_var], eqn.outvars + [output_token_var],
            eqn.primitive,
            dict(
                eqn.params,
                call_jaxpr=_rewrite_jaxpr(call_jaxpr, True,
                                          True)), eqn.source_info))
  else:
    raise NotImplementedError(f"outfeed rewrite {eqn.primitive}")


def _rewrite_while_outfeed_cond(eqn: core.JaxprEqn, eqns: List[core.JaxprEqn],
                                input_token_var: core.Var,
                                output_token_var: core.Var,
                                mk_new_var: Callable):
  """Rewrite a while whose cond has outfeed"""
  cond_jaxpr, cond_nconsts, body_jaxpr, body_nconsts = util.split_dict(
      eqn.params, ["cond_jaxpr", "cond_nconsts", "body_jaxpr", "body_nconsts"])
  transformed_cond_jaxpr = _rewrite_typed_jaxpr(cond_jaxpr, True, True)
  carry_invars = eqn.invars[cond_nconsts + body_nconsts:]
  # pred1, token1 = rewrite(COND)(cond_consts, carry_invars, input_token)
  pred1_and_token1 = [
      mk_new_var(ov.aval) for ov in transformed_cond_jaxpr.jaxpr.outvars
  ]
  eqns.append(
      core.new_jaxpr_eqn(
          eqn.invars[0:cond_nconsts] + carry_invars + [input_token_var],
          pred1_and_token1, xla.xla_call_p,
          dict(
              call_jaxpr=transformed_cond_jaxpr.jaxpr,
              name="cond_before",
              donated_invars=(False,) * (cond_nconsts + len(carry_invars) + 1)),
          eqn.source_info))
  # Make a new cond "lambda pred, carry, token: pred"
  new_cond_pred_invar = mk_new_var(cond_jaxpr.out_avals[0])
  new_cond_invars = ([new_cond_pred_invar] +
                     [mk_new_var(cv.aval) for cv in carry_invars] +
                     [mk_new_var(core.abstract_token)])
  new_cond_jaxpr = _mk_typed_jaxpr(
      core.Jaxpr([], new_cond_invars, [new_cond_pred_invar], []), [])
  # Make a new body:
  #   "lambda cond_constvars, body_constvars, pred, carry, token:
  #        carry2, token2 = rewrite(BODY)(body_constvars, carry, token)
  #        pred2, token3 = rewrite(COND)(cond_constvars, carry2, token2)
  #        (pred2, carry2, token3)
  transformed_body_jaxpr = _rewrite_typed_jaxpr(body_jaxpr, True, True)
  new_body_invars_cond_constvars = [
      mk_new_var(v.aval) for v in eqn.invars[0:cond_nconsts]
  ]
  new_body_invars_body_constvars = [
      mk_new_var(v.aval)
      for v in eqn.invars[cond_nconsts:cond_nconsts + body_nconsts]
  ]
  new_body_invars_pred = mk_new_var(cond_jaxpr.out_avals[0])
  new_body_invars_carry = [mk_new_var(cv.aval) for cv in carry_invars]
  new_body_invars_token = mk_new_var(core.abstract_token)

  new_body_carry2 = [mk_new_var(cv.aval) for cv in carry_invars]
  new_body_token2 = mk_new_var(core.abstract_token)
  new_body_pred2 = mk_new_var(cond_jaxpr.out_avals[0])
  new_body_token3 = mk_new_var(core.abstract_token)

  new_body_eqns = [
      core.new_jaxpr_eqn(
          new_body_invars_body_constvars + new_body_invars_carry +
          [new_body_invars_token], new_body_carry2 + [new_body_token2],
          xla.xla_call_p,
          dict(
              call_jaxpr=transformed_body_jaxpr.jaxpr,
              name="body",
              donated_invars=(False,) *
              (len(new_body_invars_body_constvars) +
               len(new_body_invars_carry) + 1 + len(new_body_carry2) + 1)),
          eqn.source_info),
      core.new_jaxpr_eqn(
          new_body_invars_cond_constvars + new_body_carry2 + [new_body_token2],
          [new_body_pred2, new_body_token3], xla.xla_call_p,
          dict(
              call_jaxpr=transformed_cond_jaxpr.jaxpr,
              name="cond_body",
              donated_invars=(False,) * (len(new_body_invars_cond_constvars) +
                                         len(new_body_carry2) + 1 + 2)),
          eqn.source_info)
  ]
  new_body_jaxpr = _mk_typed_jaxpr(
      core.Jaxpr([], (new_body_invars_cond_constvars +
                      new_body_invars_body_constvars + [new_body_invars_pred] +
                      new_body_invars_carry + [new_body_invars_token]),
                 ([new_body_pred2] + new_body_carry2 + [new_body_token3]),
                 new_body_eqns), [])

  pred_out = mk_new_var(cond_jaxpr.out_avals[0])
  eqns.append(
      core.new_jaxpr_eqn(
          (eqn.invars[0:cond_nconsts + body_nconsts] + [pred1_and_token1[0]] +
           carry_invars + [pred1_and_token1[1]]),
          ([pred_out] + eqn.outvars + [output_token_var]), lax.while_p,
          dict(
              cond_jaxpr=new_cond_jaxpr,
              cond_nconsts=0,
              body_jaxpr=new_body_jaxpr,
              body_nconsts=cond_nconsts + body_nconsts), eqn.source_info))


xla.outfeed_rewriter = lambda j: _rewrite_jaxpr(j, False, False)


class TapFunctionException(Exception):
  """Signals that some tap function had exceptions.

  Raised by :func:`outfeed_receiver`.
  """
  pass


@contextlib.contextmanager
def outfeed_receiver():
  """Implements a barrier after a block of code.

  DEPRECATED:
  This function is not necessary anymore, it is here for backwards compatiblity.
  At the moment it implements a ``barrier_wait`` after the body of the
  context manager finishes.
  """
  warnings.warn(
      "outfeed_receiver is unnecessary and deprecated. In the latest "
      "version the outfeer receiver mechanism is started automatically. Use "
      "barrier_wait if instead you want to wait for outfeeds after "
      "a computation", DeprecationWarning)
  _initialize_outfeed_receiver()
  # We will deprecate the outfeed_receiver context manager, but for now
  # we just turn it into a barrier.
  try:
    yield
  finally:
    # We put a barrier, which will also raise the TapFunctionException
    barrier_wait()


# For now we keep a single outfeed receiver
class _OutfeedReceiverData:
  """Keep track of the outfeed receiver data."""
  receiver: Any
  lock: threading.Lock
  num_tap_exceptions: int
  clients: Tuple[XlaLocalClient, ...]
  devices: Tuple[XlaDevice, ...]
  consumer_registry: Dict[_ConsumerCallable, int]
  consumer_registry_by_id: Dict[int, _ConsumerCallable]

  def __init__(self):
    self.receiver = None  # Initialize lazily, when first needed
    self.lock = threading.Lock()
    self.num_tap_exceptions = 0
    self.clients = ()
    self.devices = ()
    # The consumer registries must be live for the lifetime of the program,
    # because we may have cached compilations that embed consumer ids, and we
    # do not want the id reused for other shapes.
    self.consumer_registry = dict()
    self.consumer_registry_by_id = dict()

  def stop(self):
    """Wait for all pending outfeeds and stop the receiver."""
    self.receiver = None  # GC will trigger the destructor
    self.clients = ()
    self.devices = ()
    # Do not clear the consumer registries.


_outfeed_receiver = _OutfeedReceiverData()


# This function is called from C++; it must not allow exceptions through.
def _outfeed_receiver_callback(device, consumer_id, arrays):
  #logging.vlog(
  #    2, f"Outfeed received on device {device} for consumer {consumer_id} " +
  #    (" ".join([f"({a.dtype}{a.shape})" for a in arrays])))
  consumer = _outfeed_receiver.consumer_registry_by_id.get(consumer_id)
  assert consumer is not None, "We should have crashed in the runtime"
  try:
    arg = api.tree_unflatten(consumer.arg_treedef, arrays)
    consumer.func(arg,
                  **consumer.unpack_kwargs())  # type: ignore[attribute-error]
  except Exception as e:
    logging.error("Postponing exception raised in tap function: %s\n%s", str(e),
                  traceback.format_exc())
    _outfeed_receiver.num_tap_exceptions += 1
    return


def _initialize_outfeed_receiver(
    clients: Optional[List[XlaLocalClient]] = None,
    max_callback_queue_size_bytes: int = int(256 * 1e6)):
  """Creates and starts the outfeed_receiver.

  This function is called lazily only when we compile an id_tap.

  Args:
    * clients: the list of clients (backends) on whose devices to listen on.
    * max_callback_queue_size_bytes: an optional integer to bound the maximum
      size of arrays in the callback queue. When this limit is reached the
      device listener pauses.
  """
  try:
    outfeed_receiver_module = xla_extension.outfeed_receiver
  except AttributeError:
    raise NotImplementedError(
        "id_tap works only with jaxlib version 0.1.51 and higher")

  with _outfeed_receiver.lock:
    if _outfeed_receiver.receiver is not None:
      return

    if clients is None:
      # By default, all devices on all backends
      clients = xla_client._get_local_backends().values()  # type: ignore[protected-class]
      # Drop the interpreter clients
      clients = tuple([c for c in clients if c.platform != "interpreter"])  # type: ignore
    devices = list(itertools.chain(*[backend.devices() for backend in clients]))
    _outfeed_receiver.clients = clients  # type: ignore[assignment]
    _outfeed_receiver.devices = devices  # type: ignore[assignment]
    logging.vlog(
        2, f"Starting outfeed_receiver for {[str(d) for d in devices]}. "
        f"max_callback_queue_size_bytes={max_callback_queue_size_bytes}")
    _outfeed_receiver.receiver = outfeed_receiver_module.start(
        _outfeed_receiver_callback, tuple(clients),
        max_callback_queue_size_bytes)

    def exit_handler():
      # Prevent logging usage during compilation, gives errors under pytest
      xla._on_exit = True
      logging.vlog(2, "Barrier wait atexit")
      barrier_wait()

    atexit.register(exit_handler)  # We wait as long as we have callbacks


def barrier_wait():
  """Blocks the calling thread until all current outfeed is processed.

  Waits until all outfeed from computations already running on all devices
  has been received and processed by the Python callbacks. Raises
  TapFunctionException if there were exceptions while processing the callbacks.

  This works by enqueueing a special tap computation to all devices to which
  we are listening for outfeed. Once all those tap computations are done, we
  return from barrier_wait.

  Note: If any of the devices are busy and cannot accept new computations,
  this will deadlock.
  """
  logging.vlog(2, "barrier_wait: start")
  if not _outfeed_receiver.receiver:
    logging.vlog(2, "barrier_wait: receiver not started")
    return

  lock = threading.Lock()
  cv = threading.Condition(lock=lock)
  num_at_large = len(_outfeed_receiver.devices)  # Protected by lock

  def barrier_tap(dev_idx):
    nonlocal num_at_large
    logging.vlog(
        2, f"barrier_wait: thread {threading.current_thread()} for "
        f"device {_outfeed_receiver.devices[dev_idx]} at barrier_tap")
    with lock:
      num_at_large -= 1
      cv.notify()

  for d_idx, d in enumerate(_outfeed_receiver.devices):
    logging.vlog(2, f"barrier_wait: enqueueing barrier on device {d}")
    x_on_dev = api.device_put(d_idx, device=d)
    api.jit(lambda x: id_tap(barrier_tap, x), device=d)(x_on_dev)
  logging.vlog(2, "barrier_wait: waiting for calblacks")
  with lock:
    cv.wait_for(lambda: num_at_large == 0)
  logging.vlog(2, "Done barrier_wait")
  if _outfeed_receiver.num_tap_exceptions > 0:
    _outfeed_receiver.num_tap_exceptions = 0
    raise TapFunctionException(
        "There were exceptions during id_tap processing.")

def stop_outfeed_receiver():
  """Stops the outfeed receiver runtime.

  This waits for all outfeeds from computations already running on all devices,
  and then stops the outfeed receiver runtime. The runtime will be restarted
  next time you use a tap function.

  It should not be necessary to use this function, unless you want to start
  using lax.outfeed directly after having used host callbacks.
  """
  _outfeed_receiver.stop()
