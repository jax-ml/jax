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
"""Primitives for calling from JAX accelerator code to Python functions on the host.

**Experimental: please give feedback, and expect changes.**

This module introduces the host callback functions :func:`call`,
:func:`id_tap`, and :func:`id_print`, that send their arguments from the device
to the host and invoke user-defined Python functions on the host, optionally
returning results back to the device computation.

We show below how these functions can be used. We start with :func:`call`,
and we discuss examples of calling from JAX to NumPy CPU custom kernels,
or to TensorFlow functions, or to JAX running on another device. In the latter
two cases we show how we can support JAX autodiff for the host callbacks,
by deferring to the reverse-mode AD on the target platform. Then we
show uses of :func:`id_tap` and :func:`id_print`, which have the restriction
that they cannot return values from the host to the device.
These primitives are generally faster
because they are executed asynchronously with the device code and they also
support the whole spectrum of JAX transformations. In particular, they can be
used to tap into and to debug JAX-transformed code.

Using :func:`call` to call a host function and return results to device
-----------------------------------------------------------------------

Use :func:`call` to invoke a computation on the host and return
NumPy arrays to the device computation.
Host computation is useful, e.g., when a device computation needs some data
that requires I/O on the host, or it needs a library that is available on the
host and you do not want to code it in JAX.
For example, eigen decomposition for general matrices in JAX does not work on TPU.
We can call the Numpy implementation from any JAX accelerator computation,
using a host computation::

  # This function runs on the host
  def host_eig(m: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(m)

  # This function is used in JAX
  def device_fun(m):
    # We send "m" to the host, asking it to call "host_eig" and return the result.
    # We have to specify the result shape and dtype, either in the form of an
    # example return value or any object that has `shape` and `dtype` attributes,
    # e.g., a NumPy array or a `jax.ShapeDtypeStruct`.
    return hcb.call(host_eig, m,
                    # Given an input of shape (..., d, d), eig output has shape (..., d)
                    result_shape=jax.ShapeDtypeStruct(m.shape[:-1], m.dtype))


The :func:`call` function and the Python host function both take a single argument
and return a single result, but those can be pytrees. Note that we must tell
the :func:`call` what shape and dtype to expect from the host invocation, using
the ``result_shape`` kwarg.
This is important because the device code is compiled with that expectation.
There will be an error raised at runtime if the actual invocation produces a
different result shape. In general, **such errors and also exceptions raised
by the host computation may be difficult to debug**. See the Debugging section
below.
This is a problem for :func:`call` but not for :func:`id_tap`.

The :func:`call` API can be used inside a jit or pmap computation or inside
cond/scan/while control flow. When used inside :func:`jax.pmap`, there will be
separate calls to the host from each of the participating devices::

  def host_sin(x, *, device):
    print(f"Invoking host_sin with {x.shape} on {device}")
    return np.sin(x)

  # Use pmap to run the computation on two devices
  jax.pmap(lambda x: hcb.call(host_sin, x,
                              result_shape=x,
                              # Ask that the `host_sin` function be passed `device=dev`
                              call_with_device=True))(
           np.ones((2, 4), dtype=np.float32))

  # prints (in arbitrary order)
  # Invoking host_sin with (4,) on cpu:0
  # Invoking host_sin with (4,) on cpu:1

Note that :func:`call` does not (yet) support any JAX transformations, but as we
show in the next section one can make use of the
existing support for `Custom differentiation in JAX <https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_.

Using :func:`call` to call a TensorFlow function, with reverse-mode autodiff support
------------------------------------------------------------------------------------

Another possible use for host computation is to invoke a library written for
another framework, such as TensorFlow.
In this case it becomes interesting to support JAX autodiff for host callbacks
by defering to the autodiff mechanism in TensorFlow,
using the :func:`jax.custom_vjp` mechanism.

This is relatively easy to do, once one understands both the JAX custom VJP
and the TensorFlow autodiff mechanisms.
The code for how this can be done is shown in the ``call_tf_full_ad``
function in `host_callback_to_tf_test.py <https://github.com/google/jax/blob/master/tests/host_callback_to_tf_test.py>`_.
This example supports arbitrary higher-order differentiation as well.

Using :func:`call` to call a JAX function on another device, with reverse-mode autodiff support
------------------------------------------------------------------------------------------------

It should not be surprising that we can use host computation to invoke a JAX
computation on another device. The arguments are sent from the accelerator to
the host, and then to the outside device on which the JAX host
computation will run, and then the results are sent back to the original accelerator.

The code for how this can be done is shown in the ``call_jax_other_device function``
in `host_callback_test.py <https://github.com/google/jax/blob/master/tests/host_callback_test.py>`_.

Using :func:`id_tap` to call a JAX function on another device, with no returned values, but full JAX transformation support
---------------------------------------------------------------------------------------------------------------------------

The :func:`id_tap` and :func:`id_print` behave like the identity function but have the
side-effect of sending the arguments from the device to the host and
invoking a user-specified Python function (for :func:`id_tap`) or printing the
arguments on the host (for :func:`id_print`). The Python function passed
to :func:`id_tap` takes two positional arguments (the value tapped
from the device computation along with ``transforms`` sequence,
described below). Optionally, the function may be passed a keyword argument
``device`` with the Device from which the value was tapped.

A few examples::

  # calls func(2x, []) on host and returns 2x
  y = id_tap(func, 2 * x)
  # calls func((2x, 3x), []) and returns (2x, 3x)
  y, z = id_tap(func, (2 * x, 3 * x))  # The argument can be a pytree
  # calls func(2x, []) and returns y
  y = id_tap(func, 2 * x, result=y)  # override the result of id_tap
  # calls func(2x, [], device=jax.devices()[0])
  y = id_tap(func, 2 * x, tap_with_device=True)  # Pass the device to the tap
  # calls func(2x, [], what='activation') and returns 2x
  y = id_tap(functools.partial(func, what='activation'), 2 * x)
  # calls func(dict(x=x, y=y), what='data') and returns dict(x=x, y=y)
  x, y = id_tap(lambda tap, transforms: func(tap, what='data'), dict(x=x, y=y))

The above examples can all be adapted to use :func:`id_print` instead, with
the difference that :func:`id_print` takes one positional argument (to print
on the host), the optional kwarg ``result``, and possibly additional kwargs
that are also printed along with the automatic kwarg ``transforms``.

The order of execution of the callback functions is constrained by data dependency:
the arguments are tapped after all the arguments are computed and before the
result of the call is used. As of September 2020, it is not strictly necessary
anymore for the results of the tap to be used in the rest of the computation.
You can just do::

  id_tap(func, x)

The tap function will execute based on program order. However, if this code
is subject to transformations, it is possible for the tap to appear to
the transformation as dead code and to be removed from the computation. In
that case it is best to use the result of the callback.

Behavior under JAX transformations
----------------------------------

We describe the behaviour under transformations for :func:`id_tap` and
:func:`id_print` in the context of the
following function definition::

  def power3(x):
     y = x * x
     # Print both 'x' and 'x^2'
     _, y = id_print((x, y), what="x,x^2")  # Must pack multiple arguments
     return y * x

  power3(3.)
  # what: x,x^2 : (3., 9.)

(You can see these examples tested in `host_callback_test.HostCallbackIdTapTest.test_tap_transforms`.)

During JAX transformations the special parameter ``transforms`` is added to
contain a list of transformation descriptors in the form
``(transform_name, transform_params)``.

For :func:`jax.vmap` the arguments are batched, and ``transforms`` is extended
with transformation name ``batch`` and ``batch_dims`` set to the the tuple of
batched dimensions (one entry per argument, ``None`` denotes an argument that
was broadcast)::

  jax.vmap(power3)(np.arange(3.))
  # transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2 : ([0, 1, 2], [0, 1,
  4])

For :func:`jax.jvp` there will be one callback with a pair, consisting of
the values of the primals and those of the tangents::

  jax.jvp(power3, (3.,), (0.1,))
  # transforms: ['jvp'] what: x,x^2 : ( (3., 9.), (0.1, 0.6) )

For :func:`jax.vjp` or :func:`jax.grad` there will be one callback with the
values of the adjoints for the arguments. You may also see a callback with
the values of the primals from the forward pass, if those values are needed for
the backward pass::

  jax.grad(power3)(3.)
  # what=x,x^2: (3., 9.)  # from forward pass, since y is used in backward pass
  # transforms: ['jvp', 'transpose'] what: x,x^2 : (0., 3.)  # from backward pass, adjoints of _, y

And here is an example of composed transforms. For vmap of grad, we see first
a callback with the vmap of the forward pass (with just the 'batch' transform),
and another callback with the vmap of the adjoints of the arguments. Note that
the first argument is replicated (`batch_dims` is None)::

  jax.vmap(jax.grad(power3))(np.array([2., 3.]))
  # transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2
  #    ( [2. 3.]
  #      [4. 9.] )
  # transforms: ['jvp', 'transpose', ('batch', {'batch_dims': (None, 0)})] what: x,x^2
  #    ( 0.
  #      [2. 3.] )

In presence of :func:`jax.pmap` the code will run on multiple devices and
each device will tap its values independently.
It may be helpful to use the ``tap_with_device`` option for :func:`id_print`
or :func:`id_tap`, so that you see which device is sending which data::

  jax.pmap(power3, devices=jax.local_devices()[:2])(np.array([3., 4.])
  # device=cpu:0 what=x,x^2: (3., 9.)  # from the first device
  # device=cpu:1 what=x,x^2: (4., 16.)  # from the second device


When using the experimental :func:`pjit.pjit` the code will run on multiple
devices on different shards of the input. The current implementation of
host callbacks will ensure that a single device will collect and outfeed
the entire operand, in a single callback. The callback function is supposed
to return the entire array, which will then be sent in a single infeed to the
same device that issued the outfeed. This device is then responsible for
sending the required shards to the other devices::

  with maps.mesh(jax.local_devices()[:2], ["d"]):
    pjit.pjit(power3, in_axis_resources=(P("d"),),
              out_axis_resources=(P("d"),))(np.array([3., 4.]))

  # device=TPU:0 what=x,x^2: ( [3., 4.],
  #                            [9., 16.] )

See documentation for :func:`id_tap`, :func:`id_print`, and :func:`call`.
For more usage example, see tests/host_callback_test.py.

Low-level details and debugging
-------------------------------

The host callback functions will be executed for each device in the order in which
the send operations were performed on the device.

The host callback functions for multiple devices may be interleaved.
The data from the devices is received by separate threads managed by the JAX
runtime (one thread per device). The runtime maintains a buffer of
configurable size (see the flag ``--jax_host_callback_max_queue_byte_size``).
When the buffer is full, all the receiving threads are paused
which eventually pauses the computation on devices. The runtime has one
additional thread that invokes the Python user functions with the received data.
If the processing of the callbacks is slow, it may actually lead to the runtime
buffer filling up, and eventually pausing the computation on the devices
when they need to send something. For more details on the outfeed receiver
runtime mechanism see
`runtime code
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/outfeed_receiver.cc>`_.

In order to pause the execution until all data from computations already
started on devices has arrived and has been processed, use :func:`barrier_wait`.
Note that this is needed only for :func:`id_tap` and :func:`id_print`, which
are processed asyncronously with the device computation.

Exceptions from the user-defined callback functions are logged along with their
stack traces, but the receiving threads are not stopped. Instead the last
exception is recorded and the subsequent :func:`barrier_wait` will
raise :exc:`CallbackException` if any exception had occurred
in one of the tap functions. This exception will include the text and the
stack trace of the last exception encountered.

One further complication arises for callback functions that must return
results to the call origin device. In order to avoid the device computation
being stuck waiting for a result that will never arrive, in case of any
error during the processing of the callback (whether raised by the user-code
itself or due to a mismatch of the returned value and the expected return_shape)
we send the device a "fake" result of shape ``int8[12345]``. This will make the device
computation abort because the received data is different than then one that
it expects. On CPU the runtime will crash with a distinctive error message:

```
Check failed: buffer->length() == buffer_length (12345 vs. ...)
```

On GPU, the failure is more user-friendly and will be surfaced to the Python
program as:

```
RET_CHECK failure ... Mismatch between infeed source buffer shape s8[12345] ...
```

To debug the underlying cause for these messages, see the Debugging section.

On TPU, there is currently no shape check for infeed, so we take the safer
route to not send anything in case of errors, and let the computation hang.

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

Since the actual calls to your callback functions are made from the C++
receiver, it may be hard to debug the calls. In particular, the stack trace
will not include the calling code. You can use the flag
``jax_host_callback_inline`` (or the environment variable
``JAX_HOST_CALLBACK_INLINE``) to ensure that the calls to the callbacks are
inlined. This works only if the calls are outside a staging context (``jit``
or a control-flow primitive).

The C++ `receiver
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/outfeed_receiver.cc>`_
is started automatically on the first call to :func:`id_tap`. In order to stop
it properly, upon start an ``atexit`` handler is registered to call
:func:`barrier_wait` with the logging name "at_exit".

There are a few environment variables that you can use to turn on logging
for the C++ outfeed `receiver backend
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/outfeed_receiver.cc>`_.

  * ``TF_CPP_MIN_LOG_LEVEL=0``: will turn on INFO logging, needed for all below.
  * ``TF_CPP_MIN_VLOG_LEVEL=3``: will turn make all VLOG logging up to level 3
    behave like INFO logs. This may be too much, but you will see which
    modules are logging relevant info, and then you can select which modules
    to log from:
  * `TF_CPP_VMODULE=<module_name>=3`` (the module name can be either C++ or
    Python, without the extension).

You should also use the ``--verbosity=2`` flag so that you see the logs from Python.

For example, you can try to enable logging in the ``host_callback`` module:
```
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=host_callback=3 python tests/host_callback_test.py --verbosity=2 HostCallbackIdTapTest.test_tap_jit_simple
```

If you want to enable logging in lower-level implementation modules try:
```
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=outfeed_receiver=3,host_callback=3,outfeed_receiver_py=3,outfeed_thunk=3,infeed_thunk=3,cpu_transfer_manager=3,cpu_runtime=3,xfeed_manager=3,pjrt_client=3 python tests/host_callback_test.py --verbosity=2 HostCallbackIdTapTest.test_tap_jit_simple
```

(For bazel tests use --test_arg=--vmodule=...

Still to do:
  * More performance tests.
  * Explore implementation with outside compilation for TPU.
  * Explore implementation with XLA CustomCall for CPU and GPU.

"""
import atexit
import functools
import itertools
import threading
import traceback
from typing import (Any, Callable, Dict, List, Optional, Sequence,
                    Tuple, TypeVar, cast)
import typing
from absl import logging

from jax._src import api
from jax import core
from jax.config import config
from jax import custom_derivatives
from jax._src import dtypes
from jax import lax
from jax.experimental import pjit
from jax.lib import pytree
from jax.lib import xla_bridge as xb
from jax.lib import xla_client
from jax.lib import xla_extension
from jax.interpreters import ad, xla, batching, masking, pxla
from jax.interpreters import partial_eval as pe
from jax._src import pprint_util as ppu
from jax._src import source_info_util
from jax._src import util

import numpy as np


FLAGS = config.FLAGS
xops = xla_client._xla.ops

# TODO(necula): fix mypy errors if I define the type aliases below
XlaOp = Any  # xla_extension.XlaOp
XlaShape = Any  # xla_client.Shape
XlaComputationBuilder = Any  # xla_bridge._JaxComputationBuilder
XlaDevice = Any  # xla_client.Device
XlaLocalClient = Any  # xla_extension.LocalClient

T = TypeVar('T')
U = TypeVar('U')
_Transforms = Sequence[Tuple[str, Dict[str, Any]]]
_TapFunc = Callable[[T, _Transforms], Any]


@typing.overload
def id_tap(tap_func: _TapFunc, arg: T) -> T:
  ...


@typing.overload
def id_tap(tap_func: _TapFunc, arg: T, *, result: U) -> U:
  ...


@typing.overload
def id_tap(tap_func: _TapFunc, arg: T, *, result: U, tap_with_device: bool) -> U:
  ...


def id_tap(tap_func, arg, *, result=None, tap_with_device=False, **kwargs):
  """Host-callback tap primitive, like identity function with a call to ``tap_func``.

  **Experimental: please give feedback, and expect changes!**

  ``id_tap`` behaves semantically like the identity function but has the
  side-effect that a user-defined Python function is called with the runtime
  value of the argument.

  Args:
    tap_func: tap function to call like ``tap_func(arg, transforms)``, with
      ``arg`` as described below and where ``transforms`` is the sequence of
      applied JAX transformations in the form ``(name, params)``. If the
      `tap_with_device` optional argument is True, then the invocation also
      includes the device from which the value is tapped as a keyword argument:
      ``tap_func(arg, transforms, device=dev)``.
    arg: the argument passed to the tap function, can be a pytree of JAX
      types.
    result: if given, specifies the return value of ``id_tap``. This value is
      not passed to the tap function, and in fact is not sent from the device to
      the host. If the ``result`` parameter is not specified then the return
      value of ``id_tap`` is ``arg``.
    tap_with_device: if True then the tap function is invoked with the
      device from which the tap originates as a keyword argument.

  Returns:
    ``arg``, or ``result`` if given.

  The order of execution is by data dependency: after all the arguments and
  the value of ``result`` if present, are computed and before the returned
  value is used. At least one of the returned values of ``id_tap`` must be
  used in the rest of the computation, or else this operation has no effect.

  Tapping works even for code executed on accelerators and even for code under
  JAX transformations.

  For more details see the
  `module documentation
  <jax.experimental.host_callback.html>`_.
  """
  if kwargs:
    msg = (
        "Support for **kwargs in ``id_tap`` has been removed. Instead, "
        "pre-apply keyword arguments, either by using a closure or by passing "
        "``functools.partial(tap_func, **kwargs)``.")
    raise TypeError(msg)

  if result is not None:
    flat_results, result_treedef = pytree.flatten(result)
    for result in flat_results:
      api._check_arg(result)

  call_res = _call(tap_func, arg, call_with_device=tap_with_device,
                   result_shape=None, identity=True)

  if result is not None:
    # Return the results, but add a dependency on the call, to ensure it
    # is kept in the graph.
    call_flat_results, _ = pytree.flatten(call_res)
    if call_flat_results:
      call_flat_results = [id_tap_dep_p.bind(r, call_flat_results[0])
                           for r in flat_results]
    else:
      call_flat_results = flat_results
    return result_treedef.unflatten(call_flat_results)
  else:
    return call_res


def id_print(arg, *, result=None, tap_with_device=False,
             output_stream=None, threshold=None, **kwargs):
  """Like :func:`id_tap` with a printing tap function.

   **Experimental: please give feedback, and expect changes!**

   On each invocation of the printing tap, the ``kwargs`` if present
   will be printed first (sorted by keys). Then arg will be printed,
   with the arrays stringified with ``numpy.array2string``.

   See the :func:`id_tap` documentation.

   Additional keyword arguments:

   * ``tap_with_device`` if True, will print also the device from which
     the value originates.
   * ``output_stream`` if given then it will be used instead of the
     built-in ``print``. The string will be passed as
     ``output_stream.write(s)``.
   * ``threshold`` is passed to ``numpy.array2string``.
  """
  printer = functools.partial(_print_tap_func,
                              output_stream=output_stream,
                              threshold=threshold, **kwargs)
  return id_tap(printer, arg, result=result, tap_with_device=tap_with_device)


def call(callback_func: Callable, arg, *,
         result_shape=None,
         call_with_device=False):
  """Make a call to the host, and expect a result.

  **Experimental: please give feedback, and expect changes!**

  Args:
    callback_func: The Python function to invoke on the host as
      ``callback_func(arg)``. If the ``call_with_device`` optional argument is True,
      then the invocation also includes the ``device`` kwarg with the device
      from which the call originates: ``callback_func(arg, device=dev)``. This function
      must return a pytree of numpy ndarrays.

    arg: the argument passed to the callback function, can be a pytree of JAX
      types.

    result_shape: a value that describes the expected shape and dtype of the
      result. This can be a numeric scalar, from which a shape and dtype are
      obtained, or an object that has ``.shape`` and ``.dtype`` attributes.
      If the result of the callback is a pytree, then ``result_shape`` should
      also be a pytree with the same structure. In particular, ``result_shape``
      can be `()` or `None` if the function does not have any results.
      The device code containing ``call`` is compiled with the expected result shape and dtype,
      and an error will be raised at runtime if the actual ``callback_func``
      invocation returns a different kind of result.

    call_with_device: if True then the callback function is invoked with the
      device from which the call originates as a keyword argument.

  Returns:
    the result of the ``callback_func`` invocation.

  For more details see the
  `module documentation
  <jax.experimental.host_callback.html>`_.
  """
  return _call(callback_func, arg, result_shape=result_shape,
               call_with_device=call_with_device, identity=False)


# Helper function to implement both `call` and `id_tap`. The two cases are
# differentiated by the `identity` flag.
def _call(callback_func: Callable, arg, *,
          result_shape=None,
          call_with_device=False,
          identity=False):
  # Lazy initialization
  _initialize_outfeed_receiver(
      max_callback_queue_size_bytes=FLAGS.jax_host_callback_max_queue_byte_size)
  api._check_callable(callback_func)
  flat_args, arg_treedef = pytree.flatten(arg)
  for arg in flat_args:
    api._check_arg(arg)
  # See definition of outside_call_p for what parameters it takes
  params: Dict[str, Any] = {}
  # TODO: wrap function
  if identity:
    # For id_tap, we pass the transforms, for backwards compatibility
    if call_with_device:
      callback = lambda arg, device, transforms: callback_func(arg, transforms, device=device)
    else:
      callback = lambda arg, device, transforms: callback_func(arg, transforms)
  else:
    if call_with_device:
      callback = lambda arg, device, transforms: callback_func(arg, device=device)
    else:
      callback = lambda arg, device, transforms: callback_func(arg)
  params["callback"] = callback
  params["identity"] = identity
  params["arg_treedef"] = arg_treedef

  if not identity:
    # Turn abstract values into ShapesDtypeStruct
    flat_results_shape, result_treedef = pytree.flatten(result_shape)
    try:
      flat_results_aval = [core.ShapedArray(np.shape(r), dtypes.result_type(r))
                           for r in flat_results_shape]
    except Exception:
      msg = ("result_shape should be a pytree of values with structure "
             "matching the expected result of the callback function. The "
             "values must be either numeric scalars, or must have 'shape' and "
             f"'dtype' attributes. Got {result_shape}")
      raise ValueError(msg)

    params["result_treedef"] = result_treedef
    params["flat_results_aval"] = tuple(flat_results_aval)
  flat_results = outside_call_p.bind(*flat_args, **params)
  return result_treedef.unflatten(flat_results) if not identity else arg_treedef.unflatten(flat_results)


def _print_tap_func(
    arg, transforms, *, device=None,
    output_stream=None, threshold=1024, **kwargs):
  """The consumer for id_print.

  We provide this as a simple tapping function for printing.
  This is **experimental** and may not want to add many features to it;
  it should be easy for the user to roll their own printing function.

  Args:
    device: the device from which the value originates (only if
      ``tap_with_device`` was used for :func:`id_print`).
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

  if transforms:
    kwargs['transforms'] = [(name, params) if params else name
                            for name, params in transforms]
  if device is not None:
    kwargs['device'] = device
  kv_pairs = " ".join([
      f"{k}: {v}" for k, v in sorted(kwargs.items())
  ])
  if kv_pairs:
    emit_str(kv_pairs)

  def pp_val(arg) -> ppu.PrettyPrint:
    if isinstance(arg, tuple):
      return (
          ppu.pp("( ") >> ppu.vcat([pp_val(e) for e in arg]) >> ppu.pp(" )"))
    elif isinstance(arg, list):
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


def _values_to_avals(vals) -> Sequence[core.ShapedArray]:
  return tuple([core.raise_to_shaped(core.get_aval(v)) for v in vals])

### The id_tap_dep primitive
"""
The id_tap_dep_p primitive is used to create a dependency of the result of
id_tap on the actual tap operation. This is only needed when the
id_tap function is used with the `result` parameter. This primitive acts
as the identity operator on the first argument.

For example, given `id_tap(f, (a, b), result=(r, s)`, we convert this to

   a1, b1 = outside_call_p(f, a, b)
   r1 = id_tap_dep_p(r, a1)
   s1 = id_tap_dep_p(s, a1)

There are always two arguments and the result is equal to the first.
"""
id_tap_dep_p = core.Primitive("id_tap_dep")
id_tap_dep_p.multiple_results = False
id_tap_dep_p.def_impl(lambda r, _: r)
xla.translations[id_tap_dep_p] = lambda comp, a_res, a_tap: a_res
id_tap_dep_p.def_abstract_eval(lambda r_a, _: r_a)
ad.primitive_jvps[id_tap_dep_p] = (
    lambda primals, tangents: (
        id_tap_dep_p.bind(primals[0], primals[1]),
        id_tap_dep_p.bind(tangents[0], tangents[1])))


def _id_tap_dep_transpose_rule(cts, arg_res, arg_tap):
  assert ad.is_undefined_primal(arg_res)
  assert ad.is_undefined_primal(arg_tap)
  return (_instantiate_zeros(arg_res, cts), ad.Zero(arg_tap.aval))


ad.primitive_transposes[id_tap_dep_p] = _id_tap_dep_transpose_rule


def _id_tap_dep_batching_rule(batched_args, batch_dims):
  arg_res, arg_tap = batched_args
  return id_tap_dep_p.bind(arg_res, arg_tap), batch_dims[0]


batching.primitive_batchers[id_tap_dep_p] = _id_tap_dep_batching_rule


def _id_tap_dep_masking_rule(operands, operands_logical_shapes):
  arg_res, arg_tap = operands
  return id_tap_dep_p.bind(arg_res, arg_tap)


masking.masking_rules[id_tap_dep_p] = _id_tap_dep_masking_rule

### The outside_call primitive
"""
This primitive is used to implement the `call` and `id_tap` functions.
It takes several positional arguments that are the flattened
according to `arg_treedef`.
The result of the primitive is computed based on the `identity` parameter,
as follows:

  * if `identity` is True, then the results are the same as the
  positional arguments of the primitive (except perhaps the last couple of
  arguments, see `has_token`). In this case, `result_treedef` and
  `flat_results_aval` are ignored, and `args_treedef` describes the result also.
  * if `identity` is False, then the results are those from
  the call to the outside computation:

     flatten(callback(arg_treedef.unflatten(args), device=...))

   In this case, the callback results must match `result_treedef`
   and `flat_results_aval`.

It takes the following parameters:

  * callback: the function to invoke with the unflattened arguments,
    the device and the transforms: `callback(arrays, device, transforms)`
  * arg_treedef: the treedef for the argument.
  * identity: see description above.
  * result_treedef, flat_results_aval: describes the expected result of the
    callback. Only used when not `identity`.
  * transforms: a tuple of the transformations that have been applied. Each
    element of the tuple is itself a tuple with the first element the name
    of the transform. The remaining elements depend on the transform. For
    example, for `batch`, the parameters are the dimensions that have been
    batched, and for `mask` the logical shapes. These are unpacked by
    _outside_call_run_callback before passing to the user function.
  * has_token: a boolean, when True it means that the last positional argument
    is the current token. In this case, the result of the primitive is
    going to be the non-token positional arguments, along with the updated
    token. The tokens and this parameter are added after all the JAX
    transformations, just before staging XLA.
"""
outside_call_p = core.Primitive("outside_call")
outside_call_p.multiple_results = True
xla.outfeed_primitives.add(outside_call_p)


def _outside_call_abstract_eval(*args_a: pe.AbstractValue,
                                identity, **params) -> Sequence[pe.AbstractValue]:
  if identity:
    # Do some validation here
    assert "result_treedef" not in params
    assert "flat_results_aval" not in params
    return args_a

  assert params["result_treedef"] is not None
  assert params["flat_results_aval"] is not None
  flat_results_aval = params["flat_results_aval"]
  if "has_token" in params and params["has_token"]:
    assert len(args_a) >= 2 and args_a[-1] is core.abstract_token and args_a[-2] is core.abstract_token
    return flat_results_aval + (core.abstract_token, core.abstract_token)
  else:
    return flat_results_aval


outside_call_p.def_abstract_eval(_outside_call_abstract_eval)


def _outside_call_impl(*args, **params):
  assert not "has_token" in params
  if FLAGS.jax_host_callback_inline:
    device = api.devices()[0]
    results = _outside_call_run_callback(args, device, send_infeed=False, **params)
    return results
  else:
    # We use the jitted-version of the primitive even for eager execution, both
    # so that we do not duplicate logic, but also so that all outfeed is received
    # by the outfeed_listeners, in the same thread from a given device. If we were
    # to process the tap here, it would be coming from the main thread. Also,
    # even in eager execution some primitives, such as while, are compiled.
    # It would be confusing to process a sequence "id_tap; while" in two
    # different threads.
    return xla.apply_primitive(outside_call_p, *args, **params)


outside_call_p.def_impl(_outside_call_impl)


def _outside_call_translation_rule(
    comp: XlaComputationBuilder, *args_op: XlaOp, **params):
  # We expect the current tokens at the end, inserted by _rewrite_jaxpr.
  assert params["has_token"]  # type: ignore[key-error]
  current_token = args_op[-2]
  current_itoken = args_op[-1]
  # TODO: expose shape.is_token
  assert comp.get_shape(current_token).is_token() and comp.get_shape(current_itoken).is_token(), (
      "The last two arguments must be tokens")

  args_to_outfeed = args_op[:-2]
  identity = params["identity"]  # type: ignore[key-error]
  flat_results_aval = params["flat_results_aval"] if not identity else [
  ]  # type: ignore[key-error]
  # Many platforms refuse to infeed empty arrays. We generate constants
  # instead.
  non_empty_flat_results_aval = list(filter(lambda aval: not (_aval_is_empty(aval)),
                                            flat_results_aval))
  send_infeed = not identity and len(non_empty_flat_results_aval) > 0
  callback_id = _register_callback(
      functools.partial(_outside_call_run_callback, send_infeed=send_infeed, **params))
  next_token = _outfeed_receiver.receiver.add_outfeed(comp, current_token,
                                                      callback_id,
                                                      args_to_outfeed)
  expecting_infeed = False
  if identity:
    results = list(args_to_outfeed)
    next_itoken = current_itoken
  else:
    empty_results = [
        xops.ConstantLiteral(comp, np.zeros(aval.shape, aval.dtype))
        for aval in flat_results_aval
        if _aval_is_empty(aval)
    ]
    if non_empty_flat_results_aval:
      after_outfeed_itoken = xops.AfterAll(comp, [current_itoken, next_token])
      # We shard the infeed as AssignedDevice(0). This must match the
      # outfeed (from outfeed_receiver.cc). Since `lax.infeed` does not support
      # this kind of sharding, we use a custom translation for infeed.
      array_sharding_proto = xla_client.OpSharding()
      array_sharding_proto.type = xla_client.OpSharding.Type.MAXIMAL
      array_sharding_proto.tile_assignment_dimensions = [1]
      array_sharding_proto.tile_assignment_devices = [0]

      token_sharding_proto = xla_client.OpSharding()
      token_sharding_proto.type = xla_client.OpSharding.Type.REPLICATED
      infeed_sharding_proto = xb.tuple_sharding_proto(
          [array_sharding_proto] * len(non_empty_flat_results_aval) +
          [token_sharding_proto])

      shape = tuple(shape.with_major_to_minor_layout_if_absent()
                    for x in non_empty_flat_results_aval
                    for shape in xla.aval_to_xla_shapes(x))

      build_infeed = functools.partial(xops.InfeedWithToken,
                                       after_outfeed_itoken,
                                       xla_client.Shape.tuple_shape(shape))
      outs_and_token = xb.with_sharding_proto(comp, infeed_sharding_proto,
                                              build_infeed)
      outs = xops.GetTupleElement(outs_and_token, 0)
      next_itoken = xops.GetTupleElement(outs_and_token, 1)
      non_empty_results = [
          xops.GetTupleElement(outs, i)
          for i in range(len(non_empty_flat_results_aval))
      ]
      expecting_infeed = True
      results = [
          empty_results.pop(0) if _aval_is_empty(result_aval) else non_empty_results.pop(0)
          for result_aval in flat_results_aval]
    else:
      results = empty_results
      next_itoken = current_itoken

    assert len(results) == len(flat_results_aval)

  assert expecting_infeed == send_infeed
  return xops.Tuple(comp, results + [next_token, next_itoken])


xla.translations[outside_call_p] = _outside_call_translation_rule


def _outside_call_run_callback(
    arrays, device, *,
    send_infeed=True,
    # The same parameters as outside_call_p
    callback, arg_treedef,
    identity, result_treedef=None, flat_results_aval=None,
    transforms=(), has_token=False):
  """Performs the callback:
       callback(arg, device, transforms)

  Called during the device computation once we have the argument, either from
  an inlined callback or from an XLA computation outfeed.

  Returns the flat list of result arrays. If `send_infeed` then it will also send
  the flat list of results to the device.
  """

  def _unpack_transforms(transforms) -> Tuple[Tuple[str, Dict[str, Any]], ...]:
    def _unpack_transform(name, *params):
      if name == "batch":
        return name, dict(batch_dims=params[0])
      elif name == "mask":
        return name, dict(logical_shapes=5)
      else:
        assert not params, f"{name}, {params}"
        return name, dict()

    return tuple(_unpack_transform(*t) for t in transforms)

  try:
    arg = api.tree_unflatten(arg_treedef, arrays)
    unpacked_transforms = _unpack_transforms(transforms)
    logging.vlog(2,
                 f"Outside call consumer invoking call_func {callback} with {arg}, device={device}, transforms={unpacked_transforms}")
    res = callback(arg, device, unpacked_transforms)

    if identity:
      return arrays

    else:  # Check the type of the callback results
      assert result_treedef is not None
      assert flat_results_aval is not None
      actual_flat_results, actual_result_treedef = pytree.flatten(res)
      if actual_result_treedef != result_treedef:
        msg = (f"Callback func {callback} should have returned a result "
               f"with pytree {result_treedef} but returned "
               f"{actual_result_treedef}")
        raise TypeError(msg)

      canonical_flat_results = tuple(util.safe_map(xla.canonicalize_dtype, actual_flat_results))
      actual_flat_results_aval = _values_to_avals(canonical_flat_results)
      logging.vlog(
          2,
          f"Outside call consumer {callback} result {res} : {flat_results_aval}. Sending to infeed for device {device}."
      )

      if not all(ea.strip_weak_type() == ra.strip_weak_type()
                 for ea, ra in util.safe_zip(flat_results_aval,
                                             actual_flat_results_aval)):
        msg = (f"Callback func {callback} should have returned a result "
               "with abstract values "
               f"{result_treedef.unflatten(flat_results_aval)} "
               f"but returned {actual_result_treedef.unflatten(actual_flat_results_aval)}")
        raise TypeError(msg)

      if send_infeed:
        # Do not send the 0-sized arrays
        non_empty_canonical_flat_results = tuple(filter(lambda r: not _aval_is_empty(r),
                                                        canonical_flat_results))
        device.transfer_to_infeed(non_empty_canonical_flat_results)
      return canonical_flat_results

  except Exception as e:
    if send_infeed:
      # Prepare some results to send in case of error. We are sending something
      # with a distinctive shape (int8[12345]), one that is unlikely to be what the device
      # expects. This should have the effect to abort the device computation,
      # with an error message that we recognize. On TPU there seem to be no
      # such check, and if we send anything at all the device computation will
      # use some garbage data. So, on TPU we prefer to not send anything and let
      # the computation hang.
      # TODO: implement a proper error handling for TPU
      if device.platform != "tpu":
        canonical_flat_results = [xla.canonicalize_dtype(np.arange(12345, dtype=np.int8))]
        logging.vlog(2, f"Outside call consumer {callback} exception {e}. Sending to infeed the error result.")
        device.transfer_to_infeed(tuple(canonical_flat_results))
      else:
        logging.vlog(2, f"Outside call consumer {callback} exception {e}. On TPU we do not send infeed.")
    raise e  # Let the exception propagate


def _add_transform(params: Dict, name: str, *transform_params) -> Dict:
  """Adds the `transform` to the params["transforms"].

  Uses a tuple representation internally, will be unpacked before the
  callback by _ConsumerCallable.
  """
  new_transform = (name, *transform_params)
  return dict(
      params, transforms=(params.get("transforms", ()) + (new_transform,)))


def _aval_is_empty(aval) -> bool:
  return np.prod(aval.shape) == 0

# TODO(necula): there must be a better way to do this.
# The AttributeError is for regular values, the KeyError is for ConcreteArray
def _instantiate_zeros(arg, tan):
  """Turn special ad.zero tangents into arrays of 0s for sending to host."""
  # return ad.instantiate_zeros(tan)
  if type(tan) is not ad.Zero:
    return tan

  if tan.aval is core.abstract_unit:
    if ad.is_undefined_primal(arg):
      aval = arg.aval
    else:
      aval = core.raise_to_shaped(core.get_aval(arg))
  else:
    aval = tan.aval
  res = ad.instantiate_zeros_aval(aval, tan)
  return res


def _outside_call_jvp_rule(primals, tangents, **params):
  assert "has_token" not in params
  if not params["identity"]:
    raise NotImplementedError("JVP rule is implemented only for id_tap, not for call.")
  tangent_instantiated = tuple(map(_instantiate_zeros, primals, tangents))

  arg_treedef = params["arg_treedef"]
  # The argument to the jvp tap is a pair of the tapped primals and tangents
  jvp_flat_args, jvp_arg_treedef = api.tree_flatten(
      (arg_treedef.unflatten(primals),
       arg_treedef.unflatten(tangent_instantiated)))
  out_all = outside_call_p.bind(
      *jvp_flat_args,
      **dict(_add_transform(params, "jvp"),
             arg_treedef=jvp_arg_treedef,
             ))
  out_primals_tapped, out_tangents_tapped = util.split_list(out_all, [len(primals)])
  return tuple(out_primals_tapped), tuple(out_tangents_tapped)


ad.primitive_jvps[outside_call_p] = _outside_call_jvp_rule


def _outside_call_partial_eval_rule(trace, *args, **params):
  # The args have been prepared by the id_tap_jvp_rule: primals, tangents
  transforms = params.get("transforms", ())
  if not transforms or transforms[-1] != ("jvp",):
    # We are not in the process of computing VJP
    return trace.default_process_primitive(outside_call_p, args, params)

  assert "has_token" not in params
  if not params["identity"]:
    raise NotImplementedError("differentiation rules are implemented only for id_tap, not for call.")

  assert len(args) % 2 == 0
  nr_primals = len(args) // 2

  consts = [t.pval.get_known() for t in args]
  if all(c is not None for c in consts):
    return trace.default_process_primitive(outside_call_p, args, params)
  # Split into two taps, one for the knowns and one for the unknowns
  # We implement here only the case when primals are known, and we make a tap
  # with just the primals.
  primals, tangents = util.split_list(args, [nr_primals])
  c_primals_tapped, _ = util.split_list(consts, [nr_primals])
  assert all([c is not None for c in c_primals_tapped])

  prims, _ = params["arg_treedef"].unflatten(args)
  _, primals_treedef = api.tree_flatten(prims)

  outs_known = trace.default_process_primitive(
      outside_call_p, primals,
      dict(params,
           arg_treedef=primals_treedef,
           transforms=transforms[:-1]))
  # Now compute the unknowns using the whole tap, and merge them with the tapped ones
  outs_all_unknown = trace.default_process_primitive(outside_call_p, args, params)
  outs_primals_unknown, outs_tangents_unknown = util.split_list(
      outs_all_unknown, [nr_primals])
  outs_combined = (
      [pe.JaxprTracer(trace, pe.PartialVal.known(primal_known),
                      primal_unknown.recipe)
       for primal_known, primal_unknown in util.safe_zip(outs_known, outs_primals_unknown)] +
      outs_tangents_unknown)
  return tuple(outs_combined)


pe.custom_partial_eval_rules[outside_call_p] = _outside_call_partial_eval_rule


def _outside_call_transpose_rule(cts, *args, **params):
  if not params["identity"]:
    raise NotImplementedError("differentiation rules are implemented only for id_tap, not for call.")
  assert "has_token" not in params
  assert len(cts) == len(args)
  cts_instantiated = tuple(map(_instantiate_zeros, args, cts))

  # The args have been prepared by the id_tap_jvp_rule: tapped_primals, tapped_tangents, rest_primals, rest_tangents
  transforms = params.get("transforms", ())
  if not transforms or transforms[-1] != ("jvp",):
    # TODO: I should understand better when can this happen. It seems to arise
    # in scan.
    return outside_call_p.bind(
        *cts_instantiated,
        **_add_transform(params, "transpose"))

  assert len(args) % 2 == 0
  nr_primals = len(args) // 2

  args_unflat, tan_unflat = params["arg_treedef"].unflatten(args)
  _, vjp_arg_treedef = api.tree_flatten(args_unflat)
  # We want to tap the cts_tapped_tangents
  cts_primals, cts_tangents = util.split_list(cts_instantiated, [nr_primals])
  cts_tangents_through_tap = outside_call_p.bind(
      *cts_tangents,
      **dict(_add_transform(params, "transpose"),
             arg_treedef=vjp_arg_treedef))
  return (cts_primals + cts_tangents_through_tap)


ad.primitive_transposes[outside_call_p] = _outside_call_transpose_rule


def _outside_call_batching_rule(batched_args, batch_dims, **params):
  if not params["identity"]:
    raise NotImplementedError("batching rules are implemented only for id_tap, not for call.")
  assert "has_token" not in params
  new_params = _add_transform(params, "batch", batch_dims)
  res = outside_call_p.bind(*batched_args, **new_params)
  return res, batch_dims


batching.primitive_batchers[outside_call_p] = _outside_call_batching_rule


def _outside_call_masking_rule(operands, operands_logical_shapes, **params):
  if not params["identity"]:
    raise NotImplementedError("masking rules are implemented only for id_tap, not for call.")
  assert "has_token" not in params

  assert len(operands) == len(operands_logical_shapes)
  arg_treedef = params["arg_treedef"]
  # We will send the pair of (arg, arg_logical_shapes)
  packed_operands, packed_arg_tree = api.tree_flatten(
      (api.tree_unflatten(arg_treedef, operands),
       api.tree_unflatten(arg_treedef, operands_logical_shapes)))

  packed_results = outside_call_p.bind(
      *packed_operands,
      **dict(_add_transform(params, "mask"),
             arg_treedef=packed_arg_tree))
  return packed_results[:len(operands)] + packed_results[len(packed_operands):]


masking.masking_rules[outside_call_p] = _outside_call_masking_rule


####
#### Jaxpr rewriting logic to thread the tokens through stateful primitives.
####


def _rewrite_closed_jaxpr(
    cjaxpr: core.ClosedJaxpr, has_input_token: bool,
    has_output_token: bool) -> core.ClosedJaxpr:
  """Rewrites a ClosedJaxpr to thread the token, if needed."""
  new_jaxpr = _rewrite_jaxpr(cjaxpr.jaxpr, has_input_token, has_output_token)
  return core.ClosedJaxpr(new_jaxpr, cjaxpr.consts)


def _rewrite_jaxpr(jaxpr: core.Jaxpr, has_input_token: bool,
                   has_output_token: bool) -> core.Jaxpr:
  """Rewrite a Jaxpr to thread the token, if needed."""
  assert has_input_token or not has_output_token

  if not has_input_token and not xla.jaxpr_uses_outfeed(jaxpr):
    return jaxpr

  mk_new_var = core.gensym([jaxpr])

  eqns: List[core.JaxprEqn] = []
  last_token_var = mk_new_var(core.abstract_token)  # store the incoming token
  last_itoken_var = mk_new_var(core.abstract_token)  # store the incoming token
  if has_input_token:
    invars = jaxpr.invars + [last_token_var, last_itoken_var]
  else:
    invars = jaxpr.invars
    # We need tokens but none is given in input; make one depending on all invars
    eqns.append(
        core.new_jaxpr_eqn(jaxpr.invars, [last_token_var],
                           lax.create_token_p, {}, source_info_util.current()))
    eqns.append(
        core.new_jaxpr_eqn(jaxpr.invars, [last_itoken_var],
                           lax.create_token_p, {}, source_info_util.current()))

  for eqn in jaxpr.eqns:
    if not xla.primitive_uses_outfeed(eqn.primitive, eqn.params):
      eqns.append(eqn)
    else:
      output_token_var = mk_new_var(core.abstract_token)
      output_itoken_var = mk_new_var(core.abstract_token)
      _rewrite_eqn(eqn, eqns, last_token_var, output_token_var, last_itoken_var, output_itoken_var, mk_new_var)
      last_token_var = output_token_var
      last_itoken_var = output_itoken_var

  outvars = jaxpr.outvars + ([last_token_var, last_itoken_var] if has_output_token else [])
  new_jaxpr = core.Jaxpr(jaxpr.constvars, invars, outvars, eqns)
  return new_jaxpr


def _rewrite_eqn(eqn: core.JaxprEqn, eqns: List[core.JaxprEqn],
                 input_token_var: core.Var, output_token_var: core.Var,
                 input_itoken_var: core.Var, output_itoken_var: core.Var,
                 mk_new_var: Callable[[core.AbstractValue], core.Var]):
  """Rewrite an `eqn` and append equations to `eqns`.

  This is only called if the current primitive uses outfeed.
  Assume that the current token is in `input_token_var` and the resulting
  token must end in `output_token_var`.

  Append the result of rewriting to `eqns`.
  """
  if eqn.primitive is outside_call_p:
    assert "has_token" not in eqn.params
    eqns.append(
        core.new_jaxpr_eqn(eqn.invars + [input_token_var, input_itoken_var],
                           eqn.outvars + [output_token_var, output_itoken_var], eqn.primitive,
                           dict(eqn.params, has_token=True),
                           eqn.source_info))
  elif eqn.primitive is lax.while_p:
    cond_jaxpr, _, body_jaxpr, _ = util.split_dict(
        eqn.params,
        ["cond_jaxpr", "cond_nconsts", "body_jaxpr", "body_nconsts"])
    if xla.jaxpr_uses_outfeed(cond_jaxpr.jaxpr):
      _rewrite_while_outfeed_cond(eqn, eqns, input_token_var, output_token_var,
                                  input_itoken_var, output_itoken_var,
                                  mk_new_var)
      return

    eqns.append(
        core.new_jaxpr_eqn(
            eqn.invars + [input_token_var, input_itoken_var],
            eqn.outvars + [output_token_var, output_itoken_var],
            eqn.primitive,
            dict(
                eqn.params,
                body_jaxpr=_rewrite_closed_jaxpr(body_jaxpr, True, True),
                cond_jaxpr=_rewrite_closed_jaxpr(cond_jaxpr, True,
                                                 False)), eqn.source_info))
  elif eqn.primitive is lax.cond_p:
    branches, linear = util.split_dict(eqn.params, ["branches", "linear"])
    index, *operands = eqn.invars
    new_invars = [index, *operands, input_token_var, input_itoken_var]
    eqns.append(
        core.new_jaxpr_eqn(
            new_invars, eqn.outvars + [output_token_var, output_itoken_var], eqn.primitive,
            dict(
                eqn.params,
                branches=tuple(_rewrite_closed_jaxpr(jaxpr, True, True)
                               for jaxpr in branches),
                linear=(*linear, False, False)), eqn.source_info))
  elif eqn.primitive is lax.scan_p:
    num_consts, num_carry, carry_jaxpr, linear, _, _, _ = util.split_dict(
        eqn.params,
        ["num_consts", "num_carry", "jaxpr", "linear", "reverse", "length",
         "unroll"])
    # We add the tokens right at the end of carry
    nr_const_and_carry = num_consts + num_carry
    new_invars = eqn.invars[0:nr_const_and_carry] + [
        input_token_var, input_itoken_var] + eqn.invars[nr_const_and_carry:]
    new_jaxpr = _rewrite_closed_jaxpr(carry_jaxpr, True, True)
    # The rewrite has put the token at end, it has to be at end of carry
    new_jaxpr_invars = new_jaxpr.jaxpr.invars
    new_jaxpr_invars = (
        new_jaxpr_invars[0:nr_const_and_carry] + new_jaxpr_invars[-2:] +
        new_jaxpr_invars[nr_const_and_carry:-2])
    new_jaxpr.jaxpr.invars = new_jaxpr_invars

    new_jaxpr_outvars = new_jaxpr.jaxpr.outvars
    new_jaxpr_outvars = (
        new_jaxpr_outvars[0:num_carry] + new_jaxpr_outvars[-2:] +
        new_jaxpr_outvars[num_carry:-2])
    new_jaxpr.jaxpr.outvars = new_jaxpr_outvars
    eqns.append(
        core.new_jaxpr_eqn(
            new_invars,
            # Output token is at the end of carry result
            eqn.outvars[0:num_carry] + [output_token_var, output_itoken_var] +
            eqn.outvars[num_carry:],
            eqn.primitive,
            dict(
                eqn.params,
                jaxpr=new_jaxpr,
                num_carry=num_carry + 2,
                linear=linear[0:nr_const_and_carry] + (False, False) + linear[nr_const_and_carry:]),
            eqn.source_info))
  elif eqn.primitive is xla.xla_call_p:
    call_jaxpr = cast(core.Jaxpr, eqn.params["call_jaxpr"])
    eqns.append(
        core.new_jaxpr_eqn(
            eqn.invars + [input_token_var, input_itoken_var], eqn.outvars + [output_token_var, output_itoken_var],
            eqn.primitive,
            dict(
                eqn.params,
                call_jaxpr=_rewrite_jaxpr(call_jaxpr, True, True),
                donated_invars=eqn.params["donated_invars"] + (False, False)
            ),
            eqn.source_info))
  elif eqn.primitive is pxla.xla_pmap_p:
    # We broadcast the input token into an array of tokens
    call_jaxpr = cast(core.Jaxpr, eqn.params["call_jaxpr"])
    eqns.append(
        core.new_jaxpr_eqn(
            eqn.invars + [input_token_var, input_itoken_var], eqn.outvars + [output_token_var, output_itoken_var],
            eqn.primitive,
            dict(
                eqn.params,
                call_jaxpr=_rewrite_jaxpr(call_jaxpr, True, True),
                donated_invars=eqn.params["donated_invars"] + (False, False),
                # Sharding/unsharding of tokens in pmap_translation are special
                # cased to just pass-through the token
                in_axes=eqn.params["in_axes"] + (0, 0),
                out_axes=eqn.params["out_axes"] + (0, 0)
            ),
            eqn.source_info))
  elif eqn.primitive is pe.remat_call_p:
    call_jaxpr = cast(core.Jaxpr, eqn.params["call_jaxpr"])
    eqns.append(
        core.new_jaxpr_eqn(
            eqn.invars + [input_token_var, input_itoken_var],
            eqn.outvars + [output_token_var, output_itoken_var],
            eqn.primitive,
            dict(
                eqn.params,
                call_jaxpr=_rewrite_jaxpr(call_jaxpr, True, True),
            ),
            eqn.source_info))
  elif eqn.primitive is custom_derivatives.custom_jvp_call_jaxpr_p:
    fun_jaxpr = eqn.params["fun_jaxpr"]

    def unreachable_thunk():
      assert False, "Should not be reached"

    eqns.append(
        core.new_jaxpr_eqn(
            eqn.invars + [input_token_var, input_itoken_var],
            eqn.outvars + [output_token_var, output_itoken_var], eqn.primitive,
            dict(
                eqn.params,
                fun_jaxpr=_rewrite_closed_jaxpr(fun_jaxpr, True, True),
                jvp_jaxpr_thunk=unreachable_thunk
            ),
            eqn.source_info))
  elif eqn.primitive is custom_derivatives.custom_vjp_call_jaxpr_p:
    fun_jaxpr = eqn.params["fun_jaxpr"]
    new_invars = [*eqn.invars, input_token_var, input_itoken_var]

    def unreachable_thunk():
      assert False, "Should not be reached"

    eqns.append(
        core.new_jaxpr_eqn(
            new_invars, eqn.outvars + [output_token_var, output_itoken_var], eqn.primitive,
            dict(
                eqn.params,
                fun_jaxpr=_rewrite_closed_jaxpr(fun_jaxpr, True, True),
                fwd_jaxpr_thunk=unreachable_thunk,
                # The following are illegal values for the parameters, they
                # should not be needed because this rewrite is just before
                # compilation to XLA, which does not use those parameters.
                bwd="illegal param",
                out_trees="illegal param"
            ),
            eqn.source_info))
  elif eqn.primitive is core.named_call_p:
    call_jaxpr = cast(core.Jaxpr, eqn.params["call_jaxpr"])
    eqns.append(
        core.new_jaxpr_eqn(
            eqn.invars + [input_token_var, input_itoken_var],
            eqn.outvars + [output_token_var, output_itoken_var], eqn.primitive,
            dict(
                eqn.params,
                call_jaxpr=_rewrite_jaxpr(call_jaxpr, True, True),
            ), eqn.source_info))
  elif eqn.primitive is pjit.pjit_p:
    jaxpr = cast(core.ClosedJaxpr, eqn.params["jaxpr"])
    eqns.append(
        core.new_jaxpr_eqn(
            eqn.invars + [input_token_var, input_itoken_var],
            eqn.outvars + [output_token_var, output_itoken_var], eqn.primitive,
            dict(
                eqn.params,
                jaxpr=_rewrite_closed_jaxpr(jaxpr, True, True),
                donated_invars=eqn.params["donated_invars"] + (False, False),
                in_axis_resources=(eqn.params["in_axis_resources"] +
                                   (pjit.REPLICATED, pjit.REPLICATED)),
                out_axis_resources=(eqn.params["out_axis_resources"] +
                                    (pjit.REPLICATED, pjit.REPLICATED)),
            ), eqn.source_info))
  else:
    raise NotImplementedError(f"outfeed rewrite {eqn.primitive}")


def _rewrite_while_outfeed_cond(eqn: core.JaxprEqn, eqns: List[core.JaxprEqn],
                                input_token_var: core.Var, output_token_var: core.Var,
                                input_itoken_var: core.Var, output_itoken_var: core.Var,
                                mk_new_var: Callable):
  """Rewrite a while whose cond has outfeed"""
  cond_jaxpr, cond_nconsts, body_jaxpr, body_nconsts = util.split_dict(
      eqn.params, ["cond_jaxpr", "cond_nconsts", "body_jaxpr", "body_nconsts"])
  transformed_cond_jaxpr = _rewrite_closed_jaxpr(cond_jaxpr, True, True)
  carry_invars = eqn.invars[cond_nconsts + body_nconsts:]
  # pred1, token1, itoken1 = rewrite(COND)(cond_consts, carry_invars, input_token, input_itoken)
  pred1_and_token1 = [
      mk_new_var(ov.aval) for ov in transformed_cond_jaxpr.jaxpr.outvars
  ]
  eqns.append(
      core.new_jaxpr_eqn(
          eqn.invars[0:cond_nconsts] + carry_invars + [input_token_var, input_itoken_var],
          pred1_and_token1, xla.xla_call_p,
          dict(
              call_jaxpr=transformed_cond_jaxpr.jaxpr,
              name="cond_before",
              donated_invars=(False,) * len(transformed_cond_jaxpr.in_avals)),
          eqn.source_info))
  # Make a new cond "lambda pred, carry, token, itoken: pred"
  new_cond_pred_invar = mk_new_var(cond_jaxpr.out_avals[0])
  new_cond_invars = ([new_cond_pred_invar] +
                     [mk_new_var(cv.aval) for cv in carry_invars] +
                     [mk_new_var(core.abstract_token), mk_new_var(core.abstract_token)])
  new_cond_jaxpr = core.ClosedJaxpr(
      core.Jaxpr([], new_cond_invars, [new_cond_pred_invar], []), [])
  # Make a new body:
  #   "lambda cond_constvars, body_constvars, pred, carry, token, itoken:
  #        carry2, token2, itoken2 = rewrite(BODY)(body_constvars, carry, token, itoken)
  #        pred2, token3, itoken3 = rewrite(COND)(cond_constvars, carry2, token2, itoken2)
  #        (pred2, carry2, token3, itoken3)
  transformed_body_jaxpr = _rewrite_closed_jaxpr(body_jaxpr, True, True)
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
  new_body_invars_itoken = mk_new_var(core.abstract_token)

  new_body_carry2 = [mk_new_var(cv.aval) for cv in carry_invars]
  new_body_token2 = mk_new_var(core.abstract_token)
  new_body_itoken2 = mk_new_var(core.abstract_token)
  new_body_pred2 = mk_new_var(cond_jaxpr.out_avals[0])
  new_body_token3 = mk_new_var(core.abstract_token)
  new_body_itoken3 = mk_new_var(core.abstract_token)

  new_body_eqns = [
      core.new_jaxpr_eqn(
          new_body_invars_body_constvars + new_body_invars_carry +
          [new_body_invars_token, new_body_invars_itoken],
          new_body_carry2 + [new_body_token2, new_body_itoken2],
          xla.xla_call_p,
          dict(
              call_jaxpr=transformed_body_jaxpr.jaxpr,
              name="body",
              donated_invars=(False,) * len(transformed_body_jaxpr.in_avals)),
          eqn.source_info),
      core.new_jaxpr_eqn(
          new_body_invars_cond_constvars + new_body_carry2 + [new_body_token2, new_body_itoken2],
          [new_body_pred2, new_body_token3, new_body_itoken3], xla.xla_call_p,
          dict(
              call_jaxpr=transformed_cond_jaxpr.jaxpr,
              name="cond_body",
              donated_invars=(False,) * len(transformed_cond_jaxpr.in_avals)),
          eqn.source_info)
  ]
  new_body_jaxpr = core.ClosedJaxpr(
      core.Jaxpr([], (new_body_invars_cond_constvars +
                      new_body_invars_body_constvars + [new_body_invars_pred] +
                      new_body_invars_carry + [new_body_invars_token, new_body_invars_itoken]),
                 ([new_body_pred2] + new_body_carry2 + [new_body_token3, new_body_itoken3]),
                 new_body_eqns), [])

  pred_out = mk_new_var(cond_jaxpr.out_avals[0])
  eqns.append(
      core.new_jaxpr_eqn(
          (eqn.invars[0:cond_nconsts + body_nconsts] + [pred1_and_token1[0]] +
           carry_invars + pred1_and_token1[1:]),
          ([pred_out] + eqn.outvars + [output_token_var, output_itoken_var]),
          lax.while_p,
          dict(
              cond_jaxpr=new_cond_jaxpr,
              cond_nconsts=0,
              body_jaxpr=new_body_jaxpr,
              body_nconsts=cond_nconsts + body_nconsts), eqn.source_info))


# We need an identity primitive to simplify rewriting
id_p = core.Primitive("id")
id_p.multiple_results = True
id_p.def_impl(lambda *args: args)
id_p.def_abstract_eval(lambda *args: args)
xla.translations[id_p] = lambda c, *args: xops.Tuple(c, args)

xla.outfeed_rewriter = lambda j: _rewrite_jaxpr(j, False, False)


class CallbackException(Exception):
  """Signals that some callback function had exceptions.

  Raised by :func:`barrier_wait`.
  See module documentation for details.
  """
  pass

TapFunctionException = CallbackException  # For backwards compatibility

# For now we keep a single outfeed receiver
class _OutfeedReceiverData:
  """Keep track of the outfeed receiver data."""
  receiver: Any
  lock: threading.Lock
  last_callback_exception: Optional[Tuple[Exception, str]]
  clients: Tuple[XlaLocalClient, ...]
  devices: Tuple[XlaDevice, ...]
  consumer_registry: Dict[Callable, int]
  consumer_registry_by_id: Dict[int, Callable]

  def __init__(self):
    self.receiver = None  # Initialize lazily, when first needed
    self.lock = threading.Lock()
    self.last_callback_exception = None
    self.clients = ()
    self.devices = ()
    # The consumer registries must be live for the lifetime of the program,
    # because we may have cached compilations that embed consumer ids, and we
    # do not want the id reused for other shapes.
    self.callback_registry = dict()
    self.callback_registry_by_id = dict()

  def stop(self):
    """Wait for all pending outfeeds and stop the receiver."""
    self.receiver = None  # GC will trigger the destructor
    self.clients = ()
    self.devices = ()
    # Do not clear the consumer registries.


_outfeed_receiver = _OutfeedReceiverData()


# This function is called from C++; it must not allow exceptions through.
def _outfeed_receiver_callback(device, consumer_id, arrays):
  # logging.vlog(
  #    2, f"Outfeed received on device {device} for consumer {consumer_id} " +
  #    (" ".join([f"({a.dtype}{a.shape})" for a in arrays])))
  callback = _outfeed_receiver.callback_registry_by_id.get(consumer_id)
  assert callback is not None, "We should have crashed in the runtime"
  try:
    callback(arrays, device)
  except Exception as e:
    formatted_e = traceback.format_exc()
    logging.error("Postponing exception raised in callback function: %s", formatted_e)
    _outfeed_receiver.last_callback_exception = (e, formatted_e)


def _register_callback(callback: Callable) -> int:
  """Registers a callback function, cache by hash of callback.

  The callback is a function to be invoked as `callback(arrays, device)`.
  """
  callback_id = _outfeed_receiver.callback_registry.get(callback)
  if callback_id is not None:
    return callback_id
  callback_id = hash(callback) & 0xFFFFFFFC  # pybind11 has trouble here with large ints
  callback_id += 1  # Reserve the consumer ID 0
  assert callback_id not in _outfeed_receiver.callback_registry, (
      "callback id collision")
  _outfeed_receiver.callback_registry[callback] = callback_id
  _outfeed_receiver.callback_registry_by_id[callback_id] = callback
  return callback_id


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
  outfeed_receiver_module = xla_extension.outfeed_receiver

  with _outfeed_receiver.lock:
    if _outfeed_receiver.receiver is not None:
      return

    if clients is None:
      # By default, all devices on all backends
      clients = xla_client._get_local_backends().values()  # type: ignore[protected-class]
      # Drop the interpreter clients
      clients = tuple([c for c in clients if c.platform != "interpreter"])  # type: ignore
    devices = list(
        itertools.chain(*[backend.local_devices() for backend in clients]))
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
      barrier_wait("at_exit")

    atexit.register(exit_handler)  # We wait as long as we have callbacks


def barrier_wait(logging_name: Optional[str] = None):
  """Blocks the calling thread until all current outfeed is processed.

  Waits until all outfeed from computations already running on all devices
  has been received and processed by the Python callbacks. Raises
  CallbackException if there were exceptions while processing the callbacks.

  This works by enqueueing a special tap computation to all devices to which
  we are listening for outfeed. Once all those tap computations are done, we
  return from barrier_wait.

  Note: If any of the devices are busy and cannot accept new computations,
  this will deadlock.

  Args:
    logging_name: an optional string that will be used in the logging statements
      for this invocation. See `Debugging` in the module documentation.
  """
  logging_name = logging_name or ""
  logging.vlog(2, f"barrier_wait[{logging_name}]: start")
  if not _outfeed_receiver.receiver:
    logging.vlog(2, f"barrier_wait[{logging_name}]: receiver not started")
    return

  lock = threading.Lock()
  cv = threading.Condition(lock=lock)
  num_at_large = len(_outfeed_receiver.devices)  # Protected by lock

  def barrier_tap(dev_idx, _):
    nonlocal num_at_large
    logging.vlog(
        2, f"barrier_wait[{logging_name}]: at barrier_tap for device {_outfeed_receiver.devices[dev_idx]} "
           f". Thread {threading.current_thread()}")
    with lock:
      num_at_large -= 1
      logging.vlog(2, f"barrier_wait[{logging_name}]: still waiting for {num_at_large} barrier_tap")
      cv.notify()

  for d_idx, d in enumerate(_outfeed_receiver.devices):
    logging.vlog(2, f"barrier_wait[{logging_name}]: enqueueing barrier on device {d}")
    x_on_dev = api.device_put(d_idx, device=d)
    api.jit(lambda x: id_tap(barrier_tap, x), device=d)(x_on_dev)
  logging.vlog(2, f"barrier_wait[{logging_name}]: waiting for callbacks")
  with lock:
    cv.wait_for(lambda: num_at_large == 0)
  logging.vlog(2, f"barrier_wait[{logging_name}]: done")
  if _outfeed_receiver.last_callback_exception is not None:
    last_exception, formatted_last_exception = _outfeed_receiver.last_callback_exception
    _outfeed_receiver.last_callback_exception = None
    raise CallbackException(
        "There were exceptions during callback processing. "
        f"Last one was: {formatted_last_exception}") from last_exception


def stop_outfeed_receiver():
  """Stops the outfeed receiver runtime.

  This waits for all outfeeds from computations already running on all devices,
  and then stops the outfeed receiver runtime. The runtime will be restarted
  next time you use a tap function.

  It should not be necessary to use this function, unless you want to start
  using lax.outfeed directly after having used host callbacks.
  """
  _outfeed_receiver.stop()
