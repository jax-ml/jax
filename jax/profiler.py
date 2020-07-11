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

from functools import wraps
from typing import Callable, Optional

from .lib import xla_bridge
from .lib import xla_client


def start_server(port: int):
  """Starts a profiler server on port `port`.

  Using the "TensorFlow profiler" feature in `TensorBoard
  <https://www.tensorflow.org/tensorboard>`_ 2.2 or newer, you can
  connect to the profiler server and sample execution traces that show CPU and
  GPU device activity.

  Returns a profiler server object. The server remains alive and listening until
  the server object is destroyed.
  """
  return xla_client.profiler.start_server(port)


class TraceContext(xla_client.profiler.TraceMe):
  """Context manager generates a trace event in the profiler.

  The trace event spans the duration of the code enclosed by the context.

  For example:

  >>> import jax, jax.numpy as jnp
  >>> x = jnp.ones((1000, 1000))
  >>> with jax.profiler.TraceContext("acontext"):
  ...   jnp.dot(x, x.T).block_until_ready()

  This will cause an "acontext" event to show up on the trace timeline if the
  event occurs while the process is being traced by TensorBoard.
  """
  pass


class StepTraceContext(TraceContext):
  """Context manager that generates a step trace event in the profiler.

  The step trace event spans the duration of the code enclosed by the context.
  The profiler will provide the performance analysis for each step trace event.

  For example, it can be used to mark training steps and enable the profiler to
  provide the performance analysis per step:

  >>> import jax
  >>>
  >>> while global_step < NUM_STEPS:
  ...   with jax.profiler.StepTraceContext("train", step_num=global_step):
  ...     train_step()
  ...     global_step += 1

  This will cause a "train xx" event to show up on the trace timeline if the
  event occurs while the process is being traced by TensorBoard. In addition,
  if using accelerators, the device trace timeline will also show a "train xx"
  event. Note that "step_num" can be set as a keyword argument to pass the
  global step number to the profiler.

  """

  def __init__(self, name: str, **kwargs):
    super().__init__(name, _r=1, **kwargs)


def trace_function(func: Callable, name: str = None, **kwargs):
  """Decorator that generates a trace event for the execution of a function.

  For example:

  >>> import jax, jax.numpy as jnp
  >>>
  >>> @jax.profiler.trace_function
  >>> def f(x):
  ...   return jnp.dot(x, x.T).block_until_ready()
  >>>
  >>> f(jnp.ones((1000, 1000))

  This will cause an "f" event to show up on the trace timeline if the
  function execution occurs while the process is being traced by TensorBoard.

  Arguments can be passed to the decorator via :py:func:`functools.partial`.

  >>> import jax, jax.numpy as jnp
  >>> from functools import partial
  >>>
  >>> @partial(jax.profiler.trace_function, name="event_name")
  >>> def f(x):
  ...   return jnp.dot(x, x.T).block_until_ready()
  >>>
  >>> f(jnp.ones((1000, 1000))
  """

  name = name or getattr(func, '__qualname__', None)
  name = name or func.__name__
  @wraps(func)
  def wrapper(*args, **kwargs):
    with TraceContext(name, **kwargs):
      return func(*args, **kwargs)
    return wrapper
  return wrapper


def device_memory_profile(backend: Optional[str] = None) -> bytes:
  """Captures a JAX device memory profile as ``pprof``-format protocol buffer.

  A device memory profile is a snapshot of the state of memory, that describes the JAX
  :class:`jax.DeviceArray` and executable objects present in memory and their
  allocation sites.

  For more information how to use the device memory profiler, see
  :doc:`/device_memory_profiling`.

  The profiling system works by instrumenting JAX on-device allocations,
  capturing a Python stack trace for each allocation. The instrumentation is
  always enabled; :func:`device_memory_profile` provides an API to capture it.

  The output of :func:`device_memory_profile` is a binary protocol buffer that
  can be interpreted and visualized by the `pprof tool
  <https://github.com/google/pprof>`_.

  Args:
    backend: optional; the name of the JAX backend for which the device memory
      profile should be collected.

  Returns:
    A byte string containing a binary `pprof`-format protocol buffer.
  """
  return xla_client.heap_profile(xla_bridge.get_backend(backend))


def save_device_memory_profile(filename, backend: Optional[str] = None):
  """Collects a device memory profile and writes it to a file.

  :func:`save_device_memory_profile` is a convenience wrapper around :func:`device_memory_profile`
  that saves its output to a ``filename``. See the
  :func:`device_memory_profile` documentation for more information.

  Args:
    filename: the filename to which the profile should be written.
    backend: optional; the name of the JAX backend for which the device memory
      profile should be collected.
  """
  profile = device_memory_profile(backend)
  with open(filename, "wb") as f:
    f.write(profile)
