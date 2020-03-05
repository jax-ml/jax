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
from typing import Callable

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
