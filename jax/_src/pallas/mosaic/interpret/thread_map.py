# Copyright 2025 The JAX Authors.
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

from concurrent import futures
import functools

import jax
from jax._src import callback
import jax.core as jax_core
import jax.numpy as jnp


def _run_jaxpr(jaxpr, consts, *args):
  def _run(jaxpr, consts, *args):
    jax_core.eval_jaxpr(jaxpr, consts, *args)

  traced = jax.jit(_run, static_argnums=(0,)).trace(jaxpr, consts, *args)
  traced.lower().compile()(consts, *args)
  return


def _thread_map_callback(jaxpr, num_threads, consts, invals):
  num_threads = int(num_threads)
  threads = []
  with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    for i in range(num_threads):
      # `jaxpr` is the traced representation of a function whose first argument
      # is the thread ID. Hence,
      #   - prepend the thread ID onto the `invals`; and
      #   - flatten the arguments that are to be passed through to the
      #     evaluation of `jaxpr`.
      args = (jnp.int32(i), *invals)
      flat_args, _ = jax.tree.flatten(args)
      threads.append(executor.submit(_run_jaxpr, jaxpr, consts, *flat_args))
    exceptions = []
    for i in range(num_threads):
      try:
        threads[i].result()
      except Exception as e:
        exceptions.append(e)
  if exceptions:
    # TODO(jburnim): Use ExceptionGroup once JAX requires Python 3.11.
    # raise ExceptionGroup('Exceptions raised during _thread_map', exceptions)
    raise exceptions[0]


def _call_threadmap_callback(jaxpr, num_threads, consts, invals):
  # NOTE: At runtime, _thread_map_callback will lower and compile the
  # given jaxpr.  (JAX's caches should ensure the jaxpr is only lowered and
  # compiled once.)
  #
  # TODO(jburnim): Would it be worth trying to lower/compile the jaxpr at
  # lowering/compilation time?  E.g., by using a custom primitive here, could
  # we lower/compile jaxpr at lowering time, and then pass the compiled
  # function to the callback?
  return callback.io_callback(
      functools.partial(_thread_map_callback, jaxpr),
      (),
      num_threads,
      consts,
      invals,
      ordered=True,
  )


def thread_map(f, num_threads, *args):
  """Executes `f(thread_id, *args)` for `num_threads` threads."""

  if num_threads == 1:
    f(jnp.int32(0), *args)
    return

  def _f(core_or_thread_index, *args):
    f(core_or_thread_index, *args)
    return ()

  jaxpr = jax.make_jaxpr(_f)(jnp.int32(0), *args)

  _call_threadmap_callback(jaxpr.jaxpr, num_threads, jaxpr.consts, args)
