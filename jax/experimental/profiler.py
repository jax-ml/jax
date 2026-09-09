# Copyright 2023 The JAX Authors.
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

from collections.abc import Callable, Sequence
import dataclasses
import typing
import warnings
import jax
from jax._src import util
from jax._src.api import block_until_ready
from jax._src.lib import _profiler
from jax.profiler import ProfileData


def get_profiled_instructions_proto(tensorboard_dir: str) -> bytes:
  """Gets the serialized profiled instructions proto from profile.

  Restore the xplane from the tensorboard dir and convert it to profiled
  [ProfiledInstructionsProto](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/profiled_instructions.proto).
  The result is non-empty only when running on Nvidia GPU.

  Args:
    tensorboard_dir: The directory contains the profile trace, it is in the
      format of `<tb's log_dir>/plugin/profile/<run_dir>/`.

  Returns:
    Serialized
    [ProfiledInstructionsProto](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/profiled_instructions.proto).
  """
  return _profiler.get_profiled_instructions_proto(tensorboard_dir)


@dataclasses.dataclass
class _ThunkTiming:
  """Represents the timing information for a single XLA CPU thunk execution.

  Attributes:
    start_ns: The start time of the thunk in nanoseconds.
    hlo_op: The name of the HLO operation associated with the thunk.
    duration_ms: The duration of the thunk in milliseconds.
  """

  start_ns: int
  hlo_op: str
  duration_ms: float


def _parse_paired_timings(
    profile_data: ProfileData,
) -> Sequence[tuple[str, float]]:
  """Parses and pairs CPU producer and consumer thunk event timings.

  Args:
    profile_data: The captured profile data containing thunk events.

  Returns:
    A sequence of (op_name, duration_ms) pairs.
  """
  events_by_producer_id = {}
  events_by_consumer_id = {}

  for plane in profile_data.planes:
    if "CPU" not in plane.name and "Host Threads" not in plane.name:
      continue
    for line in plane.lines:
      for event in line.events:
        stats = dict(event.stats)
        if "_p" in stats:
          events_by_producer_id[stats["_p"]] = event
        elif "_c" in stats:
          events_by_consumer_id[stats["_c"]] = event

  timings = []
  for context_id, prod_event in events_by_producer_id.items():
    cons_event = events_by_consumer_id.get(context_id)
    if cons_event is None:
      continue
    duration_ms = (cons_event.end_ns - prod_event.start_ns) / 1e6
    stats = dict(prod_event.stats)
    hlo_op = stats.get("hlo_op") or prod_event.name or ""
    timings.append(_ThunkTiming(prod_event.start_ns, hlo_op, duration_ms))

  # Sort by start_ns to guarantee chronological order across multiple threads.
  timings.sort(key=lambda t: t.start_ns)

  return [(t.hlo_op, t.duration_ms) for t in timings]


def measure(
    f: Callable[..., typing.Any],
    *,
    aggregate: bool = True,
    iterations: int = 1,
    warmup: bool = True,
) -> Callable[..., tuple[typing.Any, typing.Any]]:
  """Measures the execution time of JAX operations on CPU.

  This function profiles the execution of `f` on XLA:CPU, capturing the duration
  of individual XLA thunks. It optionally excludes compilation warmup time.

  Args:
    f: The function to measure. It should be a JIT-compiled function to ensure
      XLA thunks are executed.
    aggregate: If True, returns the sum of all thunk durations. If False,
      returns a list of (op_name, duration_ms) pairs.
    iterations: The number of times to run the function. Must be >= 1.
    warmup: If True, runs the function once before profiling to ensure it is
      compiled, excluding compilation time from the measurement. If False,
      compilation is included in the profiler session (useful for exported
      traces), but the returned timings still only measure thunk execution
      (which will be "cold" execution for the first run, capturing
      initialization overhead).

  Returns:
    A wrapped function that, when called, returns a tuple `(result, timings)`:
      - `result`: The output of the first execution of `f`.
      - `timings`:
        - If `iterations == 1`:
          - If `aggregate == True`: A float representing total duration in ms.
          - If `aggregate == False`: A list of `(op_name, duration_ms)` tuples.
        - If `iterations > 1`:
          - If `aggregate == True`: A list of floats, one per iteration.
          - If `aggregate == False`: A list of lists of `(op_name, duration_ms)`
            tuples, one list per iteration.
        - If no thunks were captured (e.g. non-JIT execution), returns `None`
          for timings and emits a `RuntimeWarning`.

  Raises:
    RuntimeError: If JAX is not initialized for CPU, if another profiling
      session is active, or if the captured thunks are inconsistent across
      iterations.
  """
  if iterations < 1:
    raise ValueError(f"iterations={iterations} must be positive")

  def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    try:
      jax.devices("cpu")
    except RuntimeError as e:
      raise RuntimeError("JAX is not initialized for CPU.") from e

    if warmup:
      block_until_ready(f(*args, **kwargs))

    options = _profiler.ProfileOptions()
    options.host_tracer_level = 2
    # Note: ProfilerSession does not support the context manager protocol
    # (__enter__/__exit__). Its lifetime is explicitly managed via try...except
    # blocks ensuring .stop() / .stop_and_get_profile_data() is always invoked.
    session = None
    try:
      session = _profiler.ProfilerSession(options)
      res = f(*args, **kwargs)
      block_until_ready(res)
      for _ in range(iterations - 1):
        r = f(*args, **kwargs)
        block_until_ready(r)
      profile_data = session.stop_and_get_profile_data()
    except Exception as e:
      if session is not None:
        try:
          session.stop()
        except Exception:
          pass
      if getattr(e, "error_code_string", None) == "ALREADY_EXISTS":
        raise RuntimeError(
            "Attempted to start CPU profiler measure, but another profiler "
            "session is already active in this process."
        ) from e
      raise

    timings = _parse_paired_timings(profile_data)

    if not timings:
      warnings.warn(
          "No CPU thunk events were captured. The function may have run "
          "completely outside XLA or too fast to profile.",
          RuntimeWarning,
      )
      return res, None

    if len(timings) % iterations != 0:
      raise RuntimeError(
          "The number of captured thunks is not divisible by the number of"
          f" iterations. Expected multiple of {iterations}, got {len(timings)}"
      )

    thunks_per_iter = len(timings) // iterations
    iter_timings = util.split_list(
        timings, [thunks_per_iter] * (iterations - 1)
    )

    for thunk_idx, (name, _) in enumerate(iter_timings[0]):
      for i in range(1, iterations):
        other_name, _ = iter_timings[i][thunk_idx]
        if other_name != name:
          raise RuntimeError(
              "Captured thunk names are not consistent across iterations: "
              f"expected {name!r}, got {other_name!r}"
          )

    if aggregate:
      agg_timings = [
          sum(duration for _, duration in it_t) for it_t in iter_timings
      ]
      return res, agg_timings[0] if iterations == 1 else agg_timings

    return res, iter_timings[0] if iterations == 1 else iter_timings

  return wrapper
