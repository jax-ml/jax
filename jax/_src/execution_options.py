# Copyright 2026 The JAX Authors.
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

"""Per-execution options forwarded to the runtime."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import contextlib

from jax._src.lib import _jax

CustomOptionValue = bool | int | float | str | bytes | Sequence[int]


def current_execution_options() -> _jax.ExecutionOptions:
  """Returns the execution options in effect on the current thread."""
  return _jax.get_execution_options_thread_local()


@contextlib.contextmanager
def execution_options(
    *, custom_options: Mapping[str, CustomOptionValue] | None = None,
) -> Iterator[None]:
  """Context manager that sets per-execution options for the runtime.

  Every computation dispatched from the current thread within this context
  runs with the given execution options. Execution options are not part of the
  compiled program and can be different for every call of the same compiled
  function. Nested contexts are merged, innermost wins.

  Args:
    custom_options: runtime-defined key/value options, forwarded to the runtime
      as ``ifrt::ExecuteOptions::custom_options``. Supported value types are
      ``bool``, ``int`` (forwarded as a 64-bit integer), ``float`` (forwarded as
      a 32-bit float), ``str``, ``bytes`` and sequences of ``int``.
  """
  if custom_options is None:
    yield
    return
  previous = _jax.get_execution_options_thread_local()
  merged = dict(previous.custom_options or {})
  merged.update(custom_options)
  _jax.set_execution_options_thread_local(custom_options=merged)
  try:
    yield
  finally:
    _jax.set_execution_options_thread_local(
        custom_options=previous.custom_options)
