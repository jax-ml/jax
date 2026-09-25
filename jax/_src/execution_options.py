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
from jax._src.lib import jaxlib_extension_version

CustomOptionValue = bool | int | float | str | Sequence[int]


execution_options_context_manager = _jax.config.Config[
    dict[str, CustomOptionValue] | None
](
    'execution_options_context_manager',
    None,
    include_in_jit_key=False,
    include_in_trace_context=False,
)
if jaxlib_extension_version >= 500:
  _jax.set_execution_options_state(execution_options_context_manager)


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
      a 32-bit float), ``str`` and sequences of ``int``.
  """
  if custom_options is None:
    yield
    return
  merged = dict(execution_options_context_manager.value or {})
  merged.update(custom_options)
  previous = execution_options_context_manager.swap_local(merged)
  try:
    yield
  finally:
    execution_options_context_manager.set_local(previous)
