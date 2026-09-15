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

"""Per-call custom options forwarded to the runtime."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import contextlib

from jax._src.lib import _jax


@contextlib.contextmanager
def custom_options(
    **options: bool | int | float | str | bytes | Sequence[int],
) -> Iterator[None]:
  """Context manager that attaches custom options to executions.

  Every computation dispatched from the current thread within this context
  forwards ``options`` to the runtime as
  ``xla::ifrt::ExecuteOptions::custom_options``. Custom options are not part of
  the compiled program and can be different for every call of the same
  compiled function.

  The same options can be passed per call via the reserved ``custom_options``
  keyword argument of jitted and AOT compiled functions, e.g.
  ``f(x, custom_options={"foo": 42})``, which takes precedence over the
  enclosing contexts. Nested contexts are merged, innermost wins.

  Args:
    **options: options to forward. Supported value types are ``bool``, ``int``
      (forwarded as a 64-bit integer), ``float`` (forwarded as a 32-bit float),
      ``str``, ``bytes`` and sequences of ``int``.
  """
  previous = _jax.get_custom_options_thread_local()
  merged = dict(previous or {})
  merged.update(options)
  _jax.set_custom_options_thread_local(merged)
  try:
    yield
  finally:
    _jax.set_custom_options_thread_local(previous)
