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

"""Buffer donation recovery for JAX."""

from __future__ import annotations

from typing import Any

from jax._src import tree_util
from jax._src.lib import donation_recovery as _donation_recovery
from jax._src.lib import xla_version


class RecoveryFuture:
  """Future representing the recovery status of donated arguments."""

  def __init__(self, raw_future: Any):
    self._raw_future = raw_future

  def is_ready(self) -> bool:
    """Non-blocking check for execution completion."""
    return self._raw_future.is_ready()

  def get(self) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Awaits execution completion.

    Returns:
      None: If execution completed successfully (buffers consumed as intended).
      (recovered_args, recovered_kwargs): Full PyTree reconstructed input
        signature with newly minted jax.Array objects at donated leaf positions
        on surviving devices and original references at non-donated / static
        positions.

    Raises:
      Exception: If execution failed unrecoverably.
    """
    leaves, in_treedef = tree_util.dispatch_registry.flatten(
        self._raw_future.py_args
    )
    recovered_leaves = self._raw_future.get(leaves)
    if recovered_leaves is None:
      return None
    return in_treedef.unflatten(recovered_leaves)


class DonationRecovery:
  """Context manager for recovering donated argument buffers on failure."""

  def __init__(self):
    if _donation_recovery is None or xla_version < 91:
      raise RuntimeError(
          "DonationRecovery requires jaxlib_extension_version >= 527 and "
          "xla_version >= 91."
      )
    self._scope = _donation_recovery.DonationRecoveryScope()

  def __enter__(self) -> DonationRecovery:
    self._scope.enter()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._scope.exit()

  def future(self) -> RecoveryFuture:
    """Returns the RecoveryFuture for the single JIT execution in this scope.

    Raises:
      RuntimeError: If no JIT execution occurred within the scope.
    """
    return RecoveryFuture(self._scope.future())
