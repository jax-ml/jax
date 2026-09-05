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
import dataclasses

@dataclasses.dataclass(frozen=True, slots=True)
class LocalityDomain:
  """Identifies a locality domain for memory placement."""

  id: int

  def __post_init__(self):
    if isinstance(self.id, bool) or not isinstance(self.id, int):
      raise TypeError(
          "LocalityDomain id must be an integer, "
          f"but got {self.id!r} of type {type(self.id).__name__}."
      )
    max_locality_domain_id = ((1 << 63) - 1) - 16
    if not 0 <= self.id <= max_locality_domain_id:
      raise ValueError(
          f"LocalityDomain id must be between 0 and {max_locality_domain_id}, "
          f"but got {self.id}."
      )

  @property
  def memory_kind(self) -> str:
    """The canonical XLA memory kind for this locality domain."""
    return f"locality_domain:{self.id}"


def _parse_locality_domain_memory_kind(
    memory_kind: str,
) -> LocalityDomain:
  """Parses a canonical `locality_domain:<d>` memory kind."""
  if not memory_kind.startswith("locality_domain:"):
    raise ValueError(
      f"Locality-domain memory kind must start with 'locality_domain:', got {memory_kind!r}."
    )
  domain_id = memory_kind[16:]
  if (
      not domain_id
      or not domain_id.isascii()
      or not domain_id.isdecimal()
      or (len(domain_id) > 1 and domain_id.startswith("0"))
  ):
    raise ValueError(
        "Locality-domain memory kind must contain a canonical unsigned "
        f"decimal id, got {memory_kind!r}."
    )
  return LocalityDomain(int(domain_id))
