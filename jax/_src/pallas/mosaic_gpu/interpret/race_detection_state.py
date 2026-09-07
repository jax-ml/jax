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

import collections
import dataclasses

from jax._src import source_info_util
from jax._src.pallas.mosaic.interpret import race_detection_state
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic_gpu.interpret.shared_memory import HostAllocationKey


@dataclasses.dataclass
class GPURaceDetectionState(race_detection_state.RaceDetectionState):
  # The members of a `RefUnion` are modelled as independent buffers. Every
  # access to a member is also compared against all prior accesses through
  # *other* alias groups of the same union: an unordered pair with a write
  # races, and an ordered read through a group other than the one that last
  # wrote raises, because this violates safe usage of RefUnions according to the
  # mgpu contract. Members of one group are laid out disjointly, so between them
  # only the ordinary per-buffer check applies.

  # member key -> (union, alias group). A union is identified by the key of
  # its first member. Keys are never reused, so entries are simply left behind
  # on deallocation.
  ref_union_members: dict[HostAllocationKey, tuple[HostAllocationKey, int]] = (
      dataclasses.field(default_factory=dict)
  )

  # union -> [(thread, clock, alias group, is_write, user_frame)]
  union_accesses: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )

  def tag_ref_union_member(
      self, key: HostAllocationKey, union: HostAllocationKey, alias_group: int
  ):
    with self.lock:
      self.ref_union_members[key] = (union, alias_group)

  def check_read(
      self,
      thread,
      clock,
      buffer_key,
      rnge: interpret_utils.Access,
      source_info=None,
  ):
    super().check_read(thread, clock, buffer_key, rnge.range, source_info)
    self._check_ref_union(thread, clock, buffer_key, False, source_info)

  def check_write(
      self,
      thread,
      clock,
      buffer_key,
      rnge: interpret_utils.Access,
      source_info=None,
  ):
    super().check_write(thread, clock, buffer_key, rnge.range, source_info)
    self._check_ref_union(thread, clock, buffer_key, True, source_info)

  def _check_ref_union(self, thread, clock, buffer_key, is_write, source_info):
    with self.lock:
      membership = self.ref_union_members.get(buffer_key)
    if membership is None:
      return
    union, group = membership
    if source_info is not None:
      user_frame = source_info_util.summarize(source_info)
    else:
      user_frame = "pallas_call"

    with self.lock:
      accesses = self.union_accesses[union]
      prior = accesses[:]
      accesses.append((thread, clock, group, is_write, user_frame))

    def describe(writing, in_group, frame):
      kind = "write" if writing else "read"
      return f"{kind} of alias group {in_group}, {frame}"

    # "Last" is by arrival order. That agrees with happens-before wherever the
    # two accesses are ordered, since an access that happens-before another
    # has also executed before it here; an unordered pair is reported as a
    # race below regardless of which arrived first.
    last_write = max((i for i, a in enumerate(prior) if a[3]), default=None)
    for i, other in enumerate(prior):
      other_thread, other_clock, other_group, other_write, other_frame = other
      if other_group == group or not (is_write or other_write):
        continue
      if other_clock.ordered(clock):
        if is_write or i != last_write:
          continue
        raise ValueError(
            f"Reading `{buffer_key}` through alias group {group}, which was"
            f" last written through alias group {other_group}"
            f" ({other_frame}). Per `RefUnion`: ref unions are only safe to"
            " use when the groups of refs that we intend to alias have"
            " disjoint lifetimes (i.e. one should never attempt to read data"
            " using a different ref than the one that was used to write the"
            " data)."
        )
      print(
          f"RACE DETECTED\n  {describe(is_write, group, user_frame)} from"
          f" {thread}\n  clock: {clock}\n "
          f" {describe(other_write, other_group, other_frame)} from"
          f" {other_thread}\n  clock: {other_clock}\n"
      )
      with self.lock:
        self.races_found = True
      return
