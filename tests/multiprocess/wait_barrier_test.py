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

"""Wait barrier test."""

import jax
from jax._src import distributed
from jax._src import test_multiprocess as jt_multiprocess


class WaitBarrierTest(jt_multiprocess.MultiProcessTest):

  def test_only_participants_call_succeeds(self):
    client = distributed.global_state.client
    timeout_in_ms = 1000

    # Only even process ids will participate in the barrier.
    barrier_participants = []
    for process_index in range(jax.process_count()):
      if process_index % 2 == 0:
        barrier_participants.append(process_index)

    if jax.process_index() % 2 == 0:
      client.wait_at_barrier(
          'only_even_participants_call',
          timeout_in_ms,
          process_ids=barrier_participants,
      )
    # This test is intended to implicitly verify that no exceptions are raised
    # when the barrier is called if only the barrier participants including
    # each one of them call the barrier. Thus there are no explicit assertions.

  def test_non_participant_calls_fails(self):
    client = distributed.global_state.client
    timeout_in_ms = 1000

    process_group = []
    for process_index in range(jax.process_count()):
      if process_index % 2 == 0:
        process_group.append(process_index)

    # Processes 0, 2, 4 ... wait here.
    # Processes 1, 3, 5 ... go ahead to the barrier call.
    if jax.process_index() % 2 == 0:
      client.blocking_key_value_get('sync', timeout_in_ms=1000)

    # 1, 3, 5 ... arrive and hit an error as they are non-participating.
    # 0, 2, 4 ... has not yet arrived here. They will arrive once 1 unblocks
    # them after leaving the barrier. But when they arrive at the barrier, they
    # would see the error state even though they are participating.
    with self.assertRaisesRegex(
        jax.errors.JaxRuntimeError,
        r'INVALID_ARGUMENT: A non-participating task.*'
    ):
      client.wait_at_barrier(
          'non_participant_calls', timeout_in_ms, process_ids=process_group
      )

    # 1 unblocks 0, 2 and 4.
    if jax.process_index() == 1:
      client.key_value_set('sync', 'process 1 exiting')

  def test_all_participate_succeeds(self):
    client = distributed.global_state.client
    timeout_in_ms = 1000
    client.wait_at_barrier('all_processes_call', timeout_in_ms)

  # This test checks that processes do wait in `wait_at_barrier`and do not
  # leave until all participating processes arrive.
  def test_one_participant_never_calls_fails(self):
    client = distributed.global_state.client
    timeout_in_ms = 1000

    if jax.process_index() != 0:
      with self.assertRaisesRegex(
          jax.errors.JaxRuntimeError, r'DEADLINE_EXCEEDED: Barrier timed out.*'
      ):
        client.wait_at_barrier('one_participant_never_calls', timeout_in_ms)


if __name__ == '__main__':
  jt_multiprocess.main()
