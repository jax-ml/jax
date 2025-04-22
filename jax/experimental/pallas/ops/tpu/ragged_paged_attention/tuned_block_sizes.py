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

"""Auto-tuned block sizes for ragged paged attention."""

import jax


# TODO: add more tuned block sizes in the table
# ragged_paged_attention
# key: (num_q_head, num_kv_head, num_q_tokens, max_model_len)
# value: (num_kv_pages_per_block, num_queries_per_block)
TUNED_BLOCK_SIZES = {
    # go/keep-sorted start
    (1, 1, 1024, 128): (32, 32),
    (1, 1, 1024, 2048): (64, 32),
    (1, 1, 1024, 4096): (64, 32),
    (1, 1, 1024, 64): (32, 32),
    (32, 8, 1024, 128): (32, 32),
    (32, 8, 1024, 2048): (64, 32),
    (32, 8, 1024, 4096): (64, 32),
    (32, 8, 1024, 64): (32, 32),
    (32, 8, 2048, 128): (32, 32),
    (32, 8, 2048, 2048): (128, 32),
    (32, 8, 2048, 4096): (128, 32),
    (32, 8, 2048, 64): (32, 32),
    (32, 8, 4096, 128): (32, 32),
    (32, 8, 4096, 2048): (128, 64),
    (32, 8, 4096, 4096): (128, 64),
    (32, 8, 4096, 64): (32, 32),
    (4, 1, 2048, 128): (32, 32),
    (4, 1, 2048, 2048): (128, 64),
    (4, 1, 2048, 4096): (128, 64),
    (4, 1, 2048, 64): (32, 32),
    (4, 1, 4096, 128): (32, 32),
    (4, 1, 4096, 2048): (128, 128),
    (4, 1, 4096, 4096): (128, 128),
    (4, 1, 4096, 64): (32, 32),
    # go/keep-sorted end
}


def next_power_of_2(x: int):
  """Finds the smallest power of 2 >= x using bit manipulation.

  Args:
    x: The input number (should be an integer).

  Returns:
    The smallest integer power of 2 that is >= x.
  """
  assert x > 0
  if x == 1:
    return 1
  return 1 << (x - 1).bit_length()


def simplify_key(num_q_head, num_kv_head, num_q_tokens, max_model_len):
  num_q_tokens = next_power_of_2(num_q_tokens)
  max_model_len = next_power_of_2(max_model_len)
  return num_q_head, num_kv_head, num_q_tokens, max_model_len


def get_tpu_version() -> int:
  """Returns the numeric version of the TPU, or -1 if not on TPU."""
  kind = jax.devices()[0].device_kind
  if 'TPU' not in kind:
    return -1
  if kind.endswith(' lite'):
    kind = kind[: -len(' lite')]
  assert kind[:-1] == 'TPU v', kind
  return int(kind[-1])


def get_tuned_block_sizes(
    num_q_head, num_kv_head, num_q_tokens, page_size, pages_per_seq
) -> tuple[int, int]:
  """Searchs for best (num_kv_pages_per_blk, num_queries_per_blk)."""
  if get_tpu_version() < 4:
    raise NotImplementedError("TPU version must be 4 or higher.")
  if get_tpu_version() == 4:
    # This default block size is not tuned, only make sure there's no
    # OOM in vmem
    num_kv_pages_per_blk = 16
    num_queries_per_blk = 128
    return num_kv_pages_per_blk, num_queries_per_blk

  max_model_len = pages_per_seq * page_size
  key = simplify_key(num_q_head, num_kv_head, num_q_tokens, max_model_len)
  num_kv_pages_per_blk, num_queries_per_blk = TUNED_BLOCK_SIZES.get(
      key, (128, 32)
  )
  num_kv_pages_per_blk = min(num_kv_pages_per_blk, pages_per_seq)
  num_queries_per_blk = min(num_queries_per_blk, num_q_tokens)
  return num_kv_pages_per_blk, num_queries_per_blk
