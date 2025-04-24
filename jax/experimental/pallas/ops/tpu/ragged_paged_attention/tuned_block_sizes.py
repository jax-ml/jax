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
import jax.numpy as jnp

# The page size is too small. We only have 32 SREGs in TC. If the pages
# per seq is too large, SREGs will spill.
MAX_PAGES_PER_SEQ = 16

# key:
#     - q_dtype_name
#     - kv_dtype_name
#     - num_q_heads_per_blk
#     - num_kv_heads_per_blk
#     - head_dim
#     - page_size
#     - max_num_batched_tokens
#     - max_model_len = page_size * pages_per_seq
# value:
#     - num_kv_pages_per_block
#     - num_queries_per_block
TUNED_BLOCK_SIZES = {
    'TPU v6': {
        # go/keep-sorted start
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 1024, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 1024, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 1024, 4096): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 1024, 512): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 2048, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 2048, 2048): (16, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 2048, 4096): (32, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 2048, 512): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 4096, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 4096, 2048): (16, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 4096, 4096): (32, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 4096, 512): (4, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 512, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 512, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 512, 4096): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 512, 512): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 1024, 1024): (64, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 1024, 128): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 1024, 2048): (128, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 1024, 256): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 1024, 512): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 1024, 64): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 2048, 1024): (64, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 2048, 128): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 2048, 2048): (128, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 2048, 256): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 2048, 512): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 2048, 64): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 4096, 1024): (64, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 4096, 128): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 4096, 2048): (128, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 4096, 256): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 4096, 512): (32, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 4096, 64): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 512, 1024): (64, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 512, 128): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 512, 2048): (128, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 512, 256): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 512, 512): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 512, 64): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 1024, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 1024, 2048): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 1024, 4096): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 2048, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 2048, 2048): (8, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 2048, 4096): (16, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 4096, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 4096, 2048): (8, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 4096, 4096): (16, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 512, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 512, 2048): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 512, 4096): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 1024, 1024): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 1024, 128): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 1024, 2048): (64, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 1024, 256): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 1024, 4096): (128, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 1024, 512): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 2048, 1024): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 2048, 128): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 2048, 2048): (64, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 2048, 256): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 2048, 4096): (128, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 2048, 512): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 4096, 1024): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 4096, 128): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 4096, 2048): (64, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 4096, 256): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 4096, 4096): (128, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 4096, 512): (16, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 512, 1024): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 512, 128): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 512, 2048): (64, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 512, 256): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 512, 4096): (128, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 512, 512): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 1024, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 1024, 2048): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 1024, 256): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 1024, 4096): (64, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 1024, 512): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 2048, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 2048, 2048): (32, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 2048, 256): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 2048, 4096): (64, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 2048, 512): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 4096, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 4096, 2048): (32, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 4096, 256): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 4096, 4096): (64, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 4096, 512): (8, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 512, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 512, 2048): (32, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 512, 256): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 512, 4096): (64, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 512, 512): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 1024, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 1024, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 1024, 4096): (32, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 1024, 512): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 2048, 1024): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 2048, 2048): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 2048, 4096): (32, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 2048, 512): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 4096, 1024): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 4096, 2048): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 4096, 4096): (32, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 4096, 512): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 512, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 512, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 512, 4096): (32, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 512, 512): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 1024, 1024): (64, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 1024, 128): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 1024, 2048): (128, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 1024, 256): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 1024, 512): (32, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 1024, 64): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 2048, 1024): (64, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 2048, 128): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 2048, 2048): (128, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 2048, 256): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 2048, 512): (32, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 2048, 64): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 4096, 1024): (64, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 4096, 128): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 4096, 2048): (128, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 4096, 256): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 4096, 512): (32, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 4096, 64): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 512, 1024): (64, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 512, 128): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 512, 2048): (128, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 512, 256): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 512, 512): (32, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 512, 64): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 1024, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 1024, 2048): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 1024, 4096): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 2048, 1024): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 2048, 2048): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 2048, 4096): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 4096, 1024): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 4096, 2048): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 4096, 4096): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 512, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 512, 2048): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 512, 4096): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 1024, 1024): (32, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 1024, 128): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 1024, 2048): (64, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 1024, 256): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 1024, 4096): (128, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 1024, 512): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 2048, 1024): (32, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 2048, 128): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 2048, 2048): (64, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 2048, 256): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 2048, 4096): (64, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 2048, 512): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 4096, 1024): (32, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 4096, 128): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 4096, 2048): (64, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 4096, 256): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 4096, 4096): (64, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 4096, 512): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 512, 1024): (32, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 512, 128): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 512, 2048): (64, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 512, 256): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 512, 4096): (128, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 512, 512): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 1024, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 1024, 2048): (32, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 1024, 256): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 1024, 4096): (64, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 1024, 512): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 2048, 1024): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 2048, 2048): (32, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 2048, 256): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 2048, 4096): (64, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 2048, 512): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 4096, 1024): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 4096, 2048): (32, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 4096, 256): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 4096, 4096): (64, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 4096, 512): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 512, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 512, 2048): (32, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 512, 256): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 512, 4096): (64, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 512, 512): (8, 32),
        # go/keep-sorted end
    },
    'TPU v5': {
        # go/keep-sorted start
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 1024, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 1024, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 1024, 512): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 2048, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 2048, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 2048, 512): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 4096, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 4096, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 4096, 512): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 512, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 512, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 128, 512, 512): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 1024, 128): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 1024, 256): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 1024, 64): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 2048, 128): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 2048, 256): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 2048, 64): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 4096, 128): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 4096, 256): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 4096, 64): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 512, 128): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 512, 256): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 16, 512, 64): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 1024, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 1024, 2048): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 1024, 4096): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 2048, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 2048, 2048): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 2048, 4096): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 4096, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 4096, 2048): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 4096, 4096): (16, 64),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 512, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 512, 2048): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 256, 512, 4096): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 1024, 128): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 1024, 256): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 1024, 512): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 2048, 128): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 2048, 256): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 2048, 512): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 4096, 128): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 4096, 256): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 4096, 512): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 512, 128): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 512, 256): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 32, 512, 512): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 1024, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 1024, 256): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 1024, 512): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 2048, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 2048, 256): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 2048, 512): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 4096, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 4096, 256): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 4096, 512): (8, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 512, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 512, 256): (4, 32),
        ('bfloat16', 'bfloat16', 32, 8, 128, 64, 512, 512): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 1024, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 1024, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 1024, 512): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 2048, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 2048, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 2048, 512): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 4096, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 4096, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 4096, 512): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 512, 1024): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 512, 2048): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 128, 512, 512): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 1024, 128): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 1024, 256): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 1024, 64): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 2048, 128): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 2048, 256): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 2048, 64): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 4096, 128): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 4096, 256): (16, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 4096, 64): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 512, 128): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 512, 256): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 16, 512, 64): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 1024, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 1024, 2048): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 1024, 4096): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 2048, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 2048, 2048): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 2048, 4096): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 4096, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 4096, 2048): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 4096, 4096): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 512, 1024): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 512, 2048): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 256, 512, 4096): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 1024, 128): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 1024, 256): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 1024, 512): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 2048, 128): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 2048, 256): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 2048, 512): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 4096, 128): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 4096, 256): (8, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 4096, 512): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 512, 128): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 512, 256): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 32, 512, 512): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 1024, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 1024, 256): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 1024, 512): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 2048, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 2048, 256): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 2048, 512): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 4096, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 4096, 256): (4, 64),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 4096, 512): (8, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 512, 1024): (16, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 512, 256): (4, 32),
        ('bfloat16', 'bfloat16', 8, 1, 128, 64, 512, 512): (8, 32),
        # go/keep-sorted end
    },
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


def simplify_key(key):
  """Simplify the key to reduce the number of combinations."""
  (
      q_dtype,
      kv_dtype,
      num_q_heads_per_blk,
      num_kv_heads_per_blk,
      head_dim,
      page_size,
      max_num_batched_tokens,
      pages_per_seq,
  ) = key
  return (
      jnp.dtype(q_dtype).name,
      jnp.dtype(kv_dtype).name,
      next_power_of_2(num_q_heads_per_blk),
      next_power_of_2(num_kv_heads_per_blk),
      (head_dim + 127) // 128 * 128,
      next_power_of_2(page_size),
      next_power_of_2(max_num_batched_tokens),
      next_power_of_2(page_size * pages_per_seq),
  )


def get_tpu_version() -> int:
  """Returns the numeric version of the TPU, or -1 if not on TPU."""
  kind = jax.devices()[0].device_kind
  if 'TPU' not in kind:
    return -1
  if kind.endswith(' lite'):
    kind = kind[: -len(' lite')]
  assert kind[:-1] == 'TPU v', kind
  return int(kind[-1])


def get_device_name(num_devices:int | None = None):
  name = ' '.join(jax.devices()[0].device_kind.split()[:2])
  if num_devices is not None:
    name += f'-{num_devices}'
  return name


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    num_q_heads_per_blk,
    num_kv_heads_per_blk,
    head_dim,
    page_size,
    max_num_batched_tokens,
    pages_per_seq,
) -> tuple[int, int]:
  """Look up for the best (num_kv_pages_per_blk, num_queries_per_blk) from auto-tuned table."""
  tpu_version = get_tpu_version()
  if tpu_version < 4:
    raise NotImplementedError('TPU version must be 4 or higher.')
  key = (
      q_dtype,
      kv_dtype,
      num_q_heads_per_blk,
      num_kv_heads_per_blk,
      head_dim,
      page_size,
      max_num_batched_tokens,
      pages_per_seq,
  )
  key = simplify_key(key)
  device_name = get_device_name()

  # Default block sizes.
  bkv, bq = (128, 32)
  if tpu_version == 4:
    # This default block size is not tuned, only make sure there's no
    # OOM in vmem
    bkv, bq = (32, 32)
  elif device_name in TUNED_BLOCK_SIZES:
    if key in TUNED_BLOCK_SIZES[device_name]:
      bkv, bq = TUNED_BLOCK_SIZES[device_name][key]
  return (min(pages_per_seq, bkv), min(max_num_batched_tokens, bq))


def get_min_page_size(max_model_len, min_page_size=16):
  """Recommended min page size for high-performance kernel."""
  return max(next_power_of_2(max_model_len) // MAX_PAGES_PER_SEQ, min_page_size)
