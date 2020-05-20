# Copyright 2020 Google LLC
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

'''Extensions to lax loop-related functions that partially unroll loops.'''

from jax import api, lax

from functools import partial
from typing import Callable


def unrolled_scan(f: Callable, init, xs, length=None, reverse=False):
  with api.disable_jit():
    return lax.scan(f, init, xs, length=length, reverse=reverse)


def block_unrolled_scan(block: int, f: Callable, init, xs,
                        length=None, reverse=False):
  '''Block-unroll `lax.scan`.

  Bottoms out in a call to `lax.scan`, each of whose iterations comprises an
  unrolled scan over `block` many iterations. Any remaining iterations are
  unrolled as well.

  Currently only supports single array `xs`.
  '''
  if block <= 0:
    raise ValueError(f'non-positive unroll block length: {block}')
  if reverse or length is not None:
    raise NotImplementedError
  num_blocks, rem = divmod(xs.shape[0], block)
  xs_div = xs[:num_blocks * block]
  if rem > 0:
    xs_rem = xs[num_blocks * block:]
  carry, ys = _block_unrolled_scan(block, f, init, xs_div)
  if rem > 0:
    carry, ys_rem = unrolled_scan(f, carry, xs_rem)
    ys = lax.concatenate((ys, ys_rem), 0)
  return carry, ys


def _block_unrolled_scan(block: int, f: Callable, init, xs):
  assert xs.shape[0] % block == 0
  f_block = partial(unrolled_scan, f)
  num_blocks = xs.shape[0] // block
  xs_blocks = lax.reshape(xs, (num_blocks, block, *xs.shape[1:]))
  carry, ys_blocks = lax.scan(f_block, init, xs_blocks)
  assert ys_blocks.shape[0] == num_blocks
  return carry, lax.reshape(
      ys_blocks, (num_blocks * ys_blocks.shape[1], *ys_blocks.shape[2:]))
