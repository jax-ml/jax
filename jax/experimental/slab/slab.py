# Copyright 2024 The JAX Authors.
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

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Union
import sys

import numpy as np

import jax
import jax.numpy as jnp

from jax._src import util

map, zip = util.safe_map, util.safe_zip

Address = jax.Array
DShape = tuple[Union[int, jax.Array]]
SShape = tuple[int]

class Slab(NamedTuple):
  data: jax.Array
  cursor: Address

@jax.tree_util.register_pytree_node_class
class SlabView(NamedTuple):
  addr: Address
  shape: DShape
  dtype: jax.typing.DTypeLike

  def size(self):
    return jnp.prod(jnp.array(self.shape))

  def ndim(self):
    return len(self.shape)

  def tree_flatten(self):
    return (self.addr, self.shape), self.dtype

  @classmethod
  def tree_unflatten(cls, dtype, xs):
    addr, shape = xs
    return cls(addr, shape, dtype)

word_sz = 512

def slab_make(num_vmem_words):
  return Slab(jnp.zeros((num_vmem_words, word_sz), dtype=jnp.uint8),
              jnp.array(0, dtype=jnp.int32))

def slab_alloc(slab, shape, dtype):
  num_elts = jnp.prod(jnp.array(shape))
  mem_sz = (num_elts * dtype.itemsize + word_sz - 1) // word_sz
  new_slab = Slab(slab.data, slab.cursor + mem_sz)
  slab_val = SlabView(slab.cursor, shape, dtype)
  return new_slab, slab_val

def slab_read(slab, view, word_offset, tile_sshape_e):
  assert view.ndim() == len(tile_sshape_e)
  num_words, rem = divmod(
      np.prod(tile_sshape_e) * view.dtype.itemsize, word_sz)
  assert not rem
  mem = jax.lax.dynamic_slice_in_dim(
      slab.data, view.addr + word_offset, num_words, axis=0)
  cast = jax.lax.bitcast_convert_type(
      mem.reshape(-1, view.dtype.itemsize), view.dtype)
  return cast.reshape(tile_sshape_e)

# TODO: just take vjp of slab_read
def slab_write(slab, view, word_offset, arr):
  assert view.ndim() == arr.ndim
  assert view.dtype == arr.dtype, (view.dtype, arr.dtype)
  cast = arr.ravel()
  # TODO: bct has a different output shape if itemsizes are equal in/out.
  # Handle the special case here and in slab_read's bct use.
  mem = jax.lax.bitcast_convert_type(cast, jnp.uint8).reshape(-1, word_sz)
  data = jax.lax.dynamic_update_slice_in_dim(
      slab.data, mem, view.addr + word_offset, axis=0)
  return Slab(data, slab.cursor)

def elemwise_loop_cond(tile_sshape_e, kernel,
                       slab, cursor, end, operands, results):
  return cursor < end

def elemwise_loop_body(tile_sshape_e, kernel,
                       slab, cursor, end, operands, results):
  dtype, = {o.dtype for o in (*operands, *results)}
  in_tiles = [slab_read(slab, x, cursor, tile_sshape_e) for x in operands]
  out_tiles = kernel(*in_tiles)
  for y, r in zip(out_tiles, results):
    slab = slab_write(slab, r, cursor, y)
  inc, rem = divmod(np.prod(tile_sshape_e) * dtype.itemsize, word_sz)
  assert not rem
  cursor = cursor + inc
  return slab, cursor, end, operands, results

def while_loop(cond, body, *args):
  def c(x): return cond(*x)
  def b(x): return body(*x)
  return jax.lax.while_loop(c, b, args)

def elementwise_tile_sshape_e(rank, dtype):
  assert rank > 0, rank
  if rank >= 2:
    return (1,) * (rank - 2) + (2, word_sz // dtype.itemsize)
  elif rank == 1:
    return (word_sz // dtype.itemsize * 2,)

@partial(jax.jit, static_argnums=0)
def elementwise_tile(kernel, slab: Slab,
                     operands: tuple[SlabView], results: tuple[SlabView]):
  rank,  = {v.ndim() for v in [*operands, *results]}
  dtype, = {v.dtype for v in [*operands, *results]}
  end_w = (operands[0].size() * dtype.itemsize) // word_sz
  tile_sshape_e = elementwise_tile_sshape_e(rank, dtype)
  slab, *_ = while_loop(partial(elemwise_loop_cond, tile_sshape_e, kernel),
                        partial(elemwise_loop_body, tile_sshape_e, kernel),
                        slab, 0, end_w, operands, results)
  return slab

def make_elementwise_op(name, op):
  def kernel(*args): return [op(*args)]

  def f(slab: Slab, *operands: tuple[SlabView]):
    x, *_ = operands
    slab, result = slab_alloc(slab, x.shape, x.dtype)
    slab = elementwise_tile(kernel, slab, operands, [result])
    return slab, result

  f.__name__ = name
  return f

add = make_elementwise_op('add', jax.lax.add)
mul = make_elementwise_op('mul', jax.lax.mul)

def parse_arr(i, s):
  shape = eval(s)
  z = 3 * i + jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
  return z

def main(args):
  xs = map(parse_arr, range(len(args)), args)

  sz = xs[0].size
  shape = xs[0].shape

  slab = slab_make(1024)

  vals = []
  for x in xs:
    slab, v = slab_alloc(slab, x.shape, x.dtype)
    slab = slab_write(slab, v, 0, x)
    vals.append(v)

  def f(slab, *vals):
    slab, y = add(slab, *vals)
    slab, z = mul(slab, y, vals[0])
    return slab, y, z

  print()
  print(jax.make_jaxpr(f)(slab, *vals))

  print()
  print('-- initial args:')
  for x in xs:
    print(x)

  print()
  print('-- args allocated on slab:')
  print('slab:', slab)
  print('ptrs:', vals)

  slab, y, z = jax.jit(f)(slab, *vals)
  print()
  print('-- slab ptr results')
  print('add:', y)
  print('mul:', z)
  print()
  print('-- read off slab')
  print(slab_read(slab, y, 0, shape).astype(jnp.int32))
  print(slab_read(slab, z, 0, shape).astype(jnp.int32))


if __name__ == '__main__':
  main(sys.argv[1:])
