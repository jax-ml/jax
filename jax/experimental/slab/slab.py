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

Slab = jax.Array
Address = jax.Array
DShape = tuple[Union[int, jax.Array]]
SShape = tuple[int]

block_sz = 2

class Slab(NamedTuple):
  data: jax.Array
  cursor: Address

class SlabView(NamedTuple):
  addr: Address
  shape: DShape
  # We'll want dtypes eventually. For now, everything is f32.
  #dtype: jax.typing.DTypeLike

  def size(self):
    return jnp.prod(jnp.array(self.shape))

  def ndim(self):
    return len(self.shape)

def slab_make(sz, dtype):
  return Slab(jnp.zeros(sz, dtype=dtype), jnp.array(0, dtype=int))

def slab_alloc(slab, shape):
  sz = jnp.prod(jnp.array(shape))
  new_slab = Slab(slab.data, slab.cursor + sz)
  slab_val = SlabView(slab.cursor, shape)
  return new_slab, slab_val

def slab_read(slab, addr, shape):
  sz = np.prod(shape)
  flat = jax.lax.dynamic_slice_in_dim(slab.data, addr, sz, axis=0)
  return flat.reshape(shape)

def slab_write(slab, addr, y):
  flat = jnp.ravel(y)
  data = jax.lax.dynamic_update_slice_in_dim(slab.data, flat, addr, axis=0)
  return Slab(data, slab.cursor)

def tile_loop_bounds(operands):
  x, *_ = operands
  x_sz = x.size()
  return 0, x_sz

def tile_loop_cond(kernel, slab, cursor, end, operands, results):
  return cursor < end

def tile_loop_body(kernel, slab, cursor, end, operands, results):
  x, *_ = operands
  in_tiles = [slab_read(slab, x.addr + cursor, (block_sz,) * x.ndim())
              for x in operands]
  out_tiles = kernel(*in_tiles)
  for y, r in zip(out_tiles, results):
    slab = slab_write(slab, r.addr + cursor, y)
  cursor = cursor + block_sz * x.ndim()
  return slab, cursor, end, operands, results

def while_loop(cond, body, *args):
  def c(x): return cond(*x)
  def b(x): return body(*x)
  return jax.lax.while_loop(c, b, args)

@partial(jax.jit, static_argnums=0)
def tile(kernel, slab: Slab, operands: tuple[SlabView], results: tuple[SlabView]):
  start, end = tile_loop_bounds(operands)
  slab, *_ = while_loop(partial(tile_loop_cond, kernel),
                        partial(tile_loop_body, kernel),
                        slab, start, end, operands, results)
  return slab

def make_elementwise_op(name, op):
  def kernel(*args): return [op(*args)]

  def f(slab: Slab, *operands: tuple[SlabView]):
    x, *_ = operands
    slab, result = slab_alloc(slab, x.shape)
    slab = tile(kernel, slab, operands, [result])
    return slab, result

  f.__name__ = name
  return f

add = make_elementwise_op('add', jax.lax.add)
mul = make_elementwise_op('mul', jax.lax.mul)

def parse_arr(i, s):
  shape = eval(s)
  assert all(d % block_sz == 0 for d in shape)
  return 3 * i + jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)

def main(args):
  xs = map(parse_arr, range(len(args)), args)
  print('initial args:')
  for x in xs:
    print(x)

  sz = xs[0].size
  shape = xs[0].shape

  slab = slab_make(1024, jnp.float32)

  vals = []
  for x in xs:
    slab, v = slab_alloc(slab, x.shape)
    slab = slab_write(slab, v.addr, x)
    vals.append(v)

  print()
  print('-- args allocated on slab:')
  print('slab:', slab)
  print('ptrs:', vals)

  def f(slab, *vals):
    slab, y = add(slab, *vals)
    slab, z = mul(slab, y, vals[0])
    return slab, y, z

  print()
  print(jax.make_jaxpr(f)(slab, *vals))

  slab, y, z = jax.jit(f)(slab, *vals)
  print()
  print('-- slab ptr results')
  print('add:', y)
  print('mul:', z)
  print()
  print('-- slab space')
  print('arg:', slab.data[:sz])
  print('arg:', slab.data[sz:sz * 2])
  print('add:', slab.data[sz * 2:sz * 3])
  print('mul:', slab.data[sz * 3:sz * 4])
  print()
  print('-- read off slab')
  print(slab_read(slab, y.addr, shape))
  print(slab_read(slab, z.addr, shape))


if __name__ == '__main__':
  main(sys.argv[1:])
