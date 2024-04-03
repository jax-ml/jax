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

from functools import partial, reduce
from typing import Iterable, NamedTuple, Union
import sys

import numpy as np

import jax
import jax.numpy as jnp

from jax._src import util

map, zip = util.safe_map, util.safe_zip

DInt = jax.Array
Address = DInt
XInt = Union[int, DInt]
DShape = tuple[XInt]
SShape = tuple[int]
DType = jax.typing.DTypeLike

class Slab(NamedTuple):
  data: jax.Array
  cursor: Address

@jax.tree_util.register_pytree_node_class
class SlabView(NamedTuple):
  addr: Address
  shape: DShape
  dtype: DType

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

word_b = 4
phrase_b = 512
phrase_w = 128
tile_aspect = 8

def xceil_div(x: XInt, y: XInt) -> XInt:
  """ceil(x / y)"""
  return (x + y - 1) // y

def _xadd(x: XInt, y: XInt) -> XInt:
  return x + y

def _xmul(x: XInt, y: XInt) -> XInt:
  return x * y

def xadd(*xs: XInt) -> XInt:
  return reduce(_xadd, xs, 0)

def xmul(*xs: XInt) -> XInt:
  return reduce(_xmul, xs, 1)

def xsum(xs: Iterable[XInt]) -> XInt:
  return xadd(*list(xs))

def xprod(xs: Iterable[XInt]) -> XInt:
  return xmul(*list(xs))

def tile_shape(shape: DShape, dtype):
  # Units: (1, 1, ..., elements, 1)
  if len(shape) < 2:
    raise NotImplementedError('matrices or bust')
  num_leading = len(shape) - 2
  return (1,) * num_leading + (tile_aspect * word_b // dtype.itemsize,
                               phrase_b // word_b)

def tile_phrases(shape: DShape, dtype: DType):
  # Units: phrases
  return xprod(tile_shape(shape, dtype)) * dtype.itemsize // phrase_b

def slab_make(num_phrases):
  return Slab(jnp.zeros((num_phrases, phrase_w), dtype=jnp.uint32),
              jnp.array(0, dtype=jnp.int32))

def slab_alloc(slab: Slab, shape: DShape, dtype):
  if len(shape) < 2:
    raise NotImplementedError('matrices or bust')
  tiled_shape = map(xceil_div, shape, tile_shape(shape, dtype))
  num_p = xmul(*tiled_shape, tile_phrases(shape, dtype))
  new_slab = Slab(slab.data, slab.cursor + num_p)
  slab_val = SlabView(slab.cursor, shape, dtype)
  return new_slab, slab_val

def strides(xs):
  return tuple(reversed(jnp.cumprod(jnp.array(list(reversed(xs))))))

def slab_slices(view, slice_base_e: DShape, slice_shape_e: SShape):
  view_shape_e = tile_shape(view.shape, view.dtype)
  # dassert all(slice_base  % view_shape_t == 0)
  # dassert all(slice_shape % view_shape_t == 0)
  slice_base_t  = [s // t for s, t in zip(slice_base_e,  view_shape_e)]
  slice_shape_t = [s // t for s, t in zip(slice_shape_e, view_shape_e)]
  tiled_shape = map(xceil_div, view.shape, view_shape_e)
  tiled_strides = strides(tiled_shape)
  tp = tile_phrases(view.shape, view.dtype)
  for idx in np.ndindex(*slice_shape_t[:-1]):
    linear_idx_t = xsum(
        map(xmul, map(xadd, slice_base_t, (*idx, 0)), tiled_strides))
    yield (view.addr + linear_idx_t * tp, slice_shape_t[-1] * tp)

def reinterpret_cast(x: jax.Array, shape: SShape, dtype: DType):
  x_bytes = x.size * x.dtype.itemsize
  if -1 in shape:
    assert x_bytes % xprod(s for s in shape if s != -1) * dtype.itemsize == 0
  else:
    assert x_bytes == xprod(shape) * dtype.itemsize, (x.shape, x.dtype, shape, dtype)
  if x.dtype.itemsize != dtype.itemsize:
    # reshape(x, -1) in conversion below becomes reshape(-1, a, b) for some a,b
    raise NotImplementedError('todo')
  return jax.lax.bitcast_convert_type(x.reshape(-1), dtype).reshape(shape)

def slab_read(slab, view, slice_base: DShape, slice_shape: SShape):
  view_shape_t = tile_shape(view.shape, view.dtype)
  tiled_shape = map(xceil_div, view.shape, view_shape_t)
  slices = [
      jax.lax.dynamic_slice_in_dim(slab.data, addr, phrases)
      for addr, phrases in slab_slices(view, slice_base, slice_shape)]
  slice_mem = jnp.stack(slices, axis=0)
  return reinterpret_cast(
      slice_mem, (*tiled_shape, *view_shape_t), view.dtype
      ).swapaxes(-2, -3).reshape(slice_shape)

# TODO: just take vjp of slab_read
def slab_write(slab, view, slice_base: DShape, inval: jax.Array):
  slice_shape = inval.shape
  view_shape_t = tile_shape(view.shape, view.dtype)
  tiled_shape = map(xceil_div, view.shape, view_shape_t)
  inval_linearized = inval.reshape(
      *tiled_shape[:-1], view_shape_t[-2], tiled_shape[-1], view_shape_t[-1]
      ).swapaxes(-2, -3)
  slice_mem = reinterpret_cast(inval_linearized, (-1, phrase_w),
                               jnp.dtype('uint32'))
  slice_addr = 0
  new_slab = slab.data
  for slab_addr, slice_sz_p in slab_slices(view, slice_base, slice_shape):
    s = jax.lax.dynamic_slice_in_dim(slice_mem, slice_addr, slice_sz_p)
    slice_addr += slice_sz_p
    new_slab = jax.lax.dynamic_update_slice_in_dim(
        new_slab, s, slab_addr, axis=0)
  return Slab(new_slab, slab.cursor)

def parse_arr(i, s):
  shape = eval(s)
  z = 3 * i + jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
  return z

def print_seg(msg):
  print()
  print(f'-- {msg}')
  print()

def make_jaxpr_slab_write(slab, view, inval):
  return jax.make_jaxpr(
      lambda slab, x: slab_write(slab, view, (0, 0), x))(slab, inval)

def make_jaxpr_slab_read(slab, view, outval_shape):
  return jax.make_jaxpr(
      lambda slab: slab_read(slab, view, (0, 0), outval_shape))(slab)

def main(args):
  xs = map(parse_arr, range(len(args)), args)

  sz = xs[0].size
  shape = xs[0].shape

  slab = slab_make(1024)

  vals = []
  for x in xs:
    slab, v = slab_alloc(slab, x.shape, x.dtype)
    vals.append(v)
    print_seg('slab after init')
    print(slab)

    slab = slab_write(slab, v, (0, 0), x)
    print_seg('slab after write')
    print(slab)
    print_seg('slab_read result')
    print(slab_read(slab, v, (0, 0), x.shape))
    print_seg('slab_write jaxpr')
    print(make_jaxpr_slab_write(slab, v, x))
    print_seg('slab_read jaxpr')
    print(make_jaxpr_slab_read(slab, v, x.shape))


if __name__ == '__main__':
  main(sys.argv[1:])
