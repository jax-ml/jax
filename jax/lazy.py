# Copyright 2019 Google LLC
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


from collections import namedtuple
import functools
import operator as op
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as onp

from .util import safe_map, safe_zip, unzip2, subvals
from .lib import xla_bridge as xb
from .lib import xla_client as xc

xops = xc.ops

map = safe_map
zip = safe_zip


### util

# TODO(mattjj): replace with dataclass when Python 2 support is removed
def taggedtuple(name, fields) -> Callable[..., Any]:
  """Lightweight version of namedtuple where equality depends on the type."""
  def __new__(cls, *xs):
    return tuple.__new__(cls, (cls,) + xs)
  def __str__(self):
    return '{}{}'.format(name, tuple.__str__(self[1:]))
  class_namespace = {'__new__' : __new__, '__str__': __str__}
  for i, f in enumerate(fields):
    class_namespace[f] = property(op.itemgetter(i+1))  # type: ignore
  return type(name, (tuple,), class_namespace)


### lazy sublanguage

# There are two components to a LazyExpr: an input and a reindexing
# specification. The input represents a base array to which the reindexing
# specification is applied.
#
# An input can represent an array constructor (Iota, Eye, etc.) or it can be an
# ArrayVar which encodes that the base array is some exogenous array value (from
# an environment with only a single value in it). These LazyExprs are attached
# to DeviceArrays, so when the input part of the expression is ArrayVar that
# basically means the associated device buffer represents the input, while if
# the input is an array constructor then the associated device_buffer field of
# the DeviceArray should be set to a DeviceConstant sentinel value. For the
# array constructor expressions:
#   * Iota builds a 1D sequence [0, 1, ..., N-1],
#   * Eye builds a 2D array with ones on a (possibly offset) diagonal and zeros
#     elsewhere (like numpy.eye),
#   * Tri builds a triangular matrix with ones on and below a diagonal and zeros
#     elsewhere (like numpy.tri), and
#   * Delta builds a Kronecker delta array with ones along its multidimensional
#     main diagonal and zeros elsewhere (for use in tensor contractions).
#
# The reindexing specification encodes the shape of the final result and a list
# of dimensions, which are integers or Nones. The integer entries take on values
# 0, 1, ..., R-1 where R is the rank of the input array, and encode where the
# axes of the input array are to be mapped in the final output. When an entry is
# None that indicates that the corresponding axis of the result is a broadcasted
# one.
#
# Here are some examples of lazy expressions and the arrays they represent:
#
# LazyExpr(input=Iota(dtype=dtype('float32'), size=3),
#          shape=(3, 4), dims=(0, None))
# DeviceArray([[0., 0., 0., 0.],
#              [1., 1., 1., 1.],
#              [2., 2., 2., 2.]], dtype=float32)
#
# LazyExpr(input=Iota(dtype=dtype('float32'), size=3),
#          shape=(4, 3), dims=(None, 0))
# DeviceArray([[0., 1., 2.],
#              [0., 1., 2.],
#              [0., 1., 2.],
#              [0., 1., 2.]], dtype=float32)
#
# For performance, some functions on lazy expressions accept None as an input to
# stand for the identity lazy expression.
#
# We use the `taggedtuple` class constructor, rather than standard namedtuples,
# because two namedtuple instances of different types but equal elements hash to
# the same value, e.g.
#   A = namedtuple('A', ['x', 'y'])
#   B = namedtuple('B', ['x', 'y'])
#   hash(A(1, 2)) == hash(B(1, 2))   # True
# but we want hashes to be sensitive to the type tag (while still being fast).

# pytype: disable=wrong-arg-count
LazyExpr = namedtuple('LazyExpr', ['input', 'shape', 'dims'])
ArrayVar = taggedtuple('ArrayVar', [])
Iota = taggedtuple('Iota', ['dtype', 'size'])           # like np.arange(N)
Eye = taggedtuple('Eye', ['dtype', 'shape', 'offset'])  # like np.eye
Tri = taggedtuple('Tri', ['dtype', 'shape', 'offset'])  # like np.tri
Delta = taggedtuple('Delta', ['dtype', 'shape'])  # kronecker delta arrays
# pytype: enable=wrong-arg-count

def array(shape):
  return LazyExpr(ArrayVar(), shape, tuple(range(len(shape))))

def iota(dtype, size):
  return LazyExpr(Iota(dtype, size), (size,), (0,))

def eye(dtype, shape, offset):
  assert len(shape) == 2
  return LazyExpr(Eye(dtype, shape, offset), shape, (0, 1))

def tri(dtype, shape, offset):
  assert len(shape) == 2
  return LazyExpr(Tri(dtype, shape, offset), shape, (0, 1))

def delta(dtype, shape):
  return LazyExpr(Delta(dtype, shape), shape, tuple(range(len(shape))))

def broadcast(lexpr, shape, broadcast_dimensions):
  new_dims = [None] * len(shape)
  for i, d in enumerate(broadcast_dimensions):
    new_dims[d] = lexpr.dims[i]
  return LazyExpr(lexpr.input, shape, tuple(new_dims))

def transpose(lexpr: LazyExpr, perm: Sequence[int]):
  new_shape = tuple(lexpr.shape[i] for i in perm)
  new_dims = tuple(lexpr.dims[i] for i in perm)
  return LazyExpr(lexpr.input, new_shape, new_dims)

def is_constant(lexpr: Optional[LazyExpr]):
  return lexpr is not None and type(lexpr.input) is not ArrayVar

def is_trivial(lexpr: LazyExpr) -> bool:
  return (type(lexpr.input) is ArrayVar and
          lexpr.dims == tuple(range(len(lexpr.shape))))


def eval_lexpr(lexpr, x):
  """Evaluate a lazy expression using NumPy.
  Args:
    lexpr: the LazyExpr to evaluate.
    x: ndarray or None, representing the value of ArrayVar if present.
  Returns:
    An ndarray representing the value of the lazy expression.
  """
  if is_trivial(lexpr):
    return x

  input_, shape, dims = lexpr

  # first create a starting ndarray from input_
  t = type(input_)
  if t is ArrayVar:
    assert x is not None and type(x) is onp.ndarray
  elif t is Iota:
    assert x is None
    x = onp.arange(input_.size, dtype=input_.dtype)
  elif t is Eye:
    assert x is None
    N, M = input_.shape
    x = onp.eye(N, M, dtype=input_.dtype, k=input_.offset)
  elif t is Tri:
    assert x is None
    N, M = input_.shape
    x = onp.tri(N, M, dtype=input_.dtype, k=input_.offset)
  elif t is Delta:
    ones = [1] * len(input_.shape)
    iotas = [onp.arange(d).reshape(subvals(ones, [(i, -1)]))
             for i, d in enumerate(input_.shape)]
    eyes = [i1 == i2 for i1, i2 in zip(iotas[:-1], iotas[1:])]
    x = onp.asarray(functools.reduce(op.and_, eyes), input_.dtype)
  else:
    assert False

  # then apply the reindexing operation
  perm = [d for d in dims if d is not None]
  if perm != list(range(len(perm))):
    x = onp.transpose(x, perm)
  if shape != x.shape:
    in_shape = [1 if d is None else s for d, s in zip(dims, shape)]
    x = onp.broadcast_to(onp.reshape(x, in_shape), shape)

  return x


def stage_lexpr(c, lexpr: Optional[LazyExpr], x):
  """Stage a lazy expression into an XLA computation.
  Args:
    c: XLA ComputationBuilder into which to stage the expression.
    lexpr: a LazyExpr to evaluate (or None for the identity expression).
    x: XlaOp or None, representing the value of ArrayVar if present.
  Returns:
    An XlaOp representing the value of the lazy expression.
  """
  if lexpr is None or is_trivial(lexpr):
    return x

  input_, shape, dims = lexpr

  # first create a starting XlaOp from input_
  t = type(input_)
  if t is ArrayVar:
    assert x is not None
  elif t is Iota:
    assert x is None
    x = xops.Iota(c, xb.dtype_to_etype(input_.dtype), input_.size)
  elif t is Eye:
    assert x is None
    N, M = input_.shape
    xla_shape = xc.Shape.array_shape(xc.PrimitiveType.S32, (N, M))
    bool_eye = xops.Eq(
      xops.Add(xops.Iota(c, xla_shape, 0),
               xb.constant(c, onp.array(input_.offset, onp.int32))),
      xops.Iota(c, xla_shape, 1))
    x = xops.ConvertElementType(bool_eye, xb.dtype_to_etype(input_.dtype))
  elif t is Tri:
    assert x is None
    N, M = input_.shape
    xla_shape = xc.Shape.array_shape(xc.PrimitiveType.S32, (N, M))
    bool_tri = xops.Ge(
      xops.Add(xops.Iota(c, xla_shape, 0),
               xb.constant(c, onp.array(input_.offset, onp.int32))),
      xops.Iota(c, xla_shape, 1))
    x = xops.ConvertElementType(bool_tri, xb.dtype_to_etype(input_.dtype))
  elif t is Delta:
    etype = xb.dtype_to_etype(input_.dtype)
    iotas = [xops.Iota(c, xc.Shape.array_shape(xc.PrimitiveType.U32, input_.shape), i)
             for i in range(len(input_.shape))]
    eyes = [xops.Eq(i1, i2) for i1, i2 in zip(iotas[:-1], iotas[1:])]
    x = xops.ConvertElementType(functools.reduce(xops.And, eyes), etype)
  else:
    assert False

  # then apply the operations encoded in reindex
  bcast_dims, perm = unzip2((i, d) for i, d in enumerate(dims) if d is not None)
  if tuple(perm) != tuple(range(len(perm))):
    x = xops.Transpose(x, perm)
  if shape != c.get_shape(x).dimensions():
    x = xops.BroadcastInDim(x, shape, bcast_dims)

  return x
