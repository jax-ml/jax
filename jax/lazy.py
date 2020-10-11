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
from typing import Any, Callable, Optional, Sequence

import numpy as np

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

# A LazyExpr contains a reindexing specification. The reindexing expression is
# applied to the value represented by the device buffer backing a DeviceArray.
#
# The reindexing specification encodes the shape of the final result and a list
# of dimensions, which are integers or Nones. The integer entries take on values
# 0, 1, ..., R-1 where R is the rank of the input array, and encode where the
# axes of the input array are to be mapped in the final output. When an entry is
# None that indicates that the corresponding axis of the result is a broadcasted
# one.
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
LazyExpr = namedtuple('LazyExpr', ['shape', 'dims'])
# pytype: enable=wrong-arg-count

def array(shape):
  return LazyExpr(shape, tuple(range(len(shape))))

def broadcast(lexpr, shape, broadcast_dimensions):
  new_dims = [None] * len(shape)
  for i, d in enumerate(broadcast_dimensions):
    new_dims[d] = lexpr.dims[i]
  return LazyExpr(shape, tuple(new_dims))

def transpose(lexpr: LazyExpr, perm: Sequence[int]):
  new_shape = tuple(lexpr.shape[i] for i in perm)
  new_dims = tuple(lexpr.dims[i] for i in perm)
  return LazyExpr(new_shape, new_dims)

def is_trivial(lexpr: LazyExpr) -> bool:
  return lexpr.dims == tuple(range(len(lexpr.shape)))


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
  assert x is not None
  shape, dims = lexpr
  perm = [d for d in dims if d is not None]
  if perm != list(range(len(perm))):
    x = np.transpose(x, perm)
  if shape != x.shape:
    in_shape = [1 if d is None else s for d, s in zip(dims, shape)]
    x = np.broadcast_to(np.reshape(x, in_shape), shape)
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
  assert x is not None
  shape, dims = lexpr
  bcast_dims, perm = unzip2((i, d) for i, d in enumerate(dims) if d is not None)
  if tuple(perm) != tuple(range(len(perm))):
    x = xops.Transpose(x, perm)
  if shape != c.get_shape(x).dimensions():
    x = xops.BroadcastInDim(x, shape, bcast_dims)
  return x
