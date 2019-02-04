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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from .core import Trace, Tracer, Primitive, new_master

### library

def ptranspose(x, split_dim, concat_dim, **params):
  return ptranspose_p.bind(x, split_dim=split_dim, concat_dim=concat_dim,
                           **params)

def psplit(x, split_dim, target_dim, **params):
  return psplit_p.bind(x, split_dim=split_dim, target_dim=target_dim, **params)

def psum(x, axis_name):
  return psum_p.bind(x, axis_name=axis_name)

def all_gather(x, xdim, **params):
  x = x.broadcast((xb.get_replica_count(),))
  return ptranspose(x, 0, xdim, **params)

def gather(x, axis_name):
  return gather_p.bind(x, axis_name=axis_name)


### primitives and rules

def PmapPrimitive(name):
  prim = Primitive(name)
  prim.def_impl(partial(unbound_name_error, name))
  prim.def_abstract_eval(lambda x, *args, **kwargs: x)  # default
  return prim

def unbound_name_error(primitive_name, *args, **kwargs):
  axis_name = kwargs['axis_name']
  msg = "axis name '{}' is unbound for primitive {}."
  raise NameError(msg.format(axis_name, primitive_name))


def psum_pmap_rule(val, axis):
  return val.sum(axis), None

def psum_parallel_translation_rule(c, val, device_groups):
  if len(device_groups) > 1:
    return c.CrossReplicaSum(val, device_groups)
  else:
    return c.CrossReplicaSum(val)

psum_p = PmapPrimitive('psum')

def gather_pmap_rule(val, axis):
  return val, None

gather_p = PmapPrimitive('gather')

def ptranspose_shape_rule(x, split_dim, concat_dim, **params):
  permutation = list(range(x.ndim))
  permutation[concat_dim] = split_dim
  permutation[split_dim] = concat_dim
  return transpose_shape_rule(x, permutation)

def ptranspose_pmap_rule(x, axis, split_dim, concat_dim):
  raise NotImplementedError

def ptranspose_translation_rule(c, x, split_dim, concat_dim):
  return c.AllToAll(x, split_dim, split_dim)

ptranspose_p = PmapPrimitive('ptranspose')
ptranspose_p.def_abstract_eval(ptranspose_shape_rule)


def scatter_like(source, target):
  return scatter_like_p.bind(source, target)

def scatter_like_papply_rule(name, vals, axes):
  source, target = vals
  source_axis, target_axis = axes
  assert source_axis is None
  return _scatter(source, target, target_axis, name)

scatter_like_p = Primitive('scatter_like')
scatter_like_p.def_abstract_eval(lambda source, target: source)


## template papply rules

def vectorized_papply(prim, name, vals, axes, **params):
  assert all(axes[0] == a for a in axes[1:])
  return prim.bind(*vals, **params), axes[0]


def reducer_papply(prim, cprim, name, vals, papply_axes, input_shape, axes):
  operand, = vals
  papply_axis, = papply_axes

  other_axes = [i for i in axes if i != papply_axis]
  if other_axes:
    result = prim.bind(operand, axes=other_axes, input_shape=input_shape)
  else:
    result = operand

  if not axes or papply_axis in axes:
    return cprim.bind(result, axis_name=name), None
  else:
    new_papply_axis = papply_axis - onp.sum(onp.less(other_axes, papply_axis))
    return result, new_papply_axis


def broadcasting_papply(prim, name, vals, axes, **params):
  x, y = vals
  xdim, ydim = axes

  if xdim is None:
    return prim.bind(x, y, **params), ydim
  elif ydim is None:
    return prim.bind(x, y, **params), xdim
  elif xdim == ydim:
    return prim.bind(x, y, **params), xdim
  else:
    x = ptranspose(x, xdim, ydim, axis_name=xdim)
    return prim.bind(x, y, **params), ydim
