# Copyright 2026 The JAX Authors.
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

from typing import Any

from jax._src import config
from jax._src.lib import weakref_lru_cache
from jax._src.util import immutable


@immutable
class JaxprEqnContext:

  __slots__ = ['compute_type', 'threefry_partitionable', 'cur_abstract_mesh',
               'remove_size_one_mesh_axis', 'xla_metadata', '__weakref__']

  compute_type: str | None
  threefry_partitionable: bool
  xla_metadata: Any
  cur_abstract_mesh: Any
  remove_size_one_mesh_axis: bool

  @staticmethod
  @weakref_lru_cache.weak_value_interner
  def _create(compute_type, threefry_partitionable, cur_abstract_mesh,
              remove_size_one_mesh_axis, xla_metadata):
    obj = object.__new__(JaxprEqnContext)
    object.__setattr__(obj, 'compute_type', compute_type)
    object.__setattr__(obj, 'threefry_partitionable', threefry_partitionable)
    object.__setattr__(obj, 'cur_abstract_mesh', cur_abstract_mesh)
    object.__setattr__(obj, 'remove_size_one_mesh_axis',
                       remove_size_one_mesh_axis)
    object.__setattr__(obj, 'xla_metadata',
                       None if xla_metadata is None else dict(xla_metadata))
    return obj

  def __new__(cls, *, compute_type=None, threefry_partitionable=True,
              cur_abstract_mesh=None, remove_size_one_mesh_axis=False,
              xla_metadata=None):
    xla_meta_hashable = (None if xla_metadata is None else
                         tuple(sorted(xla_metadata.items())))
    return JaxprEqnContext._create(
        compute_type, threefry_partitionable, cur_abstract_mesh,
        remove_size_one_mesh_axis, xla_meta_hashable)

  # No __eq__ or __hash__: interned classes use object identity.

  def replace(self, **kwargs) -> JaxprEqnContext:
    return JaxprEqnContext(
        compute_type=kwargs.get('compute_type', self.compute_type),
        threefry_partitionable=kwargs.get(
            'threefry_partitionable', self.threefry_partitionable),
        cur_abstract_mesh=kwargs.get(
            'cur_abstract_mesh', self.cur_abstract_mesh),
        remove_size_one_mesh_axis=kwargs.get(
            'remove_size_one_mesh_axis', self.remove_size_one_mesh_axis),
        xla_metadata=kwargs.get('xla_metadata', self.xla_metadata),
    )

  @property
  def attributes(self) -> dict[str, Any]:
    if self.xla_metadata is None:
      return {}
    return self.xla_metadata

  def __repr__(self):
    s = [f"compute_on={self.compute_type}",
         f"threefry_partitionable={self.threefry_partitionable}",
         f"cur_abstract_mesh={self.cur_abstract_mesh}",
         f"remove_size_one_mesh_axis={self.remove_size_one_mesh_axis}"]
    s = ', '.join(s)
    return f"JaxprEqnContext({s}, xla_metadata={self.xla_metadata})"


jaxpr_eqn_ctx = config.config_ext.Config(
    'jaxpr_eqn_ctx',
    JaxprEqnContext(),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

def current_jaxpr_eqn_ctx() -> JaxprEqnContext:
  return jaxpr_eqn_ctx.value


# This context manager is fairly hot, because it is frequently called for every
# jaxpr equation. It is implemented as a class with explicit __enter__ and
# __exit__ methods since a @contextlib.contextmanager is slower.
# In effect we fuse four context managers into one, mostly to save allocations.
class JaxprEqnContextManager:
  __slots__ = ['context', 'prev']

  def __init__(self, context):
    self.context = context

  def __enter__(self):
    self.prev = jaxpr_eqn_ctx.swap_local(self.context)

  def __exit__(self, exc_type, exc_value, traceback):
    jaxpr_eqn_ctx.set_local(self.prev)


# Hooks that mirror threefry_partitionable to JaxprEqnContext.
def _update_threefry_partitionable(val):
  ctx = jaxpr_eqn_ctx.value
  if ctx is not None:
    jaxpr_eqn_ctx.set_local(ctx.replace(threefry_partitionable=val))

def _update_threefry_partitionable_global(val):
  ctx = jaxpr_eqn_ctx.get_global()
  if ctx is not None:
    jaxpr_eqn_ctx.set_global(ctx.replace(threefry_partitionable=val))

threefry_partitionable = config.bool_state(
    name='jax_threefry_partitionable',
    default=True,
    upgrade=True,
    help=('Enables internal threefry PRNG implementation changes that '
          'render it automatically partitionable in some cases. Without this '
          'flag, using the standard jax.random pseudo-random number generation '
          'may result in extraneous communication and/or redundant distributed '
          'computation. With this flag, the communication overheads disappear '
          'in some cases.'),
    update_global_hook=_update_threefry_partitionable_global,
    update_thread_local_hook=_update_threefry_partitionable,
    include_in_jit_key=False,
    include_in_trace_context=False)


# Hooks that mirror remove_size_one_mesh_axis to JaxprEqnContext.
def _update_remove_size_one_mesh_axis(val):
  ctx = jaxpr_eqn_ctx.value
  if ctx is not None:
    jaxpr_eqn_ctx.set_local(ctx.replace(remove_size_one_mesh_axis=val))

def _update_remove_size_one_mesh_axis_global(val):
  ctx = jaxpr_eqn_ctx.get_global()
  if ctx is not None:
    jaxpr_eqn_ctx.set_global(ctx.replace(remove_size_one_mesh_axis=val))

# This config is temporary and should go away since this is a user problem.
# If they don't want 1 sized mesh axis names to show up in sharding and vma
# bits on ShapedArray, then their mesh (which they pass to set_mesh) should not
# contain those axes at all.
remove_size_one_mesh_axis_from_type = config.bool_state(
    name='jax_remove_size_one_mesh_axis_from_type',
    default=False,
    help="Removes mesh axes of size 1 from ShapedArray.sharding and vma",
    update_global_hook=_update_remove_size_one_mesh_axis_global,
    update_thread_local_hook=_update_remove_size_one_mesh_axis,
    include_in_jit_key=False,
    include_in_trace_context=False)
