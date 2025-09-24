# Copyright 2023 The JAX Authors.
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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.interpreters import batching as _src_batching

from jax._src.interpreters.batching import (
  axis_primitive_batchers as axis_primitive_batchers,
  bdim_at_front as bdim_at_front,
  broadcast as broadcast,
  defbroadcasting as defbroadcasting,
  defreducer as defreducer,
  defvectorized as defvectorized,
  fancy_primitive_batchers as fancy_primitive_batchers,
  not_mapped as not_mapped,
  primitive_batchers as primitive_batchers,
  register_vmappable as register_vmappable,
  unregister_vmappable as unregister_vmappable,
)


_deprecations = {
  # Deprecated for JAX v0.7.1; finalize in JAX v0.9.0.
  "AxisSize": (
    "jax.interpreters.batching.AxisSize is deprecated.",
    _src_batching.AxisSize,
  ),
  "Array": (
    "jax.interpreters.batching.Array is deprecated. Use jax.Array directly.",
    _src_batching.Array,
  ),
  "BatchTrace": (
    "jax.interpreters.batching.BatchTrace is deprecated.",
    _src_batching.BatchTrace,
  ),
  "BatchTracer": (
    "jax.interpreters.batching.BatchTracer is deprecated.",
    _src_batching.BatchTracer,
  ),
  "BatchingRule": (
    "jax.interpreters.batching.BatchingRule is deprecated.",
    _src_batching.BatchingRule,
  ),
  "Jumble": (
    "jax.interpreters.batching.Jumble is deprecated.",
    _src_batching.Jumble,
  ),
  "JumbleAxis": (
    "jax.interpreters.batching.JumbleAxis is deprecated.",
    _src_batching.JumbleAxis,
  ),
  "JumbleTy": (
    "jax.interpreters.batching.JumbleTy is deprecated.",
    _src_batching.JumbleTy,
  ),
  "Elt": (
    "jax.interpreters.batching.Elt is deprecated.",
    _src_batching.Elt,
  ),
  "FromEltHandler": (
    "jax.interpreters.batching.FromEltHandler is deprecated.",
    _src_batching.FromEltHandler,
  ),
  "GetIdx": (
    "jax.interpreters.batching.GetIdx is deprecated.",
    _src_batching.GetIdx,
  ),
  "IndexedAxisSize": (
    "jax.interpreters.batching.IndexedAxisSize is deprecated.",
    _src_batching.IndexedAxisSize,
  ),
  "MakeIotaHandler": (
    "jax.interpreters.batching.MakeIotaHandler is deprecated.",
    _src_batching.MakeIotaHandler,
  ),
  "MapSpec": (
    "jax.interpreters.batching.MapSpec is deprecated.",
    _src_batching.MapSpec,
  ),
  "NotMapped": (
    "jax.interpreters.batching.NotMapped is deprecated.",
    _src_batching.NotMapped,
  ),
  "RaggedAxis": (
    "jax.interpreters.batching.RaggedAxis is deprecated.",
    _src_batching.RaggedAxis,
  ),
  "ToEltHandler": (
    "jax.interpreters.batching.ToEltHandler is deprecated.",
    _src_batching.ToEltHandler,
  ),
  "Vmappable": (
    "jax.interpreters.batching.Vmappable is deprecated.",
    _src_batching.Vmappable,
  ),
  "Zeros": (
    "jax.interpreters.batching.Zero is deprecated. Use jax.interpreters.ad.Zero.",
    _src_batching.Zero,
  ),
  "ZeroIfMapped": (
    "jax.interpreters.batching.ZeroIfMapped is deprecated. It is an internal type.",
    _src_batching.ZeroIfMapped,
  ),
  "batch": (
    "jax.interpreters.batching.batch is deprecated. It is an internal API.",
    _src_batching.batch,
  ),
  "batch_custom_jvp_subtrace": (
    "jax.interpreters.batching.batch_custom_jvp_subtrace is deprecated. It is an internal API.",
    _src_batching.batch_custom_jvp_subtrace,
  ),
  "batch_custom_vjp_bwd": (
    "jax.interpreters.batching.batch_custom_vjp_bwd is deprecated. It is an internal API.",
    _src_batching.batch_custom_vjp_bwd,
  ),
  "batch_jaxpr": (
    "jax.interpreters.batching.batch_jaxpr is deprecated. It is an internal API.",
    _src_batching.batch_jaxpr,
  ),
  "batch_jaxpr2": (
    "jax.interpreters.batching.batch_jaxpr2 is deprecated. It is an internal API.",
    _src_batching.batch_jaxpr2,
  ),
  "batch_jaxpr_axes": (
    "jax.interpreters.batching.batch_jaxpr_axes is deprecated. It is an internal API.",
    _src_batching.batch_jaxpr_axes,
  ),
  "batch_subtrace": (
    "jax.interpreters.batching.batch_subtrace is deprecated. It is an internal API.",
    _src_batching.batch_subtrace,
  ),
  "broadcast_batcher": (
    "jax.interpreters.batching.broadcast_batcher is deprecated. It is an internal API.",
    _src_batching.broadcast_batcher,
  ),
  "flatten_fun_for_vmap": (
    "jax.interpreters.batching.flatten_fun_for_vmap is deprecated. It is an internal API.",
    _src_batching.flatten_fun_for_vmap,
  ),
  "from_elt": (
    "jax.interpreters.batching.from_elt is deprecated. It is an internal API.",
    _src_batching.from_elt,
  ),
  "from_elt_handlers": (
    "jax.interpreters.batching.from_elt_handlers is deprecated. It is an internal API.",
    _src_batching.from_elt_handlers,
  ),
  "is_vmappable": (
    "jax.interpreters.batching.is_vmappable is deprecated. It is an internal API.",
    _src_batching.is_vmappable,
  ),
  "jumble_axis": (
    "jax.interpreters.batching.jumble_axis is deprecated. It is an internal API.",
    _src_batching.jumble_axis,
  ),
  "make_iota": (
    "jax.interpreters.batching.make_iota is deprecated. It is an internal API.",
    _src_batching.make_iota,
  ),
  "make_iota_handlers": (
    "jax.interpreters.batching.make_iota_handlers is deprecated. It is an internal API.",
    _src_batching.make_iota_handlers,
  ),
  "matchaxis": (
    "jax.interpreters.batching.matchaxis is deprecated. It is an internal API.",
    _src_batching.matchaxis,
  ),
  "moveaxis": (
    "jax.interpreters.batching.moveaxis is deprecated. Use jax.numpy.moveaxis.",
    _src_batching.moveaxis,
  ),
  "reducer_batcher": (
    "jax.interpreters.batching.reducer_batcher is deprecated. It is an internal API.",
    _src_batching.reducer_batcher,
  ),
  "spec_types": (
    "jax.interpreters.batching.spec_types is deprecated. It is an internal API.",
    _src_batching.spec_types,
  ),
  "to_elt": (
    "jax.interpreters.batching.to_elt is deprecated. It is an internal API.",
    _src_batching.to_elt,
  ),
  "to_elt_handlers": (
    "jax.interpreters.batching.to_elt_handlers is deprecated. It is an internal API.",
    _src_batching.to_elt_handlers,
  ),
  "vectorized_batcher": (
    "jax.interpreters.batching.vectorized_batcher is deprecated. It is an internal API.",
    _src_batching.vectorized_batcher,
  ),
  "vmappables": (
    "jax.interpreters.batching.vmappables is deprecated. It is an internal API.",
    _src_batching.vmappables,
  ),
  "vtile": (
    "jax.interpreters.batching.vtile is deprecated. It is an internal API.",
    _src_batching.vtile,
  ),
  "zero_if_mapped": (
    "jax.interpreters.batching.zero_if_mapped is deprecated. It is an internal API.",
    _src_batching.zero_if_mapped,
  ),
}


import typing as _typing
if _typing.TYPE_CHECKING:
  Array = _src_batching.Array
  AxisSize = _src_batching.AxisSize
  BatchTrace = _src_batching.BatchTrace
  BatchTracer = _src_batching.BatchTracer
  BatchingRule = _src_batching.BatchingRule
  Jumble = _src_batching.Jumble
  JumbleAxis = _src_batching.JumbleAxis
  JumbleTy = _src_batching.JumbleTy
  Elt = _src_batching.Elt
  FromEltHandler = _src_batching.FromEltHandler
  GetIdx = _src_batching.GetIdx
  IndexedAxisSize = _src_batching.IndexedAxisSize
  MakeIotaHandler = _src_batching.MakeIotaHandler
  MapSpec = _src_batching.MapSpec
  NotMapped = _src_batching.NotMapped
  RaggedAxis = _src_batching.RaggedAxis
  ToEltHandler = _src_batching.ToEltHandler
  Vmappable = _src_batching.Vmappable
  Zero = _src_batching.Zero
  ZeroIfMapped = _src_batching.ZeroIfMapped
  batch = _src_batching.batch
  batch_custom_jvp_subtrace = _src_batching.batch_custom_jvp_subtrace
  batch_custom_vjp_bwd = _src_batching.batch_custom_vjp_bwd
  batch_jaxpr = _src_batching.batch_jaxpr
  batch_jaxpr2 = _src_batching.batch_jaxpr2
  batch_jaxpr_axes = _src_batching.batch_jaxpr_axes
  batch_subtrace = _src_batching.batch_subtrace
  broadcast_batcher = _src_batching.broadcast_batcher
  flatten_fun_for_vmap = _src_batching.flatten_fun_for_vmap
  from_elt = _src_batching.from_elt
  from_elt_handlers = _src_batching.from_elt_handlers
  is_vmappable = _src_batching.is_vmappable
  jumble_axis = _src_batching.jumble_axis
  make_iota = _src_batching.make_iota
  make_iota_handlers = _src_batching.make_iota_handlers
  matchaxis = _src_batching.matchaxis
  moveaxis = _src_batching.moveaxis
  reducer_batcher = _src_batching.reducer_batcher
  spec_types = _src_batching.spec_types
  to_elt = _src_batching.to_elt
  to_elt_handlers = _src_batching.to_elt_handlers
  vectorized_batcher = _src_batching.vectorized_batcher
  vmappables = _src_batching.vmappables
  vtile = _src_batching.vtile
  zero_if_mapped = _src_batching.zero_if_mapped
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
