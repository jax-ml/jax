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
    None,
  ),
  "Array": (
    "jax.interpreters.batching.Array is deprecated. Use jax.Array directly.",
    None,
  ),
  "BatchTrace": (
    "jax.interpreters.batching.BatchTrace is deprecated.",
    None,
  ),
  "BatchTracer": (
    "jax.interpreters.batching.BatchTracer is deprecated.",
    None,
  ),
  "BatchingRule": (
    "jax.interpreters.batching.BatchingRule is deprecated.",
    None,
  ),
  "Elt": (
    "jax.interpreters.batching.Elt is deprecated.",
    None,
  ),
  "FromEltHandler": (
    "jax.interpreters.batching.FromEltHandler is deprecated.",
    None,
  ),
  "GetIdx": (
    "jax.interpreters.batching.GetIdx is deprecated.",
    None,
  ),
  "MakeIotaHandler": (
    "jax.interpreters.batching.MakeIotaHandler is deprecated.",
    None,
  ),
  "MapSpec": (
    "jax.interpreters.batching.MapSpec is deprecated.",
    None,
  ),
  "NotMapped": (
    "jax.interpreters.batching.NotMapped is deprecated.",
    _src_batching.NotMapped,
  ),
  "ToEltHandler": (
    "jax.interpreters.batching.ToEltHandler is deprecated.",
    None,
  ),
  "Vmappable": (
    "jax.interpreters.batching.Vmappable is deprecated.",
    None,
  ),
  "Zeros": (
    "jax.interpreters.batching.Zero is deprecated. Use jax.interpreters.ad.Zero.",
    None,
  ),
  "ZeroIfMapped": (
    "jax.interpreters.batching.ZeroIfMapped is deprecated. It is an internal type.",
    None,
  ),
  "batch": (
    "jax.interpreters.batching.batch is deprecated. It is an internal API.",
    None,
  ),
  "batch_custom_jvp_subtrace": (
    "jax.interpreters.batching.batch_custom_jvp_subtrace is deprecated. It is an internal API.",
    None,
  ),
  "batch_custom_vjp_bwd": (
    "jax.interpreters.batching.batch_custom_vjp_bwd is deprecated. It is an internal API.",
    None,
  ),
  "batch_jaxpr": (
    "jax.interpreters.batching.batch_jaxpr is deprecated. It is an internal API.",
    None,
  ),
  "batch_jaxpr_axes": (
    "jax.interpreters.batching.batch_jaxpr_axes is deprecated. It is an internal API.",
    None,
  ),
  "batch_subtrace": (
    "jax.interpreters.batching.batch_subtrace is deprecated. It is an internal API.",
    None,
  ),
  "broadcast_batcher": (
    "jax.interpreters.batching.broadcast_batcher is deprecated. It is an internal API.",
    None,
  ),
  "flatten_fun_for_vmap": (
    "jax.interpreters.batching.flatten_fun_for_vmap is deprecated. It is an internal API.",
    None,
  ),
  "from_elt": (
    "jax.interpreters.batching.from_elt is deprecated. It is an internal API.",
    None,
  ),
  "from_elt_handlers": (
    "jax.interpreters.batching.from_elt_handlers is deprecated. It is an internal API.",
    None,
  ),
  "is_vmappable": (
    "jax.interpreters.batching.is_vmappable is deprecated. It is an internal API.",
    None,
  ),
  "make_iota": (
    "jax.interpreters.batching.make_iota is deprecated. It is an internal API.",
    None,
  ),
  "make_iota_handlers": (
    "jax.interpreters.batching.make_iota_handlers is deprecated. It is an internal API.",
    None,
  ),
  "matchaxis": (
    "jax.interpreters.batching.matchaxis is deprecated. It is an internal API.",
    None,
  ),
  "moveaxis": (
    "jax.interpreters.batching.moveaxis is deprecated. Use jax.numpy.moveaxis.",
    None,
  ),
  "reducer_batcher": (
    "jax.interpreters.batching.reducer_batcher is deprecated. It is an internal API.",
    None,
  ),
  "spec_types": (
    "jax.interpreters.batching.spec_types is deprecated. It is an internal API.",
    None,
  ),
  "to_elt": (
    "jax.interpreters.batching.to_elt is deprecated. It is an internal API.",
    None,
  ),
  "to_elt_handlers": (
    "jax.interpreters.batching.to_elt_handlers is deprecated. It is an internal API.",
    None,
  ),
  "vectorized_batcher": (
    "jax.interpreters.batching.vectorized_batcher is deprecated. It is an internal API.",
    None,
  ),
  "vmappables": (
    "jax.interpreters.batching.vmappables is deprecated. It is an internal API.",
    None,
  ),
  "vtile": (
    "jax.interpreters.batching.vtile is deprecated. It is an internal API.",
    None,
  ),
  "zero_if_mapped": (
    "jax.interpreters.batching.zero_if_mapped is deprecated. It is an internal API.",
    None,
  ),
}


import typing as _typing
if _typing.TYPE_CHECKING:
  NotMapped = _src_batching.NotMapped
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
