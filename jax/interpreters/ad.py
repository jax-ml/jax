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

from __future__ import annotations

from jax._src.interpreters import ad as _src_ad

from jax._src.interpreters.ad import (
  JVPTrace as JVPTrace,
  JVPTracer as JVPTracer,
  UndefinedPrimal as UndefinedPrimal,
  Zero as Zero,
  add_jaxvals as add_jaxvals,
  add_jaxvals_p as add_jaxvals_p,
  add_tangents as add_tangents,
  defbilinear as defbilinear,
  defjvp as defjvp,
  defjvp2 as defjvp2,
  deflinear as deflinear,
  deflinear2 as deflinear2,
  get_primitive_transpose as get_primitive_transpose,
  instantiate_zeros as instantiate_zeros,
  is_undefined_primal as is_undefined_primal,
  jvp as jvp,
  linearize as linearize,
  primitive_jvps as primitive_jvps,
  primitive_transposes as primitive_transposes,
  zeros_like_aval as zeros_like_aval,
)


_deprecations = {
    # Deprecated for JAX v0.7.1; finalized in JAX v0.9.0; Remove in v0.10.0.
    "zeros_like_p": (
        "jax.interpreters.ad.zeros_like_p was removed in JAX v0.9.0.",
        None,
    ),
    "bilinear_transpose": (
        "jax.interpreters.ad.bilinear_transpose was removed in JAX v0.9.0.",
        None,
    ),
    "call_param_updaters": (
        "jax.interpreters.ad.call_param_updaters was removed in JAX v0.9.0.",
        None,
    ),
    "call_transpose": (
        "jax.interpreters.ad.call_transpose was removed in JAX v0.9.0.",
        None,
    ),
    "call_transpose_param_updaters": (
        "jax.interpreters.ad.call_transpose_param_updaters was removed in JAX v0.9.0.",
        None,
    ),
    "custom_lin_p": (
        "jax.interpreters.ad.custom_lin_p was removed in JAX v0.9.0.",
        None,
    ),
    "defjvp_zero": (
        "jax.interpreters.ad.defjvp_zero was removed in JAX v0.9.0.",
        None,
    ),
    "f_jvp_traceable": (
        "jax.interpreters.ad.f_jvp_traceable was removed in JAX v0.9.0.",
        None,
    ),
    "jvp_jaxpr": (
        "jax.interpreters.ad.jvp_jaxpr was removed in JAX v0.9.0.",
        None,
    ),
    "jvp_subtrace": (
        "jax.interpreters.ad.jvp_subtrace was removed in JAX v0.9.0.",
        None,
    ),
    "jvp_subtrace_aux": (
        "jax.interpreters.ad.jvp_subtrace_aux was removed in JAX v0.9.0.",
        None,
    ),
    "jvpfun": (
        "jax.interpreters.ad.jvpfun was removed in JAX v0.9.0.",
        None,
    ),
    "linear_jvp": (
        "jax.interpreters.ad.linear_jvp was removed in JAX v0.9.0.",
        None,
    ),
    "linear_transpose": (
        "jax.interpreters.ad.linear_transpose was removed in JAX v0.9.0.",
        None,
    ),
    "linear_transpose2": (
        "jax.interpreters.ad.linear_transpose2 was removed in JAX v0.9.0.",
        None,
    ),
    "map_transpose": (
        "jax.interpreters.ad.map_transpose was removed in JAX v0.9.0.",
        None,
    ),
    "nonzero_outputs": (
        "jax.interpreters.ad.nonzero_outputs was removed in JAX v0.9.0.",
        None,
    ),
    "nonzero_tangent_outputs": (
        "jax.interpreters.ad.nonzero_tangent_outputs was removed in JAX v0.9.0.",
        None,
    ),
    "rearrange_binders": (
        "jax.interpreters.ad.rearrange_binders was removed in JAX v0.9.0.",
        None,
    ),
    "standard_jvp": (
        "jax.interpreters.ad.standard_jvp was removed in JAX v0.9.0.",
        None,
    ),
    "standard_jvp2": (
        "jax.interpreters.ad.standard_jvp2 was removed in JAX v0.9.0.",
        None,
    ),
    "traceable": (
        "jax.interpreters.ad.traceable was removed in JAX v0.9.0.",
        None,
    ),
    "zero_jvp": (
        "jax.interpreters.ad.zero_jvp was removed in JAX v0.9.0.",
        None,
    ),
    # Deprecated for JAX v0.9.0; finalize in JAX v0.10.0.
    "reducing_transposes": (
        (
            "jax.interpreters.ad.reducing_transposes is deprecated in JAX v0.9.0."
            " It has been unused since v0.4.38."
        ),
        _src_ad.reducing_transposes,
    ),
}

import typing
if typing.TYPE_CHECKING:
  reducing_transposes = _src_ad.reducing_transposes
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
