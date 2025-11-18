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

from jax._src import ad_util as _src_ad_util
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
  reducing_transposes as reducing_transposes,
  vjp as vjp,
  zeros_like_aval as zeros_like_aval,
)


def _deprecated_backward_pass(jaxpr, reduce_axes, transform_stack,
                              consts, primals_in, cotangents_in):
  if reduce_axes:
    raise NotImplementedError("reduce_axes on ad.backward_pass is deprecated")
  del reduce_axes
  return _src_ad.backward_pass(
      jaxpr, transform_stack, consts, primals_in, cotangents_in)


_deprecations = {
    # Deprecated for JAX v0.7.1; finalize in JAX v0.9.0.
    "zeros_like_p": (
        "jax.interpreters.ad.zeros_like_p is deprecated in JAX v0.7.1. It has been unused since v0.4.24.",
        _src_ad_util.zeros_like_p,
    ),
    "backward_pass": (
        "jax.interpreters.ad.backward_pass is deprecated.",
        _deprecated_backward_pass
    ),
    "bilinear_transpose": (
        "jax.interpreters.ad.bilinear_transpose is deprecated.",
        _src_ad.bilinear_transpose,
    ),
    "call_param_updaters": (
        "jax.interpreters.ad.call_param_updaters is deprecated.",
        _src_ad.call_param_updaters,
    ),
    "call_transpose": (
        "jax.interpreters.ad.call_transpose is deprecated.",
        _src_ad.call_transpose,
    ),
    "call_transpose_param_updaters": (
        "jax.interpreters.ad.call_transpose_param_updaters is deprecated.",
        _src_ad.call_transpose_param_updaters,
    ),
    "closed_backward_pass": (
        "jax.interpreters.ad.closed_backward_pass is deprecated.",
        _src_ad.closed_backward_pass,
    ),
    "custom_lin_p": (
        "jax.interpreters.ad.custom_lin_p is deprecated.",
        _src_ad.custom_lin_p,
    ),
    "defjvp_zero": (
        "jax.interpreters.ad.defjvp_zero is deprecated.",
        _src_ad.defjvp_zero,
    ),
    "f_jvp_traceable": (
        "jax.interpreters.ad.f_jvp_traceable is deprecated.",
        _src_ad.f_jvp_traceable,
    ),
    "jvp_jaxpr": (
        "jax.interpreters.ad.jvp_jaxpr is deprecated.",
        _src_ad.jvp_jaxpr,
    ),
    "jvp_subtrace": (
        "jax.interpreters.ad.jvp_subtrace is deprecated.",
        _src_ad.jvp_subtrace,
    ),
    "jvp_subtrace_aux": (
        "jax.interpreters.ad.jvp_subtrace_aux is deprecated.",
        _src_ad.jvp_subtrace_aux,
    ),
    "jvpfun": (
        "jax.interpreters.ad.jvpfun is deprecated.",
        _src_ad.jvpfun,
    ),
    "linear_jvp": (
        "jax.interpreters.ad.linear_jvp is deprecated.",
        _src_ad.linear_jvp,
    ),
    "linear_transpose": (
        "jax.interpreters.ad.linear_transpose is deprecated.",
        _src_ad.linear_transpose,
    ),
    "linear_transpose2": (
        "jax.interpreters.ad.linear_transpose2 is deprecated.",
        _src_ad.linear_transpose2,
    ),
    "map_transpose": (
        "jax.interpreters.ad.map_transpose is deprecated.",
        _src_ad.map_transpose,
    ),
    "nonzero_outputs": (
        "jax.interpreters.ad.nonzero_outputs is deprecated.",
        _src_ad.nonzero_outputs,
    ),
    "nonzero_tangent_outputs": (
        "jax.interpreters.ad.nonzero_tangent_outputs is deprecated.",
        _src_ad.nonzero_tangent_outputs,
    ),
    "rearrange_binders": (
        "jax.interpreters.ad.rearrange_binders is deprecated.",
        _src_ad.rearrange_binders,
    ),
    "standard_jvp": (
        "jax.interpreters.ad.standard_jvp is deprecated.",
        _src_ad.standard_jvp,
    ),
    "standard_jvp2": (
        "jax.interpreters.ad.standard_jvp2 is deprecated.",
        _src_ad.standard_jvp2,
    ),
    "traceable": (
        "jax.interpreters.ad.traceable is deprecated.",
        _src_ad.traceable,
    ),
    "zero_jvp": (
        "jax.interpreters.ad.zero_jvp is deprecated.",
        _src_ad.zero_jvp,
    ),
}

import typing
if typing.TYPE_CHECKING:
  backward_pass = _deprecated_backward_pass
  bilinear_transpose = _src_ad.bilinear_transpose
  call_param_updaters = _src_ad.call_param_updaters
  call_transpose = _src_ad.call_transpose
  call_transpose_param_updaters = _src_ad.call_transpose_param_updaters
  closed_backward_pass = _src_ad.closed_backward_pass
  custom_lin_p = _src_ad.custom_lin_p
  defjvp_zero = _src_ad.defjvp_zero
  f_jvp_traceable = _src_ad.f_jvp_traceable
  jvp_jaxpr = _src_ad.jvp_jaxpr
  jvp_subtrace = _src_ad.jvp_subtrace
  jvp_subtrace_aux = _src_ad.jvp_subtrace_aux
  jvpfun = _src_ad.jvpfun
  linear_jvp = _src_ad.linear_jvp
  linear_transpose = _src_ad.linear_transpose
  linear_transpose2 = _src_ad.linear_transpose2
  map_transpose = _src_ad.map_transpose
  nonzero_outputs = _src_ad.nonzero_outputs
  nonzero_tangent_outputs = _src_ad.nonzero_tangent_outputs
  rearrange_binders = _src_ad.rearrange_binders
  standard_jvp = _src_ad.standard_jvp
  standard_jvp2 = _src_ad.standard_jvp2
  traceable = _src_ad.traceable
  zero_jvp = _src_ad.zero_jvp
  zeros_like_p = _src_ad_util.zeros_like_p
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
