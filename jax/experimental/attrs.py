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

from typing import Any

from jax._src import core
from jax._src.interpreters import partial_eval as pe

JaxVal = Any

getattr_p = core.Primitive('getattr')
setattr_p = core.Primitive('setattr')

def jax_getattr(obj: Any, attr: str):
  return getattr_p.bind(obj=obj, attr=attr)

def jax_setattr(obj: Any, attr: str, val: JaxVal):
  setattr_p.bind(val, obj=obj, attr=attr)


@getattr_p.def_impl
def _getattr_impl(*, obj, attr):
  return getattr(obj, attr)

@setattr_p.def_impl
def _setattr_impl(val, *, obj, attr):
  setattr(obj, attr, val)


def _ensure_tracked(trace: pe.DynamicJaxprTrace, obj: Any, attr: str):
  frame = trace.main.jaxpr_stack[-1]  # type: ignore
  if (obj, attr) not in frame.attrs_tracked:
    init_val = getattr(obj, attr)
    aval = core.raise_to_shaped(core.get_aval(init_val))
    tracer = pe.DynamicJaxprTracer(trace, aval, pe.source_info_util.current())
    var = frame.tracer_to_var[id(tracer)] = frame.newvar(aval)
    setattr(obj, attr, tracer)
    frame.attrs_tracked.append((obj, attr))
    frame.attrs_inits.append(init_val)
    frame.attrs_vars.append(var)
pe.DynamicJaxprTrace._ensure_tracked = _ensure_tracked

def _getattr_staging(trace, *, obj, attr):
  trace._ensure_tracked(obj, attr)
  return getattr(obj, attr)
pe.custom_staging_rules[getattr_p] = _getattr_staging

def _setattr_staging(trace, tracer, *, obj, attr):
  trace._ensure_tracked(obj, attr)
  setattr(obj, attr, tracer)
pe.custom_staging_rules[setattr_p] = _setattr_staging
