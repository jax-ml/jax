# Copyright 2025 The JAX Authors.
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

# ruff: noqa

from jax._src.ad_util import (
    Zero as Zero,
)
from jax._src.core import (
    AbstractValue as AbstractValue,
    AvalQDD as AvalQDD,
    ShapedArray as ShapedArray,
    aval_method as aval_method,
    aval_property as aval_property,
    AvalMutableQDD as AvalMutableQDD,
)
from jax._src.interpreters.ad import (
    instantiate_zeros as instantiate_zeros,
    is_undefined_primal as is_undefined_primal,
)
from jax._src.effects import (
    control_flow_allowed_effects as control_flow_allowed_effects,
)
from jax._src.hijax import (
    HiPrimitive as HiPrimitive,
    HipSpec as HipSpec,
    HiType as HiType,
    MutableHiType as MutableHiType,
    VJPHiPrimitive as VJPHiPrimitive,
    register_hitype as register_hitype,
)
from jax._src.state import (
    AbstractRef as AbstractRef,
    TransformedRef as TransformedRef
)
