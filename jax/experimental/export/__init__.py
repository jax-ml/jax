# Copyright 2023 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from jax.experimental.export._export import (
    minimum_supported_serialization_version,
    maximum_supported_serialization_version,
    Exported,
    export,
    call_exported,  # TODO: deprecate
    call,
    DisabledSafetyCheck,
    default_lowering_platform,

    args_specs,  # TODO: deprecate
)
from jax.experimental.export._shape_poly import (
    is_symbolic_dim,
    symbolic_shape,
    symbolic_args_specs,
    SymbolicScope,
)
from jax.experimental.export._serialization import (
    serialize,
    deserialize,
)
from jax.experimental.export import _shape_poly_decision
