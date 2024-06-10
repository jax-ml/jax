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
__all__ = ["DisabledSafetyCheck", "Exported", "export", "deserialize",
           "maximum_supported_serialization_version",
           "minimum_supported_serialization_version",
           "default_lowering_platform",
           "SymbolicScope", "is_symbolic_dim",
           "symbolic_shape", "symbolic_args_specs"]

from jax._src.export._export import DisabledSafetyCheck as DisabledSafetyCheck
from jax._src.export._export import Exported as Exported
from jax._src.export._export import export as export
from jax._src.export._export import deserialize as deserialize
from jax._src.export._export import maximum_supported_serialization_version as maximum_supported_serialization_version
from jax._src.export._export import minimum_supported_serialization_version as minimum_supported_serialization_version
from jax._src.export._export import default_lowering_platform as default_lowering_platform

from jax._src.export import shape_poly_decision  # Import only to set the decision procedure
del shape_poly_decision
from jax._src.export.shape_poly import SymbolicScope as SymbolicScope
from jax._src.export.shape_poly import is_symbolic_dim as is_symbolic_dim
from jax._src.export.shape_poly import symbolic_shape as symbolic_shape
from jax._src.export.shape_poly import symbolic_args_specs as symbolic_args_specs
