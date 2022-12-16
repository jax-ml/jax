# Copyright 2021 The JAX Authors.
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

# flake8: noqa

from jax._src.pjit import *
from jax._src.pjit import (_UNSPECIFIED, _prepare_axis_resources,
                           _get_op_sharding_from_executable,
                           _get_pspec_from_executable, _pjit_lower_cached,
                           _pjit_lower, _get_op_sharding,
                           _calc_is_global_sequence, _pjit_jaxpr,
                           _create_mesh_pspec_sharding_from_parsed_pspec,
                           _process_in_axis_resources)
