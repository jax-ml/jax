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

from jax.experimental.pallas.ops.tpu.ragged_paged_attention import kernel
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import tuned_block_sizes

cdiv = kernel.cdiv
dynamic_validate_inputs = kernel.dynamic_validate_inputs
ragged_paged_attention = kernel.ragged_paged_attention
ref_ragged_paged_attention = kernel.ref_ragged_paged_attention
static_validate_inputs = kernel.static_validate_inputs
get_tuned_block_sizes = tuned_block_sizes.get_tuned_block_sizes
