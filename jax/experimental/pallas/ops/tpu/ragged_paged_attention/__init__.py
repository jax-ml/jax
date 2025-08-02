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

from jax.experimental.pallas.ops.tpu.ragged_paged_attention import kernel_v2
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import tuned_block_sizes
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import util

cdiv = util.cdiv
align_to = util.align_to
get_dtype_packing = util.get_dtype_packing
next_power_of_2 = util.next_power_of_2
get_tpu_version = util.get_tpu_version

dynamic_validate_inputs = kernel_v2.dynamic_validate_inputs
static_validate_inputs = kernel_v2.static_validate_inputs
prepare_inputs = kernel_v2.prepare_inputs
prepare_outputs = kernel_v2.prepare_outputs
get_vmem_estimate_bytes = kernel_v2.get_vmem_estimate_bytes
get_smem_estimate_bytes = kernel_v2.get_smem_estimate_bytes
ragged_paged_attention = kernel_v2.ragged_paged_attention
ref_ragged_paged_attention = kernel_v2.ref_ragged_paged_attention

get_tuned_block_sizes = tuned_block_sizes.get_tuned_block_sizes
get_simplified_key = tuned_block_sizes.get_simplified_key
