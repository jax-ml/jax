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

# TODO(yashkatariya): Remove this forwarding shim.
from jax._src.tpu_info import (
    ChipVersionBase as ChipVersionBase,
    ChipVersion as ChipVersion,
    SparseCoreInfo as SparseCoreInfo,
    Tiling as Tiling,
    TpuInfo as TpuInfo,
    chip_version_from_device_kind as chip_version_from_device_kind,
    get_device_kind as get_device_kind,
    get_num_device_cores as get_num_device_cores,
    get_tpu_info_for_chip as get_tpu_info_for_chip,
    get_tpu_info as get_tpu_info,
    infer_tiling as infer_tiling,
    is_tpu_device as is_tpu_device,
    registry as registry,
)
