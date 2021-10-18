# Copyright 2018 Google LLC
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

# flake8: noqa: F401
from jax._src.lib.xla_bridge import (
  default_backend as default_backend,
  device_count as device_count,
  get_backend as get_backend,
  get_compile_options as get_compile_options,
  local_device_count as local_device_count,
  process_index as process_index,
  xla_client as xla_client,
  _backends as _backends,
)
