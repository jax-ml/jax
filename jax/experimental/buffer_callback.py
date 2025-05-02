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

from jax._src.lib import jaxlib_extension_version as _jaxlib_extension_version

if _jaxlib_extension_version >= 334:
  from jax._src.buffer_callback import (
      Buffer as Buffer,
      ExecutionContext as ExecutionContext,
      ExecutionStage as ExecutionStage,
      buffer_callback as buffer_callback,
  )

from jax._src.buffer_callback import buffer_callback as buffer_callback

del _jaxlib_extension_version
