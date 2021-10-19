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

# This .pyi file exists primarily to help pytype infer the types of reexported
# modules from jaxlib. Without an explicit type stub, many types become Any.
# (Google pytype bug b/192059119).

from typing import Any, Optional, Tuple

import jaxlib.lapack as lapack
import jaxlib.pocketfft as pocketfft
import jaxlib.xla_client as xla_client
import jaxlib.xla_extension as xla_extension
import jaxlib.xla_extension.jax_jit as jax_jit
import jaxlib.xla_extension.pmap_lib as pmap_lib
import jaxlib.xla_extension.pytree as pytree

version: Tuple[int, ...]

cuda_path: Optional[str]
cuda_linalg: Optional[Any]
cuda_prng: Optional[Any]
cusolver: Optional[Any]
cusparse: Optional[Any]
rocsolver: Optional[Any]
tpu_driver_client: Optional[Any]

_xla_extension_version: int
