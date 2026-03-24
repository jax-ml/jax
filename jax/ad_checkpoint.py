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

from jax._src.ad_checkpoint import (
  checkpoint_policies as checkpoint_policies,
  checkpoint_name as checkpoint_name,
  print_saved_residuals as print_saved_residuals,
)
from jax._src.interpreters.partial_eval import (
  Recompute as Recompute,
  Saveable as Saveable,
  Offloadable as Offloadable,
)

_deprecations = {
  # Deprecated in v0.8.2; finalized in v0.10.0.
  # TODO(jakevdp) remove entry in v0.11.0.
  "checkpoint": (
    "jax.ad_checkpoint.checkpoint was deprecated in JAX v0.8.2 and removed in"
    " JAX v0.10.0; use jax.checkpoint instead.",
    None,
  ),
  "remat": (
    "jax.ad_checkpoint.remat was deprecated in JAX v0.8.2 and removed in"
    " JAX v0.10.0; use jax.remat instead.",
    None,
  ),
}

import typing as _typing
if not _typing.TYPE_CHECKING:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
