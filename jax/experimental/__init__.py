# Copyright 2018 The JAX Authors.
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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

# Note: we discourage adding any new APIs directly here. Instead please consider
# adding them to a relevant or new submodule in jax.experimental. This approach
# gives the JAX team more granularity to manage access / visibility to
# experimental features and as a result, more flexibility to manage their status
# and lifetimes.

from jax._src.callback import (
  io_callback as io_callback
)
from jax._src.dtypes import (
    primal_tangent_dtype as primal_tangent_dtype,
)
from jax._src.earray import (
    EArray as EArray
)
from jax._src.core import (
    cur_qdd as cur_qdd,
)

_deprecations = {
  # Remove in v0.10.0
  "disable_x64": (
    ("jax.experimental.disable_x64 was removed in JAX v0.9.0;"
     " use jax.enable_x64(False) instead."),
    None,
  ),
  "enable_x64": (
    ("jax.experimental.enable_x64 was removed in JAX v0.9.0;"
     " use jax.enable_x64(True) instead."),
    None
  ),
  "mutable_array": (
    ("jax.experimental.mutable_array was removed in JAX v0.9.0;"
     " use jax.new_ref instead."),
    None,
  ),
  "MutableArray": (
    ("jax.experimental.MutableArray was removed in JAX v0.9.0;"
     " use jax.Ref instead."),
    None,
  ),
}

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
