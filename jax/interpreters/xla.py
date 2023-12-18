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

from jax._src.interpreters.xla import (
  abstractify as abstractify,
  canonicalize_dtype as canonicalize_dtype,
  canonicalize_dtype_handlers as canonicalize_dtype_handlers,
  pytype_aval_mappings as pytype_aval_mappings,
)

from jax._src.dispatch import (
  apply_primitive as apply_primitive,
)

from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc  # type: ignore

xe = xc._xla
Backend = xe.Client
