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

from jax._src import core
from jax._src.typing import Array
from jax._src.core import ArrayRef, freeze  # noqa: F401
from jax._src.state.types import AbstractRef  # noqa: F401
from jax._src.state.primitives import (
    ref_get as get,  # noqa: F401
    ref_set as set,  # noqa: F401
    ref_swap as swap,  # noqa: F401
    ref_addupdate as addupdate,  # noqa: F401
)

def array_ref(init_val: Array) -> ArrayRef:
  """Create a mutable array reference with initial value `init_val`."""
  return core.array_ref_p.bind(init_val, memory_space=None)  # noqa

del Array
