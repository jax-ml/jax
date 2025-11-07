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

from typing import Any
from jax._src import core


def new_ref(init_val: Any, *, memory_space: Any = None) -> core.Ref:
  """Create a mutable array reference with initial value ``init_val``.

  For more discussion, see the `Ref guide`_.

  Args:
    init_val: A :class:`jax.Array` representing the initial state
      of the buffer.
    memory_space: An optional memory space attribute for the Ref.

  Returns:
    A :class:`jax.ref.Ref` containing a reference to a mutable buffer.

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  return core.new_ref(init_val, memory_space=memory_space)
