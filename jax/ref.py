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

__all__ = ['AbstractRef', 'Ref', 'ArrayRef', 'addupdate', 'new_ref', 'array_ref',
           'freeze', 'get', 'set', 'swap']

from jax._src.core import Ref, ArrayRef, freeze
from jax._src.ref import new_ref, array_ref
from jax._src.state.types import AbstractRef
from jax._src.state.primitives import (
    ref_get as get,
    ref_set as set,
    ref_swap as swap,
    ref_addupdate as addupdate,
)
