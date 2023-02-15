# Copyright 2023 The JAX Authors.
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

"""
This submodule is a work in progress; to see the proposal behind the types exported
here, see https://jax.readthedocs.io/en/latest/jep/12049-type-annotations.html.

In short, for JAX-compatible functions, we recommend that input arrays be annotated with
:class:`jax.typing.ArrayLike`, and outputs be annotated with :class:`jax.Array`.
For example::

    import jax.numpy as jnp
    from jax import Array
    from jax.typing import ArrayLike

    def my_function(x: ArrayLike) -> Array:
      return 2 * jnp.sin(x) * jnp.cos(x)
"""
from jax._src.typing import (
    ArrayLike as ArrayLike
)
