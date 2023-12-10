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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/google/jax/issues/7570

from jax._src import prng as _prng


_deprecations = {
    # Added August 29, 2023
    "PRNGImpl": (
      "jax.prng.PRNGImpl is deprecated. Use jax.extend.random.PRNGImpl instead.",
      _prng.PRNGImpl,
    ),
    "seed_with_impl": (
      "jax.prng.seed_with_impl is deprecated. Use jax.extend.random.seed_with_impl instead.",
      _prng.seed_with_impl,
    ),
    "threefry2x32_p": (
      "jax.prng.threefry2x32_p is deprecated. Use jax.extend.random.threefry2x32_p instead.",
      _prng.threefry2x32_p,
    ),
    "threefry_2x32": (
      "jax.prng.threefry_2x32 is deprecated. Use jax.extend.random.threefry_2x32 instead.",
      _prng.threefry_2x32,
    ),
    "threefry_prng_impl": (
      "jax.prng.threefry_prng_impl is deprecated. Use jax.extend.random.threefry_prng_impl instead.",
      _prng.threefry_prng_impl,
    ),
    "rbg_prng_impl": (
      "jax.prng.rbg_prng_impl is deprecated. Use jax.extend.random.rbg_prng_impl instead.",
      _prng.rbg_prng_impl,
    ),
    "unsafe_rbg_prng_impl": (
      "jax.prng.unsafe_rbg_prng_impl is deprecated. Use jax.extend.random.unsafe_rbg_prng_impl instead.",
      _prng.unsafe_rbg_prng_impl,
    ),
}

import typing
if typing.TYPE_CHECKING:
  PRNGImpl = _prng.PRNGImpl
  seed_with_impl = _prng.seed_with_impl
  threefry2x32_p = _prng.threefry2x32_p
  threefry_2x32 = _prng.threefry_2x32
  threefry_prng_impl = _prng.threefry_prng_impl
  rbg_prng_impl = _prng.rbg_prng_impl
  unsafe_rbg_prng_impl = _prng.unsafe_rbg_prng_impl
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _prng
