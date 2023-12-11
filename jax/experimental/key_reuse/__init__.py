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
Experimental Key Reuse Checking
-------------------------------

This module contains **experimental** functionality for detecting re-use of random
keys within JAX programs. It is under active development and the APIs here are likely
to change.

Key reuse checking can be enabled on `jit`-compiled functions using the
:func:`jax.enable_key_reuse_checks` configuration::

  >>> import jax
  >>> @jax.jit
  ... def f(key):
  ...   return jax.random.uniform(key) + jax.random.normal(key)
  ...
  >>> key = jax.random.key(0)
  >>> with jax.enable_key_reuse_checks():
  ...   f(key)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
   ...
  KeyReuseError: In random_bits, key values a are already consumed.

This flag can also be set globally if you wish to enagle key reuse checks in
every JIT-compiled function.
"""

from jax.experimental.key_reuse._common import (
    unconsumed_copy as unconsumed_copy,
    KeyReuseError as KeyReuseError,
)
