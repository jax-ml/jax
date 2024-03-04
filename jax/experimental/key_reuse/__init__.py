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
to change. The usage below requires JAX version 0.4.26 or newer.

Key reuse checking can be enabled using the `jax_enable_key_reuse_checks` configuration::

  >>> import jax
  >>> jax.config.update('jax_enable_key_reuse_checks', True)
  >>> key = jax.random.key(0)
  >>> jax.random.normal(key)
  Array(-0.20584226, dtype=float32)
  >>> jax.random.normal(key)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
   ...
  KeyReuseError: Previously-consumed key passed to jit-compiled function at index 0

This flag can also be controlled locally using the :func:`jax.enable_key_reuse_checks`
context manager::

  >>> with jax.enable_key_reuse_checks(False):
  ...  print(jax.random.normal(key))
  -0.20584226
"""
from jax._src.prng import (
    reuse_key as reuse_key,
)

from jax.experimental.key_reuse._core import (
    KeyReuseError as KeyReuseError,
)
