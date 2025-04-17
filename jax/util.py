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

import jax._src.deprecations
import jax._src.util


_deprecations = {
    "to_dlpack": (
        (
            "jax.dlpack.to_dlpack was deprecated in JAX v0.6.0 and will be"
            " removed in JAX v0.7.0. Please use the newer DLPack API based on"
            " __dlpack__ and __dlpack_device__ instead. Typically, you can pass"
            " a JAX array directly to the `from_dlpack` function of another"
            " framework without using `to_dlpack`."
        ),
        jax._src.dlpack.to_dlpack,
    ),
    "HashableFunction": (
        (
            "HashableFunction was deprecated in JAX v0.6.0 and will be removed"
            " in JAX v0.7.0."
        ),
        jax._src.util.HashableFunction,
    ),
    "as_hashable_function": (
        (
            "as_hashable_function was deprecated in JAX v0.6.0 and will be"
            " removed in JAX v0.7.0."
        ),
        jax._src.util.as_hashable_function,
    ),
    "cache": (
        "cache was deprecated in JAX v0.6.0 and will be removed in JAX v0.7.0.",
        jax._src.util.cache,
    ),
    "safe_map": (
        (
            "safe_map was deprecated in JAX v0.6.0 and will be removed in JAX"
            " v0.7.0."
        ),
        jax._src.util.safe_map,
    ),
    "safe_zip": (
        (
            "safe_zip was deprecated in JAX v0.6.0 and will be removed in JAX"
            " v0.7.0."
        ),
        jax._src.util.safe_zip,
    ),
    "split_dict": (
        (
            "split_dict was deprecated in JAX v0.6.0 and will be removed in JAX"
            " v0.7.0."
        ),
        jax._src.util.split_dict,
    ),
    "split_list": (
        (
            "split_list was deprecated in JAX v0.6.0 and will be removed in JAX"
            " v0.7.0."
        ),
        jax._src.util.split_list,
    ),
    "split_list_checked": (
        (
            "split_list_checked was deprecated in JAX v0.6.0 and will be"
            " removed in JAX v0.7.0."
        ),
        jax._src.util.split_list_checked,
    ),
    "split_merge": (
        (
            "split_merge was deprecated in JAX v0.6.0 and will be removed in"
            " JAX v0.7.0."
        ),
        jax._src.util.split_merge,
    ),
    "subvals": (
        (
            "subvals was deprecated in JAX v0.6.0 and will be removed in JAX"
            " v0.7.0."
        ),
        jax._src.util.subvals,
    ),
    "toposort": (
        (
            "toposort was deprecated in JAX v0.6.0 and will be removed in JAX"
            " v0.7.0."
        ),
        jax._src.util.toposort,
    ),
    "unzip2": (
        (
            "unzip2 was deprecated in JAX v0.6.0 and will be removed in JAX"
            " v0.7.0."
        ),
        jax._src.util.unzip2,
    ),
    "wrap_name": (
        (
            "wrap_name was deprecated in JAX v0.6.0 and will be removed in JAX"
            " v0.7.0."
        ),
        jax._src.util.wrap_name,
    ),
    "wraps": (
        "wraps was deprecated in JAX v0.6.0 and will be removed in JAX v0.7.0.",
        jax._src.util.wraps,
    ),
}


import typing as _typing

if _typing.TYPE_CHECKING:
  HashableFunction = jax._src.util.HashableFunction
  as_hashable_function = jax._src.util.as_hashable_function
  cache = jax._src.util.cache
  safe_map = jax._src.util.safe_map
  safe_zip = jax._src.util.safe_zip
  split_dict = jax._src.util.split_dict
  split_list = jax._src.util.split_list
  split_list_checked = jax._src.util.split_list_checked
  split_merge = jax._src.util.split_merge
  subvals = jax._src.util.subvals
  toposort = jax._src.util.toposort
  unzip2 = jax._src.util.unzip2
  wrap_name = jax._src.util.wrap_name
  wraps = jax._src.util.wraps
else:
  __getattr__ = jax._src.deprecations.deprecation_getattr(
      __name__, _deprecations
  )
del _typing
