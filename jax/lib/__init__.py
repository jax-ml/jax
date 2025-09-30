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

# ruff: noqa: F401
from jax._src.lib import (
  version_str as __version__,
)

# Dynamically load submodules because they warn on import.
# TODO(jakevdp): remove this in JAX v0.9.0.
def __getattr__(attr):
  if attr in {'xla_bridge', 'xla_client', 'xla_extension'}:
    import importlib
    return importlib.import_module(f'jax.lib.{attr}')
  raise AttributeError(f"module 'jax.lib' has no attribute {attr!r}")
