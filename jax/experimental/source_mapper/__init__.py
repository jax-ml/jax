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

from jax._src.sourcemap import SourceMap as SourceMap
from jax._src.sourcemap import MappingsGenerator as MappingsGenerator
from jax.experimental.source_mapper.common import Pass as Pass
from jax.experimental.source_mapper.common import register_pass as register_pass
from jax.experimental.source_mapper.common import all_passes as all_passes
from jax.experimental.source_mapper.common import filter_passes as filter_passes
from jax.experimental.source_mapper.common import compile_with_env as compile_with_env
from jax.experimental.source_mapper.common import SourceMapDump as SourceMapDump
from jax.experimental.source_mapper.generate_map import generate_sourcemaps as generate_sourcemaps
from jax.experimental.source_mapper.mlir import create_mlir_sourcemap as create_mlir_sourcemap

# We import the jaxpr and hlo passes to register them.
import jax.experimental.source_mapper.jaxpr  # pylint: disable=unused-import # noqa: F401
from jax.experimental.source_mapper.jaxpr import canonicalize_filename as canonicalize_filename
import jax.experimental.source_mapper.hlo  # pylint: disable=unused-import # noqa: F401
