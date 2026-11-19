# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python bindings for the MLIR Thunky dialect."""

# ruff: noqa: F401
# ruff: noqa: F403
from jaxlib.mlir._mlir_libs._thunky_ext import *
from jaxlib.thunky.dialect import _thunky_ops_gen
from jaxlib.thunky.dialect._thunky_ops_gen import *

try:
  from jaxlib.mlir.dialects._ods_common import _cext
except ImportError:
  from mlir.dialects._ods_common import _cext

# Add the parent module to the search prefix
_cext.globals.append_dialect_search_prefix(__name__[: __name__.rfind(".")])
