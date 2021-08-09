# Copyright 2021 Google LLC
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

"""Concentrates IREE related imports in one place.

Different environments locate these in different ways, and indirecting through
this module provides one place to replace them.
"""

from iree.compiler import ir
from iree.compiler import passmanager
from iree.compiler.api import driver as compiler_driver
from iree.compiler.dialects import builtin, chlo, mhlo, std
from iree.compiler.dialects import iree as iree_dialect
from iree import runtime as iree_runtime

__all__ = [
  "compiler_driver",
  "ir",
  "iree_runtime",
  "passmanager",
  # Dialects.
  "builtin",
  "chlo",
  "iree_dialect",
  "mhlo",
  "std",
]
