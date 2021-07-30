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

from typing import Callable, Optional, Sequence

import logging

from jax import core, tree_util
from jax.core import DimSize, Shape
from jax._src import api
from jax._src import util

from .ir_builder import Builder


def trace_function(fun: Callable,
                   *,
                   builder: Builder,
                   exported_name: Optional[str] = None,
                   shapes=None):
  api._check_callable(fun)
  if exported_name is None:
    exported_name = getattr(fun, "__name__", "unknown")
  if not core.trace_state_clean():
    raise ValueError(
        "convert must be used outside all JAX transformations." +
        f"Trace state: {core.thread_local_state.trace_state.trace_stack}")
  fb = builder.create_function(exported_name, [], [])
  # Remove me.
  fb.emit_return([])
