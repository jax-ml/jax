# Copyright 2024 The JAX Authors.
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

from __future__ import annotations
from contextlib import contextmanager
from jax._src import config
from jax._src.lib import xla_client

config_ext = xla_client._xla.config


@contextmanager
def extend_compute_type(c_type: str | None):
  if c_type is None:
    yield
    return

  prev = config.compute_on_context_manager.swap_local(c_type)
  try:
    if prev is not None and prev is not config_ext.unset and c_type != prev:
      raise NotImplementedError(
          'Nesting `compute_on` with different compute types is not supported'
          f' yet. Current compute_on type: {prev}')
    yield c_type
  finally:
    config.compute_on_context_manager.set_local(prev)

def current_compute_type() -> str | None:
  return config.compute_on_context_manager.value

def _check_valid(c_type: str):
  if (c_type not in {'device_host', 'device', 'tpu_sparsecore'}
      and not c_type.startswith("gpu_stream:")):
    raise ValueError(
        f'Invalid compute type {c_type}. Current supported values '
        'are `device_host`, `device`, `tpu_sparsecore`, and `gpu_stream:#`.')

@contextmanager
def compute_on(compute_type: str):
  if not isinstance(compute_type, str):
    raise TypeError("`compute_on`'s compute_type argument must be a string.")
  _check_valid(compute_type)

  with extend_compute_type(compute_type):
    yield
