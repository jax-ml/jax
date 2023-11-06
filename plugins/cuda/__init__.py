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

import functools
import logging
import os
import pathlib
import platform
import sys

import jax._src.xla_bridge as xb

from jax._src.lib import cuda_plugin_extension
from jax._src.lib import xla_client
from jax._src.lib import xla_extension_version

logger = logging.getLogger(__name__)


def initialize():
  path = pathlib.Path(__file__).resolve().parent / "xla_cuda_plugin.so"
  if not path.exists():
    logger.warning(
        "WARNING: Native library %s does not exist. This most likely indicates"
        " an issue with how %s was built or installed.",
        path,
        __package__,
    )
  if xla_extension_version >= 212:
    options = xla_client.generate_pjrt_gpu_plugin_options(
        visible_devices=xb.CUDA_VISIBLE_DEVICES.value
    )
  else:
    options = {}
    visible_devices = xb.CUDA_VISIBLE_DEVICES.value
    if visible_devices != 'all':
      options['visible_devices'] = [int(x) for x in visible_devices.split(',')]

    allocator = os.getenv('XLA_PYTHON_CLIENT_ALLOCATOR', 'default').lower()
    memory_fraction = os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION', '')
    preallocate = os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE', '').lower()
    if allocator not in ('default', 'platform', 'bfc', 'cuda_async'):
      raise ValueError(
          'XLA_PYTHON_CLIENT_ALLOCATOR env var must be "default", "platform", '
          '"bfc", or "cuda_async", got "%s"' % allocator
      )
    options['allocator'] = allocator
    if memory_fraction:
      options['memory_fraction'] = float(memory_fraction)
    if preallocate:
      options['preallocate'] = preallocate not in ('false', '0')
  c_api = xb.register_plugin(
      'cuda', priority=500, library_path=str(path), options=options
  )
  if cuda_plugin_extension:
    xla_client.register_custom_call_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.register_custom_call_target, c_api
        ),
    )
    for _name, _value in cuda_plugin_extension.registrations().items():
      xla_client.register_custom_call_target(_name, _value, platform="CUDA")
  else:
    logger.warning('cuda_plugin_extension is not found.')
