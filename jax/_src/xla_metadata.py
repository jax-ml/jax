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

from contextlib import contextmanager

from jax._src import config
from jax._src.lib import xla_client
from jax._src import xla_metadata_lib

config_ext = xla_client._xla.config


class XlaMetadataContextManager:
  __slots__ = ['prev', 'updates']

  def __init__(self, updates):
    self.updates = updates

  def __enter__(self):
    if not self.updates:
      return

    self.prev = config.xla_metadata_context_manager.get_local()
    config.xla_metadata_context_manager.set_local(
        xla_metadata_lib.update_metadata(self.prev, self.updates)
    )

  def __exit__(self, exc_type, exc_value, traceback):
    if not self.updates:
      return
    config.xla_metadata_context_manager.set_local(self.prev)

@contextmanager
def set_xla_metadata(**kwargs):
  with XlaMetadataContextManager(kwargs):
    yield
