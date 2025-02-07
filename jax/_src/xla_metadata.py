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

from typing import Any
from contextlib import contextmanager

from jax._src import config
from jax._src.lib import xla_client

config_ext = xla_client._xla.config


class XlaMetadata:
  __slots__ = ['val', 'hash']

  def __init__(self, val):
    self.val = val
    self.hash = hash(tuple(sorted(self.val.items())))

  def __hash__(self):
    return self.hash

  def __eq__(self, other):
    return other is not None and self.val == other.val


def update_metadata(a, b: dict[str, Any]):
  if not b:
    return a
  if a is None or a is config_ext.unset:
    return XlaMetadata(b)
  val = a.val.copy()
  val.update(b)
  return XlaMetadata(val)


def current_xla_metadata():
  metadata = config.xla_metadata_context_manager.value
  return None if metadata is None else metadata.val


class XlaMetadataContextManager:
  __slots__ = ['prev', 'updates']

  def __init__(self, updates):
    self.updates = updates

  def __enter__(self):
    if not self.updates:
      return

    self.prev = config.xla_metadata_context_manager.get_local()
    config.xla_metadata_context_manager.set_local(
        update_metadata(self.prev, self.updates)
    )

  def __exit__(self, exc_type, exc_value, traceback):
    if not self.updates:
      return
    config.xla_metadata_context_manager.set_local(self.prev)

@contextmanager
def set_xla_metadata(**kwargs):
  with XlaMetadataContextManager(kwargs):
    yield
