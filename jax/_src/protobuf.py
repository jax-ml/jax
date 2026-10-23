# Copyright 2026 The JAX Authors.
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

"""Parsing of protobuf files written by XLA.

jaxlib ships a serialized ``FileDescriptorSet`` describing the protos that XLA
writes to disk. Any recent protobuf runtime can use this to parse those .pb files.
"""

from __future__ import annotations

import functools
import gzip
import importlib.resources
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from google.protobuf import descriptor_pool as _descriptor_pool
  from google.protobuf import message as _message

_DESCRIPTOR_SET_FILENAME = "protobuf_descriptor_set.pb"

# Some XLA protos use Editions syntax, which older protobuf runtimes either
# reject or (in the case of 4.25.x) silently misinterpret.
_MIN_PROTOBUF_VERSION = (5, 26)


def _check_protobuf_version() -> None:
  try:
    import google.protobuf
  except ImportError as e:
    raise ImportError(
        "jax.experimental.protobuf requires the protobuf package "
        f"(version >= {'.'.join(map(str, _MIN_PROTOBUF_VERSION))})."
    ) from e
  m = re.match(r"(\d+)\.(\d+)", google.protobuf.__version__)
  if m is None or (int(m[1]), int(m[2])) < _MIN_PROTOBUF_VERSION:
    raise ImportError(
        "jax.experimental.protobuf requires protobuf >= "
        f"{'.'.join(map(str, _MIN_PROTOBUF_VERSION))}, but version "
        f"{google.protobuf.__version__} is installed."
    )


def bundled_descriptor_set_bytes() -> bytes:
  """Returns the serialized ``FileDescriptorSet`` shipped with jaxlib."""
  try:
    return (importlib.resources.files("jaxlib") /
            _DESCRIPTOR_SET_FILENAME).read_bytes()
  except FileNotFoundError as e:
    raise RuntimeError(
        f"The installed jaxlib does not include {_DESCRIPTOR_SET_FILENAME}; "
        "jax.experimental.protobuf requires a newer jaxlib."
    ) from e


def load_pool(descriptor_set: bytes) -> _descriptor_pool.DescriptorPool:
  """Builds a descriptor pool from a serialized ``FileDescriptorSet``.

  Pass the result to :func:`parse` to parse protos written by a different
  version of XLA than the one bundled with the installed jaxlib. The pool is
  private, separate from protobuf's default pool.

  Args:
    descriptor_set: the contents of a serialized ``FileDescriptorSet`` file,
      such as the ``protobuf_descriptor_set.pb`` shipped with jaxlib,
      describing the message types to parse and their dependencies. To find
      that file in the environment that wrote the protos, so that it can be
      copied alongside them, run::

        python -c "import importlib.resources as r; print(r.files('jaxlib') / 'protobuf_descriptor_set.pb')"

  Returns:
    A ``google.protobuf.descriptor_pool.DescriptorPool``.
  """
  _check_protobuf_version()
  from google.protobuf import descriptor_pb2
  from google.protobuf import descriptor_database as db_lib
  from google.protobuf import descriptor_pool as pool_lib
  fds = descriptor_pb2.FileDescriptorSet.FromString(descriptor_set)
  db = db_lib.DescriptorDatabase()
  for file in fds.file:
    db.Add(file)
  return pool_lib.DescriptorPool(db)


@functools.cache
def bundled_descriptor_pool() -> _descriptor_pool.DescriptorPool:
  """Returns the descriptor pool for the descriptor set shipped with jaxlib."""
  return load_pool(bundled_descriptor_set_bytes())


def message_class(
    message_type: str, pool: _descriptor_pool.DescriptorPool | None = None
) -> type[_message.Message]:
  """Returns the protobuf message class for an XLA message type.

  Args:
    message_type: the fully-qualified message name, e.g. ``"xla.HloProto"``.
    pool: the descriptor pool to look up ``message_type`` in. Defaults to the
      pool for the descriptor set shipped with jaxlib.

  Returns:
    A subclass of ``google.protobuf.message.Message``.
  """
  if pool is None:
    pool = bundled_descriptor_pool()
  from google.protobuf import message_factory
  try:
    descriptor = pool.FindMessageTypeByName(message_type)
  except KeyError:
    raise ValueError(
        f"Unknown XLA message type {message_type!r}. Message types must be "
        "fully qualified, e.g. 'xla.HloProto'.") from None
  return message_factory.GetMessageClass(descriptor)


def parse(
    data: bytes | str,
    message_type: str,
    *,
    pool: _descriptor_pool.DescriptorPool | None = None,
) -> _message.Message:
  """Parses a serialized XLA proto.

  Args:
    data: the serialized message. ``bytes`` are parsed as a binary proto, and
      ``str`` as a text-format proto, as written by XLA when dumping as text.
    message_type: the fully-qualified message name, e.g. ``"xla.HloProto"``.
    pool: a descriptor pool from :func:`load_pool`, to parse protos written by
      a different version of XLA than the one bundled with the installed
      jaxlib. Defaults to the pool for the descriptor set shipped with jaxlib.

  Returns:
    The parsed message.
  """
  if pool is None:
    pool = bundled_descriptor_pool()
  cls = message_class(message_type, pool)
  if isinstance(data, str):
    from google.protobuf import text_format
    return text_format.Parse(data, cls(), descriptor_pool=pool)
  return cls.FromString(data)


def parse_file(
    path: str | os.PathLike[str],
    message_type: str,
    *,
    text: bool | None = None,
    pool: _descriptor_pool.DescriptorPool | None = None,
) -> _message.Message:
  """Parses an XLA proto from a file.

  Files with a ``.gz`` suffix, such as those written by XLA with
  ``--xla_dump_compress_protos``, are decompressed.

  Args:
    path: the file to read.
    message_type: the fully-qualified message name, e.g. ``"xla.HloProto"``.
    text: whether the file is a text-format proto. By default this is inferred
      from the suffix, ignoring any ``.gz``: ``.pb`` files are binary and
      ``.txt`` files are text, as written by XLA.
    pool: see :func:`parse`.

  Returns:
    The parsed message.
  """
  path_str = os.fspath(path)
  compressed = path_str.endswith(".gz")
  if text is None:
    stem = path_str.removesuffix(".gz")
    if stem.endswith(".pb"):
      text = False
    elif stem.endswith(".txt"):
      text = True
    else:
      raise ValueError(
          f"Cannot infer whether {path_str!r} is a binary or text-format "
          "proto from its suffix; pass text=False or text=True.")
  opener = gzip.open if compressed else open
  data: bytes | str
  if text:
    with opener(path, "rt", encoding="utf-8") as f:
      data = f.read()
  else:
    with opener(path, "rb") as f:
      data = f.read()
  return parse(data, message_type, pool=pool)
