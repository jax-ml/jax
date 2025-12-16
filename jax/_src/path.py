# Copyright 2022 The JAX Authors.
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

from typing import Protocol
import logging
import os
import pathlib

__all__ = ["Path"]
logger = logging.getLogger(__name__)
epath_installed: bool

class PathProtocol(Protocol):
  """A factory that creates a PurePath."""
  def __call__(self, *pathsegments: str | os.PathLike) -> pathlib.Path:
    ...

Path: PathProtocol

# If etils.epath (aka etils[epath] to pip) is present, we prefer it because it
# can read and write to, e.g., GCS buckets. Otherwise we use the builtin
# pathlib and can only read/write to the local filesystem.
try:
  from etils import epath  # type: ignore
except ImportError:
  logger.debug("etils.epath was not found. Using pathlib for file I/O.")
  Path = pathlib.Path
  epath_installed = False
else:
  logger.debug("etils.epath found. Using etils.epath for file I/O.")
  # Ultimately, epath.Path implements pathlib.Path.  See:
  # https://github.com/google/etils/blob/2083f3d932a88d8a135ef57112cd1f9aff5d559e/etils/epath/abstract_path.py#L47
  Path = epath.Path
  epath_installed = True

def make_jax_dump_dir(out_dir_path: str) -> pathlib.Path | None:
  """Make a directory or return the undeclared outputs directory if `sponge`."""
  if not out_dir_path:
    return None
  if out_dir_path == "sponge":
    out_dir_path = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
    if not out_dir_path:
      raise ValueError(
          "Got output directory (e.g., via JAX_DUMP_IR_TO) 'sponge' but"
          " TEST_UNDECLARED_OUTPUTS_DIR is not defined."
      )
  out_dir = Path(out_dir_path)
  out_dir.mkdir(parents=True, exist_ok=True)
  return out_dir
