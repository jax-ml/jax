# Copyright 2018 The JAX Authors.
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

import numpy as np

from .cpu import _lapack
from .cpu._lapack import eig
from .cpu._lapack import schur


EigComputationMode = eig.ComputationMode
SchurComputationMode = schur.ComputationMode
SchurSort = schur.Sort


LAPACK_DTYPE_PREFIX = {
    np.float32: "s",
    np.float64: "d",
    np.complex64: "c",
    np.complex128: "z",
}


def registrations() -> dict[str, list[tuple[str, Any, int]]]:
  return {"cpu": [
      (name, value, int(name.endswith("_ffi")))
      for name, value in _lapack.registrations().items()
  ]}


def batch_partitionable_targets() -> list[str]:
  return [name for name in _lapack.registrations() if name.endswith("_ffi")]


def prepare_lapack_call(fn_base, dtype):
  """Initializes the LAPACK library and returns the LAPACK target name."""
  _lapack.initialize()
  return build_lapack_fn_target(fn_base, dtype)


def build_lapack_fn_target(fn_base: str, dtype) -> str:
  """Builds the target name for a LAPACK function custom call."""
  try:
    prefix = (
        LAPACK_DTYPE_PREFIX.get(dtype, None) or LAPACK_DTYPE_PREFIX[dtype.type]
    )
    return f"lapack_{prefix}{fn_base}"
  except KeyError as err:
    raise NotImplementedError(err, f"Unsupported dtype {dtype}.") from err
