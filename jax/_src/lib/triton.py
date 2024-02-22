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

# ruff: noqa

import sys

# TODO(slebedev): Update the message to recommend jaxlib 0.4.25.
_ERROR = (
    "Cannot import the Triton bindings. You may need a newer version of"
    " jaxlib. Try installing a nightly wheel following instructions in"
    " https://jax.readthedocs.io/en/latest/installation.html#nightly-installation"
)

try:
  from jaxlib.triton import dialect  # pytype: disable=import-error
except TypeError:
  from jaxlib import version

  if sys.version_info[:2] == (3, 9) and version.__version_info__ < (0, 4, 25):
    # Triton MLIR bindings are known to be broken on Python 3.9 in jaxlib
    # prior to 0.4.25.
    raise ModuleNotFoundError(_ERROR) from None
  raise
except ImportError as e:
  raise ModuleNotFoundError(_ERROR) from e
