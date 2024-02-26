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

"""Helpers to encode and decode Pallas kernel regeneration metadata."""

import base64
import json
from typing import Any

from jaxlib.mlir import ir


def encode_kernel_regeneration_metadata(
    metadata: dict[str, Any]
) -> dict[str, bytes]:
  """Serializes the given kernel regeneration metadata.

  This function hides the serialization details from the end user.

  Args:
    metadata: dictionary with user-defined data to be serialized in the backend
      config.

  Returns:
    A dict that can be directly passed to `pallas_call` via the `*_params`
    argument.

  Raises:
    TypeError: when the input metadata is not serializable in JSON format.
  """
  serialized_metadata = base64.b64encode(json.dumps(metadata).encode())
  return dict(kernel_regeneration_metadata=serialized_metadata)


def extract_kernel_regeneration_metadata(op: ir.Operation) -> dict[str, Any]:
  """Returns the kernel regeneration metadata stored in the given operation.

  This function hides the serialization details from the end user.

  Args:
    op: the operation that contains the kernel metadata.
  """
  kernel_regeneration_metadata = ir.StringAttr(
      op.attributes["kernel_regeneration_metadata"]
  ).value
  return json.loads(base64.b64decode(kernel_regeneration_metadata))
