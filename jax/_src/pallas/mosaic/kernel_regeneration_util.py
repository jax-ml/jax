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

"""Helpers to encode and decode Mosaic kernel regeneration metadata."""

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
    A dict that can be directly passed to pallas_call as a 'mosaic_params'
    argument.

  Raises:
    TypeError: when the input metadata is not serializable in json format.
  """
  serialized_metadata = bytes(json.dumps(metadata), encoding="utf-8")
  return dict(kernel_regeneration_metadata=serialized_metadata)


def extract_kernel_regeneration_metadata(op: ir.Operation) -> dict[str, Any]:
  """Extract kernel regeneration metadata from the given Operation.

  This function hides the serialization details from the end user.

  Args:
    op: the tpu custom_call mlir Operation that contains the kernel metadata.

  Returns:
    The decoded metadata in the form of a dict. This corresponds to the dict
    in input to the 'encode' function.
  """
  kernel_regeneration_metadata = ir.StringAttr(
      op.attributes["kernel_regeneration_metadata"]
  ).value
  return json.loads(base64.b64decode(kernel_regeneration_metadata))
