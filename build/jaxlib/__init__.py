# Copyright 2018 Google LLC
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

from . import xla_client
from .version import __version__

class xla_data_pb2(object):
  """Compatibility shim for supporting Jax versions 0.1.20 or older.

  Delete when we next break Jaxlib backward compatibility."""
  GatherDimensionNumbers = xla_client.GatherDimensionNumbers
  ScatterDimensionNumbers = xla_client.ScatterDimensionNumbers
  ConvolutionDimensionNumbers = xla_client.ConvolutionDimensionNumbers
  DotDimensionNumbers = xla_client.DotDimensionNumbers

for name, value in xla_client.PrimitiveType.__members__.items():
  setattr(xla_data_pb2, name, value)
