# Copyright 2019 Google LLC
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

"""
Common neural network layer initializers, consistent with definitions
used in Keras and Sonnet.
"""

# flake8: noqa: F401
from jax._src.nn.initializers import (
  delta_orthogonal,
  glorot_normal,
  glorot_uniform,
  he_normal,
  he_uniform,
  kaiming_normal,
  kaiming_uniform,
  lecun_normal,
  lecun_uniform,
  normal,
  ones,
  orthogonal,
  uniform,
  variance_scaling,
  xavier_normal,
  xavier_uniform,
  zeros,
)
