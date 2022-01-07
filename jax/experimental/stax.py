# Copyright 2021 Google LLC
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

"""Stax has moved to jax.example_libraries.stax

jax.experimental.stax is deprecated and will delegate to
jax.example_libraries.stax with a warning for backwards-compatibility
for a limited time.
"""

import warnings

from jax.example_libraries.stax import *    # noqa: F401,F403

_HAS_DYNAMIC_ATTRIBUTES = True

warnings.warn('jax.experimental.stax is deprecated, '
              'import jax.example_libraries.stax instead',
              FutureWarning)
