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
"""pytest configuration"""

import jax
import numpy
import os
import pytest


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
  doctest_namespace["jax"] = jax
  doctest_namespace["lax"] = jax.lax
  doctest_namespace["jnp"] = jax.numpy
  doctest_namespace["np"] = numpy


@pytest.fixture(autouse=True)
def spoof_devices(doctest_namespace):
  # Set up runtime to mimic an 8-core machine
  flags = os.environ.get('XLA_FLAGS', '')
  os.environ['XLA_FLAGS'] = flags + " --xla_force_host_platform_device_count=8"
