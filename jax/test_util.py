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

# flake8: noqa: F401
from jax._src.public_test_util import (
  check_grads as check_grads,
  check_jvp as check_jvp,
  check_vjp as check_vjp,
)

# Conditional imports of private test utilities; these require their own BUILD target.
# TODO(jakevdp): remove these imports once downstream dependencies are cleaned.
try:
  from jax._src.test_util import (  # pytype: disable=import-error
    cases_from_list,
    check_close,
    check_eq,
    device_under_test,
    format_shape_dtype_string,
    rand_uniform,
    skip_on_devices,
    with_config as with_config,
    xla_bridge,
    _default_tolerance,
    DeprecatedJaxTestCase as JaxTestCase,
    DeprecatedJaxTestLoader as JaxTestLoader,
    DeprecatedBufferDonationTestCase as BufferDonationTestCase,
  )
except ImportError:
  pass
