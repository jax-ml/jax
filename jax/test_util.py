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

from jax._src.public_test_util import (
  check_grads as check_grads,
  check_jvp as check_jvp,
  check_vjp as check_vjp,
)

# TODO(jakevdp): remove everything below once downstream callers are fixed.

# Unconditionally import private test_util because it contains flag definitions.
# In bazel, jax._src.test_util requires its own BUILD target so it may not be present.
# pytype: disable=import-error
try:
  import jax._src.test_util as _private_test_util
except ImportError:
  pass
else:
  del _private_test_util

# Use module-level getattr to add warnings to imports of deprecated names.
# pylint: disable=import-outside-toplevel
def __getattr__(attr):
  try:
    from jax._src import test_util
  except ImportError:
    raise AttributeError(f"module {__name__} has no attribute {attr}")
  if attr in ['cases_from_list', 'check_close', 'check_eq', 'device_under_test',
              'format_shape_dtype_string', 'rand_uniform', 'skip_on_devices',
              'with_config', 'xla_bridge', '_default_tolerance']:
    import warnings
    warnings.warn(f"jax.test_util.{attr} is deprecated and will soon be removed.", FutureWarning)
    return getattr(test_util, attr)
  else:
    raise AttributeError(f"module {__name__} has no attribute {attr}")
