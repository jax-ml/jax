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
# TODO(phawkins): remove all exports except check_grads/check_jvp/check_vjp.
from jax._src.test_util import (
  JaxTestCase as _PrivateJaxTestCase,
  JaxTestLoader as _PrivateJaxTestLoader,
  cases_from_list,
  check_close,
  check_eq,
  check_grads as check_grads,
  check_jvp as check_jvp,
  check_vjp as check_vjp,
  device_under_test,
  format_shape_dtype_string,
  rand_uniform,
  skip_on_devices,
  with_config,
  xla_bridge,
  _default_tolerance
)

class JaxTestCase(_PrivateJaxTestCase):
  def __init__(self, *args, **kwargs):
    import warnings
    import textwrap
    warnings.warn(textwrap.dedent("""\
      jax.test_util.JaxTestCase is deprecated as of jax version 0.3.1:
      The suggested replacement is to use parametrized.TestCase directly.
      For tests that rely on custom asserts such as JaxTestCase.assertAllClose(),
      the suggested replacement is to use standard numpy testing utilities such
      as np.testing.assert_allclose(), which work directly with JAX arrays."""),
      category=DeprecationWarning)
    super().__init__(*args, **kwargs)

class JaxTestLoader(_PrivateJaxTestLoader):
  def __init__(self, *args, **kwargs):
    import warnings
    warnings.warn(
      "jax.test_util.JaxTestLoader is deprecated as of jax version 0.3.1. Use absltest.TestLoader directly.",
      category=DeprecationWarning)
    super().__init__(*args, **kwargs)

del _PrivateJaxTestCase, _PrivateJaxTestLoader