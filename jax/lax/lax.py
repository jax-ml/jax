# Copyright 2020 Google LLC
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

# TODO(phawkins): remove these aliases after fixing callers.

# flake8: noqa: F401
from jax._src.lax.lax import (
  _dilate_shape,
  _dot_general_translation_rule,
  _top_k_abstract_eval,
  _top_k_jvp,
  add,
  add_p,
  div_p,
  broadcast_in_dim_p,
  conv_general_dilated,
  conv_general_dilated_p,
  dot_general,
  dot_general_p,
  max,
  max_p,
  mul_p,
  reshape,
  reshape_p,
  standard_primitive,
  stop_gradient,
  sub,
  sub_p,
  tie_in_p,
)
