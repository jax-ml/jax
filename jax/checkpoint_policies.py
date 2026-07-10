# Copyright 2026 The JAX Authors.
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

from jax._src.ad_checkpoint import (
  SaveOnlyTheseNames as SaveOnlyTheseNames,
  SaveAnyNamesButThese as SaveAnyNamesButThese,
  SaveAndOffloadOnlyTheseNames as SaveAndOffloadOnlyTheseNames,
  everything_saveable as everything_saveable,
  nothing_saveable as nothing_saveable,
  dots_saveable as dots_saveable,
  dots_with_no_batch_dims_saveable as dots_with_no_batch_dims_saveable,
  offload_dot_with_no_batch_dims as offload_dot_with_no_batch_dims,
  save_anything_except_these_names as save_anything_except_these_names,
  save_any_names_but_these as save_any_names_but_these,
  save_only_these_names as save_only_these_names,
  save_from_both_policies as save_from_both_policies,
  save_and_offload_only_these_names as save_and_offload_only_these_names,
)

checkpoint_dots = dots_saveable
checkpoint_dots_with_no_batch_dims = dots_with_no_batch_dims_saveable
