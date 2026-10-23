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

"""External flags initialization for xla_debug_log."""

import os


def set_xla_debug_log_flags():
  args = os.environ.get("LIBTPU_INIT_ARGS", "")
  new_flags = [
      "--xla_tpu_enable_log_recorder",
      "--xla_jf_log_scopes",
      "--xla_tpu_enable_hlo_value_tracking=enabled",
  ]
  for flag in new_flags:
    if flag not in args:
      args = f"{args} {flag}" if args else flag
  os.environ["LIBTPU_INIT_ARGS"] = args
