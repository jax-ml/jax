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

"""Internal flags initialization for xla_debug_log."""

from absl import flags


def set_xla_debug_log_flags():
  flag_values = [
      ("xla_tpu_enable_log_recorder", True),
      ("xla_jf_log_scopes", True),
      ("xla_tpu_enable_hlo_value_tracking", "enabled"),
  ]
  for name, value in flag_values:
    try:
      # At JAX module load time, the C++ runtime may not be loaded yet, which
      # means C++ absl flags (e.g. xla_jf_log_scopes) are not yet present in
      # flags.FLAGS. We define them here in Python first using
      # allow_override_cpp=True so they can be configured early. When the JAX
      # C++ runtime initializes later, it binds its C++ flags to these Python
      # definitions instead of raising a DuplicateFlagError.
      if name not in flags.FLAGS:
        if isinstance(value, bool):
          flags.DEFINE_bool(
              name,
              value,
              f"Auto-set by xla_debug_log: {name}",
              allow_override_cpp=True,
          )
        else:
          flags.DEFINE_string(
              name,
              value,
              f"Auto-set by xla_debug_log: {name}",
              allow_override_cpp=True,
          )
      else:
        flags.FLAGS[name].value = value
    except Exception:
      pass
