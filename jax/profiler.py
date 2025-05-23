# Copyright 2020 The JAX Authors.
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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570
from typing import Any

from jax._src.profiler import (
    ProfileOptions as ProfileOptions,
    StepTraceAnnotation as StepTraceAnnotation,
    TraceAnnotation as TraceAnnotation,
    annotate_function as annotate_function,
    device_memory_profile as device_memory_profile,
    save_device_memory_profile as save_device_memory_profile,
    start_server as start_server,
    start_trace as start_trace,
    stop_server as stop_server,
    stop_trace as stop_trace,
    trace as trace,
)

# this is a temporary shim to please pytype in the meantime before the migration
# is complete for cl/760646494
ProfileData: Any = None
ProfileEvent: Any = None
ProfilePlane: Any = None
