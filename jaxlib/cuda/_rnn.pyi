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

def registrations() -> dict: ...
def build_rnn_descriptor(
    arg0: int,
    arg1: int,
    arg2: int,
    arg3: int,
    arg4: int,
    arg5: float,
    arg6: bool,
    arg7: bool,
    arg8: int,
    arg9: int,
    /,
) -> bytes: ...
def compute_rnn_workspace_reserve_space_sizes(
    arg0: int,
    arg1: int,
    arg2: int,
    arg3: int,
    arg4: int,
    arg5: float,
    arg6: bool,
    arg7: bool,
    /,
) -> tuple[int, int]: ...
