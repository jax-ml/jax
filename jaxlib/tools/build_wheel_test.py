# Copyright 2024 The JAX Authors.
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

# This test verifies that the build_wheel.py runs successfully.

import platform
import subprocess
import sys
import tempfile

from bazel_tools.tools.python.runfiles import runfiles

r = runfiles.Create()

with tempfile.TemporaryDirectory(prefix="jax_build_wheel_test") as tmpdir:
  subprocess.run([
    sys.executable, r.Rlocation("__main__/jaxlib/tools/build_wheel.py"),
    f"--cpu={platform.machine()}",
    f"--output_path={tmpdir}",
    "--jaxlib_git_hash=12345678"
  ], check=True)
