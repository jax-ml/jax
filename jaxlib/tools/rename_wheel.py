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

"""Script for renaming wheels."""

import argparse
import os
import shutil
import jaxlib.version

parser = argparse.ArgumentParser(description="Rename wheel script arguments")
parser.add_argument("--output-path", required=False, default="dist")
args = parser.parse_args()

new_version = jaxlib.version._get_version_for_build()
git_hash_ind = new_version.find("+") if "+" in new_version else len(new_version)

wheel_dir = os.path.join("jaxlib", "tools", os.getenv("WHEEL_DIR"))
old_file_name = ""
for f in os.listdir(wheel_dir):
  if f.endswith(".whl"):
    old_file_name = f
    break
new_file_name = old_file_name.replace(
    "{}-0".format(os.getenv("WHEEL_VERSION")), new_version[:git_hash_ind]
)

workspace_dir = os.path.realpath(os.getenv("BUILD_WORKSPACE_DIRECTORY"))
new_dir = os.path.join(workspace_dir, args.output_path)
if not os.path.exists(new_dir):
  os.mkdir(new_dir)
old_file = os.path.join(wheel_dir, old_file_name)
new_file = os.path.join(new_dir, new_file_name)
shutil.copyfile(old_file, new_file)

print("Renamed wheel path: %s" % new_file)
