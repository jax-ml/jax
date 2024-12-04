# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Converts MSYS Linux-like paths stored in env variables to Windows paths.

This is necessary on Windows, because some applications do not understand/handle
<<<<<<< HEAD
Linux-like paths MSYS uses, for example, Docker.
=======
Linux-like paths MSYS uses, for example, Bazel.
>>>>>>> 5ade371c88a1f879556ec29867b173da49ae57f0
"""
import argparse
import os
import subprocess

def msys_to_windows_path(msys_path):
  """Converts an MSYS path to a Windows path using cygpath.

  Args:
    msys_path: The MSYS path to convert.

  Returns:
    The corresponding Windows path.
  """
  try:
    # Use cygpath with the -w flag to convert to Windows format
    process = subprocess.run(['cygpath', '-w', msys_path], capture_output=True, text=True, check=True)
    windows_path = process.stdout.strip()
    return windows_path
  except FileNotFoundError:
    print("Error: cygpath not found. Make sure it's in your PATH.")
    return None
  except subprocess.CalledProcessError as e:
    print(f"Error converting path: {e}")
    return None

<<<<<<< HEAD
=======
def should_convert(var: str,
                   convert: list[str] | None):
  """Check the variable name against convert list"""
  if var in convert:
    return True
  else:
    return False

>>>>>>> 5ade371c88a1f879556ec29867b173da49ae57f0
def main(parsed_args: argparse.Namespace):
  converted_paths = {}

  for var, value in os.environ.items():
<<<<<<< HEAD
    if (parsed_args.blacklist and var in parsed_args.blacklist) or not value:
      continue
    if "_DIR" in var or (args.whitelist and var in parsed_args.whitelist):
      converted_path = msys_to_windows_path(value)
      converted_paths[var] = converted_path
=======
    if not value or not should_convert(var,
                                       parsed_args.convert):
      continue
    converted_path = msys_to_windows_path(value)
    converted_paths[var] = converted_path
>>>>>>> 5ade371c88a1f879556ec29867b173da49ae57f0

  var_str = '\n'.join(f'export {k}="{v}"'
                      for k, v in converted_paths.items())
  # The string can then be piped into `source`, to re-set the
  # 'converted' variables.
  print(var_str)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=(
      'Convert MSYS paths in environment variables to Windows paths.'))
<<<<<<< HEAD
  parser.add_argument('--blacklist',
                      nargs='*',
                      help='List of variables to ignore')
  parser.add_argument('--whitelist',
                      nargs='*',
                      help='List of variables to include')
=======
  parser.add_argument('--convert',
                      nargs='+',
                      required=True,
                      help='Space separated list of environment variables to convert. E.g: --convert env_var1 env_var2')
>>>>>>> 5ade371c88a1f879556ec29867b173da49ae57f0
  args = parser.parse_args()

  main(args)
