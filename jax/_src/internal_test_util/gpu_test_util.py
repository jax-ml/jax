# Copyright 2025 The JAX Authors.
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

import argparse
import ctypes
import importlib
import os
import pathlib
import subprocess
import sys


def parse_args():
  """Arguments parser."""
  parser = argparse.ArgumentParser(
      description="blabla",
      fromfile_prefix_chars="@",
  )
  parser.add_argument("--test_script_path", required=True, help="blabla")
  return parser.parse_known_args()


def _load(module, libraries):
  try:
    m = importlib.import_module(f"nvidia.{module}")
  except ImportError:
    m = None

  for lib in libraries:
    if m is not None:
      path = pathlib.Path(m.__path__[0]) / "lib" / lib
      try:
        ctypes.cdll.LoadLibrary(path)
        continue
      except OSError as e:
        continue


def _load_cuda_umd_libs():
  _load("cuda_driver", ["libcuda.so.1", "libnvidia-ptxjitcompiler.so.1"])
  _load("cu13", ["libcuda.so.1", "libnvidia-ptxjitcompiler.so.1"])


if __name__ == "__main__":
  args = parse_args()
  _load_cuda_umd_libs()
  try:
    subprocess.run(
        [sys.executable, args[0].test_script_path] + args[1],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ,
    )
  except subprocess.CalledProcessError as e:
    raise RuntimeError(e.stderr) from e
