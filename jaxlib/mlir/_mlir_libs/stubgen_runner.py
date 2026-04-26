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

"""Generates .pyi stubs for nanobind extensions using nanobind's stubgen."""

import argparse
import importlib.util
import sys

# pyrefly: ignore [missing-import]
from python.runfiles import Runfiles

# Recreate the MLIR namespace, because dialects might reference it.
# pyrefly: ignore [missing-import]
from jaxlib.mlir._mlir_libs import _mlir
sys.modules["mlir"] = _mlir
sys.modules["mlir.ir"] = _mlir.ir
sys.modules["mlir.passmanager"] = _mlir.passmanager
sys.modules["mlir.rewrite"] = _mlir.rewrite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--module", required=True, help="Module name to generate stubs for"
    )
    parser.add_argument("-O", "--output", required=True, help="Output directory")
    args = parser.parse_args()

    runfiles = Runfiles.Create()
    stubgen_path = runfiles.Rlocation("nanobind/src/stubgen.py")
    spec = importlib.util.spec_from_file_location("stubgen", stubgen_path)
    stubgen = importlib.util.module_from_spec(spec)  # pyrefly: ignore [bad-argument-type]
    sys.modules["stubgen"] = stubgen
    spec.loader.exec_module(stubgen)  # pyrefly: ignore [missing-attribute]
    stubgen.main(
        ["-m", args.module, "-r", "--include-private", "-O", args.output]
    )


if __name__ == "__main__":
    main()
