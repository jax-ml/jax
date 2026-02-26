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

"""Build rules for generating .pyi type stubs for nanobind extension modules."""

def py_stubgen(name, module, outs = None, stubgen = "//jaxlib/mlir/_mlir_libs:stubgen", normalize = "//jaxlib/mlir/_mlir_libs:normalize_stubs"):
    if outs == None:
        _, _, basename = module.rpartition(".")
        outs = [basename + ".pyi"]

    native.genrule(
        name = name,
        outs = outs,
        cmd = """
            $(location {stubgen}) -m {module} -O $(RULEDIR)
            $(location {normalize}) --jaxlib-build $(OUTS)
        """.format(module = module, stubgen = stubgen, normalize = normalize),
        tools = [normalize, stubgen],
    )
