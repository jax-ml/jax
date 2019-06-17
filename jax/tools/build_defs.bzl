# Copyright 2019 Google LLC
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

def jax_to_hlo(name, deps, fn, input_shapes):
    """Creates a genrule that uses jax_to_hlo.py to make HLO from a JAX func.

    Suppose we have

      $ cat your/thing/prog.py
      import jax.numpy as np

      def fn(x, y):
        return np.dot(x, y) / 2

    and

      $ cat your/thing/BUILD
      py_library(name = "prog", srcs = ["prog.py"])

    Then we can invoke this macro as follows.

      $ cat your/thing/BUILD
      load("//jax/tools:build_defs.bzl", "jax_to_hlo")

      jax_to_hlo(
        name = "prog_hlo",
        deps = ["//your/thing:prog"],
        fn = "your.thing.prog.fn",  # Fully-qualified module name
        input_shapes = {
          "x": "f32[8,128]",
          "y": "f32[128,32]",
        }
      )

    This generates two build rules, named

      //your/thing:prog_hlo.pb
      //your/thing:prog_hlo.txt

    containing HLO proto and text representations of the JAX function.  You can
    then depend on these as data dependencies from e.g. C++.

    Args:
      name: Name of this genrule, which will produce <name>.pb and <name>.txt.
      deps: Targets to depend on, one of which must contain the JAX function.
      fn: Fully-qualified Python name of function (e.g. "path.to.fn")
      input_shapes: Python dictionary mapping arg names to XLA shapes, e.g.
        {"x": "f32[10,20]", "y": "f32[]"}
    """

    # Create a py_binary for jax_to_hlo.py which links in `src`.  This is how
    # jax_to_hlo will get access to the function in `src`.
    gen_bin = name + "_jax_to_hlo_gen_binary"
    native.py_binary(
        name = gen_bin,
        srcs = [
            "//third_party/py/jax/tools:jax_to_hlo.py",
        ],
        deps = deps + [
            "//third_party/py/jax/jaxlib",
            "//third_party/py/jax/tools:jax_to_hlo",
        ],
        main = "//third_party/py/jax/tools:jax_to_hlo.py",
    )

    # Create a genrule which invokes the py_binary created above.
    #
    # Set JAX_PLATFORM_NAME to "cpu" to silence the "no GPU/TPU backend found,
    # falling back to CPU" warning.
    native.genrule(
        name = name + "_jax_to_hlo_genrule",
        outs = [name + ".pb", name + ".txt"],
        tools = [gen_bin],
        cmd = """
        JAX_PLATFORM_NAME=cpu \
        $(location {gen_bin}) \
          --fn {fn} \
          --input_shapes '{input_shapes}' \
          --hlo_proto_dest '$(location {name}.pb)' \
          --hlo_text_dest '$(location {name}.txt)' \
        """.format(
            name = name,
            fn = fn,
            input_shapes = input_shapes,
            gen_bin = gen_bin,
        ),
    )
