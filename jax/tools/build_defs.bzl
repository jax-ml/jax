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

def _shell_quote(s):
    """Copy of bazel-skylib's shell.quote.

    Quotes the given string for use in a shell command.

    This function quotes the given string (in case it contains spaces or other
    shell metacharacters.)

    Args:
      s: The string to quote.
    Returns:
      A quoted version of the string that can be passed to a shell command.
    """
    return "'" + s.replace("'", "'\\''") + "'"

def jax_to_hlo(name, deps, fn, input_shapes, constants = None):
    """Creates a genrule that uses jax_to_hlo.py to make HLO from a JAX func.

    Suppose we have

      $ cat your/thing/prog.py
      import jax.numpy as np

      def fn(x, y, z):
        return np.dot(x, y) / z

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
        input_shapes = [
          ("y", "f32[128,32]"),
          ("x", "f32[8,128]"),
        ],
        constants = {
          "z": "3.14159",
        },
      )

    This generates two build rules, named

      //your/thing:prog_hlo.pb
      //your/thing:prog_hlo.txt

    containing HLO proto and text representations of the JAX function.  You can
    then depend on these as data dependencies from e.g. C++.

    The parameters to the XLA program will be in the order of `input_shapes`.
    That is, the above macro will create a program which accepts parameters y
    and x, in that order.

    Skylark doesn't support floating-point numbers without a special option, so
    we need special rules to be able to pass fp values in `constants`.  Each
    dict value `v` is transformed as follows.

     - If `v` is a string, parse it as a Python literal:
       Let `v = ast.literal_eval(v)`.
     - After parsing, if `v` is a list, convert it to a numpy array:
       `v = np.asarray(v)`.

    For example:

      constants = {
        "a": "1.0"           # float 1.0
        "b": [1, 2]          # yields np.asarray([1, 2]) (ints)
        "c": "[2.0, 3.0]",   # yields np.asarray([1.0, 2.0]) (floats)
        "d": "'hello'"       # yields string "hello"

        "x": ["1.0"],        # yields array of strings, np.asarray(["1.0"])
        "y": "hello"         # illegal, use "'hello'" for a string.
      }

    Args:
      name: Name of this genrule, which will produce <name>.pb and <name>.txt.
      deps: Targets to depend on, one of which must contain the JAX function.
      fn: Fully-qualified Python name of function (e.g. "path.to.fn")
      input_shapes: List of tuples (param_name, shape), e.g.
        [("y", "f32[]"), ("x", "f32[10,20]")]. The XLA program's parameters
        will be in the order specified here.
      constants: Python dictionary mapping arg names to constant values they
        should take on, e.g. {"z": "float(3.14159")}.
    """
    if not constants:
        constants = {}

    # Our goal here is to create a py_binary which depends on `deps` and
    # invokes the main function defined in jax_to_hlo.py.
    #
    # At first blush it seems that we can do this in a straightforward way:
    #
    #   native.py_binary(main = "jax_to_hlo.py").
    #
    # The problem with this is that this py_binary lives in the user's package,
    # whereas jax_to_hlo.py lives inside JAX's package.  Bazel delivers a stern
    # warning if you name a file in `main` that's outside of your package.
    #
    # To avoid the warning, we generate a simple "main file" in the user's
    # package.  It only complicates the rules a bit.
    runner = name + "_jax_to_hlo_main"
    native.genrule(
        name = runner + "_gen",
        outs = [runner + ".py"],
        cmd = """cat <<EOF > '$(location {runner}.py)'
from absl import app
import jax.tools.jax_to_hlo as jax_to_hlo

jax_to_hlo.set_up_flags()
app.run(jax_to_hlo.main)
EOF
        """.format(runner = runner),
    )

    native.py_binary(
        name = runner,
        srcs = [
            runner + ".py",
        ],
        python_version = "PY3",
        deps = deps + [
            "//third_party/py/jax/jaxlib",
            "//third_party/py/jax/tools:jax_to_hlo",
        ],
    )

    # Create a genrule which invokes the py_binary created above.
    #
    # Set JAX_PLATFORM_NAME to "cpu" to silence the "no GPU/TPU backend found,
    # falling back to CPU" warning.
    native.genrule(
        name = name + "_jax_to_hlo_genrule",
        outs = [name + ".pb", name + ".txt"],
        exec_tools = [runner],
        cmd = """
        JAX_PLATFORM_NAME=cpu \
        '$(location {runner})' \
          --fn {fn} \
          --input_shapes {input_shapes} \
          --evaled_constants {constants} \
          --hlo_proto_dest '$(location {name}.pb)' \
          --hlo_text_dest '$(location {name}.txt)' \
        """.format(
            name = name,
            fn = fn,
            input_shapes = _shell_quote(str(input_shapes)),
            constants = _shell_quote(str(constants)),
            runner = runner,
        ),
    )
