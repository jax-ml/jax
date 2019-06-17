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

r"""Tool to convert a JAX function to an HLO proto.

This script is mean to be used as part of a genrule that converts a JAX program
into an HLO proto.  The HLO proto represents an XLA program, and can be run from
e.g. a C++ program, without involving any Python.

This lets you use JAX as a convenient frontend for writing "XLA programs".  From
another perspective, this script lets you make JAX into an ahead-of-time JAX ->
XLA compiler, although when you run the XLA program, it will still be compiled
just-in-time.

Usage:

  $ cat prog.py
  import jax.numpy as np

  def fn(x, y):
    return np.dot(x, y) / 2

  $ python jax_to_hlo.py \
    --fn prog.fn \
    --input_shapes '{"x": "f32[8,128]", "y": "f32[128,32]"}' \
    --hlo_text_dest /tmp/fn_hlo.txt \
    --hlo_proto_dest /tmp/fn_hlo.pb

Alternatively, you can use this script via a genrule.  This way bazel will
generate the hlo text/proto as part of compilation, and then e.g. a C++ program
can depend on this.  See jax_to_hlo macro in build_defs.bzl.

  $ cat your/thing/BUILD
  load("//jax/tools:build_defs.bzl", "jax_to_hlo")

  py_library(name = "prog", srcs = ["prog.py"])

  jax_to_hlo(
    src = "//your/thing:prog",
    fn = "your.thing.prog.fn",  # Fully-qualified module name
    input_shapes = {
      "x": "f32[8,128]",
      "y": "f32[128,32]",
    }
  )

This generates two BUILD targets:

  //your/thing:prog_hlo.pb
  //your/thing:prog_hlo.txt

You can then depend on one of these as a data dependency and invoke it using
XLA's public C++ APIs.

Note that XLA's backwards-compatibility guarantees for saved HLO are currently
(2019-06-13) best-effort.  It will mostly work, but it will occasionally break,
and the XLA team won't (and in fact will be unable to) help.  One way to be sure
it won't break is to use the same version of XLA to build the HLO as you use to
run it.  The genrule above makes this easy.

Implementation note: This script must be python2 compatible for now, because
Google's genrules still run with python2, b/66712815.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ast import literal_eval
import importlib

from absl import app
from absl import flags
import jax.api
import jax.numpy as np
from jaxlib import xla_client

FLAGS = flags.FLAGS


def jax_to_hlo(fn, input_shapes):
  """Converts a JAX function to an HLO module.

  Args:
    fn: Function to convert.
    input_shapes: Dict mapping function argument name to xla_client.Shape,
      indicating the shapes of the args to fn.

  Returns:
    A tuple (serialized_hlo_proto, hlo_text).
  """
  args = {}
  for arg_name, shape in input_shapes.items():
    if not shape.is_array():
      raise ValueError('Shape %s is not an array, but currently only arrays '
                       'are supported (i.e., no tuples).' % str(shape))

    # Check that `shape` either doesn't have a layout or has the default layout.
    #
    # TODO(jlebar): This could be simpler if the Shape class exposed its layout,
    # or if Shape exposed a function to unconditionally use the default layout.
    shape_with_default_layout = xla_client.Shape.array_shape(
        shape.xla_element_type(),
        shape.dimensions()).with_major_to_minor_layout_if_absent()
    if (shape.with_major_to_minor_layout_if_absent() !=
        shape_with_default_layout):
      raise ValueError('Shape %s has a non-default layout, but only '
                       'the default layout is allowed.' % str(shape))

    args[arg_name] = np.zeros(shape.dimensions(), dtype=shape.numpy_dtype())

  comp = jax.api.xla_computation(fn)(**args)
  return (comp.GetSerializedProto(), comp.GetHloText())


def main(argv):
  if len(argv) != 1:
    raise app.UsageError('No positional arguments are accepted.')

  if not FLAGS.hlo_proto_dest and not FLAGS.hlo_text_dest:
    raise app.Error('At least one of --hlo_proto_dest and '
                    '--hlo_text_dest is required.')

  module_name, fn_name = FLAGS.fn.rsplit('.', 1)
  module = importlib.import_module(module_name)
  fn = getattr(module, fn_name)

  input_shapes = {
    name: xla_client.Shape(shape_str)
    for name, shape_str in literal_eval(FLAGS.input_shapes).items()
  }
  hlo_proto, hlo_text = jax_to_hlo(fn, input_shapes)

  if FLAGS.hlo_proto_dest:
    with open(FLAGS.hlo_proto_dest, 'wb') as f:
      f.write(hlo_proto)

  if FLAGS.hlo_text_dest:
    with open(FLAGS.hlo_text_dest, 'w') as f:
      f.write(hlo_text)

def set_up_flags():
  flags.DEFINE_string(
      'fn', None,
      "Fully-qualified name of function that we're going to convert")
  flags.DEFINE_string('input_shapes', None,
                      'Python dict indicating XLA shapes of params')
  flags.DEFINE_string('hlo_proto_dest', None, 'File to write HLO proto')
  flags.DEFINE_string('hlo_text_dest', None, 'File to write HLO text')
  flags.mark_flag_as_required('fn')
  flags.mark_flag_as_required('input_shapes')


if __name__ == '__main__':
  set_up_flags()
  app.run(main)
