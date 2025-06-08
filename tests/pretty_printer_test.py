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

from absl.testing import absltest
from jax._src import pretty_printer as pp
from jax._src import test_util as jtu


class PrettyPrinterTest(jtu.JaxTestCase):

  def testSourceMap(self):
    doc = pp.concat([
        pp.text("abc"),
        pp.source_map(pp.text("def"), 101),
        pp.source_map(
            pp.concat([pp.text("gh"), pp.brk(""), pp.text("ijkl")]), 77
        ),
        pp.text("mn"),
    ])
    source_map = []
    out = doc.format(width=8, source_map=source_map)
    self.assertEqual(out, "abcdefgh\nijklmn")
    self.assertEqual(source_map, [[(3, 6, 101), (6, 8, 77)], [(0, 4, 77)]])

  def testBasics(self):
    self.assertEqual(pp.nil().format(), "")
    self.assertEqual(pp.text("").format(), "")
    self.assertEqual(pp.text("testing").format(), "testing")
    self.assertEqual(pp.text("\n").format(), "\n")
    self.assertEqual(pp.brk().format(), "\n")
    # Group that fits will use the space from brk()
    self.assertEqual(pp.group(pp.brk()).format(), " ")
    # Group that doesn't fit (due to width=0) will use newline
    self.assertEqual(pp.group(pp.brk()).format(width=0), "\n")

    # Custom break text
    self.assertEqual(pp.group(pp.brk("-")).format(), "-")
    self.assertEqual(pp.group(pp.brk("-")).format(width=0), "\n")

    # Concatenation
    self.assertEqual((pp.text("a") + pp.text("b")).format(), "ab")
    self.assertEqual(pp.concat([pp.text("a"), pp.text("b c")]).format(), "ab c")

    x = pp.text("x")
    y = pp.text("y")
    z = pp.text("z")

    # Join
    # Join with a break that becomes a space when fitting
    join_doc_space = pp.join(
        pp.text(",") + pp.brk(), [pp.text("a"), pp.text("b"), pp.text("c")]
    )
    self.assertEqual(pp.group(join_doc_space).format(), "a, b, c")
    self.assertEqual(pp.group(join_doc_space).format(width=5), "a,\nb,\nc")
    self.assertEqual(pp.join(pp.text(","), [x, y, z]).format(), "x,y,z")

    j = pp.join(
        pp.brk(), [pp.text("xx"), pp.text("yy"), pp.text("zz"), pp.text("ww")]
    )
    self.assertEqual(pp.group(j).format(width=3), "xx\nyy\nzz\nww")
    self.assertEqual(pp.group(j).format(width=80), "xx yy zz ww")

    bx = pp.brk() + x
    bxbx = bx + bx
    bx4 = bxbx + bxbx

    # Horizontal-like (fits)
    self.assertEqual(pp.group(bx).format(), " x")
    self.assertEqual(pp.group(bxbx).format(), " x x")
    self.assertEqual(pp.group(bx4).format(), " x x x x")

    # Vertical-like (forced by width)
    self.assertEqual(pp.group(bx).format(width=0), "\nx")
    self.assertEqual(pp.group(bxbx).format(width=0), "\nx\nx")
    self.assertEqual(pp.group(bx4).format(width=0), "\nx\nx\nx\nx")
    self.assertEqual(pp.group(bxbx).format(width=3), "\nx\nx")

    # Nesting
    xbybz = x + pp.brk() + y + pp.brk() + z
    self.assertEqual(pp.nest(2, pp.group(bx)).format(), " x")  # Stays flat
    self.assertEqual(pp.nest(2, pp.group(bxbx)).format(), " x x")  # Stays flat
    self.assertEqual(pp.nest(2, pp.group(bx)).format(width=0), "\n  x")
    self.assertEqual(
        pp.nest(2, pp.nest(2, pp.group(bx))).format(width=0), "\n    x"
    )
    self.assertEqual(pp.nest(2, pp.group(xbybz)).format(width=0), "x\n  y\n  z")
    self.assertEqual(pp.nest(2, pp.group(bxbx)).format(width=0), "\n  x\n  x")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
