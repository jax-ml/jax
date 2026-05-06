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

import unittest
from absl.testing import absltest
from jax._src import pretty_printer as pp
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version


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

  @unittest.skipIf(jaxlib_extension_version < 449, "Requires jaxlib >= 449")
  def testHtmlFormat(self):
    doc = pp.concat([
        pp.keyword(pp.text("let")),
        pp.text(" "),
        pp.type_annotation(pp.text("x:f32")),
    ])
    # HTML format without color (should only escape)
    self.assertEqual(
        doc.format(output_format=pp.OutputFormat.HTML, use_color=False),
        "let x:f32",
    )

    doc_escape = pp.text("a < b & c > d")
    self.assertEqual(
        doc_escape.format(output_format=pp.OutputFormat.HTML),
        "a &lt; b &amp; c &gt; d",
    )

    # HTML format with color
    self.assertEqual(
        doc.format(output_format=pp.OutputFormat.HTML, use_color=True),
        '<span class="ansi-fg-34 ansi-intensity-1">let</span> <span'
        ' class="ansi-fg-35">x:f32</span>',
    )

    # Breakdoc strings are html escaped
    doc_break_escape = pp.group(pp.text("a") + pp.brk("<br>") + pp.text("b"))
    self.assertEqual(
        doc_break_escape.format(output_format=pp.OutputFormat.HTML),
        "a&lt;br&gt;b",
    )

  @unittest.skipIf(jaxlib_extension_version < 449, "Requires jaxlib >= 449")
  def testHtmlSourceMapWithColors(self):
    doc = pp.color(
        pp.source_map(pp.text("a"), "source1")
        + pp.brk()
        + pp.source_map(pp.text("b"), "source2"),
        foreground=pp.Color.MAGENTA,
    )
    source_map_output = []
    rendered = doc.format(
        output_format=pp.OutputFormat.HTML,
        use_color=True,
        separable_lines=True,
        width=0,
        source_map=source_map_output,
    )
    lines = rendered.splitlines()

    self.assertEqual(len(source_map_output[0]), 1)
    start, end, source = source_map_output[0][0]
    self.assertEqual(source, "source1")
    self.assertEqual(lines[0][start:end], '<span class="ansi-fg-35">a</span>')

    self.assertEqual(len(source_map_output[1]), 1)
    start, end, source = source_map_output[1][0]
    self.assertEqual(source, "source2")
    self.assertEqual(lines[1][start:end], '<span class="ansi-fg-35">b</span>')

  def testAnsiColorBasic(self):
    doc = pp.concat([
        pp.color(
            pp.text("let"),
            foreground=pp.Color.BLUE,
            intensity=pp.Intensity.BRIGHT,
        ),
        pp.text(" "),
        pp.color(
            pp.text("x:f32"),
            foreground=pp.Color.MAGENTA,
            intensity=pp.Intensity.NORMAL,
        ),
    ])
    self.assertEqual(
        doc.format(use_color=True),
        "\x1b[34;1mlet\x1b[39;22m \x1b[35mx:f32\x1b[39m",
    )

  def testAnsiColorWithAnnotationsLinear(self):
    doc = pp.color(
        pp.text("a", annotation="annot1") + pp.text("b", annotation="annot2"),
        foreground=pp.Color.MAGENTA,
    )
    self.assertEqual(
        doc.format(use_color=True),
        "\x1b[35mab\x1b[39m\x1b[2m # annot1\x1b[22m\n\x1b[2m   #"
        " annot2\x1b[22m",
    )

  def testAnsiColorWithAnnotationsSeparable(self):
    doc = pp.color(
        pp.text("a", annotation="annot1") + pp.text("b", annotation="annot2"),
        foreground=pp.Color.MAGENTA,
    )
    self.assertEqual(
        doc.format(use_color=True, separable_lines=True),
        "\x1b[35mab\x1b[39m\x1b[2m # annot1\x1b[22m\n\x1b[2m   #"
        " annot2\x1b[22m",
    )

  def testAnsiColorMultilineLinear(self):
    doc = pp.color(
        pp.text("a", annotation="annot1")
        + pp.brk()
        + pp.text("b", annotation="annot2"),
        foreground=pp.Color.MAGENTA,
    )
    self.assertEqual(
        doc.format(use_color=True, width=0),
        "\x1b[35ma\x1b[39m\x1b[2m # annot1\x1b[22m\n\x1b[35mb\x1b[39m\x1b[2m #"
        " annot2\x1b[22m",
    )

  def testAnsiColorMultilineSeparable(self):
    doc = pp.color(
        pp.text("a", annotation="annot1")
        + pp.brk()
        + pp.text("b", annotation="annot2"),
        foreground=pp.Color.MAGENTA,
    )
    self.assertEqual(
        doc.format(use_color=True, separable_lines=True, width=0),
        "\x1b[35ma\x1b[39m\x1b[2m # annot1\x1b[22m\n\x1b[35mb\x1b[39m\x1b[2m #"
        " annot2\x1b[22m",
    )

  @unittest.skipIf(jaxlib_extension_version < 449, "Requires jaxlib >= 449")
  def testSeparableLinesHtml(self):
    doc = pp.type_annotation(pp.text("a") + pp.brk() + pp.text("b"))

    # HTML mode, separable_lines=False (default)
    self.assertEqual(
        doc.format(output_format=pp.OutputFormat.HTML, use_color=True, width=0),
        '<span class="ansi-fg-35">a\nb</span>',
    )

    # HTML mode, separable_lines=True
    self.assertEqual(
        doc.format(
            output_format=pp.OutputFormat.HTML,
            use_color=True,
            separable_lines=True,
            width=0,
        ),
        '<span class="ansi-fg-35">a</span>\n<span class="ansi-fg-35">b</span>',
    )

  def testSeparableLinesAnsi(self):
    doc = pp.type_annotation(pp.text("a") + pp.brk() + pp.text("b"))

    # ANSI mode, separable_lines=False (default)
    self.assertEqual(
        doc.format(output_format=pp.OutputFormat.TEXT, use_color=True, width=0),
        "\x1b[35ma\nb\x1b[39m",
    )

    # ANSI mode, separable_lines=True
    self.assertEqual(
        doc.format(
            output_format=pp.OutputFormat.TEXT,
            use_color=True,
            separable_lines=True,
            width=0,
        ),
        "\x1b[35ma\x1b[39m\n\x1b[35mb\x1b[39m",
    )

  @unittest.skipIf(jaxlib_extension_version < 449, "Requires jaxlib >= 449")
  def testHtmlFormatWithAnnotations(self):
    doc = pp.color(
        pp.text("a", annotation="annot1")
        + pp.brk()
        + pp.text("b", annotation="annot2"),
        foreground=pp.Color.MAGENTA,
    )
    self.assertEqual(
        doc.format(
            output_format=pp.OutputFormat.HTML,
            use_color=True,
            separable_lines=True,
            width=0,
        ),
        '<span class="ansi-fg-35">a</span><span class="ansi-intensity-2"> #'
        ' annot1</span>\n<span class="ansi-fg-35">b</span><span'
        ' class="ansi-intensity-2"> # annot2</span>',
    )
    self.assertEqual(
        doc.format(
            output_format=pp.OutputFormat.HTML,
            use_color=True,
            separable_lines=False,
            width=0,
        ),
        '<span class="ansi-fg-35">a</span><span class="ansi-intensity-2"> #'
        ' annot1</span>\n<span class="ansi-fg-35">b</span><span'
        ' class="ansi-intensity-2"> # annot2</span>',
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
