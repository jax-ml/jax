# Copyright 2021 The JAX Authors.
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
#
# Wadler-Lindig pretty printer.
#
# References:
# Wadler, P., 1998. A prettier printer. Journal of Functional Programming,
# pp.223-244.
#
# Lindig, C. 2000. Strictly Pretty.
# https://lindig.github.io/papers/strictly-pretty-2000.pdf
#
# Hafiz, A. 2021. Strictly Annotated: A Pretty-Printer With Support for
# Annotations. https://ayazhafiz.com/articles/21/strictly-annotated
#

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import sys
from typing import Any

from jax._src import config
from jax._src.lib import _pretty_printer as _pretty_printer
from jax._src.util import use_cpp_class, use_cpp_method  # pyrefly: ignore[missing-import]


_PPRINT_USE_COLOR = config.bool_state(
    'jax_pprint_use_color',
    True,
    help='Enable jaxpr pretty-printing with colorful syntax highlighting.'
)

def _can_use_color() -> bool:
  try:
    # Check if we're in IPython or Colab
    ipython = get_ipython()  # pyrefly: ignore[unknown-name]
    shell = ipython.__class__.__name__
    if shell == "ZMQInteractiveShell":
      # Jupyter Notebook
      return True
    elif "colab" in str(ipython.__class__):
      # Google Colab (external or internal)
      return True
  except NameError:
    pass
  # Otherwise check if we're in a terminal
  return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

CAN_USE_COLOR = _can_use_color()

Color = _pretty_printer.Color
Intensity = _pretty_printer.Intensity


@use_cpp_class(_pretty_printer.Doc)
class Doc:

  @use_cpp_method()
  def __add__(self, other: Doc) -> Doc:
    raise NotImplementedError

  @use_cpp_method()
  def __repr__(self) -> str:
    raise NotImplementedError

  def __str__(self) -> str:
    return self.format()

  @use_cpp_method()
  def _format(
      self,
      width: int,
      *,
      use_color: bool,
      annotation_prefix: str,
      source_map: list[list[tuple[int, int, Any]]] | None,
  ) -> str:
    raise NotImplementedError

  def format(
      self,
      width: int = 80,
      *,
      use_color: bool | None = None,
      annotation_prefix: str = " # ",
      source_map: list[list[tuple[int, int, Any]]] | None = None,
  ) -> str:
    """Formats a pretty-printer document as a string.

    Args:

    source_map: for each line in the output, contains a list of
      (start column, end column, source) tuples. Each tuple associates a
      region of output text with a source.
    """
    if use_color is None:
      use_color = CAN_USE_COLOR and _PPRINT_USE_COLOR.value
    return self._format(
        width,
        use_color=use_color,
        annotation_prefix=annotation_prefix,
        source_map=source_map,
    )


def nil() -> Doc:
  """An empty document."""
  return _pretty_printer.nil()  # pyrefly: ignore[bad-return]


def text(text: str, annotation: str | None = None) -> Doc:
  """Literal text."""
  return _pretty_printer.text(text, annotation)  # pyrefly: ignore[bad-return]


def concat(children: Sequence[Doc]) -> Doc:
  """Concatenation of documents."""
  return _pretty_printer.concat(children)  # pyrefly: ignore[bad-argument-type, bad-return]


def brk(text: str = " ") -> Doc:
  """A break.

  Prints either as a newline or as `text`, depending on the enclosing group.
  """
  return _pretty_printer.brk(text)  # pyrefly: ignore[bad-return]


def group(doc: Doc) -> Doc:
  """Layout alternative groups.

  Prints the group with its breaks as their text (typically spaces) if the
  entire group would fit on the line when printed that way. Otherwise, breaks
  inside the group as printed as newlines.
  """
  return _pretty_printer.group(doc)  # pyrefly: ignore[bad-argument-type, bad-return]


def nest(n: int, doc: Doc) -> Doc:
  """Increases the indentation level by `n`."""
  return _pretty_printer.nest(n, doc)  # pyrefly: ignore[bad-argument-type, bad-return]


def color(
    child: Doc,
    foreground: Color | None = None,
    background: Color | None = None,
    intensity: Intensity | None = None,
) -> Doc:
  """ANSI colors.

  Overrides the foreground/background/intensity of the text for the child doc.
  Requires use_colors=True to be set when printing; otherwise does nothing.
  """
  return _pretty_printer.color(child, foreground, background, intensity)  # pyrefly: ignore[bad-argument-type, bad-return]


def source_map(doc: Doc, source: Any) -> Doc:
  """Source mapping.

  A source map associates a region of the pretty-printer's text output with a
  source location that produced it. For the purposes of the pretty printer a
  ``source`` may be any object: we require only that we can compare sources for
  equality. A text region to source object mapping can be populated as a side
  output of the ``format`` method.
  """
  return _pretty_printer.source_map(doc, source)  # pyrefly: ignore[bad-argument-type, bad-return]


type_annotation = partial(color, intensity=Intensity.NORMAL,
                          foreground=Color.MAGENTA)
keyword = partial(color, intensity=Intensity.BRIGHT, foreground=Color.BLUE)


def join(sep: Doc, docs: Sequence[Doc]) -> Doc:
  """Concatenates `docs`, separated by `sep`."""
  docs = list(docs)
  if len(docs) == 0:
    return nil()
  if len(docs) == 1:
    return docs[0]
  xs = [docs[0]]
  for doc in docs[1:]:
    xs.append(sep)
    xs.append(doc)
  return concat(xs)
