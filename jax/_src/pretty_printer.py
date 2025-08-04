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


_PPRINT_USE_COLOR = config.bool_state(
    'jax_pprint_use_color',
    True,
    help='Enable jaxpr pretty-printing with colorful syntax highlighting.'
)

def _can_use_color() -> bool:
  try:
    # Check if we're in IPython or Colab
    ipython = get_ipython()  # type: ignore[name-defined]
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
Doc = _pretty_printer.Doc

def _format(
  self, width: int = 80, *, use_color: bool | None = None,
  annotation_prefix: str = " # ",
  source_map: list[list[tuple[int, int, Any]]] | None = None
) -> str:
  """
  Formats a pretty-printer document as a string.

  Args:
  source_map: for each line in the output, contains a list of
    (start column, end column, source) tuples. Each tuple associates a
    region of output text with a source.
  """
  if use_color is None:
    use_color = CAN_USE_COLOR and _PPRINT_USE_COLOR.value
  return self._format(
      width, use_color=use_color, annotation_prefix=annotation_prefix,
      source_map=source_map)
Doc.format = _format
Doc.__str__ = lambda self: self.format()  # type: ignore[method-assign]

nil = _pretty_printer.nil
text = _pretty_printer.text
concat = _pretty_printer.concat
brk = _pretty_printer.brk
group = _pretty_printer.group
nest = _pretty_printer.nest
color = _pretty_printer.color
source_map = _pretty_printer.source_map


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
