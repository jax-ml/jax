from collections.abc import Sequence
import enum


class Color(enum.Enum):
    BLACK = 30

    RED = 31

    GREEN = 32

    YELLOW = 33

    BLUE = 34

    MAGENTA = 35

    CYAN = 36

    WHITE = 37

    RESET = 39

class Intensity(enum.Enum):
    DIM = 2

    NORMAL = 22

    BRIGHT = 1

class Doc:
    def __repr__(self) -> str: ...

    def __add__(self, arg: Doc, /) -> Doc: ...

class NilDoc(Doc):
    pass

class TextDoc(Doc):
    pass

class ConcatDoc(Doc):
    pass

class BreakDoc(Doc):
    pass

class GroupDoc(Doc):
    pass

class NestDoc(Doc):
    pass

class ColorDoc(Doc):
    pass

class SourceMapDoc(Doc):
    pass

def nil() -> Doc:
    """An empty document."""

def text(text: str, annotation: str | None = None) -> Doc:
    """Literal text."""

def concat(children: Sequence[Doc]) -> Doc:
    """Concatenation of documents."""

def brk(text: str = ' ') -> Doc:
    """
    A break.

    Prints either as a newline or as `text`, depending on the enclosing group.
    """

def group(arg: Doc, /) -> Doc:
    """
    Layout alternative groups.

    Prints the group with its breaks as their text (typically spaces) if the
    entire group would fit on the line when printed that way. Otherwise, breaks
    inside the group as printed as newlines.
    """

def nest(arg0: int, arg1: Doc, /) -> Doc:
    """Increases the indentation level by `n`."""

def color(child: Doc, foreground: Color | None = None, background: Color | None = None, intensity: Intensity | None = None) -> Doc:
    """
    ANSI colors.

    Overrides the foreground/background/intensity of the text for the child doc.
    Requires use_colors=True to be set when printing; otherwise does nothing.
    """

def source_map(doc: Doc, source: object) -> Doc:
    """
    Source mapping.

    A source map associates a region of the pretty-printer's text output with a
    source location that produced it. For the purposes of the pretty printer a
    ``source`` may be any object: we require only that we can compare sources for
    equality. A text region to source object mapping can be populated as a side
    output of the ``format`` method.
    """
