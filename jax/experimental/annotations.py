from typing import (
    TypeVar,
    Union,
)

from typing_extensions import Annotated


T = TypeVar("T")

# Sentinels
# TODO: Change when we get https://peps.python.org/pep-0661/
class _StaticAnnotationType:
    NAME = "static"


class _DonateAnnotationType:
    NAME = "donate"


class _DiffAnnotationType:
    NAME = "diff"


class _NoDiffAnnotationType:
    NAME = "nodiff"


StaticAnnotation = _StaticAnnotationType()
DonateAnnotation = _DonateAnnotationType()
DiffAnnotation = _DiffAnnotationType()
NoDiffAnnotation = _NoDiffAnnotationType()

# Pass-through annotations. See: https://peps.python.org/pep-0593/
Static = Annotated[T, StaticAnnotation]
Donate = Annotated[T, DonateAnnotation]
Diff = Annotated[T, DiffAnnotation]
NoDiff = Annotated[T, NoDiffAnnotation]

Annotation = Union[
    _StaticAnnotationType,
    _DonateAnnotationType,
    _DiffAnnotationType,
    _NoDiffAnnotationType,
]
