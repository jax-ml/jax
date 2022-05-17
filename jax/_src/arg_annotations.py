from abc import ABCMeta
from typing import (
    Callable,
    Tuple,
    Optional,
    Union,
    Sequence,
    List,
)

from typing_extensions import get_args

from dataclasses import dataclass
import inspect
import warnings

from jax._src.api_util import validate_argnames, validate_argnums
from jax.experimental.annotations import Annotation


_VARIABLE_PARAMETERS = (
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
)


__MSG_VARARG_ANNO_ERR = (
    "Cannot directly annotate variable argument {arg}. "
    "You can achieve annotated variable arguments through individual "
    "argnums or argnames. These will be safely ignored if not present."
)


@dataclass
class BaseArgumentAnnotations(metaclass=ABCMeta):
    sig: Optional[inspect.Signature]
    nums: Tuple[int, ...]
    names: Tuple[str, ...]

@dataclass
class ArgumentAnnotations(BaseArgumentAnnotations):
    @classmethod
    def from_(
        cls,
        fun: Callable,
        *,
        args: Sequence[Union[str, int]],
        annotation: Annotation,
    ):
        nums_given: List[int] = []
        names_given: List[str] = []

        for arg in args:
            if isinstance(arg, int):
                nums_given.append(arg)
            elif isinstance(arg, str):
                names_given.append(arg)
            else:
                raise ValueError(
                    "Argument must be integer or string. "
                    f"Was {arg} (type={type(arg)})"
                )

        try:
            sig = inspect.signature(fun)
        except ValueError:
            # In rare cases, inspect can fail, e.g., on some builtin Python functions.
            # In these cases, don't infer any parameters.
            return cls(sig=None, nums=tuple(nums_given), names=tuple(names_given))


        nums_derived: List[int] = []
        names_derived: List[str] = []
        var_args: Optional[str] = None
        var_kwargs: Optional[str] = None
        for num, (name, param) in enumerate(sig.parameters.items()):
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                var_args = name
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                var_kwargs = name

            if param.kind not in _VARIABLE_PARAMETERS:  # Skip *args, **kwargs
                # Resolve names from nums
                if (
                    num in nums_given
                    and param.kind is not inspect.Parameter.POSITIONAL_ONLY
                ):
                    names_derived.append(name)
                    continue

                # Resolve nums from names
                if (
                    name in names_given
                    and param.kind is not inspect.Parameter.KEYWORD_ONLY
                ):
                    nums_derived.append(num)
                    continue

            # Resolve type hint annotations
            if annotation in get_args(param.annotation):
                if param.kind in _VARIABLE_PARAMETERS:
                    raise ValueError(__MSG_VARARG_ANNO_ERR.format(arg=name))

                nums_derived.append(num)
                names_derived.append(name)
                continue

        # We cannot mark *args-like or **kwargs-like as static, only individual args
        for var_arg in (var_args, var_kwargs):
            if var_arg in names_given:
                raise ValueError(__MSG_VARARG_ANNO_ERR.format(arg=var_arg))

        nums = tuple(set(nums_given + nums_derived))
        names = tuple(set(names_given + names_derived))

        validate_argnums(sig, nums, annotation.NAME + "_argnums")
        validate_argnames(sig, names, annotation.NAME + "_argnames")

        return cls(sig=sig, nums=nums, names=names)

    def __bool__(self):
        return self.nums or self.names


@dataclass
class ArgNumNamesAnnotations(BaseArgumentAnnotations):
    @classmethod
    def from_(
        cls,
        fun: Callable,
        *,
        nums: Tuple[int, ...],
        names: Tuple[str, ...],
        annotation: Annotation,
    ):
        # TODO(JeppeKlitgaard): Enable warning below. Current doctests fail
        # TODO(JeppeKlitgaard): Insert documentation URL
        # warnings.warn(
        #     (
        #         "Use of argnums and argnames is deprecated. "
        #         "Please switch to the newer args interface, which carries many additional "
        #         "benefits. It is documented at TODO"
        #     ),
        #     DeprecationWarning,
        # )

        try:
            sig = inspect.signature(fun)
        except ValueError:
            # In rare cases, inspect can fail, e.g., on some builtin Python functions.
            # In these cases, don't infer any parameters.
            return cls(sig=None, nums=nums, names=names)

        validate_argnums(sig, nums, annotation.NAME + "_argnums")
        validate_argnames(sig, names, annotation.NAME + "_argnames")

        return cls(sig=sig, nums=nums, names=names)
