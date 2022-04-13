# Copyright 2022 Google LLC
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

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple
from typing_extensions import Protocol

from jax import core
from jax import tree_util

from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


@dataclass
class ArgInfo:
  aval: core.ShapedArray
  donated: bool


class Stage:
  args_info: Any                # PyTree of ArgInfo

  @property
  def in_tree(self):
    """``PyTreeDef`` of the pair (positional arguments, keyword arguments)."""
    return tree_util.tree_structure(self.args_info)

  @property
  def in_avals(self):
    """Tree of input avals."""
    return tree_util.tree_map(lambda x: x.aval, self.args_info)

  @property
  def donate_argnums(self):
    """Flat tuple of donated argument indices."""
    return tuple([
        i for i, x in enumerate(tree_util.tree_leaves(self.args_info))
        if x.donated])


def make_args_info(in_tree, in_avals, donate_argnums):
  donate_argnums = frozenset(donate_argnums)
  flat_avals, _ = tree_util.tree_flatten(in_avals)  # todo: remove
  return in_tree.unflatten([
      ArgInfo(aval, i in donate_argnums)
      for i, aval in enumerate(flat_avals)])


class Executable(Protocol):
  """Protocol for compiled executable, which a ``Compiled`` encapsulates."""

  def call(*args_flat) -> Sequence[Any]:
    """Invoke this on the flat list of arguments, returning flat outputs."""
    raise NotImplementedError

  def hlo_modules(self) -> Sequence[Any]:
    """Return a sequence of HLO modules representing this computation."""
    raise NotImplementedError

  def runtime_executable(self) -> Any:
    """Return an opaque reference to the executable known to the runtime."""
    raise NotImplementedError


class Computation(Protocol):
  """Protocol for lowered computation, which a ``Lowered`` encapsulates."""

  def compile(self) -> Executable:
    """Compile and return a corresponding ``Exectuable``."""
    raise NotImplementedError

  def hlo(self) -> Any:
    """Return an HLO representation of this computation."""
    raise NotImplementedError

  def mhlo(self) -> Any:
    """Return an MHLO representation of this computation."""
    raise NotImplementedError


class Compiled(Stage):
  """Compiled representation of a function specialized to types/values.

  A compiled computation is associated with an executable and the
  remaining information needed to execute it. It also provides a
  common API for querying properties of compiled computations across
  JAX's various compilation paths and backends.
  """
  __slots__ = ["args_info", "out_tree", "_executable", "_no_kwargs"]

  args_info: Any                # PyTree of ArgInfo
  out_tree: tree_util.PyTreeDef
  _executable: Executable
  _no_kwargs: bool

  def __init__(self, executable, args_info, out_tree, no_kwargs=False):
    self._executable = executable
    self._no_kwargs = no_kwargs
    self.args_info = args_info
    self.out_tree = out_tree

  def compiler_ir(self):
    """Post-compilation IR.

    Compilation typically involves code transformation and
    optimization. This method exists to reflect the compiler's
    representation of the program after such passes, whenever
    possible.
    """
    return self._executable.hlo_modules()

  def runtime_executable(self):
    return self._executable.runtime_executable()

  def _xla_executable(self):
    # TODO(frostig): finalize API. For now, return the underlying
    # executable directly via this method.
    return self._executable.xla_executable

  def __call__(self, *args, **kwargs):
    if self._no_kwargs and kwargs:
      kws = ', '.join(kwargs.keys())
      raise NotImplementedError(
          "function was compiled by a transformation that does not support "
          f"keyword arguments, but called with keyword arguments: {kws}")
    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    if in_tree != self.in_tree:
      # TODO(frostig): provide more info about the source function
      # and transformation
      raise TypeError(
          f"function compiled for {self.in_tree}, called with {in_tree}")
    try:
      out_flat = self._executable.call(*args_flat)
    except TypeError:
      # We can't transform ahead-of-time compiled calls, since we've
      # lowered and compiled for a fixed function signature, and JAX
      # transformations change signatures. We interpret a Tracer
      # argument as an indication of a transformation attempt. We
      # could check this before the executable call, but we'd rather
      # avoid isinstance checks on the call path. Seeing a TypeError
      # might mean that arguments have JAX-invalid types, which in
      # turn might mean some are Tracers.
      for arg in args_flat:
        if isinstance(arg, core.Tracer):
          raise TypeError(
              "Cannot apply JAX transformations to a function lowered and "
              "compiled for a particular signature. Detected argument of "
              f"Tracer type {type(arg)}.")
      else:
        raise
    return tree_util.tree_unflatten(self.out_tree, out_flat)


class Lowered(Stage):
  """Lowering of a function specialized to argument types and values.

  A lowering is a computation ready for compilation. This class
  carries a lowering together with the remaining information needed to
  later compile and execute it. It also provides a common API for
  querying properties of lowered computations across JAX's various
  lowering paths (``jit``, ``pmap``, etc.).
  """
  __slots__ = ["args_info", "out_tree", "_lowering", "_no_kwargs"]

  args_info: Any                # PyTree of ArgInfo
  out_tree: tree_util.PyTreeDef
  _lowering: Computation
  _no_kwargs: bool

  def __init__(self,
               lowering: Computation,
               args_info,       # PyTreee of ArgInfo
               out_tree: tree_util.PyTreeDef,
               no_kwargs: bool = False):
    self._lowering = lowering
    self._no_kwargs = no_kwargs
    self.args_info = args_info
    self.out_tree = out_tree

  @classmethod
  def from_flat_info(cls,
                     lowering: Computation,
                     in_tree: tree_util.PyTreeDef,
                     in_avals,
                     donate_argnums: Tuple[int],
                     out_tree: tree_util.PyTreeDef,
                     no_kwargs: bool = False):
    """Initialize from flat info (``in_avals`` etc.) and an input PyTreeDef.

    Args:
      in_tree: The ``PyTreeDef`` of (args, kwargs).
      out_tree: The ``PyTreeDef`` of the outputs.
      no_kwargs: If ``True`` the transformation, and the
        ``Compiled`` returned from this object will not support keyword
        arguments (an error will be raised if some are provided).
    """
    return cls(lowering, make_args_info(in_tree, in_avals, donate_argnums),
               out_tree, no_kwargs=no_kwargs)

  def compile(self) -> Compiled:
    return Compiled(self._lowering.compile(), self.args_info,
                    self.out_tree, no_kwargs=self._no_kwargs)

  def compiler_ir(self, dialect: Optional[str] = None):
    if dialect is None or dialect == "mhlo":
      return self._lowering.mhlo()
    elif dialect == "hlo":
      return self._lowering.hlo()
    else:
      raise ValueError(f"Unknown dialect {dialect}")

  # TODO(frostig): remove this in favor of `compiler_ir`
  def _xla_computation(self):
    return self._lowering.hlo()


class Wrapped(Protocol):
  def __call__(self, *args, **kwargs):
    """Executes the wrapped function, lowering and compiling as needed."""
    raise NotImplementedError

  def lower(self, *args, **kwargs) -> Lowered:
    """Lower this function for the given arguments.

    A lowered function is staged out of Python and translated to a
    compiler's input language, possibly in a backend-dependent
    manner. It is ready for compilation but not yet compiled.

    Returns:
      A ``Lowered`` instance representing the lowering.
    """
    raise NotImplementedError
