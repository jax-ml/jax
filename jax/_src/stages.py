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

from typing import Any, Optional, Tuple, Union

from jax import core
from jax.interpreters import pxla
from jax.tree_util import PyTreeDef, tree_flatten, tree_unflatten

from jax._src import dispatch
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


class Lowered:
  """Lowering of a function specialized to argument types and values.

  A lowering is a computation ready for compilation. This class
  carries a lowering together with the remaining information needed to
  later compile and execute it. It also provides a common API for
  querying properties of lowered computations across JAX's various
  lowering paths (``jit``, ``pmap``, etc.).
  """
  __slots__ = [
      "in_tree", "in_avals", "out_tree", "donate_argnums", "_lowering",
      "_no_kwargs"
  ]

  # The PyTreeDef of the (positional arguments, keyword arguments).
  #
  # To get the individual PyTreeDef for the positional an keyword arguments,
  # use `in_tree.children() which will return you a sequence of 2 PyTreeDef.
  in_tree: PyTreeDef
  # The nested input tree of `ShapedArray` abstract values of (args, kwargs).
  in_avals: Any
  out_tree: PyTreeDef
  donate_argnums: Tuple[int]
  _lowering: Union[dispatch.XlaComputation,
                   pxla.MeshComputation,
                   pxla.PmapComputation]
  _no_kwargs: bool

  def __init__(self,
               lowering,
               in_tree: PyTreeDef,
               in_avals,
               out_tree: PyTreeDef,
               donate_argnums: Tuple[int],
               no_kwargs: bool = False):
    """Initializer.

    Args:
      in_tree: The `PyTreeDef` of (args, kwargs).
      out_tree: The `PyTreeDef` of the outputs.
      no_kwargs: If `True` the transformation, and the `Compiled` returned from
        this object will not support keyword arguments (an error will be raised
        if some are provided).
    """
    self._lowering = lowering
    self.in_tree = in_tree
    self.in_avals = in_avals
    self.out_tree = out_tree
    self.donate_argnums = donate_argnums
    self._no_kwargs = no_kwargs

  def compile(self) -> 'Compiled':
    return Compiled(
        self._lowering.compile(), self.in_tree, self.in_avals,
        self.out_tree, self.donate_argnums, self._no_kwargs)

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


class Compiled:
  """Compiled representation of a function specialized to types/values.

  A compiled computation is associated with an executable and the
  remaining information needed to execute it. It also provides a
  common API for querying properties of compiled computations across
  JAX's various compilation paths and backends.
  """
  __slots__ = [
      "in_tree", "in_avals", "out_tree", "donate_argnums", "_executable",
      "_no_kwargs"
  ]


  # The PyTreeDef of the (positional arguments, keyword arguments).
  in_tree: PyTreeDef
  # The nested input tree of `ShapedArray` abstract values of (args, kwargs).
  in_avals: Any
  out_tree: PyTreeDef
  donate_argnums: Tuple[int]
  _executable: Union[dispatch.XlaCompiledComputation,
                     pxla.MeshExecutable,
                     pxla.PmapExecutable]
  _no_kwargs: bool

  def __init__(self, executable, in_tree, in_avals, out_tree, donate_argnums,
               no_kwargs=False):
    self._executable = executable
    self.in_tree = in_tree
    self.in_avals = in_avals
    self.out_tree = out_tree
    self.donate_argnums = donate_argnums
    self._no_kwargs = no_kwargs

  def compiler_ir(self):
    """Post-compilation IR.

    Compilation typically involves code transformation and
    optimization. This method exists to reflect the compiler's
    representation of the program after such passes, whenever
    possible.
    """
    return self._executable.xla_executable.hlo_modules()

  def runtime_executable(self):
    return self._executable.xla_executable

  def _xla_executable(self):
    # TODO(frostig): finalize API. For now, return the underlying
    # executable directly via this method.
    return self._executable.xla_executable

  def __call__(self, *args, **kwargs):
    if self._no_kwargs and kwargs:
      kws = ', '.join(kwargs.keys())
      raise NotImplementedError(
          'function was compiled by a transformation that does not support '
          f"keyword arguments, but called with keyword arguments: {kws}")
    args_flat, in_tree = tree_flatten((args, kwargs))
    if in_tree != self.in_tree:
      # TODO(frostig): provide more info about the source function
      # and transformation
      raise TypeError(
          f'function compiled for {self.in_tree}, called with {in_tree}')
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
              'Cannot apply JAX transformations to a function lowered and '
              'compiled for a particular signature. Detected argument of '
              f'Tracer type {type(arg)}.')
      else:
        raise
    return tree_unflatten(self.out_tree, out_flat)
