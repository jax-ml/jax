# Copyright 2022 The JAX Authors.
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
"""
Interfaces to JAX's compilation steps, and utilities for conforming to them.

This module defines a set of public-facing types that reflect the output of
intermediate stages in the process of compilation. Currently there are two
stages modeled: lowering (which produces compiler input), and compilation
(which produces compiler output).

It also defines some internal-facing types to guide what JAX can present in
this common form: an internal ``Lowering`` suffices to back a public-facing
``Lowered`` and an internal ``Executable`` suffices to back a public-facing
``Compiled``.

Finally, this module defines a couple more classes to commonly adapt our
various internal XLA-backed lowerings and executables into the lowering and
executable protocols described above.
"""
from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, Union, runtime_checkable

from jax._src import core
from jax._src import config
from jax._src import sharding as sharding_lib
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src.typing import ArrayLike
from jax._src.interpreters import mlir
from jax._src.layout import Format, Layout, AutoLayout
from jax._src.sharding_impls import UnspecifiedValue, AUTO
from jax._src.lib.mlir import ir
from jax._src.lib import _jax
from jax._src.lib import xla_client as xc


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

CompilerOptions = dict[str, Union[str, bool]]


# -- Internal types


class Executable:

  def xla_extension_executable(self) -> xc.LoadedExecutable:
    raise NotImplementedError(
        "compiled executable carries no loaded XLA executable. It may be "
        f"that {type(self)} defines an incomplete implementation.")

  def call(self, *args_flat) -> Sequence[Any]:
    """Execute on the flat list of arguments, returning flat outputs."""
    raise NotImplementedError("compiled executable does not support invocation")

  def create_cpp_call(self, params: CompiledCallParams) -> Any:
    """Optionally constructs a fast c++ dispatcher."""
    return None

  def input_shardings(self) -> Sequence[sharding_lib.Sharding]:
    """Flat sequence of input shardings.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    raise NotImplementedError(
        "compiled executable carries no input sharding information")

  def output_shardings(self) -> Sequence[sharding_lib.Sharding]:
    """Flat sequence of output shardings.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    raise NotImplementedError(
        "compiled executable carries no output sharding information")

  def input_formats(self):
    raise NotImplementedError(
        "compiled executable carries no input layout information")

  def output_formats(self):
    raise NotImplementedError(
        "compiled executable carries no output layout information")

  def as_text(self) -> str:
    """A human-readable text representation of this executable.

    Intended for visualization and debugging purposes. This need not be a valid
    nor reliable serialization. It is relayed directly to external callers.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    xla_ext_exe = self.xla_extension_executable()
    err_msg = ("text view unsupported on current XLA backend: "
               f"{type(xla_ext_exe)}")
    if not hasattr(xla_ext_exe, "hlo_modules"):
      raise NotImplementedError(err_msg)
    try:
      return "\n\n".join([m.to_string() for m in xla_ext_exe.hlo_modules()])
    except _jax.XlaRuntimeError as e:
      msg, *_ = e.args
      if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
        raise NotImplementedError(err_msg) from e
      else:
        raise

  def cost_analysis(self) -> Any:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it need not be consistent across versions of JAX
    and jaxlib, or even across invocations. It is relayed directly to external
    callers.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    xla_ext_exe = self.xla_extension_executable()

    if hasattr(xla_ext_exe, "cost_analysis"):
      try:
        return xla_ext_exe.cost_analysis()
      except _jax.XlaRuntimeError as e:
        msg, *_ = e.args
        if not (type(msg) is str and msg.startswith("UNIMPLEMENTED")):
          raise

    if (
        xla_ext_exe is None
        and hasattr(self, "unsafe_call")
        and hasattr(self.unsafe_call, "compiled")
        and hasattr(self.unsafe_call.compiled, "cost_analysis")
    ):
      return self.unsafe_call.compiled.cost_analysis()

    raise NotImplementedError(
        f"cost analysis unsupported on current XLA backend: {type(xla_ext_exe)}"
    )

  def memory_analysis(self) -> Any:
    """A summary of estimated memory requirements.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it need not be consistent across versions of JAX
    and jaxlib, or even across invocations. It is relayed directly to external
    callers.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    xla_ext_exe = self.xla_extension_executable()
    err_msg = ("memory analysis unsupported on current XLA backend: "
               f"{type(xla_ext_exe)}")
    if not hasattr(xla_ext_exe, "get_compiled_memory_stats"):
      raise NotImplementedError(err_msg)
    try:
      return xla_ext_exe.get_compiled_memory_stats()
    except _jax.XlaRuntimeError as e:
      msg, *_ = e.args
      if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
        raise NotImplementedError(err_msg) from e
      else:
        raise

  def runtime_executable(self) -> Any:
    """An arbitrary object representation of this executable.

    Intended for debugging purposes. This need not be a valid nor reliable
    serialization. It is relayed directly to external callers, with no
    guarantee on type, structure, or consistency across invocations.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend or
    compiler.
    """
    return self.xla_extension_executable()


class Lowering:

  compile_args: dict[str, Any]
  # the constants that have been hoisted out and must be passed as first args,
  # after the tokens.
  # See https://docs.jax.dev/en/latest/internals/constants.html.
  const_args: list[ArrayLike]

  def hlo(self) -> xc.XlaComputation:
    """Return an HLO representation of this computation."""
    hlo = self.stablehlo()
    m: str | bytes
    m = mlir.module_to_bytecode(hlo)
    return _jax.mlir.mlir_module_to_xla_computation(
        m, use_tuple_args=self.compile_args["tuple_args"])

  def stablehlo(self) -> ir.Module:
    """Return a StableHLO representation of this computation."""
    raise NotImplementedError(
        f"cost analysis unsupported on XLA computation: {type(self)}")

  def compile(
      self, compiler_options: CompilerOptions | None = None, *,
      device_assignment: tuple[xc.Device, ...] | None = None) -> Executable:
    """Compile and return a corresponding ``Executable``."""
    raise NotImplementedError(
        f"cost analysis unsupported on XLA computation: {type(self)}")

  def as_text(self, dialect: str | None = None,
              *,
              debug_info: bool = False) -> str:
    """A human-readable text representation of this lowering.

    Intended for visualization and debugging purposes. This need not be a valid
    nor reliable serialization. It is relayed directly to external callers.
    """
    if dialect is None:
      dialect = "stablehlo"
    if dialect == "stablehlo":
      return mlir.module_to_string(self.stablehlo(),
                                   enable_debug_info=debug_info)
    elif dialect == "hlo":
      print_opts = xc._xla.HloPrintOptions.short_parsable()
      print_opts.print_metadata = debug_info
      return self.hlo().as_hlo_module().to_string(print_opts)
    else:
      raise ValueError(f"unknown dialect: {dialect}")

  def compiler_ir(self, dialect: str | None = None) -> Any:
    """An arbitrary object representation of this lowering.

    Intended for debugging purposes. This need not be a valid nor reliable
    serialization. It is relayed directly to external callers, with no
    guarantee on type, structure, or consistency across invocations.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend or
    compiler.

    Args:
      dialect: Optional string specifying a representation dialect
      (e.g. "stablehlo")
    """
    if dialect is None:
      dialect = "stablehlo"
    if dialect == "stablehlo":
      return self.stablehlo()
    elif dialect == "hlo":
      return self.hlo()
    else:
      raise ValueError(f"unknown dialect: {dialect}")

  def cost_analysis(self) -> Any:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it need not be consistent across versions of JAX
    and jaxlib, or even across invocations. It is relayed directly to external
    callers.

    This function estimates execution cost in the absence of compiler
    optimizations, which may drastically affect the cost. For execution cost
    estimates after optimizations, compile this lowering and see
    ``Compiled.cost_analysis``.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    raise NotImplementedError(
        f"cost analysis unsupported on XLA computation: {type(self)}")


# -- Public-facing API, plus helpers

@dataclass(frozen=True)
class ArgInfo:
  _aval: core.AbstractValue
  donated: bool

  @property
  def shape(self):
    return self._aval.shape  # pytype: disable=attribute-error

  @property
  def dtype(self):
    return self._aval.dtype  # pytype: disable=attribute-error


class Stage:
  args_info: Any  # PyTree of ArgInfo

  @property
  def in_tree(self) -> tree_util.PyTreeDef:
    """Tree structure of the pair (positional arguments, keyword arguments)."""
    return tree_util.tree_structure(self.args_info)

  @property
  def in_avals(self):
    """Tree of input avals."""
    return tree_util.tree_map(lambda x: x._aval, self.args_info)

  @property
  def donate_argnums(self):
    """Flat tuple of donated argument indices."""
    return tuple(
        i for i, x in enumerate(tree_util.tree_leaves(self.args_info))
        if x.donated)


def make_args_info(in_tree, in_avals, donate_argnums):
  donate_argnums = frozenset(donate_argnums)
  flat_avals, _ = tree_util.tree_flatten(in_avals)  # todo: remove
  return in_tree.unflatten([
      ArgInfo(aval, i in donate_argnums)
      for i, aval in enumerate(flat_avals)])


class CompiledCallParams(NamedTuple):
  executable: Executable
  no_kwargs: bool
  in_tree: tree_util.PyTreeDef
  out_tree: tree_util.PyTreeDef
  # See https://docs.jax.dev/en/latest/internals/constants.html
  const_args: list[ArrayLike]


class Compiled(Stage):
  """Compiled representation of a function specialized to types/values.

  A compiled computation is associated with an executable and the
  remaining information needed to execute it. It also provides a
  common API for querying properties of compiled computations across
  JAX's various compilation paths and backends.
  """
  __slots__ = ["args_info", "out_tree", "_executable", "_no_kwargs",
               "_params"]

  args_info: Any                # PyTree of ArgInfo, not including const_args
  out_tree: tree_util.PyTreeDef
  _executable: Executable
  _no_kwargs: bool
  _params: CompiledCallParams

  def __init__(self, executable, const_args: list[ArrayLike],
               args_info, out_tree, no_kwargs=False):
    self._executable = executable
    self._no_kwargs = no_kwargs
    self.args_info = args_info
    self.out_tree = out_tree
    self._params = CompiledCallParams(self._executable, self._no_kwargs,
                                      self.in_tree, self.out_tree, const_args)
    self._call = None

  def as_text(self) -> str | None:
    """A human-readable text representation of this executable.

    Intended for visualization and debugging purposes. This is not a valid nor
    reliable serialization.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    try:
      return self._executable.as_text()
    except NotImplementedError:
      return None

  def cost_analysis(self) -> Any | None:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    # TODO(frostig): improve annotation (basic pytree of arbitrary structure)
    try:
      return self._executable.cost_analysis()
    except NotImplementedError:
      return None

  def memory_analysis(self) -> Any | None:
    """A summary of estimated memory requirements.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    # TODO(frostig): improve annotation (basic pytree of arbitrary structure)
    try:
      return self._executable.memory_analysis()
    except NotImplementedError:
      return None

  @property
  def out_info(self):  # PyTree of jax.ShapeDtypeStruct
    out_avals = self._executable.out_avals
    out_formats_flat = self._output_formats_flat
    return self.out_tree.unflatten(
        [core.ShapeDtypeStruct(o.shape, o.dtype, sharding=f)
         for o, f in zip(out_avals, out_formats_flat)])

  def runtime_executable(self) -> Any | None:
    """An arbitrary object representation of this executable.

    Intended for debugging purposes. This is not valid nor reliable
    serialization. The output has no guarantee of consistency across
    invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    return self._executable.runtime_executable()

  def _input_shardings_flat(self):
    shardings_flat = self._executable._in_shardings
    # Some input shardings got DCE'd
    if self.in_tree.num_leaves > len(shardings_flat):
      iter_shardings_flat = iter(shardings_flat)
      shardings_flat = [next(iter_shardings_flat) if i in self._executable._kept_var_idx
                        else None for i in range(self.in_tree.num_leaves)]
    return shardings_flat

  @property
  def input_shardings(self):  # -> PyTree[sharding.Sharding]
    shardings_flat = self._input_shardings_flat()
    return tree_util.tree_unflatten(self.in_tree, shardings_flat)  # pytype: disable=attribute-error

  @property
  def output_shardings(self):  # -> PyTree[sharding.Sharding]
    shardings_flat = self._executable._out_shardings
    return tree_util.tree_unflatten(self.out_tree, shardings_flat)  # pytype: disable=attribute-error

  def _input_layouts_flat(self):
    layouts_flat = self._executable._xla_in_layouts
    # Some input layouts got DCE'd
    if self.in_tree.num_leaves > len(layouts_flat):
      iter_layouts_flat = iter(layouts_flat)
      layouts_flat = [next(iter_layouts_flat) if i in self._executable._kept_var_idx
                      else None for i in range(self.in_tree.num_leaves)]
    return layouts_flat

  @property
  def input_formats(self):
    layouts_flat = self._input_layouts_flat()
    shardings_flat = self._input_shardings_flat()
    formats_flat = [Format(l, s) for l, s in zip(layouts_flat, shardings_flat)]
    return tree_util.tree_unflatten(self.in_tree, formats_flat)  # pytype: disable=attribute-error

  @property
  def _output_formats_flat(self):
    layouts_flat = self._executable._xla_out_layouts
    shardings_flat = self._executable._out_shardings
    assert all(isinstance(l, Layout) for l in layouts_flat)
    return [Format(l, s) for l, s in zip(layouts_flat, shardings_flat)]

  @property
  def output_formats(self):
    formats_flat = self._output_formats_flat
    return tree_util.tree_unflatten(self.out_tree, formats_flat)  # pytype: disable=attribute-error

  @staticmethod
  def call(*args, **kwargs):
    util.test_event("stages_compiled_call")
    # This is because `__call__` passes in `self._params` as the first argument.
    # Instead of making the call signature `call(params, *args, **kwargs)`
    # extract it from args because `params` can be passed as a kwarg by users
    # which might conflict here.
    params = args[0]
    args = args[1:]  # Not including const_args
    if config.dynamic_shapes.value:
      raise NotImplementedError
    if params.no_kwargs and kwargs:
      kws = ', '.join(kwargs.keys())
      raise NotImplementedError(
          "function was compiled by a transformation that does not support "
          f"keyword arguments, but called with keyword arguments: {kws}")
    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    if in_tree != params.in_tree:
      errs = list(tree_util.equality_errors_pytreedef(in_tree, params.in_tree))
      msg = []
      msg.append(
          "Function compiled with input pytree does not match the input pytree"
          f" it was called with. There are {len(errs)} mismatches, including:")
      for path, thing1, thing2, explanation in errs:
        fst, *rest = path
        base = ['args', 'kwargs'][fst.idx]
        msg.append(
            f"    * at {base}{tree_util.keystr(tuple(rest))}, seen {thing2} but now"
            f" given {thing1}, so {explanation}")
      raise TypeError('\n'.join(msg))
    if not core.trace_state_clean():
      # We check for tracers when we are under a transformation, and skip the
      # check in the common path. We can't transform ahead-of-time compiled
      # calls, since we've lowered and compiled for a fixed function signature,
      # and JAX transformations change signatures.
      for arg in args_flat:
        if isinstance(arg, core.Tracer):
          raise TypeError(
              "Cannot apply JAX transformations to a function lowered and "
              "compiled for a particular signature. Detected argument of "
              f"Tracer type {type(arg)}.")
    out_flat = params.executable.call(*params.const_args, *args_flat)
    outs = tree_util.tree_unflatten(params.out_tree, out_flat)
    return outs, out_flat, args_flat

  def __call__(self, *args, **kwargs):
    if self._call is None:
      self._call = self._executable.create_cpp_call(self._params)
      if self._call is None:
        params = self._params
        def cpp_call_fallback(*args, **kwargs):
          outs, _, _ = Compiled.call(params, *args, **kwargs)
          return outs
        self._call = cpp_call_fallback
    return self._call(*args, **kwargs)


class Lowered(Stage):
  """Lowering of a function specialized to argument types and values.

  A lowering is a computation ready for compilation. This class
  carries a lowering together with the remaining information needed to
  later compile and execute it. It also provides a common API for
  querying properties of lowered computations across JAX's various
  lowering paths (:func:`~jax.jit`, :func:`~jax.pmap`, etc.).
  """
  __slots__ = ["_lowering", "args_info", "out_tree", "_no_kwargs"]
  _lowering: Lowering
  args_info: Any  # PyTree of ArgInfo, not including the const_args
  out_tree: tree_util.PyTreeDef
  _no_kwargs: bool

  def __init__(
      self,
      lowering: Lowering,
      args_info,
      out_tree: tree_util.PyTreeDef,
      no_kwargs: bool = False):

    self._lowering = lowering
    self.args_info = args_info
    self.out_tree = out_tree
    self._no_kwargs = no_kwargs

  @classmethod
  def from_flat_info(cls,
                     lowering: Lowering,
                     in_tree: tree_util.PyTreeDef,
                     in_avals,
                     donate_argnums: tuple[int, ...],
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
    return cls(
        lowering,
        make_args_info(in_tree, in_avals, donate_argnums),
        out_tree,
        no_kwargs=no_kwargs)

  @property
  def out_info(self):  # PyTree of OutInfo
    out_avals = self._lowering.compile_args["global_out_avals"]
    out_shardings = self._lowering.compile_args["out_shardings"]
    out_layouts = self._lowering.compile_args["out_layouts"]
    outs = []
    for o, l, s in zip(out_avals, out_layouts, out_shardings):
      s = None if isinstance(s, (UnspecifiedValue, AUTO)) else s
      l = None if isinstance(l, AutoLayout) else l
      format = Format(l, s)
      outs.append(core.ShapeDtypeStruct(o.shape, o.dtype, sharding=format))
    return self.out_tree.unflatten(outs)

  def compile(
      self, compiler_options: CompilerOptions | None = None, *,
      device_assignment: tuple[xc.Device, ...] | None = None) -> Compiled:
    """Compile, returning a corresponding ``Compiled`` instance."""

    kw: dict[str, Any] = {
        "compiler_options": compiler_options,
        "device_assignment": device_assignment
    }
    return Compiled(
        self._lowering.compile(**kw),  # pytype: disable=wrong-keyword-args
        self._lowering.const_args,
        self.args_info,
        self.out_tree,
        no_kwargs=self._no_kwargs,
    )

  def as_text(self, dialect: str | None = None, *,
              debug_info: bool = False) -> str:
    """A human-readable text representation of this lowering.

    Intended for visualization and debugging purposes. This need not be a valid
    nor reliable serialization.
    Use `jax.export` if you want reliable and portable serialization.

    Args:
      dialect: Optional string specifying a lowering dialect (e.g. "stablehlo",
        or "hlo").
      debug_info: Whether to include debugging information,
        e.g., source location.
    """
    return self._lowering.as_text(dialect, debug_info=debug_info)

  def compiler_ir(self, dialect: str | None = None) -> Any | None:
    """An arbitrary object representation of this lowering.

    Intended for debugging purposes. This is not a valid nor reliable
    serialization. The output has no guarantee of consistency across
    invocations.
    Use `jax.export` if you want reliable and portable serialization.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.

    Args:
      dialect: Optional string specifying a lowering dialect (e.g. "stablehlo",
        or "hlo").
    """
    try:
      return self._lowering.compiler_ir(dialect)
    except NotImplementedError:
      return None

  def cost_analysis(self) -> Any | None:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    # TODO(frostig): improve annotation (basic pytree of arbitrary structure)
    try:
      return self._lowering.cost_analysis()
    except NotImplementedError:
      return None


class Traced(Stage):
  """Traced form of a function specialized to argument types and values.

  A traced computation is ready for lowering. This class carries the
  traced representation with the remaining information needed to later
  lower, compile, and execute it.
  """
  __slots__ = ["jaxpr", "args_info", "fun_name", "_out_tree", "_lower_callable",
               "_args_flat", "_arg_names", "_num_consts"]

  def __init__(self, jaxpr: core.ClosedJaxpr, args_info, fun_name, out_tree,
               lower_callable, args_flat=None, arg_names=None,
               num_consts: int = 0, params_out_shardings=None):
    self.jaxpr = jaxpr
    self.args_info = args_info  # Not including the const_args
    self.fun_name = fun_name
    self._out_tree = out_tree
    self._lower_callable = lower_callable
    if args_flat is not None:
      assert len(args_flat) == len(jaxpr.in_avals)  # Not including the const_args
    self._args_flat = args_flat
    self._arg_names = arg_names  # Not including the const_args
    self._num_consts = num_consts
    self._params_out_shardings = params_out_shardings

  @property
  def out_info(self):
    out_shardings = [None if isinstance(s, UnspecifiedValue) else s
                     for s in self._params_out_shardings]
    out = []
    for a, out_s in zip(self.jaxpr.out_avals, out_shardings):
      s = (a.sharding if a.sharding.mesh.are_all_axes_explicit else out_s
           if out_s is None else out_s)
      # TODO(yashkatariya): Add `Layout` to SDS.
      out.append(
          core.ShapeDtypeStruct(
              a.shape, a.dtype, sharding=s, weak_type=a.weak_type,
              vma=(a.vma if config._check_vma.value else None)))
    return tree_util.tree_unflatten(self._out_tree, out)

  def lower(self, *, lowering_platforms: tuple[str, ...] | None = None,
            _private_parameters: mlir.LoweringParameters | None = None):
    """Lower to compiler input, returning a ``Lowered`` instance."""
    if _private_parameters is None:
      _private_parameters = mlir.LoweringParameters()
    try:
      lowering = self._lower_callable(
          lowering_platforms=lowering_platforms,
          lowering_parameters=_private_parameters)
    except DeviceAssignmentMismatchError as e:
      fails, = e.args
      msg = _device_assignment_mismatch_error(
          self.fun_name, fails, self._args_flat, 'jit', self._arg_names)
      raise ValueError(msg) from None
    return Lowered(lowering, self.args_info, self._out_tree)


@runtime_checkable
class Wrapped(Protocol):
  """A function ready to be traced, lowered, and compiled.

  This protocol reflects the output of functions such as
  ``jax.jit``. Calling it results in JIT (just-in-time) lowering,
  compilation, and execution. It can also be explicitly lowered prior
  to compilation, and the result compiled prior to execution.
  """

  def __call__(self, *args, **kwargs):
    """Executes the wrapped function, lowering and compiling as needed."""
    raise NotImplementedError

  def trace(self, *args, **kwargs) -> Traced:
    """Trace this function explicitly for the given arguments.

    A traced function is staged out of Python and translated to a jaxpr. It is
    ready for lowering but not yet lowered.

    Returns:
      A ``Traced`` instance representing the tracing.
    """
    raise NotImplementedError

  def lower(self, *args, **kwargs) -> Lowered:
    """Lower this function explicitly for the given arguments.

    This is a shortcut for ``self.trace(*args, **kwargs).lower()``.

    A lowered function is staged out of Python and translated to a
    compiler's input language, possibly in a backend-dependent
    manner. It is ready for compilation but not yet compiled.

    Returns:
      A ``Lowered`` instance representing the lowering.
    """
    raise NotImplementedError


class MismatchType(enum.Enum):
  ARG_SHARDING = enum.auto()
  CONST_SHARDING = enum.auto()
  OUT_SHARDING = enum.auto()
  SHARDING_INSIDE_COMPUTATION = enum.auto()
  CONTEXT_DEVICES = enum.auto()
  IN_SHARDING = enum.auto()

  def __str__(self):
    if self.name == 'IN_SHARDING':
      return 'explicit input sharding'
    if self.name == 'CONST_SHARDING':
      return 'closed over constant sharding'
    elif self.name == 'OUT_SHARDING':
      return 'explicit output sharding'
    elif self.name == 'CONTEXT_DEVICES':
      return 'context mesh'
    return f'{self.name}'


class SourceInfo(NamedTuple):
  source_info: source_info_util.SourceInfo
  eqn_name: str


@dataclasses.dataclass
class DeviceAssignmentMismatch:
  da: Sequence[xc.Device]
  m_type: MismatchType
  source_info: SourceInfo | None

  @property
  def device_ids(self) -> Sequence[int]:
    return [d.id for d in self.da]

  @property
  def platform(self) -> str:
    return self.da[0].platform.upper()

  def _maybe_api_name(self, api_name) -> str:
    return f" {api_name}'s" if self.m_type == MismatchType.CONTEXT_DEVICES else ""

  @property
  def source_info_str(self):
    return (
        "" if self.source_info is None
        else f" at {source_info_util.summarize(self.source_info.source_info)}"
    )

  @property
  def _dev_ids_plat_str(self):
    return f"device ids {self.device_ids} on platform {self.platform}"

  def m_type_str(self, api_name):
    return (f'{self.source_info and self.source_info.eqn_name} inside {api_name}'
            if self.m_type == MismatchType.SHARDING_INSIDE_COMPUTATION else self.m_type)

  def _str(self, api_name):
    return (f"{self._maybe_api_name(api_name)} {self.m_type_str(api_name)} with "
            f"{self._dev_ids_plat_str}{self.source_info_str}")


class DeviceAssignmentMismatchError(Exception):
  pass


def _find_arg_mismatch(arg_list, fails, fun_name):
  mismatched_args_msg = []
  def mismatch(err):
    for name, inp_da, aval in arg_list:
      if err.m_type == MismatchType.ARG_SHARDING and err.da == inp_da:
        mismatched_args_msg.append(
            f"argument {name} of {fun_name} with shape {aval.str_short()} and "
            f"{err._dev_ids_plat_str}")
        break
  first_err, second_err = fails
  mismatch(first_err)
  mismatch(second_err)
  return mismatched_args_msg


def _device_assignment_mismatch_error(fun_name, fails, args_flat, api_name,
                                      arg_names):
  arg_list = []
  if arg_names is None:
    arg_names = [''] * len(args_flat)
  for a, n in zip(args_flat, arg_names):
    da = (a.sharding._device_assignment
          if getattr(a, 'sharding', None) is not None else None)
    arg_list.append((n, da, core.shaped_abstractify(a)))

  mismatched_args_msg = _find_arg_mismatch(arg_list, fails, fun_name)

  if len(mismatched_args_msg) == 2:
    first, second = mismatched_args_msg  # pytype: disable=bad-unpacking
    extra_msg = f" Got {first} and {second}"
  elif len(mismatched_args_msg) == 1:
    first, second  = fails
    # Choose the failure left which is not already covered by ARG_SHARDING.
    left = second if first.m_type == MismatchType.ARG_SHARDING else first
    extra_msg = f" Got {mismatched_args_msg[0]} and{left._str(api_name)}"
  else:
    first, second = fails
    extra_msg = f" Got{first._str(api_name)} and{second._str(api_name)}"
  msg = (f"Received incompatible devices for {api_name}ted computation.{extra_msg}")
  return msg
