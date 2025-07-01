import gc
from typing import Any, Callable
from dataclasses import dataclass
import time
import logging

import cutlass
import cutlass.cute as cute
from cutlass.cute import AddressSpace

from cuda import cuda

import jax
import jax.numpy as jnp
from jax.experimental.buffer_callback import ExecutionContext

from .types import (
    jax_to_cutlass_dtype,
    from_dlpack,
    JaxArray,
    DEFAULT_CUTLASS_DEVICE_BUFFER_ALIGNMENT,
)

from cutlass.cutlass_dsl.cutlass import CuTeDSL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Arg:
    idx: int  # position in pytree
    shape: tuple[int, ...]
    dtype: jnp.dtype
    layout: tuple[int, ...]
    mode: tuple[int, ...]


@dataclass(frozen=True)
class FunctionSpec:
    """Contains a specification of the inputs and outputs to the kernel."""

    in_args: tuple[Arg, ...]
    input_tree: Any
    out_args: tuple[Arg, ...]
    output_tree: Any
    input_output_aliases: tuple[tuple[int, int], ...]
    input_layout: tuple[tuple[int, ...]]
    input_mode: tuple[tuple[int, ...]]
    output_mode: tuple[tuple[int, ...]]
    output_layout: tuple[tuple[int, ...]]
    convert_tensors: bool
    kwargs: tuple[tuple[str, Any]]

    def get_compile_args(self):
        """Returns the arguments to provide to cute.compile."""
        compiler_ins = [
            JaxArray.create(0, leaf.shape, leaf.dtype, leaf.layout) for leaf in self.in_args
        ]
        compiler_outs = [
            JaxArray.create(0, leaf.shape, leaf.dtype, leaf.layout) for leaf in self.out_args
        ]
        x = tuple(sum([compiler_ins, compiler_outs], []))
        return x

    def get_runtime_args(self, out, *args):
        """Returns the arguments to provide to the compiled function at runtime."""
        # We have to convert to dlpack because __cuda_device_array__ does not support
        # all of the fp8/fp6/fp4 types properly. We could also expose the device pointer
        # directly instead which is the only value needed.
        ins = [from_dlpack(args[i]) for i, spec in enumerate(self.in_args)]
        outs = [from_dlpack(out[i]) for i, spec in enumerate(self.out_args)]

        return tuple(sum([ins, outs], []))


@cute.jit
def jit_wrapper(
    stream: cuda.CUstream,
    args: tuple[JaxArray, ...],
    *,
    wrapped_fn: cutlass.Constexpr,
    spec: cutlass.Constexpr,
):
    # split buffer argument into inputs and outputs and return to tree
    ins, outs = args[: len(spec.in_args)], args[(len(spec.in_args)) :]
    if spec.convert_tensors:
        ins = jax.tree.map(lambda x, a: x.get_tensor(a.mode), ins, spec.in_args)
        outs = jax.tree.map(lambda x, a: x.get_tensor(a.mode), outs, spec.out_args)
    ins = jax.tree.unflatten(spec.input_tree, ins)
    outs = jax.tree.unflatten(spec.output_tree, outs)
    wrapped_fn(stream, *ins, *outs, **dict(spec.kwargs))


@dataclass
class CompileResult:
    """Holds reference to the compiled kernel and arguments.

    compiled_fn: The compiled function (a JitExecutor).
                 This reference keeps CUDA modules alive.

    """

    compiled_fn: cutlass.base_dsl.jit_executor.JitExecutor
    spec: FunctionSpec

    def __call__(self, ctx: ExecutionContext, out, *args):
        self.compiled_fn(ctx.stream, self.spec.get_runtime_args(out, *args))


def _check_is_valid_type(x, is_input):
    if not is_input:
        if not isinstance(x, jax.ShapeDtypeStruct):
            raise TypeError("Invalid output value passed.", x)
    else:
        if not isinstance(x, jax.Array):
            raise TypeError("Invalid type passed.", x)


def _build_arg_tree(args, specs, is_input):
    args = []
    for idx, (arg, layout) in enumerate(zip(args_flat, specs)):
        _check_is_valid_type(arg, is_input)
        args.append(Arg(idx, arg.shape, arg.dtype, layout))
    args = jax.tree.unflatten(args_tree, args)

    return args, args_tree, is_single_leaf_node


_CUTLASS_COMPILE_CACHE = {}


def build_function_spec(
    ins,
    in_tree,
    outs,
    out_tree,
    input_layout,
    output_layout,
    input_mode,
    output_mode,
    input_output_aliases,
    convert_tensors,
    kwargs,
):
    # TODO: Improve type checking and validate pytree structures.
    # TODO: Improve Pytree support for more complex or user defined structures.

    in_args = []
    for idx, (arg, layout, mode) in enumerate(zip(ins, input_layout, input_mode)):
        _check_is_valid_type(arg, is_input=True)
        in_args.append(Arg(idx, arg.shape, arg.dtype, layout, mode))

    out_args = []
    for idx, (arg, layout, mode) in enumerate(zip(outs, output_layout, output_mode)):
        _check_is_valid_type(arg, is_input=False)
        out_args.append(Arg(idx, arg.shape, arg.dtype, layout, mode))

    # Return the argument specs to the original pytree structure
    # We need this structure to sanely match index positions of the
    # arguments to the kernel.
    ins_args_structured = jax.tree.unflatten(in_tree, in_args)
    out_args_structured = jax.tree.unflatten(out_tree, out_args)

    # Assign per-leaf aliases
    input_output_aliases_per_leaf = {}
    for input_arg_alias_idx in input_output_aliases:
        flat_in, _ = jax.tree.flatten(ins_args_structured[input_arg_alias_idx])
        flat_out, _ = jax.tree.flatten(
            out_args_structured[input_output_aliases[input_arg_alias_idx]]
        )
        for i, o in zip(flat_in, flat_out):
            input_output_aliases_per_leaf[i.idx] = o.idx

    # Remove aliased arguments from output set since they are also provided
    # as inputs. This is done at the very top level of the tree to simplify
    # how we handle aliasing. The assumption is that the entire pytree is
    # aliased.
    out_args_structured = list(out_args_structured)
    for out_idx in sorted(tuple(set(input_output_aliases.values())), reverse=True):
        try:
            out_args_structured.pop(out_idx)
        except:
            raise ValueError(f"Invalid output alias in input_output_aliases.")
    out_args_structured = tuple(out_args_structured)

    in_args_flat, _ = jax.tree.flatten(ins_args_structured)
    out_args_flat, out_tree = jax.tree.flatten(out_args_structured)

    spec = FunctionSpec(
        tuple(in_args_flat),
        in_tree,
        tuple(out_args_flat),
        out_tree,
        tuple(input_output_aliases_per_leaf.items()),
        tuple(input_layout),
        tuple(input_mode),
        tuple(output_layout),
        tuple(output_mode),
        convert_tensors,
        tuple((k, kwargs[k]) for k in kwargs),
    )

    return spec


def get_or_compile_kernel(fn, spec):
    """Gets or compiles fn and returns a CutlassCompileResult.

    The function and its specification is used as a key to determine if a new
    function must be compiled.
    """
    global _CUTLASS_COMPILE_CACHE

    cache_key = (fn, spec)
    if cache_key in _CUTLASS_COMPILE_CACHE:
        return _CUTLASS_COMPILE_CACHE[cache_key]

    start = time.time()
    compiled_fn = cutlass.cute.compile(
        jit_wrapper,
        cutlass.cuda.default_stream(),
        spec.get_compile_args(),
        wrapped_fn=fn,
        spec=spec,
    )
    end = time.time()
    logger.debug(f"Took {end-start} to compile cute kernel.")

    result = CompileResult(compiled_fn=compiled_fn, spec=spec)
    _CUTLASS_COMPILE_CACHE[cache_key] = result
    return result


def release_compile_cache():
    """Released entries from the compile cache.

    Note: This does not force recompilation of currently live kernel objects but rather
    allows any unused/unreferenced kernels to be cleaned up.
    """
    global _CUTLASS_COMPILE_CACHE
    _CUTLASS_COMPILE_CACHE.clear()
    dsl = CuTeDSL._get_dsl()
    dsl.jit_cache.clear()
    gc.collect()
