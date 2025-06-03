(pallas-changelog)=

# Pallas Changelog

<!--* freshness: { reviewed: '2025-04-24' } *-->

This is the list of changes specific to {class}`jax.experimental.pallas`.
For the overall JAX change log see [here](https://docs.jax.dev/en/latest/changelog.html).

<!--
Remember to align the itemized text with the first line of an item within a list.
-->

## Unreleased

* New functionality

  * Added a new decorator {func}`jax.experimental.pallas.loop` which allows
    to write stateless loops as functions.

* Deprecations

  * {class}`jax.experimental.pallas.triton.TritonCompilerParams` has been
    renamed to {class}`jax.experimental.pallas.triton.CompilerParams`. The
    old name is deprecated and will be removed in a future release.
  * {class}`jax.experimental.pallas.tpu.TPUCompilerParams`
    and {class}`jax.experimental.pallas.tpu.TPUMemorySpace` have been
    renamed to {class}`jax.experimental.pallas.tpu.CompilerParams`
    and {class}`jax.experimental.pallas.tpu.MemorySpace`. The
    old names are deprecated and will be removed in a future release.

## Released with jax 0.6.1

* Removals

  * Removed previously deprecated {mod}`jax.experimental.pallas.gpu`. To use
    the Triton backend import {mod}`jax.experimental.pallas.triton`.

* Changes

  * {func}`jax.experimental.pallas.BlockSpec` now takes in special types in
    addition to ints/None in the `block_shape`. `indexing_mode` has been
    removed. To achieve "Unblocked", pass a `pl.Element(size)` into
    `block_shape` for each entry that needs unblocked indexing.
  * {func}`jax.experimental.pallas.pallas_call` now requires `compiler_params`
    to be a backend-specific dataclass instead of a param to value mapping.
  * {func}`jax.experimental.pallas.debug_check` is now supported both on
    TPU and Mosaic GPU. Previously, this functionality was only supported
    on TPU and required using the APIs from {mod}`jax.experimental.checkify`.
    Note that debug checks are not executed unless
    {data}`jax.experimental.pallas.enable_debug_checks` is set.

## Released with jax 0.5.0

* New functionality

  * Added vector support for {func}`jax.experimental.pallas.debug_print` on TPU.

## Released with jax 0.4.37

* New functionality

  * Added support for `DotAlgorithmPreset` precision arguments for `dot`
    lowering on Triton backend.

## Released with jax 0.4.36 (December 6, 2024)

## Released with jax 0.4.35 (October 22, 2024)

* Removals

  * Removed previously deprecated aliases
    {class}`jax.experimental.pallas.tpu.CostEstimate` and
    {func}`jax.experimental.tpu.run_scoped`. Both  are now available in
    {mod}`jax.experimental.pallas`.

* New functionality

  * Added a cost estimate tool {func}`pl.estimate_cost` for automatically
  constructing a kernel cost estimate from a JAX reference function.

## Released with jax 0.4.34 (October 4, 2024)

* Changes

  * {func}`jax.experimental.pallas.debug_print` no longer requires all arguments
    to be scalars. The restrictions on the arguments are backend-specific:
    Non-scalar arguments are currently only supported on GPU, when using Triton.
  * {class}`jax.experimental.pallas.BlockSpec` no longer supports the previously
    deprecated argument order, where `index_map` comes before `block_shape`.

* Deprecations

  * The {mod}`jax.experimental.pallas.gpu` submodule is deprecated to avoid
    ambiguite with {mod}`jax.experimental.pallas.mosaic_gpu`. To use the
    Triton backend import {mod}`jax.experimental.pallas.triton`.

* New functionality

  * {func}`jax.experimental.pallas.pallas_call` now accepts `scratch_shapes`,
    a PyTree specifying backend-specific temporary objects needed by the
    kernel, for example, buffers, synchronization primitives etc.
  * {func}`checkify.check` can now be used to insert runtime asserts when
    pallas_call is called with the `pltpu.enable_runtime_assert(True)` context
    manager.

## Released with jax 0.4.33 (September 16, 2024)

## Released with jax 0.4.32 (September 11, 2024)

* Changes
  * The kernel function is not allowed to close over constants. Instead, all the needed arrays
    must be passed as inputs, with proper block specs ({jax-issue}`#22746`).

* New functionality
  * Improved error messages for mistakes in the signature of the index map functions,
    to include the name and source location of the index map.

##  Released with jax 0.4.31 (July 29, 2024)

* Changes
  * {class}`jax.experimental.pallas.BlockSpec` now expects `block_shape` to
    be passed *before* `index_map`. The old argument order is deprecated and
    will be removed in a future release.
  * {class}`jax.experimental.pallas.GridSpec` does not have anymore the `in_specs_tree`,
    and the `out_specs_tree` fields, and the `in_specs` and `out_specs` tree now
    store the values as pytrees of BlockSpec. Previously, `in_specs` and
    `out_specs` were flattened ({jax-issue}`#22552`).
  * The method `compute_index` of {class}`jax.experimental.pallas.GridSpec` has
    been removed because it is private. Similarly, the `get_grid_mapping` and
    `unzip_dynamic_bounds` have been removed from `BlockSpec` ({jax-issue}`#22593`).
  * Fixed the interpret mode to work with BlockSpec that involve padding
    ({jax-issue}`#22275`).
    Padding in interpret mode will be with NaN, to help debug out-of-bounds
    errors, but this behavior is not present when running in custom kernel mode,
    and should not be depended on.
  * Previously it was possible to import many APIs that are meant to be
    private, as `jax.experimental.pallas.pallas`. This is not possible anymore.

* New Functionality
  * Added documentation for BlockSpec: {ref}`pallas_grids_and_blockspecs`.
  * Improved error messages for the {func}`jax.experimental.pallas.pallas_call`
    API.
  * Added lowering rules for TPU for `lax.shift_right_arithmetic` ({jax-issue}`#22279`)
    and `lax.erf_inv` ({jax-issue}`#22310`).
  * Added initial support for shape polymorphism for the Pallas TPU custom kernels\
    ({jax-issue}`#22084`).
  * Added TPU support for checkify. ({jax-issue}`#22480`)
  * Added clearer error messages when the block sizes do not match the TPU
    requirements. Previously, the errors were coming from the Mosaic backend
    and did not have useful Python stack traces.
  * Added support for TPU lowering with 1D blocks, and relaxed the requirements
    for the block sizes with at least 2 dimensions: the last 2 dimensions must
    be divisible by 8 and 128 respectively, unless they span the entire
    corresponding array dimension. Previously, block dimensions that spanned the
    entire array were allowed only if the block dimensions in the last two
    dimensions were smaller than 8 and 128 respectively.

## Released with JAX 0.4.30 (June 18, 2024)

* New Functionality
  * Added checkify support for {func}`jax.experimental.pallas.pallas_call` in
    interpret mode ({jax-issue}`#21862`).
  * Improved support for PRNG keys for TPU kernels ({jax-issue}`#21773`).
