# Converting JAX examples to TFLite/TFjs

## Overview

This directory implements a flexible evaluation framework for converting JAX
examples to TFjs/TFLIte using jax2tf, making it relatively easy to add new
examples and converters, and write results of all tests to Markdown.

The results of the conversion are written to `converters_results.md`.

See `examples_test.py` for instructions on how to run the evaluation.

### Features

*  It is easy to add new Modules of examples since each Module is specified
   using a few lines of code (see `examples.py`).

*  It is easy to add new converters since each converter is represented as a
   function (see `converters.py`).

*  The framework outputs a Markdown table (outputted below), which provides
   an overview of the missing ops for all examples and all converters.

### Limitations

*  We only evaluate whether a Module converts, we do not compare any outputs
   between the converted model and the original model.

*  If an example is missing multiple ops, then only the first missing op is
   reported.

## Code Details

### `[examples_test.py]`

This is the binary to run to execute tests. It has flags for various options.

### `[converters.py]`

This contains the functions representing different converters.

### `[all_examples.py]`

A list of all the examples to test. As one can see each example only takes a few
lines so should be quite easy to add new ones.

The file also contains several data structures:

*  `Arg`: An enum used in arguments in ModuleSpec, which depend on particular
   state (rng, module state), so these are instantiated dynamically when the
   Modules are constructed in [examples_convert.py].

*  `ModuleSpec`: An example is represented by a ModuleSpec dataclass, which
   contains information for constructing and calling a module. I have designed
   this interface by listing for all the Flax examples what the required
   arguments are for calling `init` and `apply`, which is in the end all we need
   to be able to convert a model. I expect it should be quite easy to add new
   models now.

*  `ExampleSuite`: Examples are collected in suites, which are outputted in a
   single table in the Mardown file. This is simply a groups of examples with
   some metadata.

### `[examples_converter.py]`

Takes care of all the `arg` and `kwargs` plumbing to create Modules, and tries
converting these Modules using a specified conversion function.

*  This library has two interface functions; `test_convert` and
   `write_markdown`, which are both called from `examples_test.py`.

*  The main logic of this library is in the function `make_module`, which
   converts a `ModuleSpec` into a `ModuleToConvert`, which is then the input to
   the conversion function. This function is called from `test_convert`.
