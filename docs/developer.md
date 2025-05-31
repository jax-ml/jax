(building-from-source)=
# Building from source

<!--* freshness: { reviewed: '2025-03-27' } *-->

First, obtain the JAX source code:

```
git clone https://github.com/jax-ml/jax
cd jax
```

Building JAX involves two steps:

1. Building or installing `jaxlib`, the C++ support library for `jax`.
2. Installing the `jax` Python package.

## Building or installing `jaxlib`

### Installing `jaxlib` with pip

If you're only modifying Python portions of JAX, we recommend installing
`jaxlib` from a prebuilt wheel using pip:

```
pip install jaxlib
```

See the [JAX readme](https://github.com/jax-ml/jax#installation) for full
guidance on pip installation (e.g., for GPU and TPU support).

### Building `jaxlib` from source

```{warning}
While it should typically be possible to compile `jaxlib` from source using
most modern compilers, the builds are only tested using clang. Pull requests
are welcomed to improve support for different toolchains, but other compilers
are not actively supported.
```

To build `jaxlib` from source, you must also install some prerequisites:

- A C++ compiler:

  As mentioned in the box above, it is best to use a recent version of clang
  (at the time of writing, the version we test is 18), but other compilers (e.g.
  g++ or MSVC) may work.

  On Ubuntu or Debian you can follow the instructions from the
  [LLVM](https://apt.llvm.org/) documentation to install the latest stable
  version of clang.

  If you are building on a Mac, make sure XCode and the XCode command line tools
  are installed.

  See below for Windows build instructions.

- Python: for running the build helper script. Note that there is no need to
  install Python dependencies locally, as your system Python will be ignored
  during the build; please check
  [Managing hermetic Python](#managing-hermetic-python) for details.

To build `jaxlib` for CPU or TPU, you can run:

```
python build/build.py build --wheels=jaxlib --verbose
pip install dist/*.whl  # installs jaxlib (includes XLA)
```

To build a wheel for a version of Python different from your current system
installation pass `--python_version` flag to the build command:

```
python build/build.py build --wheels=jaxlib --python_version=3.12 --verbose
```

The rest of this document assumes that you are building for Python version
matching your current system installation. If you need to build for a different
version, simply append `--python_version=<py version>` flag every time you call
`python build/build.py`. Note, the Bazel build will always use a hermetic Python
installation regardless of whether the `--python_version` parameter is passed or
not.

If you would like to build `jaxlib` and the CUDA plugins: Run
```
python build/build.py build --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt
```
to generate three wheels (jaxlib without cuda, jax-cuda-plugin, and
jax-cuda-pjrt). By default all CUDA compilation steps performed by NVCC and
clang, but it can be restricted to clang via the `--build_cuda_with_clang` flag.

See `python build/build.py --help` for configuration options. Here
`python` should be the name of your Python 3 interpreter; on some systems, you
may need to use `python3` instead. Despite calling the script with `python`,
Bazel will always use its own hermetic Python interpreter and dependencies, only
the `build/build.py` script itself will be processed by your system Python
interpreter. By default, the wheel is written to the `dist/` subdirectory of the
current directory.

*  JAX versions starting from v.0.4.32: you can provide custom CUDA and CUDNN
   versions in the configuration options. Bazel will download them and use as
   target dependencies.

   To download the specific versions of CUDA/CUDNN redistributions, you can use
   the `--cuda_version` and `--cudnn_version` flags:

   ```bash
   python build/build.py build --wheels=jax-cuda-plugin --cuda_version=12.3.2 \
   --cudnn_version=9.1.1
   ```
   or
   ```bash
   python build/build.py build --wheels=jax-cuda-pjrt --cuda_version=12.3.2 \
   --cudnn_version=9.1.1
   ```

   Please note that these parameters are optional: by default Bazel will
   download CUDA and CUDNN redistribution versions provided in `.bazelrc` in the
   environment variables `HERMETIC_CUDA_VERSION` and `HERMETIC_CUDNN_VERSION`
   respectively.

   To point to CUDA/CUDNN/NCCL redistributions on local file system, you can use
   the following command:

   ```bash
   python build/build.py build --wheels=jax-cuda-plugin \
   --bazel_options=--repo_env=LOCAL_CUDA_PATH="/foo/bar/nvidia/cuda" \
   --bazel_options=--repo_env=LOCAL_CUDNN_PATH="/foo/bar/nvidia/cudnn" \
   --bazel_options=--repo_env=LOCAL_NCCL_PATH="/foo/bar/nvidia/nccl"
   ```

   Please see the full list of instructions in [XLA documentation](https://github.com/openxla/xla/blob/main/docs/hermetic_cuda.md).

*  JAX versions prior v.0.4.32: you must have CUDA and CUDNN installed and
   provide paths to them using configuration options.

### Building jaxlib from source with a modified XLA repository.

JAX depends on XLA, whose source code is in the
[XLA GitHub repository](https://github.com/openxla/xla).
By default JAX uses a pinned copy of the XLA repository, but we often
want to use a locally-modified copy of XLA when working on JAX. There are two
ways to do this:

- use Bazel's `override_repository` feature, which you can pass as a command
  line flag to `build.py` as follows:

  ```
  python build/build.py build --wheels=jaxlib --local_xla_path=/path/to/xla
  ```

- modify the `WORKSPACE` file in the root of the JAX source tree to point to
  a different XLA tree.

To contribute changes back to XLA, send PRs to the XLA repository.

The version of XLA pinned by JAX is regularly updated, but is updated in
particular before each `jaxlib` release.

### Additional Notes for Building `jaxlib` from source on Windows

Note: JAX does not support CUDA on Windows; use WSL2 for CUDA support.

On Windows, follow [Install Visual Studio](https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019)
to set up a C++ toolchain. Visual Studio 2019 version 16.5 or newer is required.

JAX builds use symbolic links, which require that you activate
[Developer Mode](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development).

You can either install Python using its
[Windows installer](https://www.python.org/downloads/), or if you prefer, you
can use [Anaconda](https://docs.anaconda.com/anaconda/install/windows/)
or [Miniconda](https://docs.conda.io/en/latest/miniconda.html#windows-installers)
to set up a Python environment.

Some targets of Bazel use bash utilities to do scripting, so [MSYS2](https://www.msys2.org)
is needed. See [Installing Bazel on Windows](https://bazel.build/install/windows#install-compilers)
for more details. Install the following packages:

```
pacman -S patch coreutils
```

Once coreutils is installed, the realpath command should be present in your shell's path.

Once everything is installed. Open PowerShell, and make sure MSYS2 is in the
path of the current session. Ensure `bazel`, `patch` and `realpath` are
accessible. Activate the conda environment.

```
python .\build\build.py build --wheels=jaxlib
```

To build with debug information, add the flag `--bazel_options='--copt=/Z7'`.

### Additional notes for building a ROCM `jaxlib` for AMD GPUs

For detailed instructions on building `jaxlib` with ROCm support, refer to the official guide:
[Build ROCm JAX from Source](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md)

## Managing hermetic Python

To make sure that JAX's build is reproducible, behaves uniformly across
supported platforms (Linux, Windows, MacOS) and is properly isolated from
specifics of a local system, we rely on hermetic Python (provided by
[rules_python](https://github.com/bazelbuild/rules_python), see
[Toolchain Registration](https://rules-python.readthedocs.io/en/latest/toolchains.html#workspace-toolchain-registration)
for details) for all build and test commands executed via Bazel. This means that
your system Python installation will be ignored during the build and Python
interpreter itself as well as all the Python dependencies will be managed by
bazel directly.

### Specifying Python version

When you run `build/build.py` tool, the version of hermetic Python is set
automatically to match the version of the Python you used to
run `build/build.py` script. To choose a specific version explicitly you may
pass `--python_version` argument to the tool:

```
python build/build.py build --python_version=3.12
```

Under the hood, the hermetic Python version is controlled
by `HERMETIC_PYTHON_VERSION` environment variable, which is set automatically
when you run `build/build.py`. In case you run bazel directly you may need to
set the variable explicitly in one of the following ways:

```
# Either add an entry to your `.bazelrc` file
build --repo_env=HERMETIC_PYTHON_VERSION=3.12

# OR pass it directly to your specific build command
bazel build <target> --repo_env=HERMETIC_PYTHON_VERSION=3.12

# OR set the environment variable globally in your shell:
export HERMETIC_PYTHON_VERSION=3.12
```

You may run builds and tests against different versions of Python sequentially
on the same machine by simply switching the value of `--python_version` between
the runs. All the python-agnostic parts of the build cache from the previous
build will be preserved and reused for the subsequent builds.

### Specifying Python dependencies

During bazel build all JAX's Python dependencies are pinned to their specific
versions. This is necessary to ensure reproducibility of the build.
The pinned versions of the full transitive closure of JAX's dependencies
together with their corresponding hashes are specified in
`build/requirements_lock_<python version>.txt` files (
e.g. `build/requirements_lock_3_12.txt` for `Python 3.12`).

To update the lock files, make sure `build/requirements.in` contains the desired
direct dependencies list and then execute the following command (which will call
[pip-compile](https://pypi.org/project/pip-tools/) under the hood):

```
python build/build.py requirements_update --python_version=3.12
```

Alternatively, if you need more control, you may run the bazel command
directly (the two commands are equivalent):

```
bazel run //build:requirements.update --repo_env=HERMETIC_PYTHON_VERSION=3.12
```

where `3.12` is the `Python` version you wish to update.

Note, since it is still `pip` and `pip-compile` tools used under the hood, so
most of the command line arguments and features supported by those tools will be
acknowledged by the Bazel requirements updater command as well. For example, if
you wish the updater to consider pre-release versions simply pass `--pre`
argument to the bazel command:

```
bazel run //build:requirements.update --repo_env=HERMETIC_PYTHON_VERSION=3.12 -- --pre
```

### Specifying dependencies on local wheels

By default the build scans `dist` directory in the repository root for any local
`.whl` files to be included in the list of dependencies. If the wheel is Python
version specific, only the wheels that match the selected Python version will
be included.

The overall local wheel search and selection logic is controlled by the
arguments to `python_init_repositories()` macro (called directly from the
`WORKSPACE` file). You may use `local_wheel_dist_folder` to change the location
of the folder with local wheels. Use `local_wheel_inclusion_list` and
`local_wheel_exclusion_list` arguments to specify which wheels should be
included and/or excluded from the search (it supports basic wildcard matching).

If necessary, you can also depend on a local `.whl` file manually, bypassing the
automatic local wheel search mechanism. For example to depend on your newly
built jaxlib wheel, you may add a path to the wheel in `build/requirements.in`
and re-run the requirements updater command for a selected version of Python.
For example:

```
echo -e "\n$(realpath jaxlib-0.4.27.dev20240416-cp312-cp312-manylinux2014_x86_64.whl)" >> build/requirements.in
python build/build.py requirements_update --python_version=3.12
```

### Specifying dependencies on nightly wheels

To build and test against the very latest, potentially unstable, set of Python
dependencies we provide a special version of the dependency updater command as
follows:

```
python build/build.py requirements_update --python_version=3.12 --nightly_update
```

Or, if you run `bazel` directly (the two commands are equivalent):

```
bazel run //build:requirements_nightly.update --repo_env=HERMETIC_PYTHON_VERSION=3.12
```

The difference between this and the regular updater is that by default it would
accept pre-release, dev and nightly packages, it will also
search https://pypi.anaconda.org/scientific-python-nightly-wheels/simple as an
extra index url and will not put hashes in the resultant requirements lock file.

### Customizing hermetic Python (Advanced Usage)

We support all of the current versions of Python out of the box, so unless your
workflow has very special requirements (such as ability to use your own custom
Python interpreter) you may safely skip this section entirely.

In short, if you rely on a non-standard Python workflow you still can achieve
the great level of flexibility in hermetic Python setup. Conceptually there will
be only one difference compared to non-hermetic case: you will need to think in
terms of files, not installations (i.e. think what files your build actually
depends on, not what files need to be installed on your system), the rest is
pretty much the same.

So, in practice, to gain full control over your Python environment, hermetic or
not you need to be able to do the following three things:

1) Specify which python interpreter to use (i.e. pick actual `python` or
   `python3` binary and libs that come with it in the same folder).
2) Specify a list of Python dependencies (e.g. `numpy`) and their actual
   versions.
3) Be able to add/remove/update dependencies in the list easily. Each
   dependency itself could be custom too (self-built for example).

You already know how to do all of the steps above in a non-hermetic Python
environment, here is how you do the same in the hermetic one (by approaching it
in terms of files, not installations):

1) Instead of installing Python, get Python interpreter in a `tar` or `zip`
   file. Depending on your case you may simply pull one of many existing ones
   (such as [python-build-standalone](https://github.com/indygreg/python-build-standalone/releases)),
   or build your own and pack it in an archive (following official
   [build instructions](https://devguide.python.org/getting-started/setup-building/#compile-and-build)
   will do just fine). E.g. on Linux it will look something like the following:
   ```
   ./configure --prefix python
   make -j12
   make altinstall
   tar -czpf my_python.tgz python
   ```
   Once you have the tarball ready, plug it in the build by pointing
   `HERMETIC_PYTHON_URL` env var to the archive (either local one or from the
   internet):
   ```
   --repo_env=HERMETIC_PYTHON_URL="file:///local/path/to/my_python.tgz"
   --repo_env=HERMETIC_PYTHON_SHA256=<file's_sha256_sum>

   # OR
   --repo_env=HERMETIC_PYTHON_URL="https://remote/url/to/my_python.tgz"
   --repo_env=HERMETIC_PYTHON_SHA256=<file's_sha256_sum>

   # We assume that top-level folder in the tarball is called "python", if it is
   # something different just pass additional HERMETIC_PYTHON_PREFIX parameter
   --repo_env=HERMETIC_PYTHON_URL="https://remote/url/to/my_python.tgz"
   --repo_env=HERMETIC_PYTHON_SHA256=<file's_sha256_sum>
   --repo_env=HERMETIC_PYTHON_PREFIX="my_python/install"
   ```

2) Instead of doing `pip install` create `requirements_lock.txt` file with
   full transitive closure of your dependencies. You may also depend on the
   existing ones already checked in this repo (as long as they work with your
   custom Python version). There are no special instructions on how you do it,
   you may follow steps recommended in [Specifying Python dependencies](#specifying-python-dependencies)
   from this doc, just call pip-compile directly (note, the lock file must be
   hermetic, but you can always generate it from non-hermetic python if you'd
   like) or even create it manually (note, hashes are optional in lock files).


3) If you need to update or customize your dependencies list, you may once again
   follow the [Specifying Python dependencies](#specifying-python-dependencies)
   instructions to update `requirements_lock.txt`, call pip-compile directly or
   modify it manually. If you have a custom package you want to use just point
   to its `.whl` file directly (remember, work in terms of files, not
   installations) from your lock (note, `requirements.txt` and
   `requirements_lock.txt` files support local wheel references). If your
   `requirements_lock.txt` is already specified as a dependency to
   `python_init_repositories()` in `WORKSPACE` file you don't have to do
   anything else. Otherwise you can point to your custom file as follows:
   ```
   --repo_env=HERMETIC_REQUIREMENTS_LOCK="/absolute/path/to/custom_requirements_lock.txt"
   ```
   Also note if you use `HERMETIC_REQUIREMENTS_LOCK` then it fully controls list
   of your dependencies and the automatic local wheels resolution logic
   described in [Specifying dependencies on local wheels](#specifying-dependencies-on-local-wheels)
   gets disabled to not interfere with it.

That is it. To summarize: if you have an archive with Python interpreter in it
and a requirements_lock.txt file with full transitive closure of your
dependencies then you fully control your Python environment.

#### Custom hermetic Python examples

Note, for all of the examples below you may also set the environment variables
globally (i.e. `export` in your shell instead of `--repo_env` argument to your
command) so calling bazel via `build/build.py` will work just fine.

Build with custom `Python 3.13` from the internet, using default
`requirements_lock_3_13.txt` already checked in this repo (i.e. custom
interpreter but default dependencies):
```
bazel build <target>
  --repo_env=HERMETIC_PYTHON_VERSION=3.13
  --repo_env=HERMETIC_PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-3.13.0+20241016-x86_64-unknown-linux-gnu-install_only.tar.gz"
  --repo_env=HERMETIC_PYTHON_SHA256="2c8cb15c6a2caadaa98af51df6fe78a8155b8471cb3dd7b9836038e0d3657fb4"
```

Build with custom Python 3.13 from local file system and custom lock file
(assuming the lock file was put in `jax/build` folder of this repo before
running the command):
```
bazel test <target>
  --repo_env=HERMETIC_PYTHON_VERSION=3.13
  --repo_env=HERMETIC_PYTHON_URL="file:///path/to/cpython.tar.gz"
  --repo_env=HERMETIC_PYTHON_PREFIX="prefix/to/strip/in/cython/tar/gz/archive"
  --repo_env=HERMETIC_PYTHON_SHA256=<sha256_sum>
  --repo_env=HERMETIC_REQUIREMENTS_LOCK="/absolute/path/to/build:custom_requirements_lock.txt"
```

If default python interpreter is good enough for you and you just need a custom 
set of dependencies:
```
bazel test <target>
  --repo_env=HERMETIC_PYTHON_VERSION=3.13
  --repo_env=HERMETIC_REQUIREMENTS_LOCK="/absolute/path/to/build:custom_requirements_lock.txt"
```

Note, you can have multiple different `requirement_lock.txt` files corresponding
to the same Python version to support different scenarios. You can control
which one is selected by specifying `HERMETIC_PYTHON_VERSION`. For example in
`WORKSPACE` file:
```
requirements = {
  "3.10": "//build:requirements_lock_3_10.txt",
  "3.11": "//build:requirements_lock_3_11.txt",
  "3.12": "//build:requirements_lock_3_12.txt",
  "3.13": "//build:requirements_lock_3_13.txt",
  "3.13-scenario1": "//build:scenario1_requirements_lock_3_13.txt",
  "3.13-scenario2": "//build:scenario2_requirements_lock_3_13.txt",
},
```
Then you can build and test different combinations of stuff without changing
anything in your environment:
```
# To build with scenario1 dependencies:
bazel test <target> --repo_env=HERMETIC_PYTHON_VERSION=3.13-scenario1

# To build with scenario2 dependencies:
bazel test <target> --repo_env=HERMETIC_PYTHON_VERSION=3.13-scenario2

# To build with default dependencies:
bazel test <target> --repo_env=HERMETIC_PYTHON_VERSION=3.13

# To build with scenario1 dependencies and custom Python 3.13 interpreter:
bazel test <target>
  --repo_env=HERMETIC_PYTHON_VERSION=3.13-scenario1
  --repo_env=HERMETIC_PYTHON_URL="file:///path/to/cpython.tar.gz"
  --repo_env=HERMETIC_PYTHON_SHA256=<sha256_sum>
```

## Installing `jax`

Once `jaxlib` has been installed, you can install `jax` by running:

```
pip install -e .  # installs jax
```

To upgrade to the latest version from GitHub, just run `git pull` from the JAX
repository root, and rebuild by running `build.py` or upgrading `jaxlib` if
necessary. You shouldn't have to reinstall `jax` because `pip install -e`
sets up symbolic links from site-packages into the repository.

(running-tests)=

## Running the tests

There are two supported mechanisms for running the JAX tests, either using Bazel
or using pytest.

### Using Bazel

First, configure the JAX build by using the `--configure_only` flag. Pass
`--wheel_list=jaxlib` for CPU tests and CUDA/ROCM for GPU for GPU tests:

```
python build/build.py build --wheels=jaxlib --configure_only
python build/build.py build --wheels=jax-cuda-plugin --configure_only
python build/build.py build --wheels=jax-rocm-plugin --configure_only
```

You may pass additional options to `build.py` to configure the build; see the
`jaxlib` build documentation for details.

By default the Bazel build runs the JAX tests using `jaxlib` built from source.
To run JAX tests, run:

```
bazel test //tests:cpu_tests //tests:backend_independent_tests
```

`//tests:gpu_tests` and `//tests:tpu_tests` are also available, if you have the
necessary hardware.

You need to configure `cuda` to run `gpu` tests:
```
python build/build.py build --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt --configure_only
```

To use a preinstalled `jaxlib` instead of building it you first need to
make it available in the hermetic Python. To install a specific version of
`jaxlib` within hermetic Python run (using `jaxlib >= 0.4.26` as an example):

```
echo -e "\njaxlib >= 0.4.26" >> build/requirements.in
python build/build.py requirements_update
```

Alternatively, to install `jaxlib` from a local wheel (assuming Python 3.12):

```
echo -e "\n$(realpath jaxlib-0.4.26-cp312-cp312-manylinux2014_x86_64.whl)" >> build/requirements.in
python build/build.py requirements_update --python_version=3.12
```

Once you have `jaxlib` installed hermetically, run:

```
bazel test --//jax:build_jaxlib=false //tests:cpu_tests //tests:backend_independent_tests
```

A number of test behaviors can be controlled using environment variables (see
below). Environment variables may be passed to JAX tests using the
`--test_env=FLAG=value` flag to Bazel.

Some of JAX tests are for multiple accelerators (i.e. GPUs, TPUs). When JAX is
already installed, you can run GPUs tests like this:

```
bazel test //tests:gpu_tests --local_test_jobs=4 --test_tag_filters=multiaccelerator --//jax:build_jaxlib=false --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

You can speed up single accelerator tests by running them in parallel on
multiple accelerators. This also triggers multiple concurrent tests per
accelerator. For GPUs, you can do it like this:

```
NB_GPUS=2
JOBS_PER_ACC=4
J=$((NB_GPUS * JOBS_PER_ACC))
MULTI_GPU="--run_under $PWD/build/parallel_accelerator_execute.sh --test_env=JAX_ACCELERATOR_COUNT=${NB_GPUS} --test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_ACC} --local_test_jobs=$J"
bazel test //tests:gpu_tests //tests:backend_independent_tests --test_env=XLA_PYTHON_CLIENT_PREALLOCATE=false --test_tag_filters=-multiaccelerator $MULTI_GPU
```

### Using `pytest`
First, install the dependencies by
running `pip install -r build/test-requirements.txt`.

To run all the JAX tests using `pytest`, we recommend using `pytest-xdist`,
which can run tests in parallel. It is installed as a part of
`pip install -r build/test-requirements.txt` command.

From the repository root directory run:

```
pytest -n auto tests
```

### Controlling test behavior

JAX generates test cases combinatorially, and you can control the number of
cases that are generated and checked for each test (default is 10) using the
`JAX_NUM_GENERATED_CASES` environment variable. The automated tests
currently use 25 by default.

For example, one might write

```
# Bazel
bazel test //tests/... --test_env=JAX_NUM_GENERATED_CASES=25`
```

or

```
# pytest
JAX_NUM_GENERATED_CASES=25 pytest -n auto tests
```

The automated tests also run the tests with default 64-bit floats and ints
(`JAX_ENABLE_X64`):

```
JAX_ENABLE_X64=1 JAX_NUM_GENERATED_CASES=25 pytest -n auto tests
```

You can run a more specific set of tests using
[pytest](https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests)'s
built-in selection mechanisms, or alternatively you can run a specific test
file directly to see more detailed information about the cases being run:

```
JAX_NUM_GENERATED_CASES=5 python tests/lax_numpy_test.py
```

You can skip a few tests known to be slow, by passing environment variable
JAX_SKIP_SLOW_TESTS=1.

To specify a particular set of tests to run from a test file, you can pass a string
or regular expression via the `--test_targets` flag. For example, you can run all
the tests of `jax.numpy.pad` using:

```
python tests/lax_numpy_test.py --test_targets="testPad"
```

The Colab notebooks are tested for errors as part of the documentation build.

### Hypothesis tests

Some of the tests use [hypothesis](https://hypothesis.readthedocs.io/en/latest).
Normally, hypothesis will test using multiple example inputs, and on a test failure
it will try to find a smaller example that still results in failure:
Look through the test failure for a line like the one below, and add the decorator
mentioned in the message:
```
You can reproduce this example by temporarily adding @reproduce_failure('6.97.4', b'AXicY2DAAAAAEwAB') as a decorator on your test case
```

For interactive development, you can set the environment variable
`JAX_HYPOTHESIS_PROFILE=interactive` (or the equivalent flag `--jax_hypothesis_profile=interactive`)
in order to set the number of examples to 1, and skip the example
minimization phase.

### Doctests

JAX uses pytest in doctest mode to test the code examples within the documentation.
You can find the up-to-date command to run doctests in
[`ci-build.yaml`](https://github.com/jax-ml/jax/blob/main/.github/workflows/ci-build.yaml).
E.g., you can run:

```
JAX_TRACEBACK_FILTERING=off XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest -n auto --tb=short --doctest-glob='*.md' --doctest-glob='*.rst' docs --doctest-continue-on-failure --ignore=docs/multi_process.md
```

Additionally, JAX runs pytest in `doctest-modules` mode to ensure code examples in
function docstrings will run correctly. You can run this locally using, for example:

```
JAX_TRACEBACK_FILTERING=off XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest --doctest-modules jax/_src/numpy/lax_numpy.py
```


## Type checking

We use `mypy` to check the type hints. To run `mypy` with the same configuration as the
github CI checks, you can use the [pre-commit](https://pre-commit.com/) framework:

```
pip install pre-commit
pre-commit run mypy --all-files
```

Because `mypy` can be somewhat slow when checking all files, it may be convenient to
only check files you have modified. To do this, first stage the changes (i.e. `git add`
the changed files) and then run this before committing the changes:

```
pre-commit run mypy
```

## Linting

JAX uses the [ruff](https://docs.astral.sh/ruff/) linter to ensure code
quality. To run `ruff` with the same configuration as the
github CI checks, you can use the [pre-commit](https://pre-commit.com/) framework:

```
pip install pre-commit
pre-commit run ruff --all-files
```

## Update documentation

To rebuild the documentation, install several packages:

```
pip install -r docs/requirements.txt
```

And then run:

```
sphinx-build -b html docs docs/build/html -j auto
```

This can take a long time because it executes many of the notebooks in the documentation source;
if you'd prefer to build the docs without executing the notebooks, you can run:

```
sphinx-build -b html -D nb_execution_mode=off docs docs/build/html -j auto
```

You can then see the generated documentation in `docs/build/html/index.html`.

The `-j auto` option controls the parallelism of the build. You can use a number
in place of `auto` to control how many CPU cores to use.

(update-notebooks)=

### Update notebooks

We use [jupytext](https://jupytext.readthedocs.io/) to maintain two synced copies of the notebooks
in `docs/notebooks`: one in `ipynb` format, and one in `md` format. The advantage of the former
is that it can be opened and executed directly in Colab; the advantage of the latter is that
it makes it much easier to track diffs within version control.

#### Editing `ipynb`

For making large changes that substantially modify code and outputs, it is easiest to
edit the notebooks in Jupyter or in Colab. To edit notebooks in the Colab interface,
open <http://colab.research.google.com> and `Upload` from your local repo.
Update it as needed, `Run all cells` then `Download ipynb`.
You may want to test that it executes properly, using `sphinx-build` as explained above.

#### Editing `md`

For making smaller changes to the text content of the notebooks, it is easiest to edit the
`.md` versions using a text editor.

#### Syncing notebooks

After editing either the ipynb or md versions of the notebooks, you can sync the two versions
using [jupytext](https://jupytext.readthedocs.io/) by running `jupytext --sync` on the updated
notebooks; for example:

```
pip install jupytext==1.16.4
jupytext --sync docs/notebooks/thinking_in_jax.ipynb
```

The jupytext version should match that specified in
[.pre-commit-config.yaml](https://github.com/jax-ml/jax/blob/main/.pre-commit-config.yaml).

To check that the markdown and ipynb files are properly synced, you may use the
[pre-commit](https://pre-commit.com/) framework to perform the same check used
by the github CI:

```
pip install pre-commit
pre-commit run jupytext --all-files
```

#### Creating new notebooks

If you are adding a new notebook to the documentation and would like to use the `jupytext --sync`
command discussed here, you can set up your notebook for jupytext by using the following command:

```
jupytext --set-formats ipynb,md:myst path/to/the/notebook.ipynb
```

This works by adding a `"jupytext"` metadata field to the notebook file which specifies the
desired formats, and which the `jupytext --sync` command recognizes when invoked.

#### Notebooks within the Sphinx build

Some of the notebooks are built automatically as part of the pre-submit checks and
as part of the [Read the docs](https://docs.jax.dev/en/latest) build.
The build will fail if cells raise errors. If the errors are intentional, you can either catch them,
or tag the cell with `raises-exceptions` metadata ([example PR](https://github.com/jax-ml/jax/pull/2402/files)).
You have to add this metadata by hand in the `.ipynb` file. It will be preserved when somebody else
re-saves the notebook.

We exclude some notebooks from the build, e.g., because they contain long computations.
See `exclude_patterns` in [conf.py](https://github.com/jax-ml/jax/blob/main/docs/conf.py).

### Documentation building on `readthedocs.io`

JAX's auto-generated documentation is at <https://docs.jax.dev/>.

The documentation building is controlled for the entire project by the
[readthedocs JAX settings](https://readthedocs.org/dashboard/jax). The current settings
trigger a documentation build as soon as code is pushed to the GitHub `main` branch.
For each code version, the building process is driven by the
`.readthedocs.yml` and the `docs/conf.py` configuration files.

For each automated documentation build you can see the
[documentation build logs](https://readthedocs.org/projects/jax/builds/).

If you want to test the documentation generation on Readthedocs, you can push code to the `test-docs`
branch. That branch is also built automatically, and you can
see the generated documentation [here](https://docs.jax.dev/en/test-docs/). If the documentation build
fails you may want to [wipe the build environment for test-docs](https://docs.readthedocs.io/en/stable/guides/wipe-environment.html).

For a local test, I was able to do it in a fresh directory by replaying the commands
I saw in the Readthedocs logs:

```
mkvirtualenv jax-docs  # A new virtualenv
mkdir jax-docs  # A new directory
cd jax-docs
git clone --no-single-branch --depth 50 https://github.com/jax-ml/jax
cd jax
git checkout --force origin/test-docs
git clean -d -f -f
workon jax-docs

python -m pip install --upgrade --no-cache-dir pip
python -m pip install --upgrade --no-cache-dir -I Pygments==2.3.1 setuptools==41.0.1 docutils==0.14 mock==1.0.1 pillow==5.4.1 alabaster>=0.7,<0.8,!=0.7.5 commonmark==0.8.1 recommonmark==0.5.0 'sphinx<2' 'sphinx-rtd-theme<0.5' 'readthedocs-sphinx-ext<1.1'
python -m pip install --exists-action=w --no-cache-dir -r docs/requirements.txt
cd docs
python `which sphinx-build` -T -E -b html -d _build/doctrees-readthedocs -D language=en . _build/html
```
