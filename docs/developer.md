# Building from source

First, obtain the JAX source code:

```
git clone https://github.com/google/jax
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

See the [JAX readme](https://github.com/google/jax#installation) for full
guidance on pip installation (e.g., for GPU support).

### Building `jaxlib` from source

To build `jaxlib` from source, you must also install some prerequisites:

* a C++ compiler (g++, clang, or MSVC)

  On Ubuntu or Debian you can install the necessary prerequisites with:

  ```
  sudo apt install g++ python python3-dev
  ```

  If you are building on a Mac, make sure XCode and the XCode command line tools
  are installed.

  See below for Windows build instructions.
* Python packages: `numpy`, `scipy`, `six`, `wheel`.

  The `six` package is required for during the jaxlib build only, and is not
  required at install time.

You can install the necessary Python dependencies using `pip`:

```
pip install numpy scipy six wheel
```

To build `jaxlib` with CUDA support, you can run:

```
python build/build.py --enable_cuda
pip install dist/*.whl  # installs jaxlib (includes XLA)
```

See `python build/build.py --help` for configuration options, including ways to
specify the paths to CUDA and CUDNN, which you must have installed. Here
`python` should be the name of your Python 3 interpreter; on some systems, you
may need to use `python3` instead. By default, the wheel is written to the
`dist/` subdirectory of the current directory.

To build `jaxlib` without CUDA GPU support (CPU only), drop the `--enable_cuda`:

```
python build/build.py
pip install dist/*.whl  # installs jaxlib (includes XLA)
```

### Additional Notes for Building `jaxlib` from source on Windows

On Windows, follow [Install Visual Studio](https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019)
to set up a C++ toolchain. Visual Studio 2019 version 16.5 or newer is required.
If you need to build with CUDA enabled, follow the
[CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
to set up a CUDA environment.

You can either install Python using its
[Windows installer](https://www.python.org/downloads/), or if you prefer, you
can use [Anaconda](https://docs.anaconda.com/anaconda/install/windows/)
or [Miniconda](https://docs.conda.io/en/latest/miniconda.html#windows-installers)
to setup a Python environment.

Some targets of Bazel use bash utilities to do scripting, so [MSYS2](https://www.msys2.org)
is needed. See [Installing Bazel on Windows](https://docs.bazel.build/versions/master/install-windows.html#installing-compilers-and-language-runtimes)
for more details. Install the following packages:

```
pacman -S patch realpath
```

Once everything is installed. Open PowerShell, and make sure MSYS2 is in the
path of the current session. Ensure `bazel`, `patch` and `realpath` are
accessible. Activate the conda environment. The following command builds with
CUDA enabled, adjust it to whatever suitable for you:

```
python .\build\build.py `
  --enable_cuda `
  --cuda_path='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1' `
  --cudnn_path='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1' `
  --cuda_compute_capabilities='6.1' `
  --cuda_version='10.1' `
  --cudnn_version='7.6.5'
```

To build with debug information, add the flag `--bazel_options='--copt=/Z7'`.

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

# Running the tests

To run all the JAX tests, we recommend using `pytest-xdist`, which can run tests in
parallel. First, install `pytest-xdist` and `pytest-benchmark` by running
`ip install -r build/test-requirements.txt`.
Then, from the repository root directory run:

```
pytest -n auto tests
```

JAX generates test cases combinatorially, and you can control the number of
cases that are generated and checked for each test (default is 10). The automated tests
currently use 25:

```
JAX_NUM_GENERATED_CASES=25 pytest -n auto tests
```

The automated tests also run the tests with default 64-bit floats and ints:

```
JAX_ENABLE_X64=1 JAX_NUM_GENERATED_CASES=25 pytest -n auto tests
```

You can run a more specific set of tests using
[pytest](https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests)'s
built-in selection mechanisms, or alternatively you can run a specific test
file directly to see more detailed information about the cases being run:

```
python tests/lax_numpy_test.py --num_generated_cases=5
```

You can skip a few tests known as slow, by passing environment variable
JAX_SKIP_SLOW_TESTS=1.

To specify a particular set of tests to run from a test file, you can pass a string
or regular expression via the `--test_targets` flag. For example, you can run all
the tests of `jax.numpy.pad` using:

```
python tests/lax_numpy_test.py --test_targets="testPad"
```

The Colab notebooks are tested for errors as part of the documentation build.

Note that to run the full pmap tests on a (multi-core) CPU only machine, you
can run:

```
pytest tests/pmap_tests.py
```

I.e. don't use the `-n auto` option, since that effectively runs each test on a
single-core worker.

# Type checking

We use `mypy` to check the type hints. To check types locally the same way
as Travis checks them:

```
pip install mypy
mypy --config=mypy.ini --show-error-codes jax
```

# Update documentation

To rebuild the documentation, install several packages:
```
pip install -r docs/requirements.txt
```
And then run:
```
sphinx-build -b html docs docs/build/html
```
This can take a long time because it executes many of the notebooks in the documentation source;
if you'd prefer to build the docs without executing the notebooks, you can run:
```
sphinx-build -b html -D jupyter_execute_notebooks=off docs docs/build/html
```
You can then see the generated documentation in `docs/build/html/index.html`.

(update-notebooks)=

## Update notebooks

We use [jupytext](https://jupytext.readthedocs.io/) to maintain two synced copies of the notebooks
in `docs/notebooks`: one in `ipynb` format, and one in `md` format. The advantage of the former
is that it can be opened and executed directly in Colab; the advantage of the latter is that
it makes it much easier to track diffs within version control.

### Editing ipynb

For making large changes that substantially modify code and outputs, it is easiest to
edit the notebooks in Jupyter or in Colab. To edit notebooks in the Colab interface,
open <http://colab.research.google.com> and `Upload` from your local repo.
Update it as needed, `Run all cells` then `Download ipynb`.
You may want to test that it executes properly, using `sphinx-build` as explained above.

### Editing md

For making smaller changes to the text content of the notebooks, it is easiest to edit the
`.md` versions using a text editor.

### Syncing notebooks

After editing either the ipynb or md versions of the notebooks, you can sync the two versions
using [jupytext](https://jupytext.readthedocs.io/) by running:

```
$ jupytext --sync docs/notebooks/*
```

Alternatively, you can run this command via the [pre-commit](https://pre-commit.com/)
framework by executing the following in the main JAX directory:

```
$ pre-commit run --all
```

See the pre-commit framework documentation for information on how to set your local git
environment to execute this automatically.

### Creating new notebooks

If you are adding a new notebook to the documentation and would like to use the `jupytext --sync`
command discussed here, you can set up your notebook for jupytext by using the following command:

```
$ jupytext --set-formats ipynb,md:myst path/to/the/notebook.ipynb
```

This works by adding a `"jupytext"` metadata field to the notebook file which specifies the
desired formats, and which the `jupytext --sync` command recognizes when invoked.

### Notebooks within the sphinx build

Some of the notebooks are built automatically as part of the Travis pre-submit checks and
as part of the [Read the docs](https://jax.readthedocs.io/en/latest) build.
The build will fail if cells raise errors. If the errors are intentional, you can either catch them,
or tag the cell with `raises-exceptions` metadata ([example PR](https://github.com/google/jax/pull/2402/files)).
You have to add this metadata by hand in the `.ipynb` file. It will be preserved when somebody else
re-saves the notebook.

We exclude some notebooks from the build, e.g., because they contain long computations.
See `exclude_patterns` in [conf.py](https://github.com/google/jax/blob/main/docs/conf.py).

## Documentation building on readthedocs.io

JAX's auto-generated documentations is at <https://jax.readthedocs.io/>.

The documentation building is controlled for the entire project by the
[readthedocs JAX settings](https://readthedocs.org/dashboard/jax). The current settings
trigger a documentation build as soon as code is pushed to the GitHub `main` branch.
For each code version, the building process is driven by the
`.readthedocs.yml` and the `docs/conf.py` configuration files.

For each automated documentation build you can see the
[documentation build logs](https://readthedocs.org/projects/jax/builds/).

If you want to test the documentation generation on Readthedocs, you can push code to the `test-docs`
branch. That branch is also built automatically, and you can
see the generated documentation [here](https://jax.readthedocs.io/en/test-docs/). If the documentation build
fails you may want to [wipe the build environment for test-docs](https://docs.readthedocs.io/en/stable/guides/wipe-environment.html).

For a local test, I was able to do it in a fresh directory by replaying the commands
I saw in the Readthedocs logs:

```
mkvirtualenv jax-docs  # A new virtualenv
mkdir jax-docs  # A new directory
cd jax-docs
git clone --no-single-branch --depth 50 https://github.com/google/jax
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


