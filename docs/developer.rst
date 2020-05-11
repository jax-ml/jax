Building from source
====================

First, obtain the JAX source code::

    git clone https://github.com/google/jax
    cd jax

Building JAX involves two steps:

1. Building or installing ``jaxlib``, the C++ support library for ``jax``.
2. Installing the ``jax`` Python package.

Building or installing ``jaxlib``
---------------------------------

Installing ``jaxlib`` with pip
..............................

If you're only modifying Python portions of JAX, we recommend installing
``jaxlib`` from a prebuilt wheel using pip::

 pip install jaxlib

See the `JAX readme <https://github.com/google/jax#installation>`_ for full
guidance on pip installation (e.g., for GPU support).

Building ``jaxlib`` from source
...............................

To build ``jaxlib`` from source, you must also install some prerequisites:

* a C++ compiler (g++ or clang)
* Numpy
* Scipy
* Cython
* six (required for during the jaxlib build only, not required at install time)

On Ubuntu 18.04 or Debian you can install the necessary prerequisites with::

 sudo apt-get install g++ python python3-dev python3-numpy python3-scipy cython3 python3-six


If you are building on a Mac, make sure XCode and the XCode command line tools
are installed.

You can also install the necessary Python dependencies using ``pip``::

    pip install numpy scipy cython six


To build ``jaxlib`` with CUDA support, you can run::

    python build/build.py --enable_cuda
    pip install -e build  # installs jaxlib (includes XLA)


See ``python build/build.py --help`` for configuration options, including ways to
specify the paths to CUDA and CUDNN, which you must have installed. Here
``python`` should be the name of your Python 3 interpreter; on some systems, you
may need to use ``python3`` instead.

To build ``jaxlib`` without CUDA GPU support (CPU only), drop the ``--enable_cuda``::

  python build/build.py
  pip install -e build  # installs jaxlib (includes XLA)

Installing ``jax``
------------------

Once ``jaxlib`` has been installed, you can install ``jax`` by running::

  pip install -e .  # installs jax

To upgrade to the latest version from GitHub, just run ``git pull`` from the JAX
repository root, and rebuild by running ``build.py`` or upgrading ``jaxlib`` if
necessary. You shouldn't have to reinstall ``jax`` because ``pip install -e``
sets up symbolic links from site-packages into the repository.

Running the tests
=================

To run all the JAX tests, we recommend using ``pytest-xdist``, which can run tests in
parallel. First, install ``pytest-xdist`` and ``pytest-benchmark`` by running
``pip install pytest-xdist pytest-benchmark``.
Then, from the repository root directory run::

 pytest -n auto tests


JAX generates test cases combinatorially, and you can control the number of
cases that are generated and checked for each test (default is 10). The automated tests
currently use 25::

 JAX_NUM_GENERATED_CASES=25 pytest -n auto tests

The automated tests also run the tests with default 64-bit floats and ints::

 JAX_ENABLE_X64=1 JAX_NUM_GENERATED_CASES=25 pytest -n auto tests

You can run a more specific set of tests using
`pytest <https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests>`_'s
built-in selection mechanisms, or alternatively you can run a specific test
file directly to see more detailed information about the cases being run::

 python tests/lax_numpy_test.py --num_generated_cases=5

You can skip a few tests known as slow, by passing environment variable
JAX_SKIP_SLOW_TESTS=1.

The Colab notebooks are tested for errors as part of the documentation build.

Type checking
=============

We use ``mypy`` to check the type hints. To check types locally the same way
as Travis checks them::

  pip install mypy
  mypy --config=mypy.ini --show-error-codes jax


Update documentation
====================

To rebuild the documentation, install several packages::

  pip install -r docs/requirements.txt

You must also install ``pandoc`` in order to regenerate the notebooks.
See `Install Pandoc <https://pandoc.org/installing.html>`_,
or using `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ which
I have used successfully on the Mac: ``conda install -c conda-forge pandoc``.
If you do not want to install ``pandoc`` then you should regenerate the documentation
without the notebooks.

You run at top-level one of the following commands::

  sphinx-build -b html docs docs/build/html  # with the notebooks
  sphinx-build -b html -D nbsphinx_execute=never docs docs/build/html  # without the notebooks

You can then see the generated documentation in
``docs/build/html/index.html``.

Update notebooks
----------------

Open the notebook with http://colab.research.google.com (then `Upload` from your
local repo), update it as needed, ``Run all cells`` then
``Download ipynb``. You may want to test that it executes properly, using ``sphinx-build`` as
explained above.

Some of the notebooks are built automatically as part of the Travis pre-submit checks and
as part of the `Read the docs <https://jax.readthedocs.io/en/latest>`_ build.
The build will fail if cells raise errors. If the errors are intentional, you can either catch them,
or tag the cell with `raises-exceptions` metadata (`example PR <https://github.com/google/jax/pull/2402/files>`_).
You have to add this metadata by hand in the `.ipynb` file. It will be preserved when somebody else
re-saves the notebook.

We exclude some notebooks from the build, e.g., because they contain long computations.
See `exclude_patterns` in `conf.py <https://github.com/google/jax/blob/master/docs/conf.py>`_.

Documentation building on readthedocs.io
----------------------------------------

JAX's auto-generated documentations is at `jax.readthedocs.io <https://jax.readthedocs.io/>`_.

The documentation building is controlled for the entire project by the
`readthedocs JAX settings <https://readthedocs.org/dashboard/jax>`_. The current settings
trigger a documentation build as soon as code is pushed to the GitHub ``master`` branch.
For each code version, the building process is driven by the
``.readthedocs.yml`` and the ``docs/conf.py`` configuration files.

For each automated documentation build you can see the
`documentation build logs <https://readthedocs.org/projects/jax/builds/>`_.

If you want to test the documentation generation on Readthedocs, you can push code to the ``test-docs``
branch. That branch is also built automatically, and you can
see the generated documentation `here <https://jax.readthedocs.io/en/test-docs/>`_.

For a local test, I was able to do it in a fresh directory by replaying the commands
I saw in the Readthedocs logs::

    mkvirtualenv jax-docs  # A new virtualenv
    mkdir jax-docs  # A new directory
    cd jax-docs
    git clone --no-single-branch --depth 50 https://github.com/google/jax
    cd jax
    git checkout --force origin/test-docs
    git clean -d -f -f
    
    python -m pip install --upgrade --no-cache-dir pip
    python -m pip install --upgrade --no-cache-dir -I Pygments==2.3.1 setuptools==41.0.1 docutils==0.14 mock==1.0.1 pillow==5.4.1 alabaster>=0.7,<0.8,!=0.7.5 commonmark==0.8.1 recommonmark==0.5.0 'sphinx<2' 'sphinx-rtd-theme<0.5' 'readthedocs-sphinx-ext<1.1'
    python -m pip install --exists-action=w --no-cache-dir -r docs/requirements.txt
    
    python `which sphinx-build` -T -E -b html -d _build/doctrees-readthedocs -D language=en . _build/html

