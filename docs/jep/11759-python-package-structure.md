# JAX Python Package Structure

Author: phawkins@

Date: June 29, 2022

Status: draft

## What are we proposing?

This document describes a plan to change the package structure of JAX's
open-source release, splitting the hardware-specific parts of JAX into their own
Python packages.

In open-source, JAX is currently distributed as two Python wheels:

*   `jax`, which is a pure Python package that contains, e.g., the JAX API, the
    JAX function transformations, and the JAX libraries.
*   `jaxlib`, which contains the binary (C/C++) parts of JAX, including Python
    bindings, the XLA compiler, the PJRT runtime, and a handful of handwritten
    kernels. `jaxlib` contains a mixture of Python and C++. `jaxlib` is built
    for almost the entire cartesian product of {Python versions: 3.7, 3.8, 3.9}
    x {CPU, 2 CUDA versions, ROCm versions} x {OSes: Linux, Windows, Mac}.
    (e.g., `jaxlib-0.3.14-cp37-cp37m-manylinux2014_x86_64.whl`).

We propose to change to the following Python package structure:

*   `jax`, a pure Python package. (As an aside, it would be
    [preferable to distribute wheels](https://www.python.org/dev/peps/pep-0427/#rationale)
    in addition to source packages, e.g. `jax-0.3.14-py3-none-any.whl`).
*   `jaxlib`, a binary package containing primarily the Python bindings.
    `jaxlib` would become a hard dependency of `jax`: that is, `jax` lists
    `jaxlib` as a Python package dependency. `jax` and `jaxlib` would continue
    to have the same version requirements as at present, namely that the version
    of `jax` must be at least as high as the version of `jaxlib`
    (https://jax.readthedocs.io/en/latest/design\_notes/jax\_versioning.html). \
    \
    `jaxlib` would be built for the product of {Python versions} x {OSes}. For
    example `jaxlib-0.1.60-cp37-cp37m-manylinux2014_x86_64.whl`. In particular,
    this means that `jaxlib` may contain `pybind11` code. See
    [PEP 425](https://peps.python.org/pep-0425/) for an explanation of Python
    wheel tags and what they mean for compatibility.

    \
    `jaxlib` also would contain the MLIR Python bindings.

*   `jaxlib-cpu`: a wheel containing the XLA:CPU backend, exposed via a PJRT C
    API implementation. This wheel would be built once for each OS (e.g.,
    `jaxlib-cpu-cp3-abi3-manylinux2014_x86_64.whl`).

    We split `jaxlib-cpu` from `jaxlib` to reduce the size of the binary
    distribution and build time: with this split, we need only build XLA/LLVM
    for CPU once, rather than once per Python version.

    We require that the versions of `jaxlib` and `jaxlib-cpu` match exactly.

*   `jaxlib-cuda-11-1`, `jaxlib-cuda-11-4`, `jaxlib-rocm-...` etc.: wheels
    containing GPU support as optional plugins for the `jaxlib` wheel. These
    wheels would be built for the product of {CUDA/ROCm versions} x {OSes}
    (e.g., `jaxlib-cuda-11-1-0.3.14-cp3-abi3-manylinux2014_x86_64.whl`).

    We require that the versions of `jaxlib` and `jaxlib-cuda-...` match
    exactly.

    These wheels would also be structured to avoid a minor Python version
    dependency, i.e., the `cuda` should work for all Python 3 versions we
    support, rather than needing to be, say, Python 3.7 specific. This means
    that we can only target the Python stable C API and must avoid the use of
    `pybind11`.

    It is to be determined whether we need only a single `jaxlib-cuda` plugin or
    if we need multiple CUDA releases, however that is largely orthogonal to
    this proposal.

*   `jaxlib-tpu`: wheels containing TPU support as an optional plugin for the
    `jaxlib` wheel. This wheel would be built only for Linux Cloud TPU VMs
    (e.g., `jaxlib-tpu-0.3.14-cp3-abi3-linux_x86_64.whl`).


## Why do this?

Pros/cons of doing this:

*   Simpler for users: users would be able to install a complete `jax` on CPU
    with `pip install jax`. Provided we are able to get a size exception for our
    GPU wheels, GPU wheels could be uploaded to Pypi and a GPU jax installation
    could be installed with `pip install jax jaxlib-cuda-11-1`, without the need
    for
    [local versioning schemes](https://github.com/google/jax#pip-installation)
    like `jaxlib+cuda110`that are not compatible with Pypi distribution.
*   Reduced build times: `jaxlib` takes a long time to build; for example it
    requires building XLA and LLVM. At present we cannot share build work
    between, say, CUDA or Python versions, so to make a `jaxlib` release we
    build ~12 different copies of XLA on Linux alone. By refactoring the
    `jaxlib` wheels, we would remove the need to build the CPU support
    repeatedly for each CUDA version, or the need to rebuild the GPU compiler
    for each Python version.
*   Splitting the wheels will reduce the total size of the binary distribution.
    The largest `jaxlib` wheels at present are the CUDA wheels (~150MB each);
    after refactoring we would no longer need to build one CUDA wheel per Python
    version. This would reduce the size of our GPU wheel distribution by a
    factor of 4, and it would make it easier to ship GPU wheels on Pypi.
*   Making the compiler/runtime layer of JAX pluggable makes it easier to add
    support for new compilers (IREE) and new hardware. In particular, we can
    support new hardware from other vendors as new plugin packages without
    causing a further combinatorial explosion in the number of builds.

## Notes

*   Hardware plugins like `jaxlib-cuda` will be distributed as Python packages
    that act as plugins for `jaxlib`. There are standard Python mechanisms for
    [creating and discovering plugins](https://packaging.python.org/guides/creating-and-discovering-plugins/)
    which we can use here.
*   Hardware plugins, while distributed as Python packages, will primarily
    implement the PJRT C API to act as plugins for `jaxlib`. \
    \
    Hardware plugins may include Python code and Python extensions; for our
    plugins, we would use the
    [stable C ABI](https://docs.python.org/3/c-api/stable.html) of Python so the
    plugins do not need to be rebuilt for each Python version. This requires
    avoiding the use of `pybind11`; portable hardware plugins can instead use
    the stable C ABI directly, however the Python binding needs of a hardware
    plugin are slim and this should not be difficult to do.
*   Hardware plugins will be self-contained, and, for example, will contain
    their own copy of XLA and LLVM in the case of CPU and GPU. This may seem
    somewhat wasteful, but in practice it avoids the thorny problem of needing
    to slice up XLA and/or LLVM across a shared library boundary. We can do this
    safely by using private symbol visibility. TensorFlow and JAX have a similar
    relationship already, since both contain a copy of XLA. It is also necessary
    in some cases, since, for example, the CUDA and ROCM variants of XLA are
    mutually exclusive build options.
*   We do not yet promise ABI stability for hardware plugins. That is, the
    version of, say, `jaxlib` and `jaxlib-cpu` must match exactly, and this will
    be enforced by runtime checks.

## Known problems

*   The tensorboard profiler does not work well across shared library
    boundaries: each shared library gets a separate instance of the profiler.
    This is already evident, for example, when profiling a mixed TF/JAX binary.
    The fix would be to get all plugins to share a single profiler instance,
    which may require a new C API.
