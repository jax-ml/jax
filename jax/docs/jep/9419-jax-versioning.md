# Jax and Jaxlib versioning

## Why are `jax` and `jaxlib` separate packages?

We publish JAX as two separate Python wheels, namely `jax`, which is a pure
Python wheel, and `jaxlib`, which is a mostly-C++ wheel that contains libraries
such as:
* XLA,
* pieces of LLVM used by XLA,
* MLIR infrastructure, such as the StableHLO Python bindings.
* JAX-specific C++ libraries for fast JIT and PyTree manipulation.

We distribute separate `jax` and `jaxlib` packages because it makes it easy to
work on the Python parts of JAX without having to build C++ code or even having
a C++ toolchain installed. `jaxlib` is a large library that is not easy for
many users to build, but most changes to JAX only touch Python code. By
allowing the Python pieces to be updated independently of the C++ pieces, we
improve the development velocity for Python changes.

In addition `jaxlib` is not cheap to build, but we want to be able to iterate on
and run the JAX tests in environments without a lot of CPU, for example in
Github Actions or on a laptop. Many of our CI builds simply use a prebuilt
`jaxlib`, rather than needing to rebuild the C++ pieces of JAX on each PR.

As we will see, distributing `jax` and `jaxlib` separately comes with a cost, in
that it requires that changes to `jaxlib` maintain a backward compatible API.
However, we believe that on balance it is preferable to make Python changes
easy, even if at the cost of making C++ changes slightly harder.


## How are `jax` and `jaxlib` versioned?

Summary: `jax` and `jaxlib` share the same version number in the JAX source tree, but are released as separate Python packages.
When installed, the `jax` package version must be greater than or equal to `jaxlib`'s version,
and `jaxlib`'s version must be greater than or equal to the minimum `jaxlib`
version specified by `jax`.

Both `jax` and `jaxlib` releases are numbered `x.y.z`, where `x` is the major
version, and `y` is the minor version, and `z` is an optional patch release.
Version numbers must follow
[PEP 440](https://www.python.org/dev/peps/pep-0440/). Version number comparisons
are lexicographic comparisons on tuples of integers.

Each `jax` release has an associated minimum `jaxlib` version `mx.my.mz`. The
minimum `jaxlib` version for `jax` version `x.y.z` must be no greater than
`x.y.z`.

For `jax` version `x.y.z` and `jaxlib` version `lx.ly.lz` to be compatible,
the following must hold:

* The jaxlib version (`lx.ly.lz`) must be greater than or equal to the minimum
  jaxlib version (`mx.my.mz`).
* The jax version (`x.y.z`) must be greater than or equal to the jaxlib version
  (`lx.ly.lz`).

These constraints imply the following rules for releases:
* `jax` may be released on its own at any time, without updating `jaxlib`.
* If a new `jaxlib` is released, a `jax` release must be made at the same time.

These
[version constraints](https://github.com/jax-ml/jax/blob/main/jax/version.py)
are currently checked by `jax` at import time, instead of being expressed as
Python package version constraints. `jax` checks the `jaxlib` version at
runtime rather than using a `pip` package version constraint because we
[provide separate `jaxlib` wheels](https://github.com/jax-ml/jax#installation)
for a variety of hardware and software versions (e.g, GPU, TPU, etc.). Since we
do not know which is the right choice for any given user, we do not want `pip`
to install a `jaxlib` package for us automatically.

In the future, we hope to separate out the hardware-specific pieces of `jaxlib`
into separate plugins, at which point the minimum version could be expressed as
a Python package dependency. For now, we do provide
platform-specific extra requirements that install a compatible jaxlib version,
e.g., `jax[cuda]`.

## How can I safely make changes to the API of `jaxlib`?

* `jax` may drop compatibility with older `jaxlib` releases at any time, so long
  as the minimum `jaxlib` version is increased to a compatible version. However,
  note that the minimum `jaxlib`, even for unreleased versions of `jax`, must be
  a released version! This allows us to use released `jaxlib` wheels in our CI
  builds, and allows Python developers to work on `jax` at HEAD without ever
  needing to build `jaxlib`.

  For example, to remove an old backwards compatibility path in the `jax` Python
  code, it is sufficient to bump the minimum jaxlib version and then delete the
  compatibility path.

* `jaxlib` may drop compatibility with older `jax` releases lower than
  its own release version number. The version constraints enforced by `jax`
  would forbid the use of an incompatible `jaxlib`.

  For example, for `jaxlib` to drop a Python binding API used by an older `jax`
  version, the `jaxlib` minor or major version number must be incremented.

* If possible, changes to the `jaxlib` should be made in a backwards-compatible
  way.

  In general `jaxlib` may freely change its API, so long
  as the rules about `jax` being compatible with all `jaxlib`s at least as new
  as the minimum version are followed. This implies that
  `jax` must always be compatible with at least two versions of `jaxlib`,
  namely, the last release, and the tip-of-tree version, effectively
  the next release. This is easier to do if compatibility is maintained,
  although incompatible changes can be made using version tests from `jax`; see
  below.

  For example, it is usually safe to add a new function to `jaxlib`, but unsafe
  to remove an existing function or to change its signature if current `jax` is
  still using it. Changes to `jax` must work or degrade gracefully
  for all `jaxlib` releases greater than the minimum up to HEAD.


Note that the compatibility rules here only apply to *released* versions of
`jax` and `jaxlib`. They do not apply to unreleased versions; that is, it is ok
to introduce and then remove an API from `jaxlib` if it is never released, or if
no released `jax` version uses that API.

## How is the source to `jaxlib` laid out?

`jaxlib` is split across two main repositories, namely the
[`jaxlib/` subdirectory in the main JAX repository](https://github.com/jax-ml/jax/tree/main/jaxlib)
and in the
[XLA source tree, which lives inside the XLA repository](https://github.com/openxla/xla).
The JAX-specific pieces inside XLA are primarily in the
[`xla/python` subdirectory](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/python).


The reason that C++ pieces of JAX, such as Python bindings and runtime
components, are inside the XLA tree is partially
historical and partially technical.

The historical reason is that originally the
`xla/python` bindings were envisaged as general purpose Python bindings that
might be shared with other frameworks. In practice this is increasingly less
true, and `xla/python` incorporates a number of JAX-specific pieces and is
likely to incorporate more. So it is probably best to simply think of
`xla/python` as part of JAX.

The technical reason is that the XLA C++ API is not stable. By keeping the
XLA:Python bindings in the XLA tree, their C++ implementation can be updated
atomically with the C++ API of XLA. It is easier to maintain backward and forward
compatibility of Python APIs than C++ ones, so `xla/python` exposes Python APIs
and is responsible for maintaining backward compatibility at the Python
level.

`jaxlib` is built using Bazel out of the `jax` repository. The pieces of
`jaxlib` from the XLA repository are incorporated into the build
[as a Bazel submodule](https://github.com/jax-ml/jax/blob/main/WORKSPACE).
To update the version of XLA used during the build, one must update the pinned
version in the Bazel `WORKSPACE`. This is done manually on an
as-needed basis, but can be overridden on a build-by-build basis.


## How do we make changes across the `jax` and `jaxlib` boundary between releases?

The jaxlib version is a coarse instrument: it only lets us reason about
*releases*.

However, since the `jax` and `jaxlib` code is split across repositories that
cannot be updated atomically in a single change, we need to manage compatibility
at a finer granularity than our release cycle. To manage fine-grained
compatibility, we have additional versioning that is independent of the `jaxlib`
release version numbers.

We maintain an additional version number (`_version`) in
[`xla_client.py` in the XLA repository](https://github.com/openxla/xla/blob/main/xla/python/xla_client.py).
The idea is that this version number, is defined in `xla/python`
together with the C++ parts of JAX, is also accessible to JAX Python as
`jax._src.lib.jaxlib_extension_version`, and must
be incremented every time that a change is made to the XLA/Python code that has
backwards compatibility implications for `jax`. The JAX Python code can then use
this version number to maintain backwards compatibility, e.g.:

```
from jax._src.lib import jaxlib_extension_version

# 123 is the new version number for _version in xla_client.py
if jaxlib_extension_version >= 123:
  # Use new code path
  ...
else:
  # Use old code path.
```

Note that this version number is in *addition* to the constraints on the
released version numbers, that is, this version number exists to help manage
compatibility during development for unreleased code. Releases must also
follow the compatibility rules given above.

