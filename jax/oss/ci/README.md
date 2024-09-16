# JAX continuous integration

> **Warning** This folder is still under construction. It is part of an ongoing
> effort to improve the structure of CI and build related files within the
> JAX repo. This warning will be removed when the contents of this
> directory are stable and appropriate documentation around its usage is in
> place.

Maintainer: ML Velocity team @ Google

********************************************************************************

The CI folder contains the configuration files and scripts used to build, test,
and upload JAX artifacts.

## JAX's Official CI and Build/Test Scripts

JAX's official CI jobs run the scripts in this folder. The CI scripts require
an env file to be set in `JAXCI_ENV_FILE` that sets various configuration settings.
These "env" files are structured by their build type. For e.g.,
`ci/envs/build_artifacts/jaxlib` contains the configs for building the `jaxlib`
package. The scripts are intended to be used across different platforms and
architectures and currently supports the following systems: Linux x86,
Linux Arm64, Mac x86, Mac Arm64, Windows x86.


If you would like to test these scripts, follow the instructions below.

### Choose how you would like to build:
<details>
<summary> Shell Script </summary>

The artifact building script (`ci/build_artifacts.sh`) invokes the build CLI,
`ci/cli/build.py` which in turn invokes the bazel command that builds the
requested JAX artifact. Follow the instructions below to invoke the CI script
to build a JAX artifact of your choice. These scripts can build the `jax`,
`jaxlib`, `jax-cuda-plugin`, and the `jax-cuda-pjrt` artifacts. Note that all
commands are meant to be run from the root of this repository.

**Docker (soft prerequisite)**

The CI scripts are recommended to be run in Docker where possible. This ensures
the right build environment is set up before we can build the artifact. If you
would like to disable Docker, run:

```
export JAXCI_SETUP_DOCKER=0
export JAXCI_CLI_BUILD_MODE=local
```

**Changing Python version**

By default, the build will use Python 3.12. If you would like to change this,
set `JAXCI_HERMETIC_PYTHON_VERSION`. E.g.`export JAXCI_HERMETIC_PYTHON_VERSION=3.11`

**RBE support**

If you are running this on a Linux x86 or a Windows machine, you have the option
to use RBE to speed up the build. Please note this requires permissions to JAX's
remote worker pool and RBE configs. To enable RBE, run `export JAXCI_BUILD_ARTIFACT_WITH_RBE=1`.

**How to run the script**

```
1. Set JAXCI_ENV_FILE to one of the envs inside ci/build_artifacts based the artifact
you want to build and your sytem.
E.g. export JAXCI_ENV_FILE=ci/envs/build_artifacts/jaxlib
2. Run: bash ci/build_artifacts.sh
```

**Known Bugs**

1. Building `jax` fails due to Python missing the `build` dependency.
2. Auditwheel script fails on Linux Arm64's Docker image due to Python missing
the `auditwheel` dependency
3. If RBE is used to build the target for Windows, building the wheel fails
due to a permission denied error.

</details>

<details>
<summary> Build CLI </summary>

Follow the instructions below to invoke the build CLI to build a JAX artifact
of your choice. The CLI can build the `jaxlib`, `jax-cuda-plugin`, and the
`jax-cuda-pjrt` artifacts. Note that all commands are meant to be run from the
root of this repository.

By default, the CLI runs in local mode and will pick the "local_" configs in
the `ci/.bazelrc` file. On Linux systems, Bazel defaults to using GCC
as the default compiler. To change this, add `--use_clang` to your command. This
requires Clang to be present on the system and in the path. If your Clang binary
is not on the path, set its path using `--clang_path`.

**Build Modes**

If you want to run with the configs that the CI builds use, switch the mode by
setting `--mode=ci`. Please note CI mode has a dependency on a custom toolchain
that JAX uses. The build expects this toolchain to be present on the system. As
such, CI mode is usually run from within a Docker container. See `JAXCI_DOCKER_IMAGE`
inside `ci/build_artfacts` to know which image we use for each platform.

**RBE support**

If you are running this on a Linux x86 or a Windows machine, you have the option
to use RBE to speed up the build. Please note this requires permissions to JAX's
remote worker pool and RBE configs. To enable RBE, set `--use_rbe` to you command.

**Changing Python version**

If you would like to change the Python version of the artifact, add
`--python_version=<python_version>` to your command. E.g. `--python_version=3.11`.
By default, the CLI uses Python 3.12.

**Local XLA dependency**

JAX artifacts built by the CLI depend on XLA version pinned in JAX's
`workspace.bzl`. If would like to depend on the XLA from your local system,
set `--local_xla_path` to its path.

**Dry Run**

If you would like to just invoke a dry run, add `--dry_run` to your command.
This will print the `bazel` command that the CLI would have ended up invoking.

**Some example invocations**

1. For building `jaxlib`, run `python ci/cli/build.py jaxlib`
2. For building `jax-cuda-plugin` for Python 3.11, run `python ci/cli/build.py jax-cuda-pjrt --python_version=3.11`
3. For building `jax-cuda-pjrt` for Python 3.10 with RBE, run `python ci/cli/build.py jax-cuda-pjrt --use_rbe --python_version=3.10`

</details>