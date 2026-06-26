# JAX on ROCm

This directory contains the files and scripts used to **build, test, and package**
JAX with ROCm support (Docker images, CI workflows, and from-source wheel builds).

If you just want to **install and run** JAX on ROCm, see the
[AMD GPU (Linux) section of the JAX installation guide](../../docs/installation.md#amd-gpu-linux),
which covers the `jax[rocm7-local]` pip extra, ROCm version compatibility, and the
prebuilt `rocm/jax` Docker images. AMD's
[JAX on ROCm installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html)
has the authoritative, ROCm-version-specific instructions.

The rest of this document covers building JAX with ROCm support from source.

## Build ROCm JAX from Source

Follow these steps to build JAX with ROCm support from source:

### Step 1: Build the ROCm specific wheels from `rocm-jax`

Clone the `rocm-jax` repository for the desired branch:

```Bash
> git clone https://github.com/ROCm/rocm-jax.git -b <branch_name>
> cd rocm-jax
```
From the `rocm-jax` directory run:
```Bash
> python3 build/ci_build             \
    --python-version $PYTHON_VERSION \
    --rocm_version $ROCM_VERSION     \
    dist_wheels
> pip3 install jax_rocm_plugin/wheelhouse/*.whl
```
The build will produce two wheels:

* `jax-rocm-plugin` (ROCm-specific plugin)
* `jax-rocm-pjrt` (ROCm-specific runtime)

Detailed build instructions can be found
[here](https://github.com/ROCm/rocm-jax/blob/master/BUILDING.md).

### Step 2: Build `jaxlib` from the JAX Repository

Clone the ROCm-specific fork of JAX for the desired branch:

```Bash
> git clone https://github.com/ROCm/jax -b <branch_name>
> cd jax
```

Run the following command to build the `jaxlib` wheel:

```Bash
> python3 ./build/build.py build --wheels=jaxlib \
    --rocm_version=7 --rocm_path=/opt/rocm-[version]
```

This will generate the `jaxlib` wheel in the `dist/` directory. `jaxlib` is a
device agnostic library.

### Step 3: Then install custom JAX using:

```Bash
> python3 setup.py develop --user && pip3 -m pip install dist/*.whl
```

### Simplified Build Script

For a streamlined process, consider using the `jax/build/rocm/dev_build_rocm.py` script.
