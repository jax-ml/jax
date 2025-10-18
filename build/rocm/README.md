# JAX on ROCm
This directory provides setup instructions and necessary files to build, test, and run JAX with ROCm support in a Docker environment, suitable for both runtime and CI workflows. Explore the following methods to use or build JAX on ROCm!

## 1. Using Prebuilt Docker Images

The ROCm JAX team provides prebuilt Docker images, which the simplest way to use JAX on ROCm. These images are available on Docker Hub and come with JAX configured for ROCm.

To pull the latest ROCm JAX Docker image, run:

```Bash
> docker pull rocm/jax:latest
```

Once the image is downloaded, launch a container using the following command:

```Bash
> docker run -it -d --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 64G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/jax_dir --name rocm_jax rocm/jax:latest /bin/bash

> docker attach rocm_jax
```

### Notes:
1. The `--shm-size` parameter allocates shared memory for the container. Adjust it based on your system's resources if needed.
2. Replace `$(pwd)` with the absolute path to the directory you want to mount inside the container.

***For older versions please review the periodically pushed docker images at:
[ROCm JAX DockerHub](https://hub.docker.com/r/rocm/jax/tags).***

### Testing your ROCm environment with JAX:

After launching the container, test whether JAX detects ROCm devices as expected:

```Bash
> python -c "import jax; print(jax.devices())"
[RocmDevice(id=0), RocmDevice(id=1), RocmDevice(id=2), RocmDevice(id=3), RocmDevice(id=4), RocmDevice(id=5), RocmDevice(id=6), RocmDevice(id=7)]
```

If the setup is successful, the output should list all available ROCm devices.

## 2. Using a ROCm Docker Image and Installing JAX

If you prefer to use the ROCm Ubuntu image or already have a ROCm Ubuntu container, follow these steps to install JAX in the container.

### Step 1: Pull the ROCm Ubuntu Docker Image

For example, use the following command to pull the ROCm Ubuntu image:

```Bash
> docker pull rocm/dev-ubuntu-24.04:7.0.2-complete
```

### Step 2: Launch the Docker Container

After pulling the image, launch a container using this command:

```Bash
> docker run -it -d --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 64G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/jax_dir --name rocm_jax rocm/dev-ubuntu-24.04:7.0.2-complete /bin/bash
> docker attach rocm_jax
```

### Step 3: Install the Latest Version of JAX

Install the required version of JAX and the ROCm plugins using pip. Follow the
instructions for the [latest
release](https://github.com/ROCm/rocm-jax/releases). For example, on a system
with python 3.12, you will need to run the following to install `jax 0.6.2`:

```Bash
> pip3 install \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_pjrt-0.6.0-py3-none-manylinux_2_28_x86_64.whl \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_plugin-0.6.0-cp312-cp312-manylinux_2_28_x86_64.whl \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl \
    jax==0.6.2
```

### Step 4: Verify the Installed JAX Version

Check whether the correct version of JAX and its ROCm plugins are installed:

```Bash
> pip3 freeze | grep jax
jax==0.6.2
jax-rocm7-pjrt @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_pjrt-0.6.0-py3-none-manylinux_2_28_x86_64.whl#sha256=b20b6820d4701a8edd83509dcbc8dc4fb712f40eab873668ae0dd17f5194c2d6
jax-rocm7-plugin @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_plugin-0.6.0-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=cfecc2865ed450f996608b13af04189a2f9c1328ed896d71be0872d0e7d78389
jaxlib @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl#sha256=739fc2ebe28399f551a5c6daf529baae1637546a9a2a93789e3afd7ef0444e66
```

### Step 5: Set the `LLVM_PATH` Environment Variable

Explicitly set the `LLVM_PATH` environment variable (This helps XLA find `ld.lld` in the PATH during runtime):

```Bash
> export LLVM_PATH=/opt/rocm/llvm
```

### Step 6: Verify the Installation of ROCm JAX

Run the following command to verify that ROCm JAX is installed correctly:

```Bash
> python3 -c "import jax; print(jax.devices())"
[RocmDevice(id=0), RocmDevice(id=1), RocmDevice(id=2), RocmDevice(id=3), RocmDevice(id=4), RocmDevice(id=5), RocmDevice(id=6), RocmDevice(id=7)]

> python3 -c "import jax.numpy as jnp; x = jnp.arange(5); print(x)"
[0 1 2 3 4]
```

## 3. Install JAX On Bare-metal or A Custom Container

Follow these steps if you prefer to install ROCm manually on your host system or in a custom container.

### Installing ROCm Libraries Manually

### Step 1: Install ROCm

Please follow [ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) to install ROCm on your system.

Once installed, verify ROCm installation using:

```Bash
> rocm-smi
============================================ ROCm System Management Interface ============================================
====================================================== Concise Info ======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
==========================================================================================================================
0       2     0x74a1,   28851  43.0°C      142.0W    NPS1, SPX, 0        133Mhz  900Mhz  0%   auto  750.0W  0%     0%
1       3     0x74a1,   23018  37.0°C      137.0W    NPS1, SPX, 0        134Mhz  900Mhz  0%   auto  750.0W  0%     0%
2       4     0x74a1,   29122  44.0°C      140.0W    NPS1, SPX, 0        134Mhz  900Mhz  0%   auto  750.0W  0%     0%
3       5     0x74a1,   22683  38.0°C      138.0W    NPS1, SPX, 0        133Mhz  900Mhz  0%   auto  750.0W  0%     0%
4       6     0x74a1,   53458  42.0°C      143.0W    NPS1, SPX, 0        133Mhz  900Mhz  0%   auto  750.0W  0%     0%
5       7     0x74a1,   63883  39.0°C      138.0W    NPS1, SPX, 0        134Mhz  900Mhz  0%   auto  750.0W  0%     0%
6       8     0x74a1,   53667  42.0°C      140.0W    NPS1, SPX, 0        134Mhz  900Mhz  0%   auto  750.0W  0%     0%
7       9     0x74a1,   63738  38.0°C      135.0W    NPS1, SPX, 0        133Mhz  900Mhz  0%   auto  750.0W  0%     0%
==========================================================================================================================
================================================== End of ROCm SMI Log ===================================================
```

### Step 2: Install the Latest Version of JAX

Install the required version of JAX and the ROCm plugins using pip. Follow the
instructions for the [latest
release](https://github.com/ROCm/rocm-jax/releases). For example, on a system
with python 3.12, you will need to run the following to install `jax 0.6.2`:

```Bash
> pip3 install \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_pjrt-0.6.0-py3-none-manylinux_2_28_x86_64.whl \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_plugin-0.6.0-cp312-cp312-manylinux_2_28_x86_64.whl \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl \
    jax==0.6.2
```

### Step 3: Verify the Installed JAX Version

Check whether the correct version of JAX and its ROCm plugins are installed:

```Bash
> pip3 freeze | grep jax
jax==0.6.2
jax-rocm7-pjrt @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_pjrt-0.6.0-py3-none-manylinux_2_28_x86_64.whl#sha256=b20b6820d4701a8edd83509dcbc8dc4fb712f40eab873668ae0dd17f5194c2d6
jax-rocm7-plugin @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_plugin-0.6.0-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=cfecc2865ed450f996608b13af04189a2f9c1328ed896d71be0872d0e7d78389
jaxlib @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl#sha256=739fc2ebe28399f551a5c6daf529baae1637546a9a2a93789e3afd7ef0444e66
```

### Step 4: Set the `LLVM_PATH` Environment Variable

Explicitly set the `LLVM_PATH` environment variable (This helps XLA find `ld.lld` in the PATH during runtime):

```Bash
> export LLVM_PATH=/opt/rocm/llvm
```

### Step 5: Verify the Installation of ROCm JAX

Run the following command to verify that ROCm JAX is installed correctly:

```Bash
> python3 -c "import jax; print(jax.devices())"
[RocmDevice(id=0), RocmDevice(id=1), RocmDevice(id=2), RocmDevice(id=3), RocmDevice(id=4), RocmDevice(id=5), RocmDevice(id=6), RocmDevice(id=7)]

> python3 -c "import jax.numpy as jnp; x = jnp.arange(5); print(x)"
[0 1 2 3 4]
```

## 4. Build ROCm JAX from Source

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
