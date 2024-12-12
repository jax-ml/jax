# JAX on ROCm
This directory provides setup instructions and necessary files to build, test, and run JAX with ROCm support in a Docker environment, suitable for both runtime and CI workflows. Explore the following methods to use or build JAX on ROCm!

## 1. Using Prebuilt Docker Images

The ROCm JAX team provides prebuilt Docker images, which the simplest way to use JAX on ROCm. These images are available on Docker Hub and come with JAX configured for ROCm.

To pull the latest ROCm JAX Docker image, run:

```Bash
> docker pull rocm/jax-community:latest
```

Once the image is downloaded, launch a container using the following command:

```Bash
> docker run -it -d --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 64G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/jax_dir --name rocm_jax rocm/jax-community:latest /bin/bash

> docker attach rocm_jax
```

### Notes:
1. The `--shm-size` parameter allocates shared memory for the container. Adjust it based on your system's resources if needed.
2. Replace `$(pwd)` with the absolute path to the directory you want to mount inside the container.

***For older versions please review the periodically pushed docker images at:
[ROCm JAX Community DockerHub](https://hub.docker.com/r/rocm/jax-community/tags).***

### Testing your ROCm environment with JAX:

After launching the container, test whether JAX detects ROCm devices as expected:

```Bash
> python -c "import jax; print(jax.devices())"
[RocmDevice(id=0), RocmDevice(id=1), RocmDevice(id=2), RocmDevice(id=3)]
```

If the setup is successful, the output should list all available ROCm devices.

## 2. Using a ROCm Docker Image and Installing JAX

If you prefer to use the ROCm Ubuntu image or already have a ROCm Ubuntu container, follow these steps to install JAX in the container.

### Step 1: Pull the ROCm Ubuntu Docker Image

For example, use the following command to pull the ROCm Ubuntu image:

```Bash
> docker pull rocm/dev-ubuntu-22.04:6.3-complete
```

### Step 2: Launch the Docker Container

After pulling the image, launch a container using this command:

```Bash
> docker run -it -d --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 64G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/jax_dir --name rocm_jax rocm/dev-ubuntu-22.04:6.3-complete /bin/bash
> docker attach rocm_jax
```

### Step 3: Install the Latest Version of JAX

Inside the running container, install the required version of JAX with ROCm support using pip:

```Bash
> pip3 install jax[rocm]
```

### Step 4: Verify the Installed JAX Version

Check whether the correct version of JAX and its ROCm plugins are installed:

```Bash
> pip3 freeze | grep jax
jax==0.4.35
jax-rocm60-pjrt==0.4.35
jax-rocm60-plugin==0.4.35
jaxlib==0.4.35
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
[RocmDevice(id=0), RocmDevice(id=1), RocmDevice(id=2), RocmDevice(id=3)]

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

========================================== ROCm System Management Interface ==========================================
==================================================== Concise Info ====================================================
Device  [Model : Revision]    Temp        Power     Partitions      SCLK     MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
        Name (20 chars)       (Junction)  (Socket)  (Mem, Compute)
======================================================================================================================
0       [0x74a1 : 0x00]       50.0°C      170.0W    NPS1, SPX       131Mhz   900Mhz   0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
1       [0x74a1 : 0x00]       51.0°C      176.0W    NPS1, SPX       132Mhz   900Mhz   0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
2       [0x74a1 : 0x00]       50.0°C      177.0W    NPS1, SPX       132Mhz   900Mhz   0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
3       [0x74a1 : 0x00]       53.0°C      176.0W    NPS1, SPX       132Mhz   900Mhz   0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
======================================================================================================================
================================================ End of ROCm SMI Log =================================================
```

### Step 2: Install the Latest Version of JAX

Install the required version of JAX with ROCm support using pip:

```Bash
> pip3 install jax[rocm]
```

### Step 3: Verify the Installed JAX Version

Check whether the correct version of JAX and its ROCm plugins are installed:

```Bash
> pip3 freeze | grep jax
jax==0.4.35
jax-rocm60-pjrt==0.4.35
jax-rocm60-plugin==0.4.35
jaxlib==0.4.35
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
[RocmDevice(id=0), RocmDevice(id=1), RocmDevice(id=2), RocmDevice(id=3)]

> python3 -c "import jax.numpy as jnp; x = jnp.arange(5); print(x)"
[0 1 2 3 4]
```

## 4. Build ROCm JAX from Source

Follow these steps to build JAX with ROCm support from source:

### Step 1: Clone the Repository

Clone the ROCm-specific fork of JAX for the desired branch:

```Bash
> git clone https://github.com/ROCm/jax -b <branch_name>
> cd jax
```

### Step 2: Build the Wheels

Run the following command to build the necessary wheels:

```Bash
> python3 ./build/build.py build --wheels=jaxlib,jax-rocm-plugin,jax-rocm-pjrt \
    --rocm_version=60 --rocm_path=/opt/rocm-[version]
```

This will generate three wheels in the `dist/` directory:

* jaxlib (generic, device agnostic library)
* jax-rocm-plugin (ROCm-specific plugin)
* jax-rocm-pjrt (ROCm-specific runtime)

### Step 3: Then install custom JAX using:

```Bash
> python3 setup.py develop --user && pip3 -m pip install dist/*.whl
```

### Simplified Build Script

For a streamlined process, consider using the `jax/build/rocm/dev_build_rocm.py` script.
