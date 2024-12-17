# JAX Builds on ROCm
This directory contains files and setup instructions to build and test JAX for ROCm in Docker environment (runtime and CI). You can build, test and run JAX on ROCm yourself!
***
### Build JAX-ROCm in docker for the runtime

1.  Install Docker: Follow the [instructions on the docker website](https://docs.docker.com/engine/installation/).

2. Build a runtime JAX-ROCm docker container and keep this image by running the following command. Note: must pass in appropriate
options. The example below builds Python 3.12 container.

```Bash
./build/rocm/ci_build.sh --py_version 3.12
```

3. To launch a JAX-ROCm container: If the build was successful, there should be a docker image with name "jax-rocm:latest" in list of docker images (use "docker images" command to list them).

```Bash
docker run -it -d --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 64G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v ./:/jax --name rocm_jax jax-rocm:latest /bin/bash
```

***
### JAX ROCm Releases
We aim to push all ROCm-related changes to the OpenXLA repository. However, there may be times when certain JAX/jaxlib updates for
ROCm are not yet reflected in the upstream JAX repository. To address this, we maintain ROCm-specific JAX/jaxlib branches tied to JAX
releases. These branches are available in the ROCm fork of JAX at https://github.com/ROCm/jax. Look for branches named in the format
rocm-jaxlib-[jaxlib-version]. You can also find corresponding branches in https://github.com/ROCm/xla. For example, for JAX version
0.4.33, the branch is named rocm-jaxlib-v0.4.33, which can be accessed at https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.4.33.

JAX source-code and related wheels for ROCm are available here

```Bash
https://github.com/ROCm/jax/releases
```

***Note:*** Some earlier jaxlib versions on ROCm were released on ***PyPi***. 
```
https://pypi.org/project/jaxlib-rocm/#history
```
