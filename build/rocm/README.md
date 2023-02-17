# JAX Builds on ROCm
This directory contains files and setup instructions to build and test JAX for ROCm in Docker environment (runtime and CI). You can build, test and run JAX on ROCm yourself!
***
### Build JAX-ROCm in docker for the runtime

1.  Install Docker: Follow the [instructions on the docker website](https://docs.docker.com/engine/installation/).

2. Build a runtime JAX-ROCm docker container and keep this image by running the following command.

    ./build/rocm/ci_build.sh --keep_image --runtime bash -c "./build/rocm/build_rocm.sh"

3. To launch a JAX-ROCm container: If the build was successful, there should be a docker image with name "jax-rocm:latest" in list of docker images (use "docker images" command to list them).
```
sudo docker run -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --entrypoint /bin/bash jax-rocm:latest
```

***
### Build and Test JAX-ROCm in docker for CI jobs
This folder has all the scripts necessary to build and run tests for JAX-ROCm.
The following command will build JAX on ROCm and run all the tests inside docker (script should be called from JAX root folder).
```
./build/rocm/ci_build.sh
```
