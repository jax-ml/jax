# JAX Builds on ROCm
This directory contains files and setup instructions t0 build and test JAX for ROCm in Docker environment. You can build, test and run JAX on ROCm yourself!
***
### Build JAX-ROCm in docker

1.  Install Docker: Follow the [instructions on the docker website](https://docs.docker.com/engine/installation/).

  2. Build JAX by running the following command from JAX root folder.

    ./build/rocm/ci_build.sh --keep_image bash -c "./build/rocm/build_rocm.sh"

  3. Launch a container: If the build was successful, there should be a docker image with name "jax-rocm:latest" in list of docker images (use "docker images" command to list them).
  ```
  sudo docker run -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --entrypoint /bin/bash jax-rocm:latest
  ```

***
### Build and Test JAX-ROCm in docker (suitable for CI jobs)
This folder has all the scripts necessary to build and run tests for JAX-ROCm.
The following command will build JAX on ROCm and run all the tests inside docker (script should be called from JAX root folder).
```
./build/rocm/ci_build.sh bash -c "./build/rocm/build_rocm.sh&&./build/rocm/run_single_gpu.py&&build/rocm/run_multi_gpu.sh"
```
