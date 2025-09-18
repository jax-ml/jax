(installation)=
# Installation

<!--* freshness: { reviewed: '2024-06-18' } *-->

Using JAX requires installing two packages: `jax`, which is pure Python and
cross-platform, and `jaxlib` which contains compiled binaries, and requires
different builds for different operating systems and accelerators.

**Summary:** For most users, a typical JAX installation may look something like this:

* **CPU-only (Linux/macOS/Windows)**
  ```
  pip install -U jax
  ```
* **GPU (NVIDIA, CUDA 12)**
  ```
  pip install -U "jax[cuda12]"
  ```

* **TPU (Google Cloud TPU VM)**
  ```
  pip install -U "jax[tpu]"
  ```

(install-supported-platforms)=
## Supported platforms

The table below shows all supported platforms and installation options. Check if your setup is supported; and if it says _"yes"_ or _"experimental"_, then click on the corresponding link to learn how to install JAX in greater detail.

|                  | Linux, x86_64                         | Linux, aarch64                  | Mac, x86_64                           | Mac, aarch64                          | Windows, x86_64          | Windows WSL2, x86_64                     |
|------------------|---------------------------------------|---------------------------------|---------------------------------------|---------------------------------------|--------------------------|------------------------------------------|
| CPU              | {ref}`yes <install-cpu>`              | {ref}`yes <install-cpu>`        | {ref}`jax≤0.4.38 only <install-cpu>`  | {ref}`yes <install-cpu>`              | {ref}`yes <install-cpu>` | {ref}`yes <install-cpu>`                 |
| NVIDIA GPU       | {ref}`yes <install-nvidia-gpu>`       | {ref}`yes <install-nvidia-gpu>` | no                                    | n/a                                   | no                       | {ref}`experimental <install-nvidia-gpu>` |
| Google Cloud TPU | {ref}`yes <install-google-tpu>`       | n/a                             | n/a                                   | n/a                                   | n/a                      | n/a                                      |
| AMD GPU          | {ref}`experimental <install-amd-gpu>` | no                              | {ref}`experimental <install-mac-gpu>` | n/a                                   | no                       | no                                       |
| Apple GPU        | n/a                                   | no                              | n/a                                   | {ref}`experimental <install-mac-gpu>` | n/a                      | n/a                                      |
| Intel GPU        | {ref}`experimental <install-intel-gpu>`| n/a                            | n/a                                   | n/a                                     | no                       | no                                       |


(install-cpu)=
## CPU

### pip installation: CPU

Currently, the JAX team releases `jaxlib` wheels for the following
operating systems and architectures:

- Linux, x86_64
- Linux, aarch64
- macOS, Intel
- macOS, Apple ARM-based
- Windows, x86_64 (*experimental*)

To install a CPU-only version of JAX, which might be useful for doing local
development on a laptop, you can run:

```bash
pip install --upgrade pip
pip install --upgrade jax
```

On Windows, you may also need to install the
[Microsoft Visual Studio 2019 Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)
if it is not already installed on your machine.

Other operating systems and architectures require building from source. Trying
to pip install on other operating systems and architectures may lead to `jaxlib`
not being installed alongside `jax`, although `jax` may successfully install
(but fail at runtime).


(install-nvidia-gpu)=
## NVIDIA GPU

JAX supports NVIDIA GPUs that have SM version 5.2 (Maxwell) or newer.
Note that Kepler-series GPUs are no longer supported by JAX since
NVIDIA has dropped support for Kepler GPUs in its software.

You must first install the NVIDIA driver. You're
recommended to install the newest driver available from NVIDIA, but the driver
version must be >= 525.60.13 for CUDA 12 on Linux.

If you need to use a newer CUDA toolkit with an older driver, for example
on a cluster where you cannot update the NVIDIA driver easily, you may be
able to use the
[CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/)
that NVIDIA provides for this purpose.

### pip installation: NVIDIA GPU (CUDA, installed via pip, easier)

There are two ways to install JAX with NVIDIA GPU support:

- Using NVIDIA CUDA and cuDNN installed from pip wheels
- Using a self-installed CUDA/cuDNN

The JAX team strongly recommends installing CUDA and cuDNN using the pip wheels,
since it is much easier!

NVIDIA has released CUDA pip packages only for x86_64 and aarch64; on other
platforms you must use a local installation of CUDA.

```bash
pip install --upgrade pip

# NVIDIA CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12]"
```

If JAX detects the wrong version of the NVIDIA CUDA libraries, there are several things
you need to check:

* Make sure that `LD_LIBRARY_PATH` is not set, since `LD_LIBRARY_PATH` can
  override the NVIDIA CUDA libraries.
* Make sure that the NVIDIA CUDA libraries installed are those requested by JAX.
  Rerunning the installation command above should work.

### pip installation: NVIDIA GPU (CUDA, installed locally, harder)

If you prefer to use a preinstalled copy of NVIDIA CUDA, you must first
install NVIDIA [CUDA](https://developer.nvidia.com/cuda-downloads) and
[cuDNN](https://developer.nvidia.com/CUDNN).

JAX provides pre-built CUDA-compatible wheels for **Linux x86_64 and Linux aarch64 only**. Other
combinations of operating system and architecture are possible, but require
building from source (refer to {ref}`building-from-source` to learn more}.

You should use an NVIDIA driver version that is at least as new as your
[NVIDIA CUDA toolkit's corresponding driver version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions).
If you need to use a newer CUDA toolkit with an older driver, for example
on a cluster where you cannot update the NVIDIA driver easily, you may be
able to use the
[CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/)
that NVIDIA provides for this purpose.

JAX currently ships one CUDA wheel variant:

| Built with | Compatible with    |
|------------|--------------------|
| CUDA 12.3  | CUDA >=12.1        |
| CUDNN 9.1  | CUDNN >=9.1, <10.0 |
| NCCL 2.19  | NCCL >=2.18        |

JAX checks the versions of your libraries, and will report an error if they are
not sufficiently new.
Setting the `JAX_SKIP_CUDA_CONSTRAINTS_CHECK` environment variable will disable
the check, but using older versions of CUDA may lead to errors, or incorrect
results.

NCCL is an optional dependency, required only if you are performing multi-GPU
computations.

To install, run:

```bash
pip install --upgrade pip

# Installs the wheel compatible with NVIDIA CUDA 12 and cuDNN 9.0 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_local]"
```

**These `pip` installations do not work with Windows, and may fail silently; refer to the table
[above](#supported-platforms).**

You can find your CUDA version with the command:

```bash
nvcc --version
```

JAX uses `LD_LIBRARY_PATH` to find CUDA libraries and `PATH` to find binaries
(`ptxas`, `nvlink`). Please make sure that these paths point to the correct CUDA
installation.

JAX requires libdevice10.bc, which typically comes from the cuda-nvvm package.
Make sure that it is present in your CUDA installation.

Please let the JAX team know on [the GitHub issue tracker](https://github.com/jax-ml/jax/issues)
if you run into any errors or problems with the pre-built wheels.

(docker-containers-nvidia-gpu)=
### NVIDIA GPU Docker containers

NVIDIA provides the [JAX
Toolbox](https://github.com/NVIDIA/JAX-Toolbox) containers, which are
bleeding edge containers containing nightly releases of jax and some
models/frameworks.

(install-google-tpu)=
## Google Cloud TPU

### pip installation: Google Cloud TPU

JAX provides pre-built wheels for
[Google Cloud TPU](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).
To install JAX along with appropriate versions of `jaxlib` and `libtpu`, you can run
the following in your cloud TPU VM:

```bash
pip install "jax[tpu]"
```

For users of Colab (https://colab.research.google.com/), be sure you are
using *TPU v2* and not the older, deprecated TPU runtime.

(install-mac-gpu)=
## Mac GPU

### pip installation

Apple provides an experimental Metal plugin. For details,
refer to
[Apple's JAX on Metal documentation](https://developer.apple.com/metal/jax/).

**Note:** There are several caveats with the Metal plugin:

* The Metal plugin is new and experimental and has a number of
  [known issues](https://github.com/jax-ml/jax/issues?q=is%3Aissue+is%3Aopen+label%3A%22Apple+GPU+%28Metal%29+plugin%22).
  Please report any issues on the JAX issue tracker.
* The Metal plugin currently requires very specific versions of `jax` and
  `jaxlib`. This restriction will be relaxed over time as the plugin API
  matures.

(install-amd-gpu)=
## AMD GPU (Linux)

JAX has experimental ROCm support. There are two ways to install JAX:

* Use [AMD's Docker container](https://hub.docker.com/r/rocm/jax-community/tags); or
* Build from source. Refer to the section [Additional notes for building a ROCm jaxlib for AMD GPUs](https://jax.readthedocs.io/en/latest/developer.html#additional-notes-for-building-a-rocm-jaxlib-for-amd-gpus).

(install-intel-gpu)=
## Intel GPU

Intel provides an experimental OneAPI plugin: intel-extension-for-openxla for Intel GPU hardware. For more details and installation instructions, refer to one of the following two methods:
1. Pip installation: [JAX acceleration on Intel GPU](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).
2. Using [Intel's XLA Docker container](https://hub.docker.com/r/intel/intel-optimized-xla).

Please report any issues related to:
* JAX: [JAX issue tracker](https://github.com/jax-ml/jax/issues).
* Intel's OpenXLA plugin: [Intel-extension-for-openxla issue tracker](https://github.com/intel/intel-extension-for-openxla/issues).

## Conda (community-supported)

### Conda installation

There is a community-supported Conda build of `jax`. To install it using `conda`,
simply run:

```bash
conda install jax -c conda-forge
```

If you run this command on machine with an NVIDIA GPU, this should install a CUDA-enabled package of `jaxlib`.

To ensure that the jax version you are installing is indeed CUDA-enabled, run:

```bash
conda install "jaxlib=*=*cuda*" jax -c conda-forge
```

If you would like to override which release of CUDA is used by JAX, or to
install the CUDA build on a machine without GPUs, follow the instructions in the
[Tips & tricks](https://conda-forge.org/docs/user/tipsandtricks.html#installing-cuda-enabled-packages-like-tensorflow-and-pytorch)
section of the `conda-forge` website.

Go to the `conda-forge`
[jaxlib](https://github.com/conda-forge/jaxlib-feedstock#installing-jaxlib) and
[jax](https://github.com/conda-forge/jax-feedstock#installing-jax) repositories
for more details.


## JAX nightly installation

Nightly releases reflect the state of the main JAX repository at the time they are
built, and may not pass the full test suite.

Unlike the instructions for installing a JAX release, here we name all of JAX's
packages explicitly on the command line, so `pip` will upgrade them if a newer
version is available.

- CPU only:

```bash
pip install -U --pre jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
```

- Google Cloud TPU:

```bash
pip install -U --pre jax jaxlib libtpu requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

- NVIDIA GPU (CUDA 12):

```bash
pip install -U --pre jax jaxlib "jax-cuda12-plugin[with_cuda]" jax-cuda12-pjrt -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
```

- NVIDIA GPU (CUDA 12) legacy:

Use the following for historical nightly releases of monolithic CUDA jaxlibs.
You most likely do not want this; no further monolithic CUDA jaxlibs will be
built and those that exist will expire by Sep 2024. Use the "CUDA 12" option above.

```bash
pip install -U --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda12_releases.html
```

(building-jax-from-source)=
## Building JAX from source

Refer to {ref}`building-from-source`.

## Installing older `jaxlib` wheels

Due to storage limitations on the Python package index, the JAX team periodically removes
older `jaxlib` wheels from the releases on http://pypi.org/project/jax. These can
still be installed directly via the URLs here. For example:

```bash
# Install jaxlib on CPU via the wheel archive
pip install "jax[cpu]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install the jaxlib 0.3.25 CPU wheel directly
pip install jaxlib==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
For specific older GPU wheels, be sure to use the `jax_cuda_releases.html` URL; for example
```bash
pip install jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
