# Changelog

These are the release notes for JAX.

## jax 0.1.59 (unreleased)

### Breaking changes

* The minimum jaxlib version is now 0.1.38.
* Simplified `Jaxpr` by removing the `Jaxpr.freevars` and changing the 
  representation of `Jaxpr.bound_subjaxprs` to drop the environment values. 

### New features

* Reverse-mode automatic differentiation (e.g. `grad`) of `lax.cond`, making it
  now differentiable in both modes (https://github.com/google/jax/pull/2091)
* JAX now supports DLPack, which allows sharing CPU and GPU arrays in a
  zero-copy way with other libraries, such as PyTorch.
* JAX GPU DeviceArrays now support `__cuda_array_interface__`, which is another
  zero-copy protocol for sharing GPU arrays with other libraries such as CuPy
  and Numba.
* JAX CPU device buffers now implement the Python buffer protocol, which allows
  zero-copy buffer sharing between JAX and NumPy.
* Added JAX_SKIP_SLOW_TESTS environment variable to skip tests known as slow.

## jaxlib 0.1.39 (February 10, 2020)

* Bumped version so JAX can use `PyLocalExecutable::local_logical_device_ids`.

## jaxlib 0.1.38 (January 29, 2020)

* CUDA 9.0 is no longer supported.
* CUDA 10.2 wheels are now built by default.

## jax 0.1.58 (January 28, 2020)

### Breaking changes

* JAX has dropped Python 2 support, because Python 2 reached its end of life on
  January 1, 2020. Please update to Python 3.5 or newer.

### New features

* Forward-mode automatic differentiation (`jvp`) of while loop
  (https://github.com/google/jax/pull/1980)
* New NumPy and SciPy functions:
  * `jax.numpy.fft.fft2`
  * `jax.numpy.fft.ifft2`
  * `jax.numpy.fft.rfft`
  * `jax.numpy.fft.irfft`
  * `jax.numpy.fft.rfft2`
  * `jax.numpy.fft.irfft2`
  * `jax.numpy.fft.rfftn`
  * `jax.numpy.fft.irfftn`
  * `jax.numpy.fft.fftfreq`
  * `jax.numpy.fft.rfftfreq`
  * `jax.numpy.linalg.matrix_rank`
  * `jax.numpy.linalg.matrix_power`
  * `jax.scipy.special.betainc`
* Batched Cholesky decomposition on GPU now uses a more efficient batched
  kernel.


### Notable bug fixes

* With the Python 3 upgrade, JAX no longer depends on `fastcache`, which should
  help with installation.
