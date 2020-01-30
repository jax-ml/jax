# Changelog

These are the release notes for JAX.

## jax 0.1.59 (unreleased)

### Breaking changes

* The minimum jaxlib version is now 0.1.38.

### New features

* Reverse-mode automatic differentiation (e.g. `grad`) of `lax.cond`, making it
  now differentiable in both modes (https://github.com/google/jax/pull/2091)

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
