from jax._src import numpy as jnp
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.ufuncs import rint
from jax._src.numpy.linalg import vecdot
from jax._src.numpy.util import check_arraylike


def _expand_to_ndim(a: Array, b: Array) -> Array:
  axes = range(a.ndim, b.ndim)
  return jnp.expand_dims(a, axes)


def _sample_image_nearest(
    img: Array,
    loc: Array,
    padding: str,
    fill_value: Array,
) -> Array:
  loc = rint(loc).astype(int)
  new_img = img[tuple(jnp.moveaxis(loc, -1, 0))]
  if padding == "fill":
    limit = jnp.array(img.shape[: loc.shape[-1]])
    valid = ((0 <= loc) & (loc < limit)).all(-1)
    new_img = jnp.where(_expand_to_ndim(valid, new_img), new_img, fill_value)
    assert isinstance(new_img, Array)
  return new_img


def _get_all_bit_strings(n: int) -> Array:
  return (jnp.arange(1 << n)[:, None] >> jnp.arange(n)[None, :]) & 1


def _sample_image_linear(
    img: Array,
    loc: Array,
    padding: str,
    fill_value: Array,
) -> Array:
  offsets = _get_all_bit_strings(loc.shape[-1])

  ip = jnp.floor(loc)
  fp = loc - ip

  ip = ip.astype(int)

  ip = ip[..., None, :]
  fp = fp[..., None, :]

  ip += offsets
  fp = jnp.where(offsets, fp, 1 - fp)

  value = img[tuple(jnp.moveaxis(ip, -1, 0))]
  weight = fp.prod(-1)

  if padding == "fill":
    limit = jnp.array(img.shape[: loc.shape[-1]])
    valid = ((0 <= ip) & (ip < limit)).all(-1)
    value = jnp.where(_expand_to_ndim(valid, value), value, fill_value)

  new_img = vecdot(_expand_to_ndim(weight, value), value, axis=loc.ndim - 1)
  return new_img


def sample_image(
    img: ArrayLike,
    loc: ArrayLike,
    interpolation: str = "linear",
    padding: str = "fill",
    fill_value: ArrayLike = 0,
    rescale: bool = False,
) -> Array:
  """Sample an image at a given location.

  Supports the following interpolation modes:

  - ``'nearest'``: Nearest interpolation.

  - ``'linear'``: Linear interpolation.

  Supports the following padding modes:

  - ``'fill'``: Fills points outside the image with a constant value.

    Example: ``v v v v v | a b c | v v v v v``.

  - ``'clip'``: Clips points outside the image to the edges of the image.

    Example: ``a a a a a | a b c | c c c c c``.

  - ``'wrap'``: Wraps points outside the image to the opposite edge.

    Example: ``b c a b c | a b c | a b c a b``.

  - ``'reflect'``: Reflects points outside the image into the image.

    Example: ``b a b c b | a b c | b a c a b``.

  Args:
    img: Array-like of shape ``(n1, n2, ..., nd, *img_pixel_shape)``. Image to sample from.
    loc: Array-like of shape ``(*loc_batch_shape, d)``. Location to sample at.
    interpolation: Interpolation mode to use.
    padding: Padding mode to use.
    fill_value: Array-like. Fill value to use. Must be broadcast-compatible with ``img_pixel_shape``.
    rescale: Boolean. Rescale coordinates from [0, 1] to the image shape.

  Returns:
    Array of shape ``(*loc_batch_shape, *img_pixel_shape)``. Sampled pixels.
  """
  check_arraylike("sample_image", img)
  check_arraylike("sample_image", loc)
  check_arraylike("sample_image", fill_value)

  img = jnp.asarray(img)
  loc = jnp.asarray(loc)
  fill_value = jnp.asarray(fill_value)

  if loc.shape[-1] > img.ndim:
    raise ValueError("Size of last axis of loc cannot exceed img.ndim.")

  limit = jnp.array(img.shape[: loc.shape[-1]])

  if rescale:
    loc *= limit - 1

  if padding == "wrap":
    loc %= limit
  elif padding == "clip":
    loc = loc.clip(0, limit - 1)
  elif padding == "reflect":
    loc = limit - jnp.abs(loc % (limit * 2) - limit)
  elif padding == "fill":
    pass
  else:
    raise ValueError("Invalid padding mode {padding!r}")

  if interpolation == "nearest":
    new_img = _sample_image_nearest(
      img, loc, padding=padding, fill_value=fill_value
    )
  elif interpolation == "linear":
    new_img = _sample_image_linear(
      img, loc, padding=padding, fill_value=fill_value
    )
  else:
    raise ValueError(f"Invalid interpolation mode {interpolation!r}")

  assert new_img.shape == loc.shape[:-1] + img.shape[loc.shape[-1] :]

  return new_img
