---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(jax-101-convolutions)=
# Convolutions

<!--* freshness: { reviewed: '2026-08-26' } *-->

For basic convolution operations, JAX provides NumPy- and SciPy-style
interfaces: {func}`jax.numpy.convolve` (with {func}`jax.numpy.correlate`) and
{func}`jax.scipy.signal.convolve` (with its 2D and correlation variants).
Those work like their NumPy and SciPy counterparts, and their API docs
describe them.

This page is about the general case: the batched, N-dimensional convolutions
used in deep neural networks, computed by
{func}`jax.lax.conv_general_dilated`. It's a very general function, and it's
not very obvious how to use it, so we'll work through the common use-cases.

A survey of the family of convolutional operators, [a guide to convolutional
arithmetic](https://arxiv.org/abs/1603.07285), is highly recommended reading.

## Setup: a kernel and an image

Let's define a simple diagonal edge kernel, and a synthetic image of colored
squares to convolve it with:

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
from jax import lax

# 2D kernel - HWIO layout
kernel = jnp.zeros((3, 3, 3, 3), dtype=jnp.float32)
kernel += jnp.array([[1, 1, 0],
                     [1, 0,-1],
                     [0,-1,-1]])[:, :, jnp.newaxis, jnp.newaxis]

# NHWC layout
img = jnp.zeros((1, 200, 198, 3), dtype=jnp.float32)
for k in range(3):
  x = 30 + 60*k
  y = 20 + 60*k
  img = img.at[0, x:x+10, y:y+10, k].set(1.0)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(kernel[:, :, 0, 0])
ax[0].set_title('edge conv kernel')
ax[1].imshow(img[0])
ax[1].set_title('original image');
```

## Dimension numbers define dimensional layout

The important argument to {func}`~jax.lax.conv_general_dilated` is the
3-tuple of axis layout arguments: (Input Layout, Kernel Layout, Output
Layout), where

 - __N__ - batch dimension
 - __H__ - spatial height
 - __W__ - spatial width
 - __C__ - channel dimension
 - __I__ - kernel _input_ channel dimension
 - __O__ - kernel _output_ channel dimension

Here we choose a __NHWC__ image and __HWIO__ kernel convention, matching the
arrays above:

```{code-cell}
dn = lax.conv_dimension_numbers(img.shape,     # only ndim matters, not shape
                                kernel.shape,  # only ndim matters, not shape
                                ('NHWC', 'HWIO', 'NHWC'))  # the important bit
print(dn)
```

With dimension numbers in hand, here's a basic convolution. `'SAME'` padding
preserves the spatial shape:

```{code-cell}
out = lax.conv_general_dilated(img,    # lhs = image tensor
                               kernel, # rhs = conv kernel tensor
                               (1,1),  # window strides
                               'SAME', # padding mode
                               (1,1),  # lhs/image dilation
                               (1,1),  # rhs/kernel dilation
                               dn)     # dimension_numbers
print("out shape: ", out.shape)
plt.imshow(np.array(out)[0,:,:,0]);
```

Strides downsample the output. With `(2,2)` window strides, the output has
half the spatial size (and `'VALID'` padding, which applies no padding at
all, would shrink it further):

```{code-cell}
out = lax.conv_general_dilated(img, kernel,
                               (2,2),  # window strides
                               'SAME', (1,1), (1,1), dn)
print("out shape: ", out.shape, " <-- half the size")
plt.imshow(np.array(out)[0,:,:,0]);
```

Kernel (rhs) dilation spreads the kernel taps apart, giving *atrous* or
*dilated* convolution. Here we use an excessive dilation to make the effect
obvious:

```{code-cell}
out = lax.conv_general_dilated(img, kernel,
                               (1,1), 'VALID',
                               (1,1),
                               (12,12),  # rhs/kernel dilation
                               dn)
print("out shape: ", out.shape)
plt.imshow(np.array(out)[0,:,:,0]);
```

```{note}
JAX also provides the convenience functions `lax.conv` and
`lax.conv_with_general_padding`, which fix the layout convention instead of
taking dimension numbers: they assume __NCHW__ images and __OIHW__ kernels.
```

## Transposed convolutions

Transposed convolution (sometimes called deconvolution, or fractionally
strided convolution) upsamples rather than downsamples: it's the transpose of
a strided convolution, and it's what the backward pass of a strided
convolution computes. JAX provides it directly as
{func}`jax.lax.conv_transpose`:

```{code-cell}
out = lax.conv_transpose(img, kernel,
                         (2,2),   # strides of the conv being transposed
                         'SAME',
                         dimension_numbers=dn,
                         transpose_kernel=True)
print("out shape: ", out.shape, "<-- larger than the input")
plt.imshow(np.array(out)[0,:,:,0]);
```

Internally, this is another `conv_general_dilated`: a transposed
convolution is a 180° kernel rotation plus *lhs* (input) dilation, which
spreads the input values apart:

```{code-cell}
# transposed conv = 180deg kernel rotation plus LHS dilation
kernel_rot = jnp.rot90(jnp.rot90(kernel, axes=(0,1)), axes=(0,1))
manual = lax.conv_general_dilated(img, kernel_rot,
                                  (1,1),
                                  ((2, 1), (2, 1)),  # custom output padding
                                  (2,2),             # lhs/image dilation
                                  (1,1), dn)
print("same as conv_transpose:", jnp.allclose(out, manual))
```

## 1D convolutions

You aren't limited to 2D convolutions; the same dimension-number machinery
handles any spatial rank. A simple 1D demo:

```{code-cell}
# 1D kernel - WIO layout
kernel = jnp.array([[[1, 0, -1], [-1,  0,  1]],
                    [[1, 1,  1], [-1, -1, -1]]],
                    dtype=jnp.float32).transpose([2,1,0])
# 1D data - NWC layout
data = np.zeros((1, 200, 2), dtype=jnp.float32)
for i in range(2):
  for k in range(2):
      x = 35*i + 30 + 60*k
      data[0, x:x+30, k] = 1.0

print("in shapes:", data.shape, kernel.shape)

plt.figure(figsize=(10,5))
plt.plot(data[0]);
dn = lax.conv_dimension_numbers(data.shape, kernel.shape,
                                ('NWC', 'WIO', 'NWC'))
print(dn)

out = lax.conv_general_dilated(data, kernel,
                               (1,),   # 1D window strides
                               'SAME', (1,), (1,), dn)
print("out shape: ", out.shape)
plt.figure(figsize=(10,5))
plt.plot(out[0]);
```

## 3D convolutions

```{code-cell}
import matplotlib as mpl

# Random 3D kernel - HWDIO layout
kernel = jnp.array([
  [[0, 0,  0], [0,  1,  0], [0,  0,   0]],
  [[0, -1, 0], [-1, 0, -1], [0,  -1,  0]],
  [[0, 0,  0], [0,  1,  0], [0,  0,   0]]],
  dtype=jnp.float32)[:, :, :, jnp.newaxis, jnp.newaxis]

# 3D data - NHWDC layout
data = jnp.zeros((1, 30, 30, 30, 1), dtype=jnp.float32)
x, y, z = np.mgrid[0:1:30j, 0:1:30j, 0:1:30j]
data += (jnp.sin(2*x*jnp.pi)*jnp.cos(2*y*jnp.pi)*jnp.cos(2*z*jnp.pi))[None,:,:,:,None]

print("in shapes:", data.shape, kernel.shape)
dn = lax.conv_dimension_numbers(data.shape, kernel.shape,
                                ('NHWDC', 'HWDIO', 'NHWDC'))
print(dn)

out = lax.conv_general_dilated(data, kernel,
                               (1,1,1), 'SAME', (1,1,1), (1,1,1), dn)
print("out shape: ", out.shape)

# Make some simple 3d density plots:
def make_alpha(cmap):
  my_cmap = cmap(jnp.arange(cmap.N))
  my_cmap[:,-1] = jnp.linspace(0, 1, cmap.N)**3
  return mpl.colors.ListedColormap(my_cmap)
my_cmap = make_alpha(plt.cm.viridis)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=data.ravel(), cmap=my_cmap)
ax.axis('off')
ax.set_title('input')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=out.ravel(), cmap=my_cmap)
ax.axis('off')
ax.set_title('3D conv output');
```
