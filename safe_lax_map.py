import jax
import jax.numpy as jnp

def consistent_lax_map(f, x, fixed_batch_size=128, **map_kwargs):
    """
    Applies the function f over the leading axis of x using jax.lax.map in fixed-size chunks.
    This ensures that the output is consistent regardless of the batch_size parameter passed
    to jax.lax.map.

    How it works:
      1. It records the original size of x along axis 0.
      2. If x.shape[0] is not a multiple of fixed_batch_size, it pads x along axis 0 with zeros.
      3. It reshapes x into chunks of shape (num_chunks, fixed_batch_size, ...).
      4. It applies jax.lax.map (with the fixed batch size) to each chunk via jax.vmap.
      5. It reshapes the result back to a single batch dimension.
      6. It slices off any extra padded entries to recover the original size.

    Args:
        f: A function to map over each element of x.
        x: A JAX array, where x.shape[0] is the number of items.
        fixed_batch_size: The chunk size to use (default 128).
        **map_kwargs: Additional keyword arguments for jax.lax.map.

    Returns:
        A JAX array resulting from applying f over x, with any padded values removed.
    """
    # 1. Record original size.
    original_size = x.shape[0]

    # 2. Calculate padding if needed.
    remainder = original_size % fixed_batch_size
    if remainder != 0:
        pad_size = fixed_batch_size - remainder
        # Pad only axis 0. For an array of shape (N, ...), pad with ((0, pad_size), (0,0), ...)
        pad_width = [(0, pad_size)] + [(0, 0)] * (x.ndim - 1)
        x_padded = jnp.pad(x, pad_width, mode="constant", constant_values=0)
    else:
        x_padded = x

    # 3. Reshape x_padded into chunks: shape becomes (num_chunks, fixed_batch_size, ...)
    num_chunks = x_padded.shape[0] // fixed_batch_size
    new_shape = (num_chunks, fixed_batch_size) + x_padded.shape[1:]
    x_chunks = x_padded.reshape(new_shape)

    # 4. Define a helper to apply jax.lax.map on one chunk.
    def map_chunk(chunk):
        # Here we use the fixed_batch_size as the batch_size parameter.
        return jax.lax.map(f, chunk, batch_size=fixed_batch_size, **map_kwargs)

    # 5. Use jax.vmap to apply the helper to each chunk.
    mapped_chunks = jax.vmap(map_chunk)(x_chunks)

    # 6. Reshape mapped_chunks back to (num_chunks * fixed_batch_size, ...)
    result_padded = mapped_chunks.reshape(-1, *mapped_chunks.shape[2:])
    # 7. Slice off the extra padded entries.
    result = result_padded[:original_size]
    return result