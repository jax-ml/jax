# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Serializations routines for pytrees including array and non-array serialization.
"""

from __future__ import annotations

from os import PathLike
import os
import re
from typing import Any
from uuid import uuid4, UUID
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import shutil
import logging

import jax
from jax._src import distributed
from jax._src.api_util import flatten_axes

from jax.experimental import multihost_utils
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
import jax.experimental.array_serialization.pytree_serialization_utils as utils
from jax._src import path as pathlib
import numpy as np

logger = logging.getLogger(__name__)

_THREADING_SAVE_LOCK = threading.Lock()

_REMOTE_URL_PREFIXES = ['gs://', 's3://']
_PYTREEDEF_FILE = "pytreedef.json"
_ARCHIVE_NAME = "archive.zip"
_USE_OCDBT = True  # a lot of the code relies on this being True
_MAX_PATH_LENGTH = 4096
_ARRAY_STORE_DIRNAME = "array_store"
_ARRAY_TYPE_FORMAT = "Array({dtype}[{shape}])"
_ARRAY_TYPE_REGEX = r"Array\(([a-zA-Z0-9_]+)\[([0-9, ]*)\]\)"
_MAX_CONCURRENCY = 32
_TIMEOUT_SEC = 30

PyTreeT = Any

__all__ = ["save", "load", "load_pytreedef",
           "nonblocking_load", "nonblocking_save"]


def _get_unique_sync_key() -> str | None:
  """Generate a thread-local key for ensuring all host finish (de)serializing"""
  if jax.process_count() == 1:
    return None
  # broadcast a thread-local unique barrier name
  sync_key_unique = multihost_utils.broadcast_one_to_all(
      np.frombuffer(uuid4().bytes, dtype=np.int32))
  sync_key_id = UUID(bytes=np.array(sync_key_unique).tobytes())
  return f"jax_sync_key_{str(sync_key_id)}"


def _is_str_same_on_all_hosts(path: str | PathLike[str]) -> bool:
  """All-gather the location of the checkpoint and check if it's the same."""
  if jax.process_count() <= 1:
    return False
  path_b = str(path).encode("utf-8")
  if len(path_b) > _MAX_PATH_LENGTH:
    raise ValueError(f"Path exceeds maximum length of {_MAX_PATH_LENGTH} in"
                     " multiprocess case.")
  path_array = np.concatenate([
      np.frombuffer(path_b, dtype=np.uint8), np.zeros(
          _MAX_PATH_LENGTH - len(path_b), dtype=np.uint8)])
  path_array = multihost_utils.process_allgather(path_array)
  return bool(np.all(path_array[0] == path_array[1:]))


def _sync_on_key(key: str | None, extra_tag: str = "") -> None:
  if key is None:
    return
  full_key = f"{key}-{extra_tag}" if extra_tag else key
  if (client := distributed.global_state.client) is not None:
    client.wait_at_barrier(full_key, timeout_in_ms=_TIMEOUT_SEC * 1000)


def _is_array_like(x):
  return isinstance(x, (jax.Array, np.ndarray))


def _leaf_to_desc(leaf) -> str:
  if leaf is None:
    return "null"
  elif _is_array_like(leaf):
    return _ARRAY_TYPE_FORMAT.format(
        dtype=leaf.dtype.name, shape=", ".join(map(str, leaf.shape)))
  else:
    return type(leaf).__name__


def _desc_to_leaf(leaf_desc: str | None) -> str | None | jax.ShapeDtypeStruct:
  if leaf_desc is None:
    return None
  if not re.match(_ARRAY_TYPE_REGEX, leaf_desc):
    return leaf_desc
  shape_dtype_match = re.match(_ARRAY_TYPE_REGEX, leaf_desc)
  assert shape_dtype_match is not None
  dtype_str, shape_str = shape_dtype_match.groups()
  shape = [int(x.strip()) for x in shape_str.strip("]").strip().split(",")
            if len(x.strip()) > 0]
  return jax.ShapeDtypeStruct(shape, jax.numpy.dtype(dtype_str))


def _is_remote_path(path: str | PathLike[str]):
  """Check whether a path is remote by examining the prefix."""
  # we need to truncate e.g., gs:// to gs:/ because pathlib.Path collapses //
  return any(str(path).startswith(prefix[:-1])
             for prefix in _REMOTE_URL_PREFIXES)


def _norm_path(path: str | PathLike[str]) -> Any:
  if _is_remote_path(path):
    return pathlib.Path(path)
  return pathlib.Path(path).expanduser().resolve()


def _rm_dir(root: Any) -> None:
  if _is_remote_path(root):
    root.rmtree()  # pytype: disable=attribute-error
  else:
    shutil.rmtree(root)


def _set_up_destination(root: str | PathLike[str], overwrite: bool,
                        pytree_repr: dict[str, Any], distinct_locations: bool,
                        sync_key: str | None) -> dict[str, Any]:
  """Inspect the destination, set it up for writing, potentially read existing data."""
  root = _norm_path(root)
  if overwrite:
    if root.exists() and len(list(root.iterdir())) > 0:
      # check that we're only deleting things that come from JAX
      # refuse to rm directories containing additional entries
      extra_member_paths = [
          path for path in list(root.iterdir()) if path.name not in
          (_PYTREEDEF_FILE, _ARCHIVE_NAME, _ARRAY_STORE_DIRNAME)]

      if len(extra_member_paths) != 0:
        raise RuntimeError(
            "Refusing to work on a directory that is not a previous checkpoint."
            f" Unrecognized paths: {extra_member_paths}. Remove them manually"
            f" if you're sure you want to use {root} as the checkpoint"
            " directory.")

      if (jax.process_index() == 0 or distinct_locations) and root.exists():
        _rm_dir(root)
    _sync_on_key(sync_key, "overwrite")
    return pytree_repr
  else:
    if (root.exists() and len(list(root.iterdir())) > 0):  # not empty
      raise ValueError(f"Files already exist at path: `{root}`, but you"
                       f" specified `{overwrite=}`")
    return pytree_repr


def _prepare_directory(root: str | PathLike[str], overwrite: bool,
                       pytreedef_repr: dict[str, Any], distinct_locations: bool,
                       sync_key: str | None):
  """Prepare the directory: check destination, potentially read existing data
  and overwrite.

  Raises:
    RuntimeError: If the destination directory cannot be created.
  """
  root = _norm_path(root)
  # prepare the destination directory, overwrite destination directory or error
  pytreedef_repr = _set_up_destination(
      root, overwrite, pytreedef_repr, distinct_locations, sync_key)

  if not _is_remote_path(root) and (distinct_locations
                                    or jax.process_index() == 0):
    root.mkdir(exist_ok=True)  # do not make parents, that's too much
    if not root.exists() or not root.is_dir():
      raise RuntimeError(f"Could not create destination directory at {root}")
  _sync_on_key(sync_key, "mkdir")
  return pytreedef_repr


def _write_arrays(array_store_path: Any, arrs: list[Any],
                  arr_leaf_ids: list[int], ts_specs: list[Any | None],
                  distinct_locations: bool):
  paths = [array_store_path / str(leaf_id) for leaf_id in arr_leaf_ids]
  process_idx = None
  if not distinct_locations and jax.process_count() > 1:
    process_idx = jax.process_index()
  default_ts_specs = [ts_impl.get_tensorstore_spec(path, ocdbt=_USE_OCDBT,
                                                   process_idx=process_idx,
                                                   arr=arr)
                      for (path, arr) in zip(paths, arrs)]
  ts_specs = [ts_impl.merge_nested_ts_specs(default_ts_spec, ts_spec)
              for (default_ts_spec, ts_spec) in zip(default_ts_specs, ts_specs)]

  # sanity check the ts specs
  if len(ts_specs) > 0:  # verify the base path is shared for all arrays
    expected_path = ts_specs[0]["kvstore"]["base"]["path"]  # shared base path
    for ts_spec, arr in zip(ts_specs, arrs):
      ts_impl.verify_tensorstore_spec(ts_spec, arr, expected_path,
                                      ocdbt=_USE_OCDBT, check_metadata=True)

  async def _serialize_arrays():
    await asyncio.gather(*[
        ts_impl.async_serialize(arr, ts_spec, primary_host=None)
        for (arr, ts_spec) in zip(arrs, ts_specs)])

  asyncio.run(_serialize_arrays())


def _finalize_array_store(kvstore_path, distinct_locations: bool):
  """When multiple processes are writing, they must write to a per-process
  location followed by combining them via no-copy links to the final location.
  """
  # only in multiprocess case and only process 0
  if distinct_locations or jax.process_count() == 1 or jax.process_index() != 0:
    return
  dummy_key_path = os.path.join(kvstore_path, "dummy_key")
  combined_kvstore = ts_impl.get_tensorstore_spec(
      dummy_key_path, ocdbt=True, process_idx=None)["kvstore"]
  children_kvstores = [ts_impl.get_tensorstore_spec(
      dummy_key_path, ocdbt=True, process_idx=i)["kvstore"]
      for i in range(jax.process_count())]
  _ = combined_kvstore.pop("path")
  _ = [kvstore.pop("path") for kvstore in children_kvstores]
  asyncio.run(ts_impl.combine_kvstores(combined_kvstore, children_kvstores))


def _write_pytreedef(directory: Any, pytree_repr: dict[str, Any],
                     distinct_locations: bool):
  """Write the pytreedef to the destination directory and aux data to the archive."""
  if not (jax.process_index() == 0 or distinct_locations):
    return
  root = _norm_path(directory)
  (root / _PYTREEDEF_FILE).write_text(json.dumps(pytree_repr, indent=2))


def _tree_broadcast(a, b, is_leaf=lambda x: x is None):
  """Broadcast the prefix tree `a` to the full tree `b`

  Uses `flatten_axes` for better error messages on mismatched arity but allowing
  for custom is_leaf in the `a` and `b` trees.
  """
  a_leaves, a_struct = jax.tree.flatten(a, is_leaf=is_leaf)
  a_idx2leaf_map = dict(enumerate(a_leaves))
  a_idx = jax.tree.unflatten(a_struct, a_idx2leaf_map.keys())
  a_idx_broadcast = flatten_axes("tree_broadcast",
                                 jax.tree.structure(b, is_leaf=is_leaf), a_idx)
  return jax.tree.map(lambda i: a_idx2leaf_map[i], a_idx_broadcast)


_serialization_executor = ThreadPoolExecutor(max_workers=_MAX_CONCURRENCY)


def save(data: PyTreeT, directory: str | PathLike[str], *,
         overwrite: bool = True, ts_specs: PyTreeT | None = None) -> None:
  """Saves the given data structure to the provided directory path.

  This function provides functionality to serialize and save a data structure
  comprising JAX arrays, along with its structure to a given directory. It
  leverages `PyTree` for flattening and reconstructing the data structure.

  This is a simple experimental array serialization API, for anything more
  complex and for all checkpointing prefer: https://github.com/google/orbax

  Args:
    data: The data structure to be saved. Arbitrary composition of JAX arrays,
      including nested structures.
    directory: The directory path where the data will be saved. A local path or
      a remote URL (e.g., gs://, s3://). For remote URLs, `etils` is required.
    overwrite: If True, any existing directory with the same name will be
      overwritten.
    ts_specs: Optional tensorstore specs to use for serialization. If None,
      defaults to using the default tensorstore specs.

  Example:
    >>> data = {"a": jnp.array([1, 2]), "b": None}
    >>> save(data, directory)
  """
  with _THREADING_SAVE_LOCK:
    return _save(data, directory, overwrite=overwrite, ts_specs=ts_specs)


def _save(data: PyTreeT, directory: str | PathLike[str], *,
          overwrite: bool = True, ts_specs: PyTreeT | None = None) -> None:
  sync_key = _get_unique_sync_key()  # get a synchronization key for multi-host

  if _is_remote_path(directory) and not pathlib.epath_installed:
    raise RuntimeError("For saving to remote URLs (e.g., gs, s3) you need the"
                       " `etils` module installed. You can install it using"
                       " `pip install etils`.")
  ts_specs = _tree_broadcast(ts_specs, data,
                             is_leaf=ts_impl.is_tensorstore_spec_leaf)
  data_flat, pytreedef = jax.tree.flatten(data, is_leaf=lambda x: x is None)
  if not all(x is None or _is_array_like(x) for x in data_flat):
    raise ValueError("For serialization, all leaves must be either None or"
                     " jax.Array-like objects.")
  distinct_locations = not _is_str_same_on_all_hosts(directory)
  if jax.process_count() > 1 and distinct_locations:
    raise ValueError(
        "Saving to different locations on different hosts is not supported,"
        " because it is extremely fragile. Consider using a single location.")
  root = _norm_path(directory)

  # 1. serialize the pytree #################################
  pytreedef_repr = utils.serialize_pytreedef(pytreedef)
  pytreedef_repr[utils._LEAF_IDS_KEY] = jax.tree.map(_leaf_to_desc, data_flat)

  pytreedef_repr = _prepare_directory(
      root, overwrite, pytreedef_repr, distinct_locations, sync_key)
  futures = []
  futures.append(_serialization_executor.submit(
      _write_pytreedef, root, pytreedef_repr, distinct_locations))

  # 2. serialize arrays #####################################
  array_store_path = root / _ARRAY_STORE_DIRNAME
  arrs = [data for data in data_flat if _is_array_like(data)]
  arr_leaf_ids = [i for i, data in enumerate(data_flat) if _is_array_like(data)]
  ts_specs_flat = jax.tree.leaves(ts_specs,
                                  is_leaf=ts_impl.is_tensorstore_spec_leaf)
  ts_specs_flat = [ts_specs_flat[i] for i in arr_leaf_ids]
  futures.append(_serialization_executor.submit(
      _write_arrays, array_store_path, arrs, arr_leaf_ids, ts_specs_flat,
      distinct_locations))

  # 3. wait for all futures to complete #####################
  _ = [fut.result() for fut in futures]
  _sync_on_key(sync_key, "array_serialization")

  # 4. finalize the array writing ###########################
  if len(arr_leaf_ids) > 0 and _USE_OCDBT:
    _finalize_array_store(array_store_path, distinct_locations)
  # we are done with all async ops here, we can block ####
  _sync_on_key(sync_key, "end")


def _read_arrays(array_store_path: str | PathLike[str], arr_leaf_ids: list[int],
                 ts_specs: list[Any], shardings: list[Any]):
  # array_store_path = root / _LEAF_DATA_DIR / _ARRAY_STORE_DIRNAME
  arr_store_path = _norm_path(array_store_path)
  arr_paths = [arr_store_path / str(leaf_id) for leaf_id in arr_leaf_ids]

  # byte limiter to limit number of parallel reads, resizes to largest read
  byte_limiter = ts_impl._LimitInFlightBytes(10 * 1024 ** 3)  # 10 GB

  default_ts_specs = [ts_impl.get_tensorstore_spec(path, ocdbt=_USE_OCDBT,
                                                   process_idx=None)
                      for path in arr_paths]
  ts_specs = [ts_impl.merge_nested_ts_specs(default_ts_spec, ts_spec)
              for (default_ts_spec, ts_spec) in zip(default_ts_specs, ts_specs)]

  if len(ts_specs) > 0:  # verify the base path is shared for all arrays
    expected_path = ts_specs[0]["kvstore"]["base"]["path"]  # shared base path
    for ts_spec in ts_specs:
      ts_impl.verify_tensorstore_spec(ts_spec, arr=None, path=expected_path,
                                      ocdbt=_USE_OCDBT, check_metadata=False)

  async def _deserialize_arrays():
    return await asyncio.gather(*[
        ts_impl.async_deserialize(sharding, ts_spec, byte_limiter=byte_limiter)
        for (sharding, ts_spec) in zip(shardings, ts_specs)])

  return dict(zip(arr_leaf_ids, asyncio.run(_deserialize_arrays())))


def load_pytreedef(directory: str | PathLike[str]) -> PyTreeT:
  """Loads a pytree from the given directory.

  This is a simple experimental array serialization API, for anything more
  complex and for all checkpointing prefer: https://github.com/google/orbax

  Args:
    directory: Directory path to load from.
  Returns:
    The loaded pytree with arrays represented as jax.ShapeDtypeStruct's.
  """
  assert not _is_remote_path(directory) or pathlib.epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")
  json_content = (_norm_path(directory) / _PYTREEDEF_FILE).read_text()
  raw_tree = json.loads(json_content)
  leaves = map(_desc_to_leaf, raw_tree[utils._LEAF_IDS_KEY])
  return jax.tree.unflatten(utils.deserialize_pytreedef(raw_tree), leaves)


def load(directory: str | PathLike[str], shardings: PyTreeT, *,
         mask: PyTreeT | None = None, ts_specs: PyTreeT | None = None
         ) -> PyTreeT:
  """Loads and reconstructs a data structure from a directory.

  This is a simple experimental array serialization API, for anything more
  complex and for all checkpointing prefer: https://github.com/google/orbax

  Args:
    directory: Directory path where the data is stored.
    shardings: Sharding strategy for array objects. If None, defaults to
      single device sharding on the default device.
    mask: boolean prefix tree for partial loading, will return None for False
      leaves.
    ts_specs: Optional tensorstore specs to use for deserialization. If None,
      defaults to using the default tensorstore specs.

  Returns:
    Reconstructed data.

  Example:
    >>> save(data, directory)
    >>> restored_data = load(directory, SingleDeviceSharding(jax.devices()[0]))
  """
  assert not _is_remote_path(directory) or pathlib.epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")

  root = _norm_path(directory)
  assert root.is_dir(), f"Checkpoint directory {root} does not exist"
  is_leaf = lambda x: x is None

  # deserialize PyTreeDef
  pytree = load_pytreedef(directory)
  # broadcast the (prefix) shardings and tensorstore specs to the full pytree
  shardings = _tree_broadcast(shardings, pytree)
  ts_specs = _tree_broadcast(ts_specs, pytree,
                             is_leaf=ts_impl.is_tensorstore_spec_leaf)
  if mask is not None:
    _prefix_mask = lambda m, x: jax.tree.map(lambda _: None, x) if not m else x
    pytree = jax.tree.map(_prefix_mask, mask, pytree)
  pytreedef = jax.tree.structure(pytree, is_leaf=is_leaf)
  leaf_ids_flat = jax.tree.leaves(pytree, is_leaf=is_leaf)
  shardings_flat = jax.tree.leaves(shardings, is_leaf=is_leaf)
  ts_specs_flat = jax.tree.leaves(ts_specs,
                                  is_leaf=ts_impl.is_tensorstore_spec_leaf)

  # deserialize array objects
  arr_leaf_ids = [i for i, leaf_id in enumerate(leaf_ids_flat)
                  if leaf_id is not None]
  shardings_flat = [shardings_flat[i] for i in arr_leaf_ids]
  ts_specs_flat = [ts_specs_flat[i] for i in arr_leaf_ids]

  arrs_fut = _serialization_executor.submit(
      _read_arrays, root / _ARRAY_STORE_DIRNAME, arr_leaf_ids, ts_specs_flat,
      shardings_flat)

  arrs = arrs_fut.result()
  filled_values = [arrs.get(i, None) for i, _ in enumerate(leaf_ids_flat)]
  return jax.tree.unflatten(pytreedef, filled_values)


def nonblocking_save(data: PyTreeT, directory: str | PathLike[str], *,
                     overwrite: bool = True, ts_specs: PyTreeT | None = None
                     ) -> utils.PyTreeFuture:
  """Nonblocking alias of save, return an awaitable future with a pytree stub.

  This is a simple experimental array serialization API, for anything more
  complex and for all checkpointing prefer: https://github.com/google/orbax

  Examples:
    >>> fut = nonblocking_save(data, directory)
    >>> print(fut.pytree)  # a pytree of jax.ShapeDtypeStruct's
    >>> print(fut.result())  # None, blocking until the serialization is done
  """
  # start serialization immediately
  fut = utils.PyTreeFuture(_serialization_executor.submit(
      save, data, directory, overwrite=overwrite, ts_specs=ts_specs))
  # construct a nice looking pytree representing the nodes being read
  fut.pytree = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype)
                            if _is_array_like(x) else x, data)
  return fut


def nonblocking_load(directory: str | PathLike[str], shardings: PyTreeT, *,
                     mask: PyTreeT | None = None,
                     ts_specs: PyTreeT | None = None) -> utils.PyTreeFuture:
  """Nonblocking alias of load, return an awaitable future with a pytree stub.

  This is a simple experimental array serialization API, for anything more
  complex and for all checkpointing prefer: https://github.com/google/orbax

  Examples:
    >>> fut = nonblocking_load(directory)
    >>> print(fut.pytree)  # a pytree of jax.ShapeDtypeStruct
    >>> print(fut.result())  # the fully populated pytree
  """
  # TODO(rdyro): the awaitable future output is a workaround
  # it should return the fully populated pytree instead of just
  # jax.ShapeDtypeStruct for arrays by constructing them asynchronously
  fut = utils.PyTreeFuture(_serialization_executor.submit(
      load, directory, shardings, mask=mask, ts_specs=ts_specs))
  fut.pytree = load_pytreedef(directory)
  return fut
