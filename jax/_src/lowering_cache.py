# Copyright 2026 The JAX Authors.
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

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import logging
import re
import sys
import threading
from typing import Any, Callable

from jax._src import config, core, stages, xla_bridge
from jax._src.cache_interface import CacheInterface, compress_bytes, compression_algorithm, decompress_bytes
from jax._src.lib import version_str as jaxlib_version_str
from jax._src.lru_cache import LRUCache
from jax.version import __version__ as jax_version

logger = logging.getLogger(__name__)

_cache: CacheInterface | None = None
_cache_initialized: bool = False
_cache_mutex = threading.Lock()


def custom_hook() -> str:
  return ""


def get_file_cache(path: str) -> tuple[CacheInterface, str] | None:
  return LRUCache(path, max_size=-1), path


def set_cache_dir(path: str) -> None:
  config.config.update("jax_lowering_cache_dir", path)


def get_cache_dir() -> str | None:
  return config.lowering_cache_dir.value


def is_cache_enabled() -> bool:
  return (
      config.enable_lowering_cache.value
      and (config.lowering_cache_dir.value is not None or _cache is not None)
      and core.trace_state_clean()
  )


def _get_cache() -> CacheInterface | None:
  global _cache, _cache_initialized
  if not is_cache_enabled():
    return None
  with _cache_mutex:
    if not _cache_initialized and _cache is None:
      _cache_initialized = True
      path = config.lowering_cache_dir.value
      if path is not None and config.enable_lowering_cache.value:
        res = get_file_cache(path)
        if res is not None:
          _cache, _ = res
  return _cache


def reset_cache() -> None:
  global _cache, _cache_initialized
  with _cache_mutex:
    _cache, _cache_initialized = None, False


def explain_cannot_cache(lowered: stages.Lowered) -> list[str]:
  """Returns an explanation list of all reasons why a lowered computation cannot be cached."""
  reasons = []
  lowering = getattr(lowered, "_lowering", None)
  if (
      lowering is None
      or getattr(type(lowering), "__name__", "") != "MeshComputation"
  ):
    reasons.append("non-MeshComputation lowering")

  ca = getattr(lowering, "compile_args", {}) if lowering is not None else {}
  if ca.get("unordered_effects"):
    reasons.append(f"contains unordered_effects: {ca.get('unordered_effects')}")
  if ca.get("ordered_effects"):
    reasons.append(f"contains ordered_effects: {ca.get('ordered_effects')}")
  if ca.get("host_callbacks"):
    reasons.append(f"contains host_callbacks: {ca.get('host_callbacks')}")
  if ca.get("keepalive"):
    reasons.append(f"contains keepalive: {ca.get('keepalive')}")
  if ca.get("mut"):
    reasons.append(f"contains mutable arguments: {ca.get('mut')}")

  return reasons


def can_cache(lowered: stages.Lowered, fun_name: str = "") -> bool:
  reasons = explain_cannot_cache(lowered)
  if reasons:
    if config.explain_cache_misses.value:
      name = fun_name or getattr(
          getattr(lowered, "_lowering", None), "_name", "lowered"
      )
      logger.warning(
          "Cannot cache lowered %s because: %s", name, "; ".join(reasons)
      )
    return False
  return True


def serialize_lowered(lowered: stages.Lowered, *, vjp_order: int = 0) -> bytes:
  from jax._src.export import _export  # pyrefly: ignore[missing-import]

  exported = _export._export_lowered(
      lowered,
      jaxpr=None,  # pyrefly: ignore[bad-argument-type]
      fun_name=getattr(lowered._lowering, "_name", "lowered"),
  )
  return bytes(exported.serialize(vjp_order=vjp_order))


def deserialize_lowered(data: bytes) -> stages.Lowered:
  from jax._src.export import _export  # pyrefly: ignore[missing-import]
  from jax._src.pjit import pjit as jit  # pyrefly: ignore[missing-import]

  exported = _export.deserialize(bytearray(data))
  in_args, in_kwargs = exported.in_tree.unflatten(exported.in_avals)
  with config.enable_lowering_cache(False):
    return jit(exported.call).lower(*in_args, **in_kwargs)


def get_cache_key(
    fun: Callable[..., Any],
    ji: Any,
    arg_signature: Any,
    avals: Sequence[core.AbstractValue],
    ctx_mesh: Any,
) -> str:
  h = hashlib.sha256()
  fun_name = getattr(fun, "__qualname__", getattr(fun, "__name__", str(fun)))
  h.update(
      f"{fun_name};{getattr(fun, '__module__', '')};{arg_signature}".encode()
  )
  code = getattr(fun, "__code__", None)
  if code is not None:
    h.update(
        code.co_code
        + str(code.co_consts).encode()
        + str(code.co_varnames).encode()
    )
  for a in avals:
    h.update(
        f"{getattr(a, 'shape', None)};{getattr(a, 'dtype', None)};{getattr(a, 'sharding', None)}"
        .encode()
    )
  if ctx_mesh is not None and not getattr(ctx_mesh, "empty", True):
    h.update(f"{ctx_mesh.shape};{ctx_mesh.axis_names}".encode())
  try:
    h.update(xla_bridge.get_backend().platform.encode())
  except Exception:
    pass
  if ji is not None:
    for attr in (
        "in_shardings_treedef",
        "out_shardings_treedef",
        "compiler_options_kvs",
    ):
      h.update(f"{attr}={getattr(ji, attr, None)};".encode())
  h.update(f"{jax_version};{jaxlib_version_str}".encode())
  for arg in sys.argv:
    h.update(arg.encode())
  hook = custom_hook()
  if hook:
    h.update(hook.encode())
  h.update(compression_algorithm().encode())
  return f"lowered_{re.sub(r'[^a-zA-Z0-9_]', '_', fun_name)}-{h.hexdigest()}"


def get_lowered(cache_key: str, fun_name: str = "") -> stages.Lowered | None:
  cache = _get_cache()
  if cache is None:
    return None
  data = cache.get(cache_key)
  if data is None:
    return None
  try:
    return deserialize_lowered(decompress_bytes(data))
  except Exception:
    if config.raise_persistent_cache_errors.value:
      raise
    return None


def put_lowered(
    cache_key: str, lowered: stages.Lowered, fun_name: str = ""
) -> None:
  cache = _get_cache()
  if cache is None:
    return
  try:
    cache.put(cache_key, compress_bytes(serialize_lowered(lowered)))
  except Exception as err:
    if config.explain_cache_misses.value:
      name = fun_name or getattr(
          getattr(lowered, "_lowering", None), "_name", "lowered"
      )
      logger.warning(
          "Failed to store lowering cache entry for %s: %s", name, err
      )
    if config.raise_persistent_cache_errors.value:
      raise
