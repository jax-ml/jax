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

"""Helpers for `mhlo.frontend_attributes` in TPU lowerings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from jax._src.lib.mlir import ir

_FRONTEND_ATTRIBUTES = "mhlo.frontend_attributes"


def attributes(
    params: Any, query: Callable[[Any], Mapping[str, str]]
) -> dict[str, str]:
  """Returns `query(params)` as a str to str dict, or `{}` for `None` params."""
  if params is None:
    return {}
  attrs = dict(query(params))
  if not all(
      isinstance(k, str) and isinstance(v, str) for k, v in attrs.items()
  ):
    raise TypeError(
        f"{type(params).__name__} frontend attributes must map str to str,"
        f" got {attrs!r}."
    )
  return attrs


def frontend_attributes(op: ir.Operation) -> dict[str, str]:
  """Returns the `mhlo.frontend_attributes` of `op` as a dict."""
  if (attrs := op.attributes.get(_FRONTEND_ATTRIBUTES)) is None:
    return {}
  return {a.name: ir.StringAttr(a.attr).value for a in ir.DictAttr(attrs)}


def attach_frontend_attributes(
    op: ir.Operation, attrs: Mapping[str, str]
) -> None:
  """Merges `attrs` into the frontend attributes of `op`; `attrs` wins."""
  if not attrs:
    return
  merged: dict[str, ir.Attribute] = {}
  if (existing := op.attributes.get(_FRONTEND_ATTRIBUTES)) is not None:
    merged |= {a.name: a.attr for a in ir.DictAttr(existing)}
  merged |= {k: ir.StringAttr.get(v) for k, v in attrs.items()}
  op.attributes[_FRONTEND_ATTRIBUTES] = ir.DictAttr.get(merged)


def attach_missing_frontend_attributes(
    op: ir.Operation, attrs: Mapping[str, str]
) -> None:
  """Merges `attrs` into the frontend attributes of `op`; `op` wins."""
  have = frontend_attributes(op)
  attach_frontend_attributes(
      op, {k: v for k, v in attrs.items() if k not in have}
  )
