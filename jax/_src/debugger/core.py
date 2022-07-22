# Copyright 2022 Google LLC
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

import dataclasses
import inspect
import threading

from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Protocol

import jax.numpy as jnp
from jax import core
from jax import tree_util
from jax._src import debugging
from jax._src import traceback_util
from jax._src import util
import numpy as np

@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class DebuggerFrame:
  """Encapsulates Python frame information."""
  filename: str
  locals: Dict[str, Any]
  code_context: str
  source: List[str]
  lineno: int
  offset: Optional[int]

  def tree_flatten(self):
    flat_locals, locals_tree = tree_util.tree_flatten(self.locals)
    is_valid = [
        isinstance(l, (core.Tracer, jnp.ndarray, np.ndarray))
        for l in flat_locals
    ]
    invalid_locals, valid_locals = util.partition_list(is_valid, flat_locals)
    return valid_locals, (is_valid, invalid_locals, locals_tree, self.filename,
                          self.code_context, self.source, self.lineno,
                          self.offset)

  @classmethod
  def tree_unflatten(cls, info, valid_locals):
    (is_valid, invalid_locals, locals_tree, filename, code_context, source,
     lineno, offset) = info
    flat_locals = util.merge_lists(is_valid, invalid_locals, valid_locals)
    locals_ = tree_util.tree_unflatten(locals_tree, flat_locals)
    return DebuggerFrame(filename, locals_, code_context, source, lineno,
                         offset)

  @classmethod
  def from_frameinfo(cls, frame_info) -> DebuggerFrame:
    try:
      _, start = inspect.getsourcelines(frame_info.frame)
      source = inspect.getsource(frame_info.frame).split('\n')
      offset = frame_info.lineno - start
    except OSError:
      source = []
      offset = None
    return DebuggerFrame(
        filename=frame_info.filename,
        locals=frame_info.frame.f_locals,
        code_context=frame_info.code_context,
        source=source,
        lineno=frame_info.lineno,
        offset=offset)


class Debugger(Protocol):

  def __call__(self, frames: List[DebuggerFrame], thread_id: Optional[int],
      **kwargs: Any) -> None:
    ...
_debugger_registry: Dict[str, Tuple[int, Debugger]] = {}


def get_debugger() -> Debugger:
  debuggers = sorted(_debugger_registry.values(), key=lambda x: -x[0])
  if not debuggers:
    raise ValueError("No debuggers registered!")
  return debuggers[0][1]


def register_debugger(name: str, debugger: Debugger, priority: int) -> None:
  if name in _debugger_registry:
    raise ValueError(f"Debugger with name \"{name}\" already registered.")
  _debugger_registry[name] = (priority, debugger)


debug_lock = threading.Lock()


def breakpoint(*, ordered: bool = False, **kwargs):  # pylint: disable=redefined-builtin
  """Enters a breakpoint at a point in a program."""
  frame_infos = inspect.stack()
  # Filter out internal frames
  frame_infos = [
      frame_info for frame_info in frame_infos
      if traceback_util.include_frame(frame_info.frame)
  ]
  frames = [
      DebuggerFrame.from_frameinfo(frame_info) for frame_info in frame_infos
  ]
  # Throw out first frame corresponding to this function
  frames = frames[1:]
  flat_args, frames_tree = tree_util.tree_flatten(frames)

  def _breakpoint_callback(*flat_args):
    frames = tree_util.tree_unflatten(frames_tree, flat_args)
    thread_id = None
    if threading.current_thread() is not threading.main_thread():
      thread_id = threading.get_ident()
    debugger = get_debugger()
    # Lock here because this could be called from multiple threads at the same
    # time.
    with debug_lock:
      debugger(frames, thread_id, **kwargs)

  if ordered:
    effect = debugging.DebugEffect.ORDERED_PRINT
  else:
    effect = debugging.DebugEffect.PRINT
  debugging.debug_callback(_breakpoint_callback, effect, *flat_args)
