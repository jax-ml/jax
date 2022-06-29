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

import cmd
import dataclasses
import inspect
import sys
import threading
import traceback

from typing import Any, Callable, Dict, IO, List, Optional

import numpy as np
from jax import core
from jax import tree_util
from jax._src import debugging
from jax._src import traceback_util
from jax._src import util
import jax.numpy as jnp


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
    with debug_lock:
      TextDebugger(frames, thread_id, **kwargs).run()

  if ordered:
    effect = debugging.DebugEffect.ORDERED_PRINT
  else:
    effect = debugging.DebugEffect.PRINT
  debugging.debug_callback(_breakpoint_callback, effect, *flat_args)


class TextDebugger(cmd.Cmd):
  """A text-based debugger."""
  prompt = '(jaxdb) '
  use_rawinput: bool = False

  def __init__(self, frames: List[DebuggerFrame], thread_id,
      stdin: Optional[IO[str]] = None, stdout: Optional[IO[str]] = None,
      completekey: str = "tab"):
    super().__init__(stdin=stdin, stdout=stdout, completekey=completekey)
    self.frames = frames
    self.frame_index = 0
    self.thread_id = thread_id
    self.intro = 'Entering jaxdb:'

  def current_frame(self):
    return self.frames[self.frame_index]

  def evaluate(self, expr):
    env = {}
    curr_frame = self.frames[self.frame_index]
    env.update(curr_frame.locals)
    return eval(expr, {}, env)

  def print_backtrace(self):
    self.stdout.write('Traceback:\n')
    for frame in self.frames[::-1]:
      self.stdout.write(f'  File "{frame.filename}", line {frame.lineno}\n')
      if frame.offset is None:
        self.stdout.write('    <no source>\n')
      else:
        line = frame.source[frame.offset]
        self.stdout.write(f'    {line}\n')

  def print_context(self, num_lines=2):
    curr_frame = self.frames[self.frame_index]
    self.stdout.write(f'> {curr_frame.filename}({curr_frame.lineno})\n')
    for i, line in enumerate(curr_frame.source):
      assert curr_frame.offset is not None
      if (curr_frame.offset - 1 - num_lines <= i <=
          curr_frame.offset + num_lines):
        if i == curr_frame.offset:
          self.stdout.write(f'->  {line}\n')
        else:
          self.stdout.write(f'    {line}\n')

  def do_p(self, arg):
    try:
      self.stdout.write(repr(self.evaluate(arg)) + "\n")
    except Exception:
      traceback.print_exc(limit=1)

  def do_u(self, arg):
    if self.frame_index == len(self.frames) - 1:
      self.stdout.write('At topmost frame.\n')
    else:
      self.frame_index += 1
    self.print_context()

  def do_d(self, arg):
    if self.frame_index == 0:
      self.stdout.write('At bottommost frame.\n')
    else:
      self.frame_index -= 1
    self.print_context()

  def do_l(self, arg):
    self.print_context(num_lines=5)

  def do_ll(self, arg):
    self.print_context(num_lines=5)

  def do_c(self, arg):
    return True

  def do_EOF(self, arg):
    sys.exit(0)

  def do_bt(self, arg):
    self.print_backtrace()

  def run(self):
    while True:
      try:
        self.cmdloop()
        break
      except KeyboardInterrupt:
        self.stdout.write('--KeyboardInterrupt--\n')
