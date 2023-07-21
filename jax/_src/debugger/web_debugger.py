# Copyright 2022 The JAX Authors.
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

import os
from typing import Any
import weakref

from jax._src.debugger import cli_debugger
from jax._src.debugger import core as debugger_core

web_pdb_version: tuple[int, ...] | None = None
try:
  import web_pdb  # pytype: disable=import-error
  web_pdb_version = tuple(map(int, web_pdb.__version__.split(".")))
  WEB_PDB_ENABLED = True
except:
  WEB_PDB_ENABLED = False


_web_consoles: dict[tuple[str, int], web_pdb.WebConsole] = {}

class WebDebugger(cli_debugger.CliDebugger):
  """A web-based debugger."""
  prompt = '(jdb) '
  use_rawinput: bool = False

  def __init__(self, frames: list[debugger_core.DebuggerFrame], thread_id,
               completekey: str = "tab", host: str = "", port: int = 5555):
    if (host, port) not in _web_consoles:
      _web_consoles[host, port] = web_pdb.WebConsole(host, port, self)
    # Clobber the debugger in the web console
    _web_console = _web_consoles[host, port]
    _web_console._debugger = weakref.proxy(self)
    super().__init__(frames, thread_id, stdin=_web_console, stdout=_web_console,
                     completekey=completekey)

  def get_current_frame_data(self):
    # Constructs the info needed for the web console to display info
    current_frame = self.current_frame()
    filename = current_frame.filename
    lines = current_frame.source
    current_line = None
    if current_frame.offset is not None:
      current_line = current_frame.offset + 1
    if web_pdb_version and web_pdb_version < (1, 4, 4):
      return {
        'filename': filename,
        'listing': '\n'.join(lines),
        'curr_line': current_line,
        'total_lines': len(lines),
        'breaklist': [],
      }
    return {
        'dirname': os.path.dirname(os.path.abspath(filename)) + os.path.sep,
        'filename': os.path.basename(filename),
        'file_listing': '\n'.join(lines),
        'current_line': current_line,
        'breakpoints': [],
        'globals': self.get_globals(),
        'locals': self.get_locals(),
    }

  def get_globals(self):
    current_frame = self.current_frame()
    globals = "\n".join([f"{key} = {value}" for key, value in
      sorted(current_frame.globals.items())])
    return globals

  def get_locals(self):
    current_frame = self.current_frame()
    locals = "\n".join([f"{key} = {value}" for key, value in
      sorted(current_frame.locals.items())])
    return locals

  def run(self):
    return self.cmdloop()

def run_debugger(frames: list[debugger_core.DebuggerFrame],
                 thread_id: int | None, **kwargs: Any):
  WebDebugger(frames, thread_id, **kwargs).run()

if WEB_PDB_ENABLED:
  debugger_core.register_debugger("web", run_debugger, -2)
