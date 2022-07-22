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
"""Module for Colab-specific debugger."""
from __future__ import annotations

import html
import inspect
import traceback

from typing import List

import uuid

from jax._src.debugger import colab_lib
from jax._src.debugger import core as debugger_core
from jax._src.debugger import cli_debugger

# pylint: disable=g-import-not-at-top
# pytype: disable=import-error
if colab_lib.IS_COLAB_ENABLED:
  from google.colab import output
try:
  import pygments
  IS_PYGMENTS_ENABLED = True
except ImportError:
  IS_PYGMENTS_ENABLED = False
# pytype: enable=import-error
# pylint: enable=g-import-not-at-top


class CodeViewer(colab_lib.DynamicDOMElement):
  """A mutable DOM element that displays code as HTML."""

  def __init__(self, code_: str, highlights: List[int], linenostart: int = 1):
    self._code = code_
    self._highlights = highlights
    self._view = colab_lib.dynamic(colab_lib.div())
    self._linenostart = linenostart

  def render(self):
    self.update_code(
        self._code, self._highlights, linenostart=self._linenostart)

  def clear(self):
    self._view.clear()

  def append(self, child):
    raise NotImplementedError

  def update(self, elem):
    self._view.update(elem)

  def _highlight_code(self, code: str, highlights, linenostart: int):
    is_dark_mode = output.eval_js(
        'document.documentElement.matches("[theme=dark]");')
    code_style = "monokai" if is_dark_mode else "default"
    hl_color = "#4e56b7" if is_dark_mode else "#fff7c1"
    if IS_PYGMENTS_ENABLED:
      lexer = pygments.lexers.get_lexer_by_name("python")
      formatter = pygments.formatters.HtmlFormatter(
          full=False,
          hl_lines=highlights,
          linenos=True,
          linenostart=linenostart,
          style=code_style)
      if hl_color:
        formatter.style.highlight_color = hl_color
      css_ = formatter.get_style_defs()
      code = pygments.highlight(code, lexer, formatter)
    else:
      return "";
    return code, css_

  def update_code(self, code_, highlights, *, linenostart: int = 1):
    """Updates the code viewer to use new code."""
    self._code = code_
    self._view.clear()
    code_, css_ = self._highlight_code(self._code, highlights, linenostart)
    uuid_ = uuid.uuid4()
    code_div = colab_lib.div(
        colab_lib.css(css_),
        code_,
        id=f"code-{uuid_}",
        style=colab_lib.style({
            "max-height": "500px",
            "overflow-y": "scroll",
            "background-color": "var(--colab-border-color)",
            "padding": "5px 5px 5px 5px",
        }))
    if highlights:
      percent_scroll = highlights[0] / len(self._code.split("\n"))
    else:
      percent_scroll = 0.
    self.update(code_div)
    # Scroll to where the line is
    output.eval_js("""
    console.log("{id}")
    var elem = document.getElementById("{id}")
    var maxScrollPosition = elem.scrollHeight - elem.clientHeight;
    elem.scrollTop = maxScrollPosition * {percent_scroll}
    """.format(id=f"code-{uuid_}", percent_scroll=percent_scroll))


class FramePreview(colab_lib.DynamicDOMElement):
  """Displays information about a stack frame."""

  def __init__(self, frame):
    super().__init__()
    self._header = colab_lib.dynamic(
        colab_lib.div(colab_lib.pre(colab_lib.code(""))))
    self._code_view = CodeViewer("", highlights=[])
    self.frame = frame
    self._file_cache = {}

  def clear(self):
    self._header.clear()
    self._code_view.clear()

  def append(self, child):
    raise NotImplementedError

  def update(self, elem):
    raise NotImplementedError

  def update_frame(self, frame):
    """Updates the frame viewer to use a new frame."""
    self.frame = frame
    lineno = self.frame.lineno or None
    filename = self.frame.filename.strip()
    if inspect.getmodulename(filename):
      if filename not in self._file_cache:
        try:
          with open(filename, "r") as fp:
            self._file_cache[filename] = fp.read()
          source = self._file_cache[filename]
          highlight = lineno
          linenostart = 1
        except FileNotFoundError:
          source = "\n".join(frame.source)
          highlight = min(frame.offset + 1, len(frame.source) - 1)
          linenostart = lineno - frame.offset
    else:
      source = "\n".join(frame.source)
      highlight = min(frame.offset + 1, len(frame.source) - 1)
      linenostart = lineno - frame.offset
    self._header.clear()
    self._header.update(
        colab_lib.div(
            colab_lib.pre(colab_lib.code(f"{html.escape(filename)}({lineno})")),
            style=colab_lib.style({
                "padding": "5px 5px 5px 5px",
                "background-color": "var(--colab-highlighted-surface-color)",
            })))
    self._code_view.update_code(source, [highlight], linenostart=linenostart)

  def render(self):
    self.update_frame(self.frame)


class DebuggerView(colab_lib.DynamicDOMElement):
  """Main view for the Colab debugger."""

  def __init__(self, frame, *, log_color=""):
    super().__init__()
    self._interaction_log = colab_lib.dynamic(colab_lib.div())
    self._frame_preview = FramePreview(frame)
    self._header = colab_lib.dynamic(
        colab_lib.div(
            colab_lib.span("Breakpoint"),
            style=colab_lib.style({
                "background-color": "var(--colab-secondary-surface-color)",
                "color": "var(--colab-primary-text-color)",
                "padding": "5px 5px 5px 5px",
                "font-weight": "bold",
            })))

  def render(self):
    self._header.render()
    self._frame_preview.render()
    self._interaction_log.render()

  def append(self, child):
    raise NotImplementedError

  def update(self, elem):
    raise NotImplementedError

  def clear(self):
    self._header.clear()
    self._interaction_log.clear()
    self._frame_preview.clear()

  def update_frame(self, frame):
    self._frame_preview.update_frame(frame)

  def log(self, text):
    self._interaction_log.append(colab_lib.pre(text))

  def read(self):
    with output.use_tags(["stdin"]):
      user_input = input()
    output.clear(output_tags=["stdin"])
    return user_input


class ColabDebugger(cli_debugger.CliDebugger):
  """A JAX debugger for a Colab environment."""

  def __init__(self,
               frames: List[debugger_core.DebuggerFrame],
               thread_id: int):
    super().__init__(frames, thread_id)
    self._debugger_view = DebuggerView(self.current_frame())

  def read(self):
    return self._debugger_view.read()

  def cmdloop(self, intro=None):
    self.preloop()
    stop = None
    while not stop:
      if self.cmdqueue:
        line = self.cmdqueue.pop(0)
      else:
        try:
          line = self.read()
        except EOFError:
          line = "EOF"
      line = self.precmd(line)
      stop = self.onecmd(line)
      stop = self.postcmd(stop, line)
    self.postloop()

  def do_u(self, _):
    if self.frame_index == len(self.frames) - 1:
      self.log("At topmost frame.")
      return False
    self.frame_index += 1
    self._debugger_view.update_frame(self.current_frame())
    return False

  def do_d(self, _):
    if self.frame_index == 0:
      self.log("At bottommost frame.")
      return False
    self.frame_index -= 1
    self._debugger_view.update_frame(self.current_frame())
    return False

  def do_bt(self, _):
    self.log("Traceback:")
    for frame in self.frames[::-1]:
      filename = frame.filename.strip()
      filename = filename or "<no filename>"
      self.log(f"  File: {filename}, line ({frame.lineno})")
      if frame.offset < len(frame.source):
        line = frame.source[frame.offset]
        self.log(f"    {line.strip()}")
      else:
        self.log(" ")

  def do_c(self, _):
    return True

  def do_q(self, _):
    return True

  def do_EOF(self, _):
    return True

  def do_p(self, arg):
    try:
      value = self.evaluate(arg)
      self.log(repr(value))
    except Exception:  # pylint: disable=broad-except
      self.log(traceback.format_exc(limit=1))
    return False

  do_pp = do_p

  def log(self, text):
    self._debugger_view.log(html.escape(text))

  def run(self):
    self._debugger_view.render()
    try:
      self.cmdloop()
    except KeyboardInterrupt:
      self.log("--Keyboard-Interrupt--")
      pass
    self._debugger_view.clear()


def _run_debugger(frames, thread_id, **kwargs):
  try:
    ColabDebugger(frames, thread_id, **kwargs).run()
  except Exception:
    traceback.print_exc()


if colab_lib.IS_COLAB_ENABLED:
  debugger_core.register_debugger("colab", _run_debugger, 1)
