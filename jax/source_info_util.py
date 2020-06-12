# Copyright 2020 Google LLC
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

import contextlib
import inspect
import os.path
import threading
from typing import List, NamedTuple, Optional

class Frame(NamedTuple):
  file_name: str
  line_num: int
  function_name: str

SourceInfo = List[Frame]


def _get_stacktrace() -> List[Frame]:
  return [Frame(f.filename, f.lineno, f.function) for f in inspect.stack(0)]


_jax_path = os.path.dirname(_get_stacktrace()[0].file_name)

def user_frame(source_info: Optional[SourceInfo]) -> Optional[Frame]:
  """Heuristic that guesses the identity of the user's code in a stack trace."""
    # Guess that the user's frame is the innermost stack frame that isn't in the
    # jax source tree.
  return next((x for x in (source_info or [])
               if not x.file_name.startswith(_jax_path)), None)


def summarize(source_info: Optional[SourceInfo]) -> str:
  frame = user_frame(source_info)
  return (f"{frame.file_name}:{frame.line_num} ({frame.function_name})"
          if frame else "unknown")


class _SourceInfoContext(threading.local):
  context: Optional[SourceInfo]

  def __init__(self):
    self.context = None

_source_info_context = _SourceInfoContext()

def current() -> Optional[SourceInfo]:
  return _source_info_context.context or _get_stacktrace()

@contextlib.contextmanager
def user_context(c):
  prev = _source_info_context.context
  _source_info_context.context = c or _source_info_context.context
  try:
    yield
  finally:
    _source_info_context.context = prev
