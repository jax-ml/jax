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
import itertools
import os.path
import threading
from typing import Optional, Iterator, NamedTuple

import jax.version
from jax._src.lib import xla_client

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)


Traceback = xla_client.Traceback
Frame = xla_client.Frame

_exclude_paths = [os.path.dirname(jax.version.__file__)]

def register_exclusion(path):
  _exclude_paths.append(path)

class SourceInfo(NamedTuple):
  traceback: Optional[Traceback]

  def replace(self, *, traceback: Optional[Traceback] = None) -> 'SourceInfo':
    traceback = traceback or self.traceback
    return self._replace(traceback=traceback)

def new_source_info() -> SourceInfo:
  return SourceInfo(None)

def user_frames(source_info: SourceInfo) -> Iterator[Frame]:
  """Heuristic that guesses the identity of the user's code in a stack trace."""
  # Guess the user's frame is the innermost frame not in the jax source tree
  # We don't use traceback_util.path_starts_with because that incurs filesystem
  # access, which may be slow; we call this function when e.g. adding source
  # provenance annotations to XLA lowerings, so we don't want to incur the cost.
  # We consider files that end with _test.py as user frames, to allow testing
  # this mechanism from tests.
  traceback = source_info.traceback
  return (x for x in (traceback.frames if traceback else [])
          if x.file_name.endswith("_test.py") or not any(x.file_name.startswith(p) for p in _exclude_paths))

def user_frame(source_info: SourceInfo) -> Optional[Frame]:
  return next(user_frames(source_info), None)

def summarize(source_info: SourceInfo, num_frames=1) -> str:
  frames = itertools.islice(user_frames(source_info), num_frames)
  frame_strs = [f"{frame.file_name}:{frame.line_num} ({frame.function_name})"
                if frame else "unknown" for frame in frames]
  return '\n'.join(reversed(frame_strs))

class _SourceInfoContext(threading.local):
  context: SourceInfo

  def __init__(self):
    self.context = new_source_info()

_source_info_context = _SourceInfoContext()

def current() -> SourceInfo:
  context = _source_info_context.context
  if not context.traceback:
    return context.replace(traceback=xla_client.Traceback.get_traceback())
  return context


class JaxStackTraceBeforeTransformation(Exception): pass

_message = (
    'The preceding stack trace is the source of the JAX operation that, once '
    'transformed by JAX, triggered the following exception.\n'
    '\n--------------------')

def has_user_context(e):
  while e is not None:
    if isinstance(e, JaxStackTraceBeforeTransformation):
      return True
    e = e.__cause__
  return False

@contextlib.contextmanager
def user_context(c: Optional[Traceback]):
  prev = _source_info_context.context
  _source_info_context.context = _source_info_context.context.replace(traceback=c)
  filtered_tb = None
  try:
    yield
  except Exception as e:
    if c is None or has_user_context(e):
      raise
    filtered_tb = traceback_util.filter_traceback(c.as_python_traceback())
    if filtered_tb:
      msg = traceback_util.format_exception_only(e)
      msg = f'{msg}\n\n{_message}'
      exp = JaxStackTraceBeforeTransformation(msg).with_traceback(filtered_tb)
      exp.__context__ = e.__context__
      exp.__cause__ = e.__cause__
      exp.__suppress_context__ = e.__suppress_context__
      e.__context__ = None
      e.__cause__ = exp
    raise
  finally:
    _source_info_context.context = prev
    del filtered_tb
