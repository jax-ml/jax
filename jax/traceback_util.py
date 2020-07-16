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

import os
import sys
import traceback
import types

from .api_util import wraps

_jax_path = os.path.dirname(__file__)
_include_paths = [
    os.path.join(_jax_path, path) for path in (
        'config.py', 'dlpack.py', 'experimental', 'lax', 'lax_linalg.py',
        'lax_reference.py', 'nn', 'numpy', 'ops', 'profiler.py', 'random.py',
        'scipy', 'test_util.py', 'third_party', 'tools',
    )]

_jax_message_append = (
    'The stack trace above excludes JAX-internal frames.\n'
    'The following is the original exception that occurred, unmodified.\n'
    '\n--------------------')

def include_frame(f):
  return (not f.f_code.co_filename.startswith(_jax_path) or
          any(f.f_code.co_filename.startswith(path) for path in _include_paths))

# When scanning stack traces, we might encounter frames from cpython that are
# removed from printed stack traces, such as frames from parts of importlib. We
# ignore these frames heuristically based on source and name match.
def ignore_known_hidden_frame(f):
  return 'importlib._bootstrap' in f.f_code.co_filename

def filter_traceback_and_stack(e):
  out = None

  # Scan the traceback and collect relevant frames.

  for f, lineno in reversed(list(traceback.walk_tb(e.__traceback__))):
    if include_frame(f):
      out = types.TracebackType(out, f, f.f_lasti, lineno)  # pytype: disable=wrong-arg-count

  # Continue up the call stack.
  #
  # We would like to avoid stepping too far up, e.g. past the exec/eval point of
  # a REPL such as IPython. To that end, we stop past the first contiguous bunch
  # of module-level frames, if we reach any such frames at all. This is a
  # heuristic that might stop in advance of the REPL boundary. For example, if
  # the call stack includes module-level frames from the current module A, and
  # the current module A was imported from within a function F elsewhere, then
  # the stack trace we produce will be truncated at F's frame.

  reached_module_level = False
  for f, lineno in traceback.walk_stack(e.__traceback__.tb_frame):
    if ignore_known_hidden_frame(f):
      continue
    if reached_module_level and f.f_code.co_name != '<module>':
      break
    if include_frame(f):
      out = types.TracebackType(out, f, f.f_lasti, lineno)  # pytype: disable=wrong-arg-count
    if f.f_code.co_name == '<module>':
      reached_module_level = True

  return out

def is_reraiser_frame(f):
  return (f.filename == __file__ and
          f.name == 'reraise_with_filtered_traceback')

def is_under_reraiser(e):
  tb = traceback.extract_stack(e.__traceback__.tb_frame)
  return any(is_reraiser_frame(f) for f in tb[:-1])

def format_exception_only(e):
  return ''.join(traceback.format_exception_only(type(e), e)).strip()

def last_cause(e):
  prev, cur = e, e.__cause__
  while cur is not None:
    prev, cur = cur, cur.__cause__
  return prev

class FilteredStackTrace(Exception): pass

def filtered_tracebacks_supported():
  return sys.version_info >= (3, 7)

def api_boundary(fun):
  if not filtered_tracebacks_supported():
    return fun

  @wraps(fun)
  def reraise_with_filtered_traceback(*args, **kwargs):
    try:
      return fun(*args, **kwargs)
    except Exception as e:
      if not is_under_reraiser(e):
        filtered_tb = filter_traceback_and_stack(e)
        if filtered_tb:
          msg = format_exception_only(e)
          msg = f'{msg}\n\n{_jax_message_append}'
          filtered = FilteredStackTrace(msg).with_traceback(filtered_tb)
          cause = last_cause(e)
          cause.__cause__ = filtered
          raise
        else:
          raise
      else:
        raise
  return reraise_with_filtered_traceback
