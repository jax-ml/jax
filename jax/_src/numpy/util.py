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

import re
import textwrap


def update_numpydoc(docstr, fun, op):
  '''Transforms the numpy docstring to remove references of
     parameters that are supported by the numpy version but not the JAX version'''

  #Some numpy functions have an extra tab at the beginning of each line,
  #If this function is one of those we remove this extra tab from all the lines
  if not hasattr(op, '__code__'):
    return docstr
  if docstr[:4] == '    ':
    lines = docstr.split('\n')
    for idx, line in enumerate(lines):
      lines[idx] = line.replace('    ', '', 1)
    docstr = '\n'.join(lines)

  begin_idx = docstr.find("Parameters")
  begin_idx = docstr.find("--\n", begin_idx) + 2
  end_idx = docstr.find("Returns", begin_idx)

  parameters = docstr[begin_idx:end_idx]
  param_list = parameters.replace('\n    ', '@@').split('\n')
  for idx, p in enumerate(param_list):
    param = p[:p.find(' : ')].split(", ")[0]
    if param not in op.__code__.co_varnames:
      param_list[idx] = ''
  param_list = [param for param in param_list if param != '']
  parameters = '\n'.join(param_list).replace('@@', '\n    ')
  return docstr[:begin_idx + 1] + parameters + docstr[end_idx - 2:]

_numpy_signature_re = re.compile(r'^([\w., ]+=)?\s*[\w\.]+\([\w\W]*\)$')


def _wraps(fun, update_doc=True, lax_description=""):
  """Like functools.wraps but works with numpy.ufuncs.
     It is important that when wrapping numpy functions the parameters names
     in the original function and in the JAX version are the same
    Parameters:
      fun: The function being wrapped
      update_doc: whether to transform the numpy docstring to remove references of
      parameters that are supported by the numpy version but not the JAX version.
      If False, include the numpy docstring verbatim.
  """
  def wrap(op):
    if not hasattr(fun, '__doc__') or fun.__doc__ is None:
      return op
    try:
      # Numpy doc comments have the form:
      # fn(x, y, z)          (optional)
      #
      # A one-line summary
      #
      # ... everything else ...
      # We (a) move the summary to the top, since it is what the Sphinx
      # autosummary extension expects, and (b) add a comment below the summary
      # to the effect that this is a LAX wrapper of a Numpy function.

      # Dedent docstring, handling case where summary is on the first line.
      doc = fun.__doc__
      if doc[0].isspace():
        doc = textwrap.dedent(doc)
      else:
        firstline, rest = doc.split('\n', 1)
        doc = firstline + '\n' + textwrap.dedent(rest)
      sections = doc.split("\n\n")

      # We remove signatures from the docstrings, because they redundant at best and
      # misleading at worst: JAX wrappers don't implement all ufunc keyword arguments.
      for i, section in enumerate(sections):
        if not _numpy_signature_re.match(section):
          sections = sections[i:]
          break
      summary = sections[0].strip() if sections else ""
      body = "\n\n".join(sections[1:])

      if update_doc:
        body = update_numpydoc(body, fun, op)
      desc = lax_description.strip()

      desc = desc + "\n\n" if desc else ""
      summary = summary + "\n\n" if summary else ""
      docstr = (
          f"{summary}"
          f"LAX-backend implementation of :func:`{fun.__name__}`.\n\n"
          f"{desc}"
          f"*Original docstring below.*\n\n"
          f"{body}")

      op.__name__ = fun.__name__
      op.__qualname__ = fun.__qualname__
      op.__doc__ = docstr
      op.__np_wrapped__ = fun
    finally:
      return op
  return wrap
