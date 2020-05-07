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
      sections = fun.__doc__.split("\n\n")

      signatures = []
      summary = None
      for i in range(len(sections)):
        if _numpy_signature_re.match(sections[i]):
          signatures.append(sections[i])
        else:
          summary = sections[i].strip()
          break
      body = "\n\n".join(signatures + sections[i + 1:])
      if update_doc:
        body = update_numpydoc(body, fun, op)
      desc = lax_description + "\n" if lax_description else ""
      docstr = (
          "{summary}\n\nLAX-backend implementation of :func:`{fun}`.\n"
          "{lax_description}Original docstring below.\n\n{body}"
          .format(summary=summary, lax_description=desc,
                  fun=fun.__name__, body=body))

      op.__name__ = fun.__name__
      op.__doc__ = docstr
    finally:
      return op
  return wrap
