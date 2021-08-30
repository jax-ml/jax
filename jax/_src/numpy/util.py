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

from collections import OrderedDict
import re
import textwrap
from typing import Callable, NamedTuple, Optional, Dict, Sequence

_parameter_break = re.compile("\n(?=[A-Za-z_])")
_section_break = re.compile(r"\n(?=[^\n]{3,15}\n-{3,15})", re.MULTILINE)
_numpy_signature_re = re.compile(r'^([\w., ]+=)?\s*[\w\.]+\([\w\W]*?\)$', re.MULTILINE)
_versionadded = re.compile(r'^\s+\.\.\s+versionadded::', re.MULTILINE)
_docreference = re.compile(r':doc:`(.*?)\s*<.*?>`')

class ParsedDoc(NamedTuple):
  """
  docstr: full docstring
  signature: signature from docstring.
  summary: summary from docstring.
  front_matter: front matter before sections.
  sections: dictionary of section titles to section content.
  """
  docstr: Optional[str]
  signature: str = ""
  summary: str = ""
  front_matter: str = ""
  sections: Dict[str, str] = OrderedDict()


def _parse_numpydoc(docstr: Optional[str]) -> ParsedDoc:
  """Parse a standard numpy-style docstring.

  Args:
    docstr: the raw docstring from a function
  Returns:
    ParsedDoc: parsed version of the docstring
  """
  if docstr is None or not docstr.strip():
    return ParsedDoc(docstr)

  # Remove any :doc: directives in the docstring to avoid sphinx errors
  docstr = _docreference.sub(
    lambda match: f"{match.groups()[0]}", docstr)

  signature, body = "", docstr
  match = _numpy_signature_re.match(body)
  if match:
    signature = match.group()
    body = docstr[match.end():]

  firstline, _, body = body.partition('\n')
  body = textwrap.dedent(body.lstrip('\n'))

  match = _numpy_signature_re.match(body)
  if match:
    signature = match.group()
    body = body[match.end():]

  summary = firstline
  if not summary:
    summary, _, body = body.lstrip('\n').partition('\n')
    body = textwrap.dedent(body.lstrip('\n'))

  front_matter = ""
  body = "\n" + body
  section_list = _section_break.split(body)
  if not _section_break.match(section_list[0]):
    front_matter, *section_list = section_list
  sections = OrderedDict((section.split('\n', 1)[0], section) for section in section_list)

  return ParsedDoc(docstr=docstr, signature=signature, summary=summary,
                   front_matter=front_matter, sections=sections)


def _parse_parameters(body: str) -> Dict[str, str]:
  """Parse the Parameters section of a docstring."""
  title, underline, content = body.split('\n', 2)
  assert title == 'Parameters'
  assert underline and not underline.strip('-')
  parameters = _parameter_break.split(content)
  return OrderedDict((p.partition(' : ')[0].partition(', ')[0], p) for p in parameters)


def _wraps(fun: Callable, update_doc: bool = True, lax_description: str = "",
           sections: Sequence[str] = ('Parameters', 'Returns', 'References'),
           skip_params: Sequence[str] = ()):
  """Specialized version of functools.wraps for wrapping numpy functions.

  This produces a wrapped function with a modified docstring. In particular, if
  `update_doc` is True, parameters listed in the wrapped function that are not
  supported by the decorated function will be removed from the docstring. For
  this reason, it is important that parameter names match those in the original
  numpy function.

  Args:
    fun: The function being wrapped
    update_doc: whether to transform the numpy docstring to remove references of
      parameters that are supported by the numpy version but not the JAX version.
      If False, include the numpy docstring verbatim.
    lax_description: a string description that will be added to the beginning of
      the docstring.
    sections: a list of sections to include in the docstring. The default is
      ["Parameters", "returns", "References"]
    skip_params: a list of strings containing names of parameters accepted by the
      function that should be skipped in the parameter list.
  """
  def wrap(op):
    docstr = getattr(fun, "__doc__", None)
    if docstr:
      try:
        parsed = _parse_numpydoc(docstr)

        if update_doc and hasattr(op, '__code__') and 'Parameters' in parsed.sections:
          # Remove unrecognized parameter descriptions.
          parameters = _parse_parameters(parsed.sections['Parameters'])
          parsed.sections['Parameters'] = (
            "Parameters\n"
            "----------\n" +
            "\n".join(_versionadded.split(desc)[0].rstrip() for p, desc in parameters.items()
                      if p in op.__code__.co_varnames and p not in skip_params)
          )

        docstr = parsed.summary.strip() + "\n" if parsed.summary else ""
        docstr += f"\nLAX-backend implementation of :func:`{fun.__name__}`.\n"
        if lax_description:
          docstr += "\n" + lax_description.strip() + "\n"
        docstr += "\n*Original docstring below.*\n"

        # We remove signatures from the docstrings, because they redundant at best and
        # misleading at worst: e.g. JAX wrappers don't implement all ufunc keyword arguments.
        # if parsed.signature:
        #   docstr += "\n" + parsed.signature.strip() + "\n"

        if parsed.front_matter:
          docstr += "\n" + parsed.front_matter.strip() + "\n"
        kept_sections = (content.strip() for section, content in parsed.sections.items()
                         if section in sections)
        if kept_sections:
          docstr += "\n" + "\n\n".join(kept_sections) + "\n"
      except:
        docstr = fun.__doc__

    op.__doc__ = docstr
    op.__np_wrapped__ = fun
    for attr in ['__name__', '__qualname__']:
      try:
        value = getattr(fun, attr)
      except AttributeError:
        pass
      else:
        setattr(op, attr, value)
    return op
  return wrap
