# Copyright 2021 The JAX Authors.
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

from docutils import nodes, utils
from docutils.parsers.rst import Directive

from sphinx.util.nodes import split_explicit_title

def jax_issue_role(name, rawtext, text, lineno, inliner, options=None,
                   content=()):
  """Generate links to jax issues or PRs in sphinx.

  Usage::

      :jax-issue:`1234`

  This will output a hyperlink of the form
  `#1234 <http://github.com/jax-ml/jax/issues/1234>`_. These links work even
  for PR numbers.
  """
  text = text.lstrip('#')
  if not text.isdigit():
      raise RuntimeError(f"Invalid content in {rawtext}: expected an issue or PR number.")
  options = {} if options is None else options
  url = f"https://github.com/jax-ml/jax/issues/{text}"
  node = nodes.reference(rawtext, '#' + text, refuri=url, **options)
  return [node], []

def doi_role(typ, rawtext, text, lineno, inliner, options=None, content=()):
  text = utils.unescape(text)
  has_explicit_title, title, part = split_explicit_title(text)
  full_url = 'https://doi.org/' + part
  if not has_explicit_title:
    title = 'DOI:' + part
  pnode = nodes.reference(title, title, internal=False, refuri=full_url)
  return [pnode], []


class LegacyDirective(Directive):
  """A noop variant of the ``legacy`` directive from SciPy."""
  has_content = True
  node_class = nodes.admonition
  optional_arguments = 1

  def run(self):
    return []


def setup(app):
  app.add_directive("legacy", LegacyDirective)
  app.add_role('jax-issue', jax_issue_role)
  app.add_role('doi', doi_role)
  return {
    'version': 1.0,
    'parallel_read_safe': True,
    'parallel_write_safe': True,
  }
