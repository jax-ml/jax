# Copyright 2021 Google LLC
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

from docutils import nodes

def jax_issue_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
  """Generate links to jax issues or PRs in sphinx.

  Usage::

      :jax-issue:`1234`

  This will output a hyperlink of the form
  `#1234 <http://github.com/google/jax/issues/1234>`_. These links work even
  for PR numbers.
  """
  text = text.lstrip('#')
  if not text.isdigit():
      raise RuntimeError(f"Invalid content in {rawtext}: expected an issue or PR number.")
  url = "https://github.com/google/jax/issues/{}".format(text)
  node = nodes.reference(rawtext, '#' + text, refuri=url, **options)
  return [node], []

def setup(app):
  app.add_role('jax-issue', jax_issue_role)
