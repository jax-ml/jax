# Copyright 2018 Google LLC
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
import re
import subprocess
import sys
import tempfile

import nbformat

from jax import test_util as jtu
from jax.config import config
config.parse_flags_with_absl()


class NotebooksTest(jtu.JaxTestCase):
  """Tests the notebooks."""

  def _execute_notebook(self,
                        notebook_path,
                        expected_errors=None):
    """Execute a notebook, check for errors.

    Params:
      notebook_path: the path to the .ipynb notebook to execute, relative to
        the "jax" project root.
      expected_errors: a list of expected errors. Each element of the list
        is a pair of a pattern present in the cell source and
        a pattern present in the expected error output. The patterns
        are searched as regular expressions.

    Returns:
      asserts that exactly the expected errors are found.
    """
    jax_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected_errors = expected_errors if expected_errors else []
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as notebook_out:
      args = ["jupyter", "nbconvert",
              "--to", "notebook", "--execute",
              "--ExecutePreprocessor.timeout=60",
              "--allow-errors",
              "--output", notebook_out.name, os.path.join(jax_path, notebook_path)]
      subprocess.check_call(args)

      notebook_out.seek(0)
      nb = nbformat.read(notebook_out, nbformat.current_nbformat)

    # Keep a dict with the errors we are still expecting. Indexed by the
    # pair of (cell_pattern, error_pattern). Value is the error message.
    still_expecting_errors = {
      expected:
        "Did not find error matching '{}' in cell matching '{}".format(expected[1],
                                                                       expected[0])
      for expected in expected_errors}
    error_messages = []  # Collect error messages for unexpected errors
    for cell in nb.cells:
      if "outputs" in cell:
        for output in cell["outputs"]:
          if output.output_type == "error":
            # Found an error
            for (cell_pattern, error_pattern) in expected_errors:
              if (re.search(cell_pattern, cell.source, re.MULTILINE) and
                  re.search(error_pattern, output.evalue, re.MULTILINE)):
                # This was an expected error
                del still_expecting_errors[(cell_pattern, error_pattern)]
                break
            else:
              error_messages.append("Unexpected error in cell (execution count {}):\n{}\n\ngot error:\n{}\n".format(
                cell.execution_count,
                cell.source,
                output.evalue))

    if still_expecting_errors:
      error_messages.extend(still_expecting_errors.values())

    self.assertMultiLineEqual("", "\n".join(error_messages))

  def test_quickstart_notebook(self):
    self._execute_notebook(os.path.abspath("docs/notebooks/quickstart.ipynb"))

  def test_common_gotchas_notebook(self):
    self._execute_notebook(
      os.path.abspath("docs/notebooks/Common_Gotchas_in_JAX.ipynb"),
      expected_errors=[
        ("In place update of JAX's array will yield an error",
         "JAX arrays are immutable")
      ])

  def test_autodiff_cookbook_notebook(self):
    self._execute_notebook(
      os.path.abspath("docs/notebooks/autodiff_cookbook.ipynb"),
      expected_errors=[])