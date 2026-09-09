# Copyright 2026 The JAX Authors.
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
import io
import json
import os
import tempfile
import unittest

from ci.postprocess import xml2json


class Xml2JsonTest(unittest.TestCase):

  def _process_bazel_xml(self, xml):
    with tempfile.NamedTemporaryFile(
        "w", prefix="tests__example_test__", suffix=".xml", delete=False
    ) as f:
      f.write(xml)
      path = f.name

    try:
      output = io.StringIO()
      with contextlib.redirect_stdout(output):
        xml2json.process_xml(path, 123, "2026-06-20T00:00:00+00:00")
      records = [json.loads(line) for line in output.getvalue().splitlines()]
      return records[0] if len(records) == 1 else records
    finally:
      os.unlink(path)

  def testBazelErrorElementIsReportedAsError(self):
    record = self._process_bazel_xml("""\
<testsuite>
  <testcase name="x">
    <error message="boom">stack trace</error>
  </testcase>
</testsuite>
""")

    self.assertEqual(record["status"], "ERROR")
    self.assertEqual(record["message"], "boom")
    self.assertEqual(record["detail"], "stack trace")

  def testBazelSkippedElementIsReportedAsSkipped(self):
    record = self._process_bazel_xml("""\
<testsuite>
  <testcase name="x">
    <skipped message="ignored"/>
  </testcase>
</testsuite>
""")

    self.assertEqual(record["status"], "SKIPPED")
    self.assertEqual(record["message"], "ignored")


if __name__ == "__main__":
  unittest.main()
