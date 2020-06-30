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

import functools
import operator as op


class PrettyPrint:
  """Crude Hughes-inspired pretty printer."""

  def __init__(self, lines):
    self.lines = lines

  def indent(self, indent):
    return PrettyPrint([(indent + orig_indent, s)
                        for orig_indent, s in self.lines])

  def annotate(self, length, msg):
    (i, s), *rest = self.lines
    return PrettyPrint([(i, s.ljust(length) + f" [{msg}]")] + list(rest))

  def __add__(self, rhs):
    return PrettyPrint(self.lines + rhs.lines)

  def __rshift__(self, rhs):
    if not rhs.lines:
      return self
    if not self.lines:
      return rhs

    indent, s = self.lines[-1]
    indented_block = rhs.indent(indent + len(s))
    common_line = s + ' ' * rhs.lines[0][0] + rhs.lines[0][1]
    return PrettyPrint(self.lines[:-1]
                       + [(indent, common_line)]
                       + indented_block.lines[1:])

  def __str__(self):
    return '\n'.join(' ' * indent + s for indent, s in self.lines)


def pp(s):
  return PrettyPrint([(0, line) for line in str(s).splitlines()])

def hcat(ps):
  return functools.reduce(op.rshift, ps)

def vcat(ps):
  return sum(ps, pp(''))
