# ---
# Copyright 2024 The JAX Authors.
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

import builtins
from contextlib import contextmanager

# === fix python3 regressions ===

def map(f, *xs):
  return list(builtins.map(f, *xs))

def zip(*args):
  fst, *rest = args = map(list, args)
  n = len(fst)
  for arg in rest:
    assert len(arg) == n
  return list(builtins.zip(*args))

def unzip2(pairs):
  lst1, lst2 = [], []
  for x1, x2 in pairs:
    lst1.append(x1)
    lst2.append(x2)
  return lst1, lst2

# === pretty printer ===

class PrettyPrinter:
  def __init__(self):
    self.fragments = []
    self.cur_indent = ''

  # string shouldn't contain newline characters
  def emit(self, s):
    self.fragments.append((self.cur_indent, s))

  @contextmanager
  def indent(self, new_indent='  '):
    old_indent = self.cur_indent
    self.cur_indent = old_indent + new_indent
    yield
    self.cur_indent = old_indent

  def build_str(self):
    return '\n'.join(indent + s for indent, s in self.fragments)

  @staticmethod
  def to_str(x) -> str:
    p = PrettyPrinter()
    x.pretty_print(p)
    return p.build_str()

def arglist_str(xs):
  args = ', '.join(map(str, xs))
  return f'({args})'
