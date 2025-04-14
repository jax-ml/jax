# ---
# Copyright 2025 The JAX Authors.
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
#
# ---

from absl.testing import absltest
from absl.testing import parameterized



from core import make_jaxpr, f32, jit
import ops

class CoreTest(parameterized.TestCase):
  def test_make_jaxpr(self):
    def foo(x): return x * (x + 3.0)
    expected = (
      "(v1:f32[],) =>\n"
      "  v2:f32[] = add(v1:f32[], 3.0:f32[])\n"
      "  v3:f32[] = mul(v1:f32[], v2:f32[])\n"
      "  return v3:f32[]\n")
    jaxpr = make_jaxpr(foo, (f32[()],))
    assert str(jaxpr) == expected

  def test_jit(self):
    @jit
    def f(x):
      return x + x

    assert len(f.cache) == 0
    assert f(1.0) == 2.0
    assert len(f.cache) == 1
    assert f(1) == 2
    assert len(f.cache) == 2
    assert f(2.0) == 4.0


# # === Testing it out ===

# def foo(x):
#   return x * (x + 3.0)

# # print(add(1.0, 2.0))
# # print(foo(2.0))
# japxr = make_jaxpr(foo, (f32[()],))
# print(japxr)
# # eval_jaxpr(jaxpr, (2.0,))


# m = MutableArray(1.0)

# def doit(m):
#   m[...] = 2.0
#   return m[...]

# japxr = make_jaxpr(doit, (MutableArrayTy(f32[()]),))
# print(japxr)
# print(m.val)
# eval_jaxpr(jaxpr, (m,))
# print(m.val)

