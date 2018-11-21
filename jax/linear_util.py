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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .util import curry, partial

def thunk(f):
  store = Store()
  def f_memoized():
    if not store:
      # TODO(dougalm): save/restore relevant environment state too
      store.store(f())
    return store.val

  return f_memoized

class StoreException(Exception): pass

class Store(object):
  def store(self, val):
    assert not self, "Store occupied"
    self._val = val

  @property
  def val(self):
    if not self:
      raise StoreException("Store empty")
    return self._val

  def __nonzero__(self):
    return hasattr(self, '_val')

  __bool__ = __nonzero__


@curry
def staged(f, *init_args):
  store = Store()
  def f_partial(*rest):
    ans, aux = f(*(init_args + rest))
    store.store(aux)
    return ans

  f_partial.__name__ = f.__name__ + "_staged"
  return f_partial, thunk(lambda: store.val)


class WrappedFun(object):
  def __init__(self, f, transforms, kwargs):
    self.f = f
    self.transforms = transforms
    self.kwargs = kwargs

  def wrap(self, *transformation):
    return WrappedFun(self.f, [transformation] + self.transforms, self.kwargs)

  def populate_stores(self, other):
    for (_, _, self_store), (_, _, other_store) in zip(self.transforms,
                                                       other.transforms):
      if self_store is not None:
        self_store.store(other_store.val)

  def call_wrapped(self, *args):
    stack = []
    for gen, gen_args, out_store in self.transforms:
      gen = gen(*(gen_args + tuple(args)))
      args = next(gen)
      stack.append((gen, out_store))

    del gen
    ans = self.f(*args, **self.kwargs)
    del args
    while stack:
      gen, out_store = stack.pop()
      ans = gen.send(ans)
      if out_store is not None:
        ans, side = ans
        out_store.store(side)

    return ans

  def __repr__(self):
    def transform_to_str(x):
      i, (gen, args, _) = x
      return "{}   : {}   {}".format(i, fun_name(gen), fun_name(args))
    transformation_stack = map(transform_to_str, enumerate(self.transforms))
    return "Wrapped function:\n" + '\n'.join(transformation_stack) + '\nCore: ' + fun_name(self.f) + '\n'

  def hashable_payload(self):
    return (self.f,
            tuple((gen, tuple(gen_args)) for gen, gen_args, _ in self.transforms),
            tuple(sorted(self.kwargs.items())))

  def __hash__(self):
    return hash(self.hashable_payload())

  def __eq__(self, other):
    return self.hashable_payload() == other.hashable_payload()

@curry
def transformation(gen, fun, *transformation_args):
  return fun.wrap(gen, transformation_args, None)

@curry
def transformation_with_aux(gen, fun, *transformation_args):
  out_store = Store()
  out_thunk = lambda: out_store.val
  return fun.wrap(gen, transformation_args, out_store), out_thunk

def fun_name(f):
  try:
    return f.__name__
  except:
    return str(f)

def wrap_init(f, kwargs={}):
  return WrappedFun(f, [], kwargs)


def memoize(call):
  cache = {}
  def memoized_fun(f, *args):
    key = (f, args)
    if key in cache:
      ans, f_prev = cache[key]
      f.populate_stores(f_prev)
    else:
      ans = call(f, *args)
      cache[key] = (ans, f)
    return ans

  return memoized_fun
