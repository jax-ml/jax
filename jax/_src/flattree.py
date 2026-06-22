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
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, partial
import itertools as it

from jax._src import tree
from jax._src import util
from jax._src.util import (Either, unzip2, safe_map, safe_zip)
from jax._src import tree_util

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

hole = util.Singleton("_")

# TODO: eventually all our internal APIs whould work with flattree-accepting
# functions and we shouldn't need to use these conversion functions much.
def fun_ft_to_pt(f_ft):
  def f_pt(*arg_pts): return f_ft(*map(flatten, arg_pts)).unflatten()
  return f_pt

def fun_pt_to_ft(f_pt):
  def f_ft(*arg_fts): return flatten(f_pt(*(x.unflatten() for x in arg_fts)))
  return f_ft

def flatten(pytree):
  if type(pytree) is tuple:
    return FTTuple(map(flatten, pytree))
  else:
    xs, treedef = tree.flatten(pytree)
    return FTPyTree(xs, treedef)
def flatten_list(xs): return FTTuple(map(FTSingleton, xs))

def flatten_args(*args): return flatten_args_and_kwargs(args)

def flatten_args_and_kwargs(args, kwargs={}, static_argnums=(), static_argnames=()):
  def handle_arg(statics, i, arg):
    return Either.left(arg) if i in statics else Either.right(flatten(arg))
  args_ = tuple(handle_arg(static_argnums, i, arg) for i, arg in enumerate(args))
  kwargs_ = tuple((k, handle_arg(static_argnames, k, arg))
                  for k, arg in sorted(kwargs.items()))
  return FTArgsAndKwargs(args_, *unzip2(kwargs_))

def pack(*trees): return FTTuple(trees)
def pack2(*trees): return FTTuple(FTTuple(t) for t in trees)
def pack_args_and_kwargs(args_ft):
  args_ft = tuple(args_ft)
  assert all(isinstance(arg, FlatTree) for arg in args_ft)
  return FTArgsAndKwargs(tuple(Either.right(arg_ft) for arg_ft in args_ft), (), ())

class FlatTree:
  """FlatTree is a Python OOP version of this functor:

    data FlatTree a = Tuple [FlatTree a]
                    | WithAux (FlatTree a) Aux
                    | List [a]
                    | Filtered [a] (FlatTree (Either Hole Aux))
                    | Pytree [a] PyTreeDef
                    | ArgsAndKwargs [Either Aux (FlatTree a)]
                                    {Either Aux (FlatTree a)]
  """
  def unpack(self): raise TypeError(f"Not a FlatTree tuple: {self}")
  def from_list(self): raise TypeError(f"Not a FlatTree list: {self}")
  def unflatten(self): raise TypeError(f"Not a flattened PyTree: {self}")
  def unfilter(self): raise TypeError(f"Not a filtered FlatTree: {self}")
  def map(self, f): return self.update(map(f, self))
  def map2(self, ys, f): return self.update(map(f, self, ys))
  def map3(self, ys, zs, f): return self.update(map(f, self, ys, zs))
  def unzip2(self): return self.map(lambda x: x[0]), self.map(lambda x: x[1])
  def with_aux(self, aux): return FTWithAux(self, aux)
  def update(self, xs):
    xs = list(xs)
    assert len(xs) == len(self)
    xs_iter = iter(xs)
    return self._iter_update(xs_iter)

  def filter(self, keeps): return self.map2(keeps, lambda x, k: (x, k))._filter()
  def _filter(self):
    kept = [x for x, keep in self if keep]
    put_aside = self.map(lambda x_keep: hole if x_keep[1] else x_keep[0])
    return FTFiltered(kept, put_aside)
  @property # TODO: remove this shim
  def tree(self): return self.treedef

  @property
  def paths(self) -> FlatTree:
    # TODO(dougalm): find a way to do this without roundtripping
    try:
      paths, _ = unzip2(self.registry.flatten_with_path(self.unflatten())[0])
      assert len(paths) == len(self.xs)
      return self.update(paths)
    except:
      return self.update([()] * len(self))  # not our fault

  def __eq__(self, other): assert False, "subclass should implement"
  def __hash__(self): assert False, "subclass should implement"

class FTTuple(FlatTree):
  def __init__(self, trees):
    trees = trees if isinstance(trees, tuple) else tuple(trees)
    for t in trees: assert isinstance(t, FlatTree)
    self.trees = trees
  def unflatten(self): return tuple(t.unflatten() for t in self.trees)
  @cached_property
  def treedef(self): return tree_util.treedef_tuple(t.treedef for t in self.trees)
  def unpack(self): return self.trees
  def _filter(self): return FTTuple(t._filter() for t in self.trees)
  def unfilter(self): return FTTuple(t.unfilter() for t in self.trees)
  def __iter__(self): return (x for tree in self.trees for x in tree)
  def __len__(self): return sum(len(t) for t in self.trees)
  def _iter_update(self, xs_iter):
    return FTTuple(tuple(t._iter_update(xs_iter) for t in self.trees))
  def __repr__(self): return repr(self.trees)
  def __eq__(self, other): return isinstance(other, FTTuple) and self.trees == other.trees
  def __hash__(self): return hash(self.trees)

class FTWithAux(FlatTree):
  def __init__(self, ft, aux):
    assert isinstance(ft, FlatTree)
    self.ft = ft
    self.aux = aux
  def unpack_aux(self): return self.ft, self.aux
  def __iter__(self): return iter(self.ft)
  def __len__(self): return len(self.ft)
  def _iter_update(self, xs_iter):
    return FTWithAux(self.ft._iter_update(xs_iter), self.aux)
  def __repr__(self): return f"WithAux(ft={self.ft}, aux={self.aux})"
  def __eq__(self, other):
    return (isinstance(other, FTWithAux) and
            self.ft == other.ft and self.aux == other.aux)
  def __hash__(self): return hash((self.xs, self.aux))

class FTSingleton(FlatTree):
  def __init__(self, x): self.x = x
  def from_list(self): return list(self.xs)
  def unflatten(self): return self.x
  def __iter__(self): return iter([self.x])
  def __len__(self): return 1
  def _iter_update(self, xs_iter): return FTSingleton(next(xs_iter))
  def __repr__(self): return repr(self.x)
  def __eq__(self, other): return isinstance(other, FTSingleton) and self.x == other.x
  def __hash__(self): return hash(self.x)

class FTFiltered(FlatTree):
  def __init__(self, xs, ft_statics):
    assert isinstance(ft_statics, FlatTree)
    xs = xs if isinstance(xs, tuple) else tuple(xs)
    self.xs = xs
    self.ft_statics = ft_statics
  def unfilter(self):
    xs_iter = iter(self.xs)
    return self.ft_statics.map(lambda s: next(xs_iter) if s is hole else s)
  def __iter__(self): return iter(self.xs)
  def __len__(self): return len(self.xs)
  def _iter_update(self, xs_iter):
    return FTFiltered(it.islice(xs_iter, len(self.xs)), self.ft_statics)
  def __repr__(self): return f"Filtered(vals={self.xs}, ft={self.ft_statics})"
  def __eq__(self, other):
    return (isinstance(other, FTFiltered) and
            self.xs == other.xs and self.ft_statics == other.ft_statics)
  def __hash__(self): return hash((self.xs, self.ft_statics))

class FTPyTree(FlatTree):
  def __init__(self, xs, treedef):
    assert isinstance(treedef, tree_util.PyTreeDef)
    xs = xs if isinstance(xs, tuple) else tuple(xs)
    self.xs = xs
    self.treedef = treedef

  def unflatten(self):
    return tree_util.tree_unflatten(self.treedef, self.xs)
  def __iter__(self): return iter(self.xs)
  def __len__(self): return len(self.xs)
  def _iter_update(self, xs_iter):
    return FTPyTree(it.islice(xs_iter, len(self.xs)), self.treedef)
  def __repr__(self): return f"Pytree(vals={list(self.xs)}, tree={self.treedef})"
  def __eq__(self, other):
    return (isinstance(other, FTPyTree) and
            self.xs == other.xs and self.treedef == other.treedef)
  def __hash__(self): return hash((self.xs, self.treedef))

class FTArgsAndKwargs(FlatTree):
  args : tuple[Either[Any, FlatTree[T]], ...]
  kwarg_keys : tuple[Any, ...]
  kwarg_vals : tuple[Either[Any, FlatTree[T]], ...]

  def __init__(self, args, kwarg_keys, kwarg_vals):
    # static args as vals (left); dynamic args as avals (right)
    self.args = args
    self.kwarg_keys = kwarg_keys
    self.kwarg_vals = kwarg_vals

  def _iter_update(self, xs_iter):
    def doit(x):
      return x if x.is_left else Either.right(x.from_right()._iter_update(xs_iter))
    return FTArgsAndKwargs(
        tuple(map(doit, self.args)),
        self.kwarg_keys,
        tuple(map(doit, self.kwarg_vals)))

  def unflatten(self):
    def doit(x): return x.from_left() if x.is_left else x.from_right().unflatten()
    return (tuple(map(doit, self.args)),
            {k : doit(v) for k, v in zip(self.kwarg_keys, self.kwarg_vals)})

  @cached_property
  def vals(self):
    return [val for arg in it.chain(self.args, self.kwarg_vals)
            if arg.is_right for val in arg.from_right()]

  def __len__(self): return len(self.vals)
  def __iter__(self): return iter(self.vals)
  def __eq__(self, other): return (
      isinstance(other, FTArgsAndKwargs) and
      self.args == other.args and
      self.kwarg_keys == other.kwarg_keys and
      self.kwarg_vals == other.kwarg_vals)
  def __hash__(self): return hash((self.args, self.kwarg_keys, self.kwarg_vals))

  # TODO: revise this away
  @cached_property
  def tree(self): return self.tree_without_statics

  # TODO: revise this away
  @cached_property
  def tree_without_statics(self):
    args_tree = tree_util.treedef_tuple(
        ft.from_right().treedef for ft in self.args if ft.is_right)
    # TOOD: better way to do this? dict version of treedef_tuple?
    _, kwargs_tree = tree_util.tree_flatten(
        {k : v.from_right().map(lambda _: 0).unflatten()
         for k, v in zip(self.kwarg_keys, self.kwarg_vals)
         if v.is_right})
    return tree_util.treedef_tuple((args_tree, kwargs_tree))

  def bitvector(self, argnums, argnames):
    bits = []
    def handle_arg(i_chosen, i_arg):
      i, arg = i_arg
      if arg.is_right: bits.extend((i in i_chosen,) * len(arg.from_right()))
    map(partial(handle_arg, argnums) , enumerate(self.args))
    map(partial(handle_arg, argnames), zip(self.kwarg_keys, self.kwarg_vals))
    return tuple(bits)
