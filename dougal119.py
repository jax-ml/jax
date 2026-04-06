from __future__ import annotations
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.tree_util import FlatTree, tree_flatten, tree_unflatten
from jax._src.hijax import MutableHiType, QDD, register_hitype, HiPrimitive, box_effect, LoVal
from jax._src.util import safe_map as map, safe_zip as zip
log_effect = box_effect

def log_extend(log, dct):
  leaves, treedef = tree_flatten(dct)
  log_extend_p.bind(log, *leaves, treedef=treedef)

def log_append(log, key, val):
  log_extend(log, {key: [val]})

class Log:
  _dct: dict  # dict[str, list[PyTree[Array]]]
  __qdd__: LogQDD  # only used at lowering time

  def __init__(self):
    self._dct = {}

  def cur_qdd(self):
    return LogQDD(FlatTree.flatten(self._dct).map(core.typeof))

  append = log_append
  extend = log_extend

  def __repr__(self) -> str:
    return f'Log({self._dct})'

class LogTy(MutableHiType):
  has_qdd = True
  is_writer = True

  append = core.aval_method(log_append)
  extend = core.aval_method(log_extend)

  def __hash__(self): return hash(Log)
  def __eq__(self, other): return isinstance(other, Log)
  def str_short(self, short_dtypes=False, **_) -> str: return 'Log'

  def lo_ty_qdd(self, state: LogQDD, /) -> list[core.AbstractValue]:
    return list(state.ft)

  def preallocate(self, state: LogQDD) -> list[LoVal]:
    return [jax.lax.empty(a.shape, a.dtype) for a in state.ft]

  def filter(self, cur_qdd: LogQDD, update_qdd: LogQDD, log: Log) -> list[LoVal]:
    dct = {}
    qdd = cur_qdd.ft.unflatten()
    for k, v in update_qdd.ft.unflatten().items():
      i = len(qdd.get(k, []))
      dct[k] = log._dct[k][i:i+len(v)]
    return list(FlatTree.flatten(dct))

  def new_empty(self, state, *lo_vals) -> Log:
    log = Log()
    update_dct = state.ft.update(lo_vals).unflatten()
    for k, v in update_dct.items():
      log._dct[k] = v
    return log

  def empty_qdd(self) -> LogQDD:
    return LogQDD(FlatTree.flatten({}))

  def extend_qdd(self, qdd1, qdd2):
    dct = qdd1.ft.unflatten()
    update_dct = qdd2.ft.unflatten()
    for k, v in update_dct.items():
      dct.setdefault(k, []).extend(v)
    return LogQDD(FlatTree.flatten(dct), qdd1.in_scope)

  def read_loval(self, _state: LogQDD, log: Log) -> list[LoType]:
    return list(FlatTree.flatten(log._dct))

  def update_from_loval(self, update_qdd: LogQDD, log: Log, *lo_vals) -> None:
    updates = update_qdd.ft.update(lo_vals).unflatten()
    if hasattr(log, '__qdd__'):  # HTLV/lowering-time Log, preallocated, dus
      cur_qdd = log.__qdd__.ft.unflatten()
      for k, v in updates.items():
        sl = slice(len(cur_qdd.setdefault(k, [])), len(cur_qdd[k]) + len(v))
        log._dct[k][sl] = [u for x, u in zip(log._dct[k][sl], v)]
      log.__qdd__ = self.extend_qdd(log.__qdd__, update_qdd)
    else:  # at-rest Log, not preallocated, assign allocated
      for k, v in updates.items():
        log._dct.setdefault(k, []).extend(v)


register_hitype(Log, lambda _: LogTy())

@dataclass(frozen=True)
class LogQDD(QDD):
  ft: FlatTree  # FlatTree of dict[str, list[PyTree[Aval]]]
  in_scope: bool = False

  def __repr__(self):
    return f'LogQDD({repr(self.ft.unflatten())})'
  def inc_rank(self, length):
    return LogQDD(self.ft.map(partial(core.unmapped_leading_aval, length)))

class LogExtend(HiPrimitive):
  multiple_results = True  # no results
  is_effectful = lambda *_, **__: True

  def abstract_eval(self, log_ty, *val_tys, treedef):
    cur_qdd = log_ty.mutable_qdd.cur_val
    dct = cur_qdd.ft.unflatten()
    new_dct = tree_unflatten(treedef, val_tys)
    for k, v in new_dct.items():
      for length in reversed(core.scan_env() or ()):
        v = map(partial(core.unmapped_leading_aval, length), v)
      dct.setdefault(k, []).extend(v)
    log_ty.mutable_qdd.update(LogQDD(FlatTree.flatten(dct), cur_qdd.in_scope))
    return [], {log_effect}

  def to_lojax(_, log, *vals, treedef):
    updates = tree_unflatten(treedef, vals)
    if hasattr(log, '__qdd__'):
      qdd = log.__qdd__.ft.unflatten()
      idx = core.scan_env()
      for k, v in updates.items():
        sl = slice(len(qdd.setdefault(k, [])), len(qdd[k]) + len(v))
        log._dct[k][sl] = [x.at[idx].set(u) for x, u in zip(log._dct[k][sl], v)]
        qdd[k].extend(map(core.typeof, v))  # qdd update
      log.__qdd__ = LogQDD(FlatTree.flatten(qdd), log.__qdd__.in_scope)
    else:  # top-level
      for k, v in updates.items():
        log._dct.setdefault(k, []).extend(v)
    return []
log_extend_p = LogExtend('log_extend')

class NewLog(HiPrimitive):
  def is_high(self) -> bool: return True

  def abstract_eval(self):
    ty = LogTy()
    qdd = replace(ty.empty_qdd(), in_scope=True)
    return core.AvalQDD(ty, qdd), {log_effect}

  def to_lojax(_):
    return Log()
new_log_p = NewLog('new_log')

def new_log():
  return new_log_p.bind()


class ReadLog(HiPrimitive):
  multiple_results = True

  def is_high(self, _) -> bool: return True

  def abstract_eval(self, log_qdd):
    if not log_qdd.mutable_qdd.cur_val.in_scope: raise Exception
    return list(log_qdd.mutable_qdd.cur_val.ft), {log_effect}

  def to_lojax(_, log):
    return list(FlatTree.flatten(log._dct))
read_log_p = ReadLog('read_log')

def read_log(log):
  flat = read_log_p.bind(log)
  return core.cur_qdd(log).ft.update(flat).unflatten()


##

# l = Log()

# @jax.jit
# def f(l, x):
#   l.append('x', x + 1)

#   @jax.jit
#   def g():
#     l.append('x', x + 2)
#     l.append('x', x + 3)

#   g()
#   g()

# f(l, 0)
# print(l._dct)


# l = Log()
# def body(_, x):
#   l.append('x', x + 1)
#   l.append('x', 2 * x)
#   l.append('x', 2 * x + 1)
#   l.append('x', x + 10)
#   l.append('x', x + 20)
#   return (), ()
# (), () = jax.lax.scan(body, (), jnp.arange(3))
# print(l._dct)

# l = Log()
# def body(_, x):
#   l.append('x', x + 1)
#   jax.jit(lambda: l.append('x', 2 * x) or l.append('x', 2 * x + 1))()
#   l.append('x', x + 10)
#   jax.jit(jax.jit(lambda: l.append('x', x + 20)))()
#   return (), ()
# (), () = jax.lax.scan(body, (), jnp.arange(3))
# print(l._dct)


# def loop(*ns):
#   def wrap(f):
#     for n in reversed(ns):
#       f = (lambda f, n: lambda: jax.lax.scan(lambda _, __: f() or ((), ()), (), (), length=n))(f, n)
#     f()
#   return wrap

# from jax._src.hijax import Box
# l = Log()
# b = Box(0)

# @loop(3, 5)
# def f():
#   i = b.get()
#   l.append('x', i)
#   jax.jit(lambda: l.append('y', i * 2))()
#   b.set(i + 1)
# print(l._dct)



@jax.jit
def f(x):
  log = new_log()
  log.append('x', x + 1)

  @jax.jit
  def g():
    try:
      read_log(log)
    except:
      print('good!')
    else:
      raise Exception
    # log.append('x', x + 2)
    # log.append('x', x + 3)

  print(core.cur_qdd(log))
  g()
  g()
  return read_log(log)

print(f(0))
