from collections import namedtuple
from functools import partial
from itertools import count
from threading import Thread
from Queue import Queue

import numpy as onp

from jax.util import unzip2


# utils

new_queue = partial(Queue, maxsize=10)

def spawn(fun):
  thread = Thread(target=fun)
  thread.daemon = True  # exit when main thread exits
  thread.start()
  return thread

def curry(f):
  return partial(partial, f)

def memoize_on_id(fun):
  cache = {}
  def memoized_fun(x):
    key = id(x)
    if key in cache:
      val, _ = cache[key]
    else:
      val, _ = cache[key] = (fun(x), x)
    return val
  return memoized_fun


# combinators

@curry
def producer(fun, q_out):
  def produce():
    while True:
      q_out.put(fun())
  spawn(produce)

@curry
def producer_consumer(fun, q_in, q_out):
  def produce_consume():
    while True:
      q_out.put(fun(q_in.get()))
  spawn(produce_consume)

@curry
def consumer(fun, q_in):
  def consume():
    while True:
      fun(q_in.get())
  spawn(consume)

def round_robin_splitter(q_in, qs_out):
  def round_robin_split():
    while True:
      for q_out in qs_out:
        q_out.put(q_in.get())
  spawn(round_robin_split)

def fan_in_concat(qs_in, q_out):
  def fan_in_concat():
    while True:
      q_out.put([q_in.get() for q_in in qs_in])
  spawn(fan_in_concat)

@curry
def parallel(makers, qs_in, qs_out):
  assert len(funs) == len(qs_in) == len(qs_out)
  map(makers, qs_in, qs_out)

@curry
def source_pipeline(makers, q_out):
  qs = [new_queue() for _ in makers[:-1]] + [q_out]
  makers[0](qs[0])
  for maker, q_in, q_out in zip(makers[1:], qs[:-1], qs[1:]):
    maker(q_in, q_out)


# using the combinators directly

if __name__ == '__main__':
  counter = count().next
  square = lambda x: x**2

  make_source = source_pipeline([producer(counter), producer_consumer(square)])
  stream = new_queue()
  make_source(stream)

  for _ in range(10):
    print stream.get()
  print


# ast constructors for a daxpr language

Source = SourceLiteral = namedtuple('Source', ['pyfun'])
Sink = SinkLiteral = namedtuple('Sink', ['pyfun'])
ProducerConsumer = ProducerConsumerLiteral = namedtuple('ProducerConsumer', ['pyfun'])

FanOutRoundRobin = namedtuple('FanOutRoundRobin', ['arity'])
FanInConcatenate = namedtuple('FanInConcatenate', ['arity'])

Pipeline = namedtuple('Pipeline', ['daxprs'])
Parallel = namedtuple('Parallel', ['daxprs'])


# a typer

DaxprType = namedtuple('DaxprType', ['arity_in', 'arity_out'])

literals = {
    SourceLiteral: lambda _: DaxprType(0, 1),
    SinkLiteral: lambda _: DaxprType(1, 0),
    ProducerConsumerLiteral: lambda _: DaxprType(1, 1),
    FanOutRoundRobin: lambda daxpr: DaxprType(1, daxpr.arity),
    FanInConcatenate: lambda daxpr: DaxprType(daxpr.arity, 1),
}

@memoize_on_id
def type_daxpr(daxpr):
  s = type(daxpr)  # syntax analysis
  if s in literals:
    return literals[s](daxpr)
  else:
    assert daxpr.daxprs  # TODO raise syntax error in ast constructors
    daxpr_types = map(type_daxpr, daxpr.daxprs)
    ins, outs = unzip2((dt.arity_in, dt.arity_out) for dt in daxpr_types)
    if s is Parallel:
      if not all(ain == aout == 1 for ain, aout in zip(ins, outs)):
        raise DaxprTypeError("parallel combination of multi-arity daxprs")
      return DaxprType(sum(ins), sum(outs))
    elif s is Pipeline:
      if ins[1:] != outs[:-1]:
        raise DaxprTypeError("pipeline arity mismatch: {}".format(daxpr_types))
      return DaxprType(ins[0], outs[-1])
    else:
      raise DaxprSyntaxError(s)

class DaxprTypeError(TypeError): pass
class DaxprSyntaxError(SyntaxError): pass


# an interpreter

def build_threaded(daxpr):
  dt = type_daxpr(daxpr)
  arity_in, arity_out = dt.arity_in, dt.arity_out
  qs_in = [new_queue() for _ in range(arity_in)]
  qs_out = [new_queue() for _ in range(arity_out)]
  _build_threaded(daxpr, qs_in, qs_out)

  # convenience unpacking
  qs_in = qs_in[0] if len(qs_in) == 1 else qs_in
  qs_out = qs_out[0] if len(qs_out) == 1 else qs_out
  if arity_in == 0:
    return qs_out
  elif arity_out == 0:
    return qs_in
  else:
    return qs_in, qs_out

def _build_threaded(daxpr, qs_in, qs_out):
  s = type(daxpr)  # syntax analysis
  if s is SourceLiteral:
    q_out, = qs_out
    producer(daxpr.pyfun)(q_out)
  elif s is SinkLiteral:
    q_in, = qs_in
    consumer(daxpr.pyfun)(q_in)
  elif s is ProducerConsumerLiteral:
    q_in, = qs_in
    q_out, = qs_out
    producer_consumer(daxpr.pyfun)(q_in, q_out)
  elif s is FanOutRoundRobin:
    q_in, = qs_in
    round_robin_splitter(q_in, qs_out)
  elif s is FanInConcatenate:
    q_out, = qs_out
    fan_in_concat(qs_in, q_out)
  else:
    daxpr_types = map(type_daxpr, daxpr.daxprs)
    ins, outs = unzip2((dt.arity_in, dt.arity_out) for dt in daxpr_types)
    if s is Parallel:
      for d, q_in, q_out in zip(daxpr.daxprs, qs_in, qs_out):
        _build_threaded(d, [q_in], [q_out])
    elif s is Pipeline:
      qs = [[new_queue() for _ in range(arity)] for arity in ins[1:]]
      map(_build_threaded, daxpr.daxprs, [qs_in] + qs, qs + [qs_out])
    else:
      assert False  # unreachable, syntax errors should be caught earlier


# using the language

if __name__ == '__main__':

  # ast constructors
  daxpr = Pipeline([
      Source(count().next),
      ProducerConsumer(square),
      ProducerConsumer(onp.sqrt),
  ])

  # typer
  print type_daxpr(daxpr)
  print

  # interpreter
  q = build_threaded(daxpr)

  for _ in range(10):
    print q.get()
  print


  # another example
  daxpr = Pipeline([
      Source(count().next),
      ProducerConsumer(square),
      FanOutRoundRobin(2),
      Parallel([ProducerConsumer(lambda x: x), ProducerConsumer(onp.sqrt)]),
      FanInConcatenate(2),
  ])

  q = build_threaded(daxpr)

  for _ in range(10):
    print q.get()
  print
