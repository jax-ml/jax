from collections import namedtuple
from functools import partial
from itertools import count
from threading import Thread
from Queue import Queue

import numpy as onp


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
def source_pipeline(makers, q_out):
  qs = [new_queue() for _ in makers[:-1]] + [q_out]
  makers[0](qs[0])
  for maker, q_in, q_out in zip(makers[1:], qs[:-1], qs[1:]):
    maker(q_in, q_out)

@curry
def sink_pipeline(makers, q_in):
  raise NotImplementedError

@curry
def producer_consumer_pipeline(makers, q_in, q_out):
  raise NotImplementedError

@curry
def parallel(makers, qs_in, qs_out):
  assert len(funs) == len(qs_in) == len(qs_out)
  map(makers, qs_in, qs_out)


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

FanOutRoundRobin = namedtuple('FanOutRoundRobin', [])()
FanInConcatenate = namedtuple('FanInConcatenate', [])()

Pipeline = namedtuple('Pipeline', ['daxprs'])
Parallel = namedtuple('Parallel', ['daxprs'])


# a typer

class DaxprType(object):
  def __init__(self, name):
    self.name =  name
  def __repr__(self):
    return self.name

SourceT = DaxprType('SourceT')
SinkT = DaxprType('SinkT')
ProducerConsumerT = DaxprType('ProducerConsumerT')

literals = {
  SourceLiteral: SourceT,
  SinkLiteral: SinkT,
  ProducerConsumerLiteral: ProducerConsumerT,
}

@memoize_on_id
def type_daxpr(daxpr):
  t = type(daxpr)  # syntax analysis
  if t in literals:
    return literals[t]
  elif t is Pipeline:
    types = map(type_daxpr, daxpr.daxprs)
    if all(d is ProducerConsumerT for d in types):
      return ProducerConsumerT
    elif types[0] is SourceT and all(d is ProducerConsumerT for d in types[1:]):
      return SourceT
    elif types[-1] is SinkT and all(d is ProducerConsumerT for d in types[:-1]):
      return SinkT
    else:
      raise DaxprTypeError(types)
  elif t is Parallel:
    raise NotImplementedError
  else:
    raise DaxprSyntaxError(t)

class DaxprTypeError(TypeError): pass
class DaxprSyntaxError(SyntaxError): pass


# an interpreter

def build_threaded(daxpr):
  t = type_daxpr(daxpr)
  if t is SourceT:
    return build_source(daxpr, new_queue())
  elif t is SinkT:
    return build_sink(daxpr, new_queue())
  elif t is ProducerConsumerT:
    return build_producer_consumer(daxpr, new_queue(), new_queue())

def build_source(daxpr, q_out):
  t = type(daxpr)
  if t is SourceLiteral:
    producer(daxpr.pyfun)(q_out)
    return q_out
  elif t is Pipeline:
    qs = [new_queue() for _ in daxpr.daxprs[:-1]] + [q_out]
    build_source(daxpr.daxprs[0], qs[0])
    map(build_producer_consumer, daxpr.daxprs[1:], qs[:-1], qs[1:])
    return q_out
  else:
    raise NotImplementedError

def build_sink(daxpr, q_in):
  raise NotImplementedError

def build_producer_consumer(daxpr, q_in, q_out):
  t = type(daxpr)
  if t is ProducerConsumerLiteral:
    producer_consumer(daxpr.pyfun)(q_in, q_out)
    return q_in, q_out
  else:
    raise NotImplementedError


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
