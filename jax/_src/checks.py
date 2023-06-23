import contextlib

checks = frozenset({})
@contextlib.contextmanager
def instrument(new_checks=frozenset({}), remove_checks=frozenset({})):
  # todo: make threadsafe etc etc
  global checks
  prev_val = checks
  new_val = (prev_val | new_checks) - remove_checks
  checks = new_val
  try:
    yield
  finally:
    checks = prev_val
