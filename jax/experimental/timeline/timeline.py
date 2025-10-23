import jax
from jax.experimental.xla_metadata import set_xla_metadata
import uuid

class Future:
    def __init__(self, aval):
        self.aval = aval

class Timeline:
    def __enter__(self):
        self.prev = None
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.prev = None

    def launch(self, val):
        tag = uuid.uuid4().hex
        if self.prev:
          val = set_xla_metadata(
              val,
              start_wait_tag=tag,
              start_wait_for=self.prev)
        else:
          val = set_xla_metadata(
              val,
              start_wait_tag=tag)
        self.prev = tag
        return Future(val)
    
    def finish(self, val):
        val = val.aval
        tag = uuid.uuid4().hex
        val = set_xla_metadata(
            val,
            done_wait_tag=tag,
            done_wait_for=self.prev)
        self.prev = tag
        return val

    def ready(self, future):
        done = jax.jit(lambda a: a)(future.aval)
        tag = uuid.uuid4().hex
        tagged_done = set_xla_metadata(done, _async_done="", inlineable="false", wait_tag=tag, wait_for=self.prev)
        self.prev = tag
        return tagged_done

