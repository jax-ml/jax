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

    def async_call(self, f, stream=None) -> callable:
        def wrapper(*args, **kwargs) -> Future:
            result = jax.jit(f)(*args, **kwargs)
            tag = uuid.uuid4().hex
            if self.prev:
              tagged_result = set_xla_metadata(
                      result, 
                      _async_start=str(stream) if stream else "",
                      wait_tag=tag,
                      wait_for=self.prev,
                      inlineable="false")
            else:
              tagged_result = set_xla_metadata(
                      result,
                      _async_start=str(stream) if stream else "",
                      wait_tag=tag,
                      inlineable="false")
            self.prev = tag
            return Future(tagged_result)
        return wrapper

    def ready(self, future):
        done = jax.jit(lambda a: a)(future.aval)
        tag = uuid.uuid4().hex
        tagged_done = set_xla_metadata(done, _async_done="", inlineable="false", wait_tag=tag, wait_for=self.prev)
        self.prev = tag
        return tagged_done

