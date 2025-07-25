import jax
from jax.experimental.xla_metadata import set_xla_metadata


class Future:
    def __init__(self, aval):
        self.aval = aval

class Timeline:
    def __enter__(self):
        self.token = jax.lax.create_token()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.token = None

    def async_call(self, f, stream=None) -> callable:
        tokenized_f = jax.jit(lambda token, *a, **k: (token, f(*a, **k)))
        def wrapper(*args, **kwargs) -> Future:
            if self.token is None:
                raise ValueError("Timeline exited already")
            token, result = tokenized_f(self.token, *args, **kwargs)
            tagged_result = set_xla_metadata(
                    result, 
                    _async_start=str(stream) if stream else "",
                    inlineable="false")
            self.token = token
            return Future(tagged_result)
        return wrapper

    def ready(self, future):
        token, done = jax.jit(lambda token, a: (token, a))(self.token, future.aval)
        self.token = token
        tagged_done = set_xla_metadata(done, _async_done="", inlineable="false")
        return tagged_done


