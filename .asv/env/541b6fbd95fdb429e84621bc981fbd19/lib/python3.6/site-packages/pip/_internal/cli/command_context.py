# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

from contextlib import contextmanager

from pip._vendor.contextlib2 import ExitStack


class CommandContextMixIn(object):
    def __init__(self):
        super(CommandContextMixIn, self).__init__()
        self._in_main_context = False
        self._main_context = ExitStack()

    @contextmanager
    def main_context(self):
        assert not self._in_main_context

        self._in_main_context = True
        try:
            with self._main_context:
                yield
        finally:
            self._in_main_context = False

    def enter_context(self, context_provider):
        assert self._in_main_context

        return self._main_context.enter_context(context_provider)
