"""
Background file writes: submit(fn, *args) runs fn on a writer thread so the
caller keeps computing; drain() waits for all of them and re-raises the
first error. Callers must pass data they will not modify afterwards (the
Scripts pass a fresh float32 copy).

run_pipeline_new_spocker.py drains after every step (forked branches end
with os._exit, which skips atexit). A Script run on its own needs nothing:
the executor's threads are joined at interpreter exit.
"""

from concurrent.futures import ThreadPoolExecutor

_pool = ThreadPoolExecutor(4)
_pending = []


def submit(fn, *args, **kwargs):
    _pending.append(_pool.submit(fn, *args, **kwargs))


def drain():
    errors = [f.exception() for f in _pending]
    _pending.clear()
    for e in errors:
        if e is not None:
            raise e
