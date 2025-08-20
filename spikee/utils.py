"""Small timeout utilities.

Provides a :class:`TimeoutException` and a callable ``Timeout`` helper that is
primarily intended to be used as a decorator: ``@Timeout(seconds)``. The
implementation uses a signal/SIGALRM fast path on POSIX main-thread and
falls back to a thread-join approach on platforms without SIGALRM (for
example, Windows).

Notes:
- The decorator fast-path uses SIGALRM and therefore only works in the main
  thread on POSIX systems.
- The fallback thread-based implementation detects timeouts but cannot
  forcibly interrupt blocking native calls inside the decorated function.

This module intentionally keeps the API small and dependency-free.
"""
from __future__ import annotations

import functools
import signal
import threading
import warnings
from typing import Any, Callable, Optional

__all__ = ["Timeout", "TimeoutException", "timeout"]


class TimeoutException(TimeoutError):
  """Raised when an operation times out."""


def _signal_available() -> bool:
  """Return True when signal-based timeout (SIGALRM) is available.

  This is only true on POSIX-like systems and when running in the main
  thread. Signal alarm cannot be used from other threads.
  """
  try:
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
  except Exception:
    return False


class Timeout:
  """Decorator-style timeout helper.

  Use as ``@Timeout(seconds)`` to limit execution time of the decorated
  function. The implementation prefers a SIGALRM-based fast path on POSIX
  main-thread but falls back to a thread-join approach on other platforms.

  Args:
      seconds: timeout in seconds (float). Must be > 0.
      on_timeout: optional callable invoked when a timeout occurs (fallback
          thread path only). It will be called with no arguments.
  """

  def __init__(self, seconds: float, on_timeout: Optional[Callable[[], None]] = None) -> None:
    if seconds <= 0:
      raise ValueError("timeout seconds must be > 0")
    self.seconds = float(seconds)
    self.on_timeout = on_timeout

  def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
    """Return a wrapped callable that enforces the timeout.

    Uses signal-based timing on POSIX main thread for low-overhead
    interruption; otherwise falls back to running the function in a
    background thread and joining with a timeout.
    """

    if _signal_available():
      @functools.wraps(func)
      def _wrapped_signal(*args, **kwargs):
        prev_handler = signal.getsignal(signal.SIGALRM)
        prev_timer = signal.setitimer(signal.ITIMER_REAL, self.seconds)

        def _handler(signum, frame):
          raise TimeoutException(f"operation exceeded {self.seconds} seconds")

        signal.signal(signal.SIGALRM, _handler)
        try:
          return func(*args, **kwargs)
        finally:
          # restore previous timer/handler
          signal.setitimer(signal.ITIMER_REAL, prev_timer[0], prev_timer[1])
          signal.signal(signal.SIGALRM, prev_handler)

      return _wrapped_signal

    # Fallback: thread-based wrapper. Works cross-platform but cannot
    # interrupt blocking calls inside the target function.
    @functools.wraps(func)
    def _wrapped_thread(*args, **kwargs):
      result: dict = {}

      def _runner():
        try:
          result["value"] = func(*args, **kwargs)
        except BaseException as e:
          result["exc"] = e

      thread = threading.Thread(target=_runner, daemon=True)
      thread.start()
      thread.join(self.seconds)
      if thread.is_alive():
        # optional callback
        if callable(self.on_timeout):
          try:
            self.on_timeout()
          except Exception:
            warnings.warn(
              "on_timeout callback raised an exception", RuntimeWarning)
        raise TimeoutException(f"operation exceeded {self.seconds} seconds")

      if "exc" in result:
        # re-raise original exception with correct traceback
        raise result["exc"]
      return result.get("value")

    return _wrapped_thread


def timeout(seconds: float, on_timeout: Optional[Callable[[], None]] = None) -> Timeout:
  """Convenience factory matching common usage: ``@timeout(1.0)``.

  Returns a :class:`Timeout` instance.
  """

  return Timeout(seconds, on_timeout=on_timeout)


if __name__ == "__main__":
  # Example usage
  @Timeout(2.0)
  def example_function():
    import time
    time.sleep(3)  # This will raise TimeoutException in the SIGALRM fast path

  try:
    example_function()
  except TimeoutException as e:
    print(f"Caught timeout: {e}")
