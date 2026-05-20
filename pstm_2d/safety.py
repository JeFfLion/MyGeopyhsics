# Copyright (c) LJF. All Rights Reserved. | Licensed under GPL-3.0.
# Unauthorized commercial use is strictly prohibited.
"""Circuit breaker and emergency cleanup utilities for HPC execution safety."""

import gc
import os
import sys
import signal
import torch
import traceback
from typing import Optional, Callable


class CircuitBreaker:
    """Detects repetitive failures and triggers abort to avoid infinite loops.

    If the same failure type and location occur >= max_retries times
    consecutively, the circuit breaker trips and raises RuntimeError
    with a diagnostic report.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._failure_stack: list = []

    def record_failure(self, error: Exception, location: str) -> None:
        key = (type(error).__name__, location)
        self._failure_stack.append(key)
        recent = self._failure_stack[-self.max_retries:]
        if len(recent) >= self.max_retries and all(k == recent[0] for k in recent):
            self._trip(error, location)

    def _trip(self, error: Exception, location: str) -> None:
        report = (
            "\n[Execution Terminated]\n"
            f"  Cause: {self.max_retries} consecutive identical failures\n"
            f"  Location: {location}\n"
            f"  Error: {type(error).__name__}: {error}\n"
            f"  Failed paths: {[f'{t}({l})' for t, l in self._failure_stack[-self.max_retries:]]}\n"
            f"  Suggestion: Check GPU memory, DDP configuration, or data integrity.\n"
        )
        raise RuntimeError(report) from error

    def reset(self) -> None:
        self._failure_stack.clear()


class EmergencyCleanup:
    """Context manager ensuring GPU and process cleanup on exit."""

    def __init__(self, on_cleanup: Optional[Callable] = None):
        self._on_cleanup = on_cleanup

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._on_cleanup:
                self._on_cleanup()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_val, exc_tb)


def memory_snapshot(device: int = 0) -> str:
    """Return a formatted GPU memory usage snapshot."""
    try:
        alloc = torch.cuda.memory_allocated(device) / 1024**3
        reserv = torch.cuda.memory_reserved(device) / 1024**3
        return f"GPU[{device}] alloc={alloc:.2f}GB reserved={reserv:.2f}GB"
    except Exception:
        return "memory_snapshot unavailable"


def kill_all_children(logger=None) -> None:
    """Terminate all child processes and clean CUDA contexts."""
    import multiprocessing as mp
    for p in mp.active_children():
        try:
            p.terminate()
        except Exception:
            pass
    for p in mp.active_children():
        try:
            p.join(timeout=5)
        except Exception:
            pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
    except Exception:
        pass


def install_signal_handlers(cleanup_fn: Callable) -> None:
    """Register cleanup on SIGTERM / SIGINT."""

    def handler(signum, frame):
        del signum, frame
        try:
            cleanup_fn()
        except Exception:
            pass
        os._exit(1)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
