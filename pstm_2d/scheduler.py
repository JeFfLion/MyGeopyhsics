"""Resource-safe multi-GPU scheduler for PSTM shot distribution.

Spawns 4 GPU workers via torch.multiprocessing.spawn and collects partial
images via a multiprocessing.Queue.  Implements a timeout watchdog and
cleanup-on-failure protocol.
"""

import gc
import logging
import multiprocessing as mp
import os
import threading
import time
from typing import List, Callable, Any, Optional
import numpy as np

from .geometry import Geometry, CmpGrid, ChunkSpec
from .worker import worker_entry
from .safety import kill_all_children

logger = logging.getLogger(__name__)

mp_ctx = mp.get_context("spawn")


class ResourceSafeScheduler:
    """Distributes 601 shots across 4 GPUs with overlap halos.

    Parameters
    ----------
    n_shots : int
        Total shots (601).
    n_gpus : int
        Number of GPU workers (4).
    chunk_overlap : int
        Number of overlapping shots between adjacent chunks (10).
    timeout_seconds : int
        Watchdog timeout, after which emergency cleanup fires (7200).
    """

    def __init__(
        self,
        n_shots: int = 601,
        n_gpus: int = 4,
        chunk_overlap: int = 10,
        timeout_seconds: int = 7200,
    ):
        self.n_shots = n_shots
        self.n_gpus = n_gpus
        self.chunk_overlap = chunk_overlap
        self.timeout_seconds = timeout_seconds

        self._pool = None
        self._watchdog: Optional[threading.Timer] = None
        self._aborted = mp_ctx.Event()

    # ------------------------------------------------------------------
    #  Chunking
    # ------------------------------------------------------------------
    def compute_chunking(self) -> List[ChunkSpec]:
        """Divide shots into chunks with halo overlap."""
        overlap = min(self.chunk_overlap, self.n_shots // (self.n_gpus * 2))
        if self.n_shots <= self.n_gpus or overlap == 0:
            return self._simple_chunking(self.n_shots)

        base = self.n_shots // self.n_gpus
        rem = self.n_shots % self.n_gpus
        chunks = []
        start = 1

        for rank in range(self.n_gpus):
            size = base + (1 if rank < rem else 0)
            end = start + size - 1

            hl = overlap if rank > 0 else 0
            hr = overlap if rank < self.n_gpus - 1 else 0

            chunks.append(ChunkSpec(
                rank=rank,
                shot_start=max(1, start - hl),
                shot_end=min(self.n_shots, end + hr),
                halo_left=hl,
                halo_right=hr,
            ))
            start = end + 1

        for c in chunks:
            logger.info("  GPU %d: shots %d→%d (halo L=%d R=%d)",
                         c.rank, c.shot_start, c.shot_end, c.halo_left, c.halo_right)
        return chunks

    def _simple_chunking(self, ng: int) -> List[ChunkSpec]:
        chunks = []
        base = self.n_shots // ng
        rem = self.n_shots % ng
        start = 1
        for rank in range(ng):
            size = base + (1 if rank < rem else 0)
            end = start + size - 1
            chunks.append(ChunkSpec(
                rank=rank,
                shot_start=start,
                shot_end=end,
                halo_left=0,
                halo_right=0,
            ))
            start = end + 1
        return chunks

    # ------------------------------------------------------------------
    #  Launch
    # ------------------------------------------------------------------
    def _get_free_port(self) -> int:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def launch(
        self,
        segy_shot_path: str,
        geometry: Geometry,
        cmp_grid: CmpGrid,
        statics: np.ndarray,
        vel_traces: np.ndarray,
        vel_x: np.ndarray,
        **worker_kwargs,
    ) -> np.ndarray:
        """Launch GPU workers, wait for completion, return merged result.

        Returns:
            result: [N_cmp, N_t] float32 merged migration image.
        """
        chunks = self.compute_chunking()
        queue = mp_ctx.Queue()

        # Set up NCCL environment for distributed init
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self._get_free_port())

        # Start watchdog
        self._start_watchdog()

        try:
            logger.info("Spawning %d workers for %d shots ...", self.n_gpus, self.n_shots)
            t0 = time.perf_counter()

            procs = []
            for rank in range(self.n_gpus):
                p = mp_ctx.Process(
                    target=_launcher_fn,
                    args=(
                        rank,
                        self.n_gpus,
                        segy_shot_path,
                        geometry,
                        cmp_grid,
                        statics,
                        vel_traces,
                        vel_x,
                        chunks,
                        queue,
                        worker_kwargs,
                    ),
                )
                p.start()
                procs.append(p)

            logger.info("All workers launched, waiting for rank 0 result ...")

            # Wait for rank 0 to send result (interruptible by watchdog)
            poll_interval = 5
            elapsed = 0
            result = None
            while result is None and elapsed < self.timeout_seconds:
                try:
                    result = queue.get(timeout=poll_interval)
                except Exception:
                    pass
                elapsed += poll_interval
                if self._aborted.is_set():
                    raise RuntimeError("Watchdog aborted")

            if result is None:
                raise RuntimeError(f"Timed out waiting for result after {self.timeout_seconds}s")

            logger.info("Result received from rank 0, shape=%s", result.shape)

            # Wait for all workers to finish
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    logger.warning("Worker still alive, terminating")
                    p.terminate()
                    p.join(timeout=10)

            logger.info("All workers completed in %.0fs", time.perf_counter() - t0)

            logger.info("Final image shape: %s, min=%.4f max=%.4f",
                         result.shape, result.min(), result.max())

        except Exception as e:
            logger.exception("Scheduler failure, triggering emergency cleanup")
            self.emergency_cleanup()
            raise
        finally:
            self._stop_watchdog()

        return result

    # ------------------------------------------------------------------
    #  Watchdog
    # ------------------------------------------------------------------
    def _start_watchdog(self):
        if self.timeout_seconds > 0:
            self._aborted.clear()
            self._watchdog = threading.Timer(self.timeout_seconds, self._timeout_handler)
            self._watchdog.daemon = True
            self._watchdog.start()
            logger.info("Watchdog started (timeout=%ds)", self.timeout_seconds)

    def _stop_watchdog(self):
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    def _timeout_handler(self):
        logger.critical("TIMEOUT after %ds → emergency abort", self.timeout_seconds)
        self._aborted.set()
        self.emergency_cleanup()

    # ------------------------------------------------------------------
    #  Emergency cleanup
    # ------------------------------------------------------------------
    def emergency_cleanup(self):
        """Kill all children, free resources, hard exit if necessary."""
        logger.warning("Emergency cleanup initiated")
        kill_all_children()
        gc.collect()
        logger.warning("Emergency cleanup done")


def _launcher_fn(
    rank: int,
    world_size: int,
    segy_shot_path: str,
    geometry: Geometry,
    cmp_grid: CmpGrid,
    statics: np.ndarray,
    vel_traces: np.ndarray,
    vel_x: np.ndarray,
    chunks: list,
    output_queue,
    worker_kwargs: dict,
):
    """Internal wrapper called by mp.spawn on each rank."""
    import sys as _sys
    try:
        chunk = chunks[rank]
        worker_entry(
            rank=rank,
            world_size=world_size,
            segy_shot_path=segy_shot_path,
            geometry=geometry,
            cmp_grid=cmp_grid,
            statics=statics,
            vel_traces=vel_traces,
            vel_x=vel_x,
            chunk=chunk,
            output_queue=output_queue,
            **worker_kwargs,
        )
    finally:
        _sys.exit(0)
