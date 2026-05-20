"""Single-GPU Kirchhoff PSTM worker engine.

Each worker processes its assigned shot chunk, producing a partial stacked
image.  Memory is aggressively managed: after every shot the intermediate
tensors are deleted and CUDA cache is emptied.
"""

import gc
import logging
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from typing import Tuple, Optional
from contextlib import nullcontext

from .geometry import Geometry, CmpGrid, ChunkSpec, GeomPreProcessor
from .traveltime import migrate_one_shot, interp_velocity
from .safety import memory_snapshot

logger = logging.getLogger(__name__)


class KirchhoffCUDAWorker:
    """Processes a chunk of shot gathers on one GPU.

    Parameters
    ----------
    rank : int
        GPU rank (0..3).
    world_size : int
        Total number of GPU workers.
    device : torch.device
        Target CUDA device for this worker.
    geometry : Geometry
        Full trace geometry.
    cmp_grid : CmpGrid
        Output imaging grid definition.
    statics : np.ndarray
        Per-trace static shift in seconds.
    vel_traces : np.ndarray
        [N_vel, n_t] RMS velocity field.
    vel_x : np.ndarray
        [N_vel] velocity trace CMP positions (m).
    segy_shot_path : str
        Path to input SEGY shot file.
    chunk : ChunkSpec
        Shot range assigned to this worker.
    max_aperture_m : float, optional
        Maximum migration aperture in metres (default 3000).
    rec_batch_size : int, optional
        Receiver batch size for GPU memory control (default 64).
    f_max : float, optional
        Maximum frequency for anti-aliasing filter (default 125 Hz).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        geometry: Geometry,
        cmp_grid: CmpGrid,
        statics: np.ndarray,
        vel_traces: np.ndarray,
        vel_x: np.ndarray,
        segy_shot_path: str,
        chunk: ChunkSpec,
        max_aperture_m: float = 3000.0,
        rec_batch_size: int = 64,
        f_max: float = 125.0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.geometry = geometry
        self.cmp_grid = cmp_grid
        self.statics = statics
        self.segy_shot_path = segy_shot_path
        self.chunk = chunk
        self.max_aperture_m = max_aperture_m
        self.rec_batch_size = rec_batch_size
        self.f_max = f_max

        # GPU-resident arrays
        self.x_cmp_gpu: torch.Tensor = torch.as_tensor(
            cmp_grid.x_array.astype(np.float32), device=self.device
        )
        self.t0_gpu: torch.Tensor = torch.as_tensor(
            cmp_grid.t_array.astype(np.float32), device=self.device
        )
        self.vel_traces_gpu: torch.Tensor = torch.as_tensor(vel_traces, device=self.device)
        self.vel_x_gpu: torch.Tensor = torch.as_tensor(
            vel_x.astype(np.float32), device=self.device
        )

        self._partial_image: Optional[torch.Tensor] = None
        self._segy_handle = None

    # ------------------------------------------------------------------
    #  Output buffer
    # ------------------------------------------------------------------
    def _ensure_output_buffer(self) -> torch.Tensor:
        if self._partial_image is None:
            self._partial_image = torch.zeros(
                self.cmp_grid.n_cmp, self.cmp_grid.n_t,
                dtype=torch.float32, device=self.device
            )
        return self._partial_image

    # ------------------------------------------------------------------
    #  SEGY read helpers
    # ------------------------------------------------------------------
    def _open_segy(self):
        import segyio
        self._segy_handle = segyio.open(
            self.segy_shot_path, "r", strict=False, ignore_geometry=True
        )

    def _read_shot_gather(self, shot_id: int) -> np.ndarray:
        """Read one shot gather from SEGY (all receivers)."""
        n_rec = self.geometry.n_rec_per_shot
        start = (shot_id - 1) * n_rec
        data = np.zeros((n_rec, self.cmp_grid.n_t), dtype=np.float32)
        for i in range(n_rec):
            data[i] = self._segy_handle.trace[start + i]
        return data

    # ------------------------------------------------------------------
    #  Chunk processing
    # ------------------------------------------------------------------
    @torch.no_grad()
    def process_chunk(self, total_shots: int) -> torch.Tensor:
        """Process all shots in the assigned chunk, returning partial image."""
        output = self._ensure_output_buffer()
        self._open_segy()

        shots = list(range(self.chunk.shot_start, self.chunk.shot_end + 1))
        shot_count = 0

        t_start = time.perf_counter()
        for shot_id in shots:
            shot_count += 1
            t0_shot = time.perf_counter()
            fail_key = f"shot_{shot_id}_rank{self.rank}"
            error_count = 0

            while error_count < 3:
                try:
                    self._migrate_one_shot_in_place(shot_id, output)
                    break
                except torch.cuda.OutOfMemoryError as e:
                    error_count += 1
                    self._emergency_free()
                    if error_count >= 3:
                        logger.critical("OOM after 3 retries on shot %d rank %d", shot_id, self.rank)
                        raise RuntimeError(
                            f"FATAL OOM on shot {shot_id} rank {self.rank} "
                            f"after 3 retries. {memory_snapshot(self.rank)}"
                        ) from e
                    # Reduce rec_batch_size and retry
                    old_bs = self.rec_batch_size
                    self.rec_batch_size = max(8, self.rec_batch_size // 2)
                    logger.warning("OOM on %s: reduced rec_batch %d→%d", fail_key, old_bs, self.rec_batch_size)
                except Exception:
                    raise

            t_shot = time.perf_counter() - t0_shot
            if shot_count % 10 == 0 or shot_count == 1:
                done = shot_count
                elapsed = time.perf_counter() - t_start
                eta = (elapsed / done) * (len(shots) - done) if done > 0 else 0
                msg = (f"[rank{self.rank}] shot {shot_id:04d} ({done}/{len(shots)}) "
                       f"| {t_shot:.2f}s/shot | ETA {eta:.0f}s "
                       f"| {memory_snapshot(self.rank)}")
                logger.info(msg)

        self._close_segy()
        return output

    @torch.no_grad()
    def _migrate_one_shot_in_place(self, shot_id: int, output: torch.Tensor) -> None:
        """Read one gather, migrate it, and add to output buffer in-place."""
        # 1. Read gather
        gather_np = self._read_shot_gather(shot_id)              # [Nr, Nt_samp]
        gather = torch.as_tensor(gather_np, device=self.device)

        # Static shift for this shot's first trace
        tr0 = (shot_id - 1) * self.geometry.n_rec_per_shot
        static_s = float(self.statics[tr0])

        # 2. Shot position
        sx_m = float(self.geometry.sx[tr0])

        # 3. Determine aperture
        x_cmp = self.cmp_grid.x_array
        ap_mask = np.abs(x_cmp - sx_m) < self.max_aperture_m
        ap_indices = np.where(ap_mask)[0]
        if len(ap_indices) == 0:
            del gather
            return
        x_ap = torch.as_tensor(x_cmp[ap_indices].astype(np.float32), device=self.device)

        # 4. Velocity at aperture positions
        vrms_ap = interp_velocity(self.vel_x_gpu, self.vel_traces_gpu, x_ap)

        # 5. Run migration
        try:
            partial = migrate_one_shot(
                sx=torch.tensor(sx_m, device=self.device),
                gather=gather,
                x_img=x_ap,
                t0=self.t0_gpu,
                dt=self.cmp_grid.dt,
                vrms_img=vrms_ap,
                static_shift=static_s,
                rec_batch_size=self.rec_batch_size,
            )
        except torch.cuda.OutOfMemoryError:
            del gather, x_ap, vrms_ap
            gc.collect()
            torch.cuda.empty_cache()
            raise

        # 6. Add to output (partial indices → global CMP grid)
        ap_idx_gpu = torch.as_tensor(ap_indices, dtype=torch.long, device=self.device)
        output.index_add_(0, ap_idx_gpu, partial)

        # 7. Free ALL intermediate GPU memory
        del gather, partial, x_ap, vrms_ap, ap_idx_gpu
        gc.collect()
        torch.cuda.empty_cache()

    def _emergency_free(self):
        """Aggressive memory cleanup on OOM."""
        del self._partial_image
        self._partial_image = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.warning("Emergency free on rank %d: %s", self.rank, memory_snapshot(self.rank))

    def _close_segy(self):
        if self._segy_handle is not None:
            try:
                self._segy_handle.close()
            except Exception:
                pass
            self._segy_handle = None

    def get_partial_image(self) -> torch.Tensor:
        return self._partial_image if self._partial_image is not None else self._ensure_output_buffer()


# ---------------------------------------------------------------------------
#  Entry point for torch.multiprocessing.spawn
# ---------------------------------------------------------------------------
def worker_entry(
    rank: int,
    world_size: int,
    segy_shot_path: str,
    geometry: Geometry,
    cmp_grid: CmpGrid,
    statics: np.ndarray,
    vel_traces: np.ndarray,
    vel_x: np.ndarray,
    chunk: ChunkSpec,
    output_queue,
    **kwargs,
) -> None:
    """Entry point called by mp.spawn on each GPU rank."""
    import torch.distributed as dist

    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        worker = KirchhoffCUDAWorker(
            rank=rank,
            world_size=world_size,
            device=device,
            geometry=geometry,
            cmp_grid=cmp_grid,
            statics=statics,
            vel_traces=vel_traces,
            vel_x=vel_x,
            segy_shot_path=segy_shot_path,
            chunk=chunk,
            **kwargs,
        )

        total_shots = geometry.n_shots
        partial = worker.process_chunk(total_shots)

        logger.info("[rank%d] local processing done, starting all_reduce", rank)

        dist.all_reduce(partial, op=dist.ReduceOp.SUM)

        if rank == 0:
            output_queue.put(partial.cpu().numpy())

        dist.barrier()
        dist.destroy_process_group()

    except Exception as e:
        logger.exception("[rank%d] FATAL: %s", rank, e)
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        os._exit(1)
