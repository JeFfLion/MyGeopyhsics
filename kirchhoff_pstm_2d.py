#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Kirchhoff PSTM 2D — Single-file standalone
#  Copyright (c) LJF. All Rights Reserved. | Licensed under GPL-3.0.
#  Unauthorized commercial use is strictly prohibited.
#
#  Usage:
#      python kirchhoff_pstm_2d.py --shot-path SHOT.sgy --vel-path VEL.sgy
#      python kirchhoff_pstm_2d.py --test-shots 4
#      python kirchhoff_pstm_2d.py --n-gpus 2 --help
# =============================================================================

import argparse
import gc
import logging
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any

import numpy as np
import torch
import torch.distributed as dist


__version__ = "1.0.0"
__author__ = "LJF"
__copyright__ = "Copyright (c) LJF. All Rights Reserved."


# =============================================================================
#  Loggers
# =============================================================================

_log_safety = logging.getLogger("kirchhoff.safety")
_log_geom   = logging.getLogger("kirchhoff.geometry")
_log_agg    = logging.getLogger("kirchhoff.aggregator")
_log_worker = logging.getLogger("kirchhoff.worker")
_log_sched  = logging.getLogger("kirchhoff.scheduler")
_log_engine = logging.getLogger("kirchhoff.engine")

# =============================================================================
#  Section 1 — Circuit breaker & emergency cleanup   (safety.py)
# =============================================================================

class CircuitBreaker:
    """Detects repetitive failures and triggers abort to avoid infinite loops."""

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


# =============================================================================
#  Section 2 — Geometry data structures   (geometry.py)
# =============================================================================

@dataclass
class Geometry:
    """Per-trace geometry extracted from SEGY shot gathers."""
    sx: np.ndarray          # [N_traces] source x, meters
    gx: np.ndarray          # [N_traces] group (receiver) x, meters
    offset: np.ndarray      # [N_traces] offset, meters
    cmp_x: np.ndarray       # [N_traces] CMP x coordinate, meters
    fldr: np.ndarray        # [N_traces] field record (shot) index
    n_shots: int = 0
    n_rec_per_shot: int = 0

    @property
    def n_traces(self) -> int:
        return len(self.sx)

    def get_shot_indices(self, shot_id: int) -> Tuple[int, int]:
        base = (shot_id - 1) * self.n_rec_per_shot
        return base, base + self.n_rec_per_shot


@dataclass
class CmpGrid:
    """Output CMP imaging grid definition."""
    x_min: float
    x_max: float
    spacing: float
    n_cmp: int
    n_t: int
    dt: float

    @property
    def x_array(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.n_cmp, dtype=np.float64)

    @property
    def t_array(self) -> np.ndarray:
        return np.arange(self.n_t, dtype=np.float64) * self.dt


@dataclass
class ChunkSpec:
    """Shot chunk distribution specification with optional halos."""
    rank: int
    shot_start: int
    shot_end: int
    halo_left: int = 0
    halo_right: int = 0
    output_x_slice: Optional[slice] = None

    @property
    def core_shots(self) -> slice:
        return slice(self.shot_start + self.halo_left,
                     self.shot_end - self.halo_right + 1)

    @property
    def all_shots(self) -> slice:
        return slice(self.shot_start, self.shot_end + 1)


class GeomPreProcessor:
    """Reads SEGY shot headers, builds geometry, computes statics and CMP grid."""

    def __init__(
        self,
        segy_shot_path: str,
        segy_vel_path: str,
        src_depth: float = 9.0,
        rec_depth: float = 10.0,
        datum_elev: float = 0.0,
        replacement_vel: float = 1800.0,
        cmp_spacing: float = 25.0,
        max_aperture_m: float = 3000.0,
        coord_scale: float = 100.0,
        n_shots_expected: int = 0,
        n_rec_expected: int = 0,
    ):
        self.segy_shot_path = segy_shot_path
        self.segy_vel_path = segy_vel_path
        self.src_depth = src_depth
        self.rec_depth = rec_depth
        self.datum_elev = datum_elev
        self.replacement_vel = replacement_vel
        self.cmp_spacing = cmp_spacing
        self.max_aperture_m = max_aperture_m
        self.coord_scale = coord_scale
        self.n_shots_expected = n_shots_expected
        self.n_rec_expected = n_rec_expected

        self.geometry: Optional[Geometry] = None
        self.cmp_grid: Optional[CmpGrid] = None
        self.static_shifts: Optional[np.ndarray] = None
        self.vel_traces: Optional[np.ndarray] = None
        self.vel_x_array: Optional[np.ndarray] = None
        self.dt: float = 0.0
        self.n_t: int = 0

    @staticmethod
    def _open_segy(path: str):
        import segyio
        return segyio.open(path, "r", strict=False, ignore_geometry=True)

    @staticmethod
    def _coord_to_m(val_cm, scale):
        if abs(scale) > 1e-6:
            return np.asarray(val_cm, dtype=np.float64) / abs(scale)
        return np.asarray(val_cm, dtype=np.float64)

    def load_geometry(self) -> Geometry:
        _log_geom.info("Loading geometry from %s ...", self.segy_shot_path)
        import segyio
        TF = segyio.TraceField

        with self._open_segy(self.segy_shot_path) as f:
            n_traces = f.tracecount
            self.n_t = len(f.samples)
            self.dt = float(f.samples[1] - f.samples[0]) / 1000.0
            _log_geom.info("  %d traces, nt=%d, dt=%.4fs", n_traces, self.n_t, self.dt)

            sx_raw = np.zeros(n_traces, dtype=np.int32)
            gx_raw = np.zeros(n_traces, dtype=np.int32)
            fldr_raw = np.zeros(n_traces, dtype=np.int32)

            for i in range(n_traces):
                h = f.header[i]
                sx_raw[i] = h[TF.SourceX]
                gx_raw[i] = h[TF.GroupX]
                fldr_raw[i] = h[TF.FieldRecord]

        s = self.coord_scale
        sx_m = self._coord_to_m(sx_raw, s)
        gx_m = self._coord_to_m(gx_raw, s)
        offset_m = sx_m - gx_m
        cmp_x_m = (sx_m + gx_m) / 2.0

        fldr_ids = np.unique(fldr_raw)
        n_shots = len(fldr_ids)
        shot_size = int(n_traces / n_shots) if n_shots > 0 else self.n_rec_expected

        self.geometry = Geometry(
            sx=sx_m, gx=gx_m, offset=offset_m, cmp_x=cmp_x_m,
            fldr=fldr_raw, n_shots=n_shots, n_rec_per_shot=shot_size,
        )

        _log_geom.info("  n_shots=%d, n_rec_per_shot=%d", n_shots, shot_size)
        _log_geom.info("  sx: %.1f → %.1f m", sx_m.min(), sx_m.max())
        _log_geom.info("  gx: %.1f → %.1f m", gx_m.min(), gx_m.max())
        _log_geom.info("  offset: %.1f → %.1f m", offset_m.min(), offset_m.max())
        _log_geom.info("  cmp_x: %.1f → %.1f m", cmp_x_m.min(), cmp_x_m.max())
        return self.geometry

    def compute_static_corrections(self) -> np.ndarray:
        if self.geometry is None:
            raise RuntimeError("Call load_geometry() first.")
        ds = abs(self.src_depth - self.datum_elev)
        dr = abs(self.rec_depth - self.datum_elev)
        dt_static = (ds + dr) / self.replacement_vel
        self.static_shifts = np.full(self.geometry.n_traces, dt_static, dtype=np.float32)
        _log_geom.info("Static shift: %.4f ms per trace", dt_static * 1000)
        return self.static_shifts

    def build_cmp_grid(self) -> CmpGrid:
        if self.geometry is None:
            raise RuntimeError("Call load_geometry() first.")
        cmp_min = self.geometry.cmp_x.min()
        cmp_max = self.geometry.cmp_x.max()
        n_cmp = int(round((cmp_max - cmp_min) / self.cmp_spacing)) + 1
        self.cmp_grid = CmpGrid(
            x_min=cmp_min, x_max=cmp_max, spacing=self.cmp_spacing,
            n_cmp=n_cmp, n_t=self.n_t, dt=self.dt,
        )
        _log_geom.info("CMP grid: %.1f → %.1f m, %d points @ %.0fm spacing",
                       cmp_min, cmp_max, n_cmp, self.cmp_spacing)
        return self.cmp_grid

    def load_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        _log_geom.info("Loading velocity from %s ...", self.segy_vel_path)
        import segyio

        with self._open_segy(self.segy_vel_path) as f:
            n_vel = f.tracecount
            n_samp = len(f.samples)
            vel_dt = float(f.samples[1] - f.samples[0]) / 1000.0
            _log_geom.info("  %d velocity traces, nt=%d, dt=%.4fs", n_vel, n_samp, vel_dt)
            vel_data = np.zeros((n_vel, n_samp), dtype=np.float32)
            for i in range(n_vel):
                vel_data[i] = f.trace[i].astype(np.float32)

        with self._open_segy(self.segy_vel_path) as f:
            h0 = f.header[0]
            TF = segyio.TraceField
            cdp_x0 = self._coord_to_m(h0[TF.CDP_X], self.coord_scale)
            sx0 = self._coord_to_m(h0[TF.SourceX], self.coord_scale)

        first_x = cdp_x0 if cdp_x0 > 0 else sx0
        if first_x == 0 and n_vel > 1:
            with self._open_segy(self.segy_vel_path) as f:
                hL = f.header[n_vel - 1]
                TF = segyio.TraceField
                last_x = self._coord_to_m(hL[TF.SourceX], self.coord_scale)
                if last_x == 0:
                    last_x = self._coord_to_m(hL[TF.CDP_X], self.coord_scale)
            spacing = last_x / (n_vel - 1) if last_x > 0 else 12.5
        else:
            spacing = self.cmp_spacing

        vel_x = np.arange(n_vel, dtype=np.float64) * spacing + first_x

        if abs(vel_dt - self.dt) > 1e-6 or n_samp != self.n_t:
            _log_geom.warning("Velocity time axis mismatch (dt=%.4f vs %.4f, nt=%d vs %d)",
                              vel_dt, self.dt, n_samp, self.n_t)
            t_old = np.arange(n_samp) * vel_dt
            t_new = np.arange(self.n_t) * self.dt
            from scipy.interpolate import interp1d
            vel_resampled = np.zeros((n_vel, self.n_t), dtype=np.float32)
            for i in range(n_vel):
                vel_resampled[i] = interp1d(
                    t_old, vel_data[i], kind="linear",
                    bounds_error=False, fill_value="extrapolate",
                )(t_new).astype(np.float32)
            vel_data = vel_resampled

        self.vel_traces = vel_data
        self.vel_x_array = vel_x
        _log_geom.info("  Velocity spacing: %.2f m, range: %.1f → %.1f m",
                       spacing, vel_x[0], vel_x[-1])
        return vel_data, vel_x

    def compute_chunking(self, n_gpus: int = 4, chunk_overlap: int = 10) -> List[ChunkSpec]:
        if self.geometry is None:
            raise RuntimeError("Call load_geometry() first.")
        n_shots = self.geometry.n_shots
        if n_shots < n_gpus:
            n_gpus = max(1, n_shots)
            chunk_overlap = 0

        base = n_shots // n_gpus
        rem = n_shots % n_gpus
        chunks = []
        start = 1
        for rank in range(n_gpus):
            size = base + (1 if rank < rem else 0)
            end = start + size - 1
            hl = chunk_overlap if rank > 0 else 0
            hr = chunk_overlap if rank < n_gpus - 1 else 0
            chunks.append(ChunkSpec(
                rank=rank,
                shot_start=max(1, start - hl),
                shot_end=min(n_shots, end + hr),
                halo_left=hl, halo_right=hr,
            ))
            start = end + 1
        for c in chunks:
            _log_geom.info("  Rank %d: shots %d→%d (halo L=%d R=%d)",
                           c.rank, c.shot_start, c.shot_end, c.halo_left, c.halo_right)
        return chunks

    def run(self) -> Tuple[Geometry, CmpGrid, np.ndarray, np.ndarray, np.ndarray]:
        geom = self.load_geometry()
        statics = self.compute_static_corrections()
        cmp_grid = self.build_cmp_grid()
        vel, vel_x = self.load_velocity()
        return geom, cmp_grid, statics, vel, vel_x


# =============================================================================
#  Section 3 — Traveltime kernels   (traveltime.py)
# =============================================================================

def interp_velocity(
    vel_x: torch.Tensor,
    vel_traces: torch.Tensor,
    x_img: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate velocity traces to imaging positions."""
    if len(vel_x) > 1:
        idx = (x_img - vel_x[0]) / (vel_x[1] - vel_x[0])
    else:
        idx = torch.zeros_like(x_img)
    i0 = idx.long().clamp(0, len(vel_x) - 2)
    i1 = i0 + 1
    frac = (idx - i0.float()).clamp(0.0, 1.0)
    v0 = vel_traces[i0]
    v1 = vel_traces[i1]
    return v0 * (1.0 - frac.unsqueeze(-1)) + v1 * frac.unsqueeze(-1)


@torch.no_grad()
def migrate_one_shot(
    sx: torch.Tensor,
    gather: torch.Tensor,
    x_img: torch.Tensor,
    t0: torch.Tensor,
    dt: float,
    vrms_img: torch.Tensor,
    shot_depth: float = 9.0,
    rec_depth: float = 10.0,
    min_offset: float = 250.0,
    dx_rec: float = 25.0,
    static_shift: float = 0.0,
    rec_batch_size: int = 64,
) -> torch.Tensor:
    """Migrate one shot gather using geometric ray tracing."""
    PI = 3.141592653589793
    Nr = gather.shape[0]
    Na = x_img.shape[0]
    Nt = t0.shape[0]
    device = gather.device
    sx_val = float(sx.item())
    nt_samp = gather.shape[1]

    itrace_idx = torch.arange(Nr, device=device, dtype=torch.float32)
    rx_abs = sx_val - min_offset - itrace_idx * dx_rec

    depth = t0.unsqueeze(0) * vrms_img / 2.0

    dx_src_sq = (x_img - sx_val) ** 2
    dz_src_sq = (depth - shot_depth) ** 2
    dx_src = torch.sqrt(dx_src_sq.unsqueeze(-1) + dz_src_sq + 1e-12)

    partial = torch.zeros(Na, Nt, dtype=torch.float32, device=device)

    for r_start in range(0, Nr, rec_batch_size):
        r_end = min(r_start + rec_batch_size, Nr)
        rx_b = rx_abs[r_start:r_end]
        gather_b = gather[r_start:r_end]
        batch = r_end - r_start

        dx_horiz_sq = (rx_b.unsqueeze(1) - x_img.unsqueeze(0)) ** 2
        dz_rec_sq = (depth.unsqueeze(0) - rec_depth) ** 2
        _dx_rec = torch.sqrt(dx_horiz_sq.unsqueeze(-1) + dz_rec_sq + 1e-12)

        T = (dx_src.unsqueeze(0) + _dx_rec) / (vrms_img.unsqueeze(0) + 1e-12)
        T = T + static_shift

        n_idx = (T / dt).long()
        n_idx = n_idx.clamp(1, nt_samp - 1)

        denom = (vrms_img.unsqueeze(0) + 1e-12) * (dx_src.unsqueeze(0) + 1e-12) * _dx_rec * _dx_rec + 1e-8
        W = depth.unsqueeze(0) / denom

        r_idx = torch.arange(batch, device=device).unsqueeze(1).unsqueeze(2)
        r_exp = r_idx.expand(batch, Na, Nt)
        a_n = gather_b[r_exp, n_idx]
        a_n1 = gather_b[r_exp, n_idx - 1]
        deriv = (a_n - a_n1) / dt

        contribution = W * deriv / (2.0 * PI)
        contribution = torch.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0)
        partial += contribution.sum(dim=0)

        del T, n_idx, W, deriv, contribution, a_n, a_n1, _dx_rec, r_exp
        torch.cuda.empty_cache()

    return partial


# =============================================================================
#  Section 4 — Result aggregator   (aggregator.py)
# =============================================================================

class ResultAggregator:
    """Accumulates partial migration results and writes SEGY output."""

    def __init__(self, cmp_grid: CmpGrid, output_path: str):
        self.cmp_grid = cmp_grid
        self.output_path = output_path
        self._global_image: np.ndarray = np.zeros(
            (cmp_grid.n_cmp, cmp_grid.n_t), dtype=np.float32
        )

    def accumulate(self, image: np.ndarray) -> None:
        if image.shape != self._global_image.shape:
            raise ValueError(
                f"Shape mismatch: got {image.shape}, expected {self._global_image.shape}"
            )
        self._global_image += image.astype(np.float32)

    def normalize_by_fold(self, fold_map: np.ndarray) -> None:
        mask = fold_map > 0
        self._global_image[mask] /= fold_map[mask, np.newaxis]
        _log_agg.info("Normalized by fold (max_fold=%.0f)", fold_map.max())

    def get_image(self) -> np.ndarray:
        return self._global_image.copy()

    def write_segy(self, sample_interval_ms: float = 4.0) -> str:
        _log_agg.info("Writing SEGY output to %s ...", self.output_path)
        import segyio

        spec = segyio.spec()
        spec.sorting = 2
        spec.format = 1
        spec.iline = 189
        spec.xline = 193
        spec.samples = list(range(self.cmp_grid.n_t))
        spec.tracecount = self.cmp_grid.n_cmp

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        with segyio.create(self.output_path, spec) as dst:
            dst.bin[segyio.BinField.Traces] = self.cmp_grid.n_cmp
            dst.bin[segyio.BinField.Samples] = self.cmp_grid.n_t
            dst.bin[segyio.BinField.Interval] = int(sample_interval_ms * 1000)

            for i in range(self.cmp_grid.n_cmp):
                x = self.cmp_grid.x_array[i]
                header = {
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                    segyio.TraceField.CDP: i + 1,
                    segyio.TraceField.CDP_X: int(x * 100),
                    segyio.TraceField.CDP_Y: 0,
                    segyio.TraceField.offset: 0,
                    segyio.TraceField.TRACE_SAMPLE_COUNT: self.cmp_grid.n_t,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(sample_interval_ms * 1000),
                }
                dst.header[i] = header
                dst.trace[i] = self._global_image[i]

        _log_agg.info("SEGY written: %d traces, %d samples", self.cmp_grid.n_cmp, self.cmp_grid.n_t)
        return self.output_path


# =============================================================================
#  Section 5 — GPU worker   (worker.py)
# =============================================================================

class KirchhoffCUDAWorker:
    """Processes a chunk of shot gathers on one GPU."""

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

        self.x_cmp_gpu = torch.as_tensor(cmp_grid.x_array.astype(np.float32), device=self.device)
        self.t0_gpu = torch.as_tensor(cmp_grid.t_array.astype(np.float32), device=self.device)
        self.vel_traces_gpu = torch.as_tensor(vel_traces, device=self.device)
        self.vel_x_gpu = torch.as_tensor(vel_x.astype(np.float32), device=self.device)

        self._partial_image: Optional[torch.Tensor] = None
        self._segy_handle = None

    def _ensure_output_buffer(self) -> torch.Tensor:
        if self._partial_image is None:
            self._partial_image = torch.zeros(
                self.cmp_grid.n_cmp, self.cmp_grid.n_t,
                dtype=torch.float32, device=self.device,
            )
        return self._partial_image

    def _open_segy(self):
        import segyio
        self._segy_handle = segyio.open(
            self.segy_shot_path, "r", strict=False, ignore_geometry=True,
        )

    def _read_shot_gather(self, shot_id: int) -> np.ndarray:
        n_rec = self.geometry.n_rec_per_shot
        start = (shot_id - 1) * n_rec
        data = np.zeros((n_rec, self.cmp_grid.n_t), dtype=np.float32)
        for i in range(n_rec):
            data[i] = self._segy_handle.trace[start + i]
        return data

    @torch.no_grad()
    def process_chunk(self, total_shots: int) -> torch.Tensor:
        output = self._ensure_output_buffer()
        self._open_segy()

        shots = list(range(self.chunk.shot_start, self.chunk.shot_end + 1))
        shot_count = 0
        t_start = time.perf_counter()

        for shot_id in shots:
            shot_count += 1
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
                        _log_worker.critical("OOM after 3 retries on shot %d rank %d", shot_id, self.rank)
                        raise RuntimeError(
                            f"FATAL OOM on shot {shot_id} rank {self.rank} "
                            f"after 3 retries. {memory_snapshot(self.rank)}"
                        ) from e
                    old_bs = self.rec_batch_size
                    self.rec_batch_size = max(8, self.rec_batch_size // 2)
                    _log_worker.warning("OOM on %s: reduced rec_batch %d→%d", fail_key, old_bs, self.rec_batch_size)
                except Exception:
                    raise

            t_shot = time.perf_counter() - t_start if shot_count == 1 else 0.0
            if shot_count % 10 == 0 or shot_count == 1:
                done = shot_count
                elapsed = time.perf_counter() - t_start
                eta = (elapsed / done) * (len(shots) - done) if done > 0 else 0
                _log_worker.info("[rank%d] shot %04d (%d/%d) | %.2fs/shot | ETA %.0fs | %s",
                                 self.rank, shot_id, done, len(shots),
                                 t_shot if shot_count > 1 else (elapsed), eta,
                                 memory_snapshot(self.rank))

        self._close_segy()
        return output

    @torch.no_grad()
    def _migrate_one_shot_in_place(self, shot_id: int, output: torch.Tensor) -> None:
        gather_np = self._read_shot_gather(shot_id)
        gather = torch.as_tensor(gather_np, device=self.device)

        tr0 = (shot_id - 1) * self.geometry.n_rec_per_shot
        static_s = float(self.statics[tr0])
        sx_m = float(self.geometry.sx[tr0])

        x_cmp = self.cmp_grid.x_array
        ap_mask = np.abs(x_cmp - sx_m) < self.max_aperture_m
        ap_indices = np.where(ap_mask)[0]
        if len(ap_indices) == 0:
            del gather
            return
        x_ap = torch.as_tensor(x_cmp[ap_indices].astype(np.float32), device=self.device)

        vrms_ap = interp_velocity(self.vel_x_gpu, self.vel_traces_gpu, x_ap)

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

        ap_idx_gpu = torch.as_tensor(ap_indices, dtype=torch.long, device=self.device)
        output.index_add_(0, ap_idx_gpu, partial)

        del gather, partial, x_ap, vrms_ap, ap_idx_gpu
        gc.collect()
        torch.cuda.empty_cache()

    def _emergency_free(self):
        del self._partial_image
        self._partial_image = None
        gc.collect()
        torch.cuda.empty_cache()
        _log_worker.warning("Emergency free on rank %d: %s", self.rank, memory_snapshot(self.rank))

    def _close_segy(self):
        if self._segy_handle is not None:
            try:
                self._segy_handle.close()
            except Exception:
                pass
            self._segy_handle = None

    def get_partial_image(self) -> torch.Tensor:
        return self._partial_image if self._partial_image is not None else self._ensure_output_buffer()


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
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        worker = KirchhoffCUDAWorker(
            rank=rank, world_size=world_size, device=device,
            geometry=geometry, cmp_grid=cmp_grid, statics=statics,
            vel_traces=vel_traces, vel_x=vel_x,
            segy_shot_path=segy_shot_path, chunk=chunk, **kwargs,
        )

        total_shots = geometry.n_shots
        partial = worker.process_chunk(total_shots)

        _log_worker.info("[rank%d] local processing done, starting all_reduce", rank)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)

        if rank == 0:
            output_queue.put(partial.cpu().numpy())

        dist.barrier()
        dist.destroy_process_group()

    except Exception as e:
        _log_worker.exception("[rank%d] FATAL: %s", rank, e)
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        os._exit(1)


# =============================================================================
#  Section 6 — Multi-GPU scheduler   (scheduler.py)
# =============================================================================

_mp_ctx = mp.get_context("spawn")


class ResourceSafeScheduler:
    """Distributes shots across N GPUs with overlap halos."""

    def __init__(
        self,
        n_shots: int = 0,
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
        self._aborted = _mp_ctx.Event()

    def compute_chunking(self) -> List[ChunkSpec]:
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
                halo_left=hl, halo_right=hr,
            ))
            start = end + 1

        for c in chunks:
            _log_sched.info("  GPU %d: shots %d→%d (halo L=%d R=%d)",
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
                rank=rank, shot_start=start, shot_end=end,
                halo_left=0, halo_right=0,
            ))
            start = end + 1
        return chunks

    @staticmethod
    def _get_free_port() -> int:
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
        chunks = self.compute_chunking()
        queue = _mp_ctx.Queue()

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self._get_free_port())

        self._start_watchdog()

        try:
            _log_sched.info("Spawning %d workers for %d shots ...", self.n_gpus, self.n_shots)
            t0 = time.perf_counter()

            procs = []
            for rank in range(self.n_gpus):
                p = _mp_ctx.Process(
                    target=_launcher_fn,
                    args=(rank, self.n_gpus, segy_shot_path, geometry, cmp_grid,
                          statics, vel_traces, vel_x, chunks, queue, worker_kwargs),
                )
                p.start()
                procs.append(p)

            _log_sched.info("All workers launched, waiting for rank 0 result ...")

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

            _log_sched.info("Result received from rank 0, shape=%s", result.shape)

            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    _log_sched.warning("Worker still alive, terminating")
                    p.terminate()
                    p.join(timeout=10)

            _log_sched.info("All workers completed in %.0fs", time.perf_counter() - t0)
            _log_sched.info("Final image shape: %s, min=%.4f max=%.4f",
                            result.shape, result.min(), result.max())

        except Exception as e:
            _log_sched.exception("Scheduler failure, triggering emergency cleanup")
            self.emergency_cleanup()
            raise
        finally:
            self._stop_watchdog()

        return result

    def _start_watchdog(self):
        if self.timeout_seconds > 0:
            self._aborted.clear()
            self._watchdog = threading.Timer(self.timeout_seconds, self._timeout_handler)
            self._watchdog.daemon = True
            self._watchdog.start()
            _log_sched.info("Watchdog started (timeout=%ds)", self.timeout_seconds)

    def _stop_watchdog(self):
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    def _timeout_handler(self):
        _log_sched.critical("TIMEOUT after %ds → emergency abort", self.timeout_seconds)
        self._aborted.set()
        self.emergency_cleanup()

    def emergency_cleanup(self):
        _log_sched.warning("Emergency cleanup initiated")
        kill_all_children()
        gc.collect()
        _log_sched.warning("Emergency cleanup done")


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
    try:
        chunk = chunks[rank]
        worker_entry(
            rank=rank, world_size=world_size,
            segy_shot_path=segy_shot_path,
            geometry=geometry, cmp_grid=cmp_grid,
            statics=statics, vel_traces=vel_traces, vel_x=vel_x,
            chunk=chunk, output_queue=output_queue,
            **worker_kwargs,
        )
    finally:
        sys.exit(0)


# =============================================================================
#  Section 7 — Top-level engine   (engine.py)
# =============================================================================

class PSTM2DEngine:
    """Enterprise 2D PSTM migration engine.

    Full pipeline:
        1. Parse SEGY headers → Geometry, CmpGrid
        2. Compute statics, load velocity
        3. Schedule shots across N GPUs
        4. Aggregate and write SEGY output
    """

    def __init__(
        self,
        segy_shot_path: str,
        segy_vel_path: str,
        segy_output_path: str,
        *,
        src_depth: float = 9.0,
        rec_depth: float = 10.0,
        datum_elev: float = 0.0,
        replacement_vel: float = 1800.0,
        cmp_spacing_m: float = 25.0,
        max_aperture_m: float = 3000.0,
        dt_ms: float = 4.0,
        n_gpus: int = 4,
        chunk_overlap: int = 10,
        coord_scale: float = 100.0,
        rec_batch_size: int = 64,
        f_max: float = 125.0,
        timeout_seconds: int = 7200,
        log_level: str = "INFO",
    ):
        self.segy_shot_path = segy_shot_path
        self.segy_vel_path = segy_vel_path
        self.segy_output_path = segy_output_path
        self.src_depth = src_depth
        self.rec_depth = rec_depth
        self.datum_elev = datum_elev
        self.replacement_vel = replacement_vel
        self.cmp_spacing_m = cmp_spacing_m
        self.max_aperture_m = 1500.0
        self.dt_ms = dt_ms
        self.n_gpus = n_gpus
        self.chunk_overlap = chunk_overlap
        self.coord_scale = coord_scale
        self.rec_batch_size = rec_batch_size
        self.f_max = f_max
        self.timeout_seconds = timeout_seconds

        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        self._check_gpus()
        self.preprocessor = GeomPreProcessor(
            segy_shot_path=segy_shot_path,
            segy_vel_path=segy_vel_path,
            src_depth=src_depth,
            rec_depth=rec_depth,
            datum_elev=datum_elev,
            replacement_vel=replacement_vel,
            cmp_spacing=cmp_spacing_m,
            max_aperture_m=max_aperture_m,
            coord_scale=coord_scale,
        )

    def _check_gpus(self):
        available = torch.cuda.device_count()
        if available < self.n_gpus:
            _log_engine.warning("Requested %d GPUs but only %d available; using %d",
                                self.n_gpus, available, available)
            self.n_gpus = max(1, available)
        _log_engine.info("Using %d / %d GPUs available", self.n_gpus, available)

    def run(self) -> str:
        """Execute full PSTM pipeline and return output file path."""
        t_total = time.perf_counter()

        with EmergencyCleanup(on_cleanup=lambda: kill_all_children()):
            _log_engine.info("=" * 60)
            _log_engine.info("Phase 1: Geometry & Velocity Preprocessing")
            _log_engine.info("=" * 60)

            geom, cmp_grid, statics, vel_traces, vel_x = self.preprocessor.run()

            _log_engine.info("=" * 60)
            _log_engine.info("Phase 2: Multi-GPU PSTM Migration")
            _log_engine.info("=" * 60)

            scheduler = ResourceSafeScheduler(
                n_shots=geom.n_shots,
                n_gpus=self.n_gpus,
                chunk_overlap=self.chunk_overlap,
                timeout_seconds=self.timeout_seconds,
            )

            result = scheduler.launch(
                segy_shot_path=self.segy_shot_path,
                geometry=geom,
                cmp_grid=cmp_grid,
                statics=statics,
                vel_traces=vel_traces,
                vel_x=vel_x,
                max_aperture_m=self.max_aperture_m,
                rec_batch_size=self.rec_batch_size,
                f_max=self.f_max,
            )

            _log_engine.info("=" * 60)
            _log_engine.info("Phase 3: SEGY Output")
            _log_engine.info("=" * 60)

            aggregator = ResultAggregator(cmp_grid, self.segy_output_path)
            aggregator.accumulate(result)

            fold_map = self._compute_fold_map(geom, cmp_grid)
            aggregator.normalize_by_fold(fold_map)

            output_file = aggregator.write_segy(sample_interval_ms=self.dt_ms)

            elapsed = time.perf_counter() - t_total
            _log_engine.info("=" * 60)
            _log_engine.info("PSTM complete in %.0fs (%.1f min)", elapsed, elapsed / 60)
            _log_engine.info("Output: %s", output_file)
            _log_engine.info("=" * 60)

            return output_file

    @staticmethod
    def _compute_fold_map(geom: Geometry, cmp_grid: CmpGrid) -> np.ndarray:
        fold = np.zeros(cmp_grid.n_cmp, dtype=np.float32)
        cmp_x = geom.cmp_x
        for x in cmp_x:
            idx = int(round((x - cmp_grid.x_min) / cmp_grid.spacing))
            if 0 <= idx < cmp_grid.n_cmp:
                fold[idx] += 1.0
        fold = np.maximum(fold, 1.0)
        _log_engine.info("Fold map: min=%.0f max=%.0f", fold.min(), fold.max())
        return fold

    def emergency_shutdown(self, reason: str = "unknown") -> None:
        _log_engine.critical("EMERGENCY SHUTDOWN: %s", reason)
        kill_all_children()
        gc.collect()
        for i in range(self.n_gpus):
            try:
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            except Exception:
                pass
        sys.exit(1)


# =============================================================================
#  Section 8 — CLI entry point   (run_pstm.py)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="[LJF] 2D Pre-Stack Time Migration (Kirchhoff PSTM) on Multi-GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shot-path", default="shot_data.sgy",
                        help="Input SEGY shot gathers")
    parser.add_argument("--vel-path", default="vel_data.sgy",
                        help="Input SEGY RMS velocity")
    parser.add_argument("--output", default="pstm_result.sgy",
                        help="Output SEGY migrated section")
    parser.add_argument("--work-dir", default=None,
                        help="Working directory (default: script directory)")
    parser.add_argument("--src-depth", type=float, default=9.0,
                        help="Source depth (m)")
    parser.add_argument("--rec-depth", type=float, default=10.0,
                        help="Receiver depth (m)")
    parser.add_argument("--datum-elev", type=float, default=0.0,
                        help="Datum elevation (m)")
    parser.add_argument("--replacement-vel", type=float, default=1800.0,
                        help="Replacement velocity (m/s)")
    parser.add_argument("--cmp-spacing", type=float, default=25.0,
                        help="CMP output spacing (m)")
    parser.add_argument("--max-aperture", type=float, default=3000.0,
                        help="Max migration aperture (m)")
    parser.add_argument("--dt-ms", type=float, default=4.0,
                        help="Sample interval (ms)")
    parser.add_argument("--n-gpus", type=int, default=4,
                        help="Number of GPUs")
    parser.add_argument("--chunk-overlap", type=int, default=10,
                        help="Shot overlap between workers")
    parser.add_argument("--rec-batch-size", type=int, default=64,
                        help="Receiver batch size (GPU memory control)")
    parser.add_argument("--f-max", type=float, default=125.0,
                        help="Anti-aliasing max frequency (Hz)")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Watchdog timeout (seconds)")
    parser.add_argument("--test-shots", type=int, default=None,
                        help="Process only first N shots (for validation)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--coord-scale", type=float, default=100.0,
                        help="SEGY coordinate divisor (100 = cm→m)")

    args = parser.parse_args()

    if args.work_dir:
        work_dir = args.work_dir
    else:
        work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)
    sys.path.insert(0, work_dir)

    shot_path = os.path.join(work_dir, args.shot_path)
    vel_path = os.path.join(work_dir, args.vel_path)
    output_path = os.path.join(work_dir, args.output)

    for p, label in [(shot_path, "shot"), (vel_path, "velocity")]:
        if not os.path.exists(p):
            print(f"[LJF] ERROR: {label} file not found: {p}")
            sys.exit(1)

    print("=" * 60)
    print("  Kirchhoff PSTM 2D — Author: LJF | GPL-3.0 License")
    print("=" * 60)

    engine = PSTM2DEngine(
        segy_shot_path=shot_path,
        segy_vel_path=vel_path,
        segy_output_path=output_path,
        src_depth=args.src_depth,
        rec_depth=args.rec_depth,
        datum_elev=args.datum_elev,
        replacement_vel=args.replacement_vel,
        cmp_spacing_m=args.cmp_spacing,
        max_aperture_m=args.max_aperture,
        dt_ms=args.dt_ms,
        n_gpus=args.n_gpus,
        chunk_overlap=args.chunk_overlap,
        coord_scale=args.coord_scale,
        rec_batch_size=args.rec_batch_size,
        f_max=args.f_max,
        timeout_seconds=args.timeout,
        log_level=args.log_level,
    )

    if args.test_shots is not None:
        print(f"\n  [LJF] TEST MODE: processing first {args.test_shots} shots only\n")
        orig_run = engine.preprocessor.load_geometry

        def patched_load():
            geom = orig_run()
            geom.n_shots = min(geom.n_shots, args.test_shots)
            return geom

        engine.preprocessor.load_geometry = patched_load
        if hasattr(engine, "n_gpus"):
            engine.n_gpus = min(engine.n_gpus, max(1, args.test_shots))

    try:
        result_path = engine.run()
        print(f"\n  [LJF] SUCCESS — output written to: {result_path}")
    except Exception as e:
        print(f"\n  [LJF] FAILED: {e}")
        engine.emergency_shutdown(str(e))
        sys.exit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
