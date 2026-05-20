# Copyright (c) LJF. All Rights Reserved. | Licensed under GPL-3.0.
# Unauthorized commercial use is strictly prohibited.
"""Geometry data structures and SEGY header pre-processing."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


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
        """Return (start, end) trace indices for a given 1-based shot_id."""
        base = (shot_id - 1) * self.n_rec_per_shot
        return base, base + self.n_rec_per_shot


@dataclass
class CmpGrid:
    """Output CMP imaging grid definition."""
    x_min: float
    x_max: float
    spacing: float           # meters
    n_cmp: int
    n_t: int                 # time samples
    dt: float                # seconds

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
    shot_start: int          # 1-based
    shot_end: int            # 1-based, inclusive
    halo_left: int = 0       # number of overlapping shots on left
    halo_right: int = 0      # number of overlapping shots on right
    output_x_slice: Optional[slice] = None  # CMP output slice for this chunk

    @property
    def core_shots(self) -> slice:
        """Return slice for core shots (1-based), excluding halos."""
        return slice(self.shot_start + self.halo_left,
                     self.shot_end - self.halo_right + 1)

    @property
    def all_shots(self) -> slice:
        """Return slice for all shots including halos (1-based)."""
        return slice(self.shot_start, self.shot_end + 1)


class GeomPreProcessor:
    """Reads SEGY shot headers, builds geometry, computes statics and CMP grid.

    Coordinate convention:
        SEGY stores coordinates in centimetres. The ElevationScalar header
        value (-100) indicates a divisor of 100 for depth/elevation.
        We convert all spatial coordinates to metres by dividing by 100.
    """

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
        coord_scale: float = 100.0,          # divisor: SEGY coords → metres
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
        self.static_shifts: Optional[np.ndarray] = None   # [N_traces] seconds
        self.vel_traces: Optional[np.ndarray] = None       # [N_vel, Nt]
        self.vel_x_array: Optional[np.ndarray] = None      # [N_vel] CMP positions
        self.dt: float = 0.0
        self.n_t: int = 0

    # ------------------------------------------------------------------
    #  SEGY I/O helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _open_segy(path: str):
        import segyio
        return segyio.open(path, "r", strict=False, ignore_geometry=True)

    @staticmethod
    def _coord_to_m(val_cm, scale):
        """Convert SEGY coordinate (cm) to metres."""
        if abs(scale) > 1e-6:
            return np.asarray(val_cm, dtype=np.float64) / abs(scale)
        return np.asarray(val_cm, dtype=np.float64)

    # ------------------------------------------------------------------
    #  Geometry extraction
    # ------------------------------------------------------------------
    def load_geometry(self) -> Geometry:
        """Extract source, receiver, offset, CMP geometry from SEGY headers."""
        logger.info("Loading geometry from %s ...", self.segy_shot_path)
        import segyio
        TF = segyio.TraceField

        with self._open_segy(self.segy_shot_path) as f:
            n_traces = f.tracecount
            self.n_t = len(f.samples)
            self.dt = float(f.samples[1] - f.samples[0]) / 1000.0  # ms → s
            logger.info("  %d traces, nt=%d, dt=%.4fs", n_traces, self.n_t, self.dt)

            sx_raw = np.zeros(n_traces, dtype=np.int32)
            gx_raw = np.zeros(n_traces, dtype=np.int32)
            fldr_raw = np.zeros(n_traces, dtype=np.int32)

            for i in range(n_traces):
                h = f.header[i]
                sx_raw[i] = h[TF.SourceX]
                gx_raw[i] = h[TF.GroupX]
                fldr_raw[i] = h[TF.FieldRecord]

        # Convert to metres
        s = self.coord_scale
        sx_m = self._coord_to_m(sx_raw, s)
        gx_m = self._coord_to_m(gx_raw, s)
        offset_m = sx_m - gx_m
        cmp_x_m = (sx_m + gx_m) / 2.0

        # Determine n_shots / n_rec
        fldr_ids = np.unique(fldr_raw)
        n_shots = len(fldr_ids)
        shot_size = int(n_traces / n_shots) if n_shots > 0 else self.n_rec_expected

        self.geometry = Geometry(
            sx=sx_m,
            gx=gx_m,
            offset=offset_m,
            cmp_x=cmp_x_m,
            fldr=fldr_raw,
            n_shots=n_shots,
            n_rec_per_shot=shot_size,
        )

        logger.info("  n_shots=%d, n_rec_per_shot=%d", n_shots, shot_size)
        logger.info("  sx: %.1f → %.1f m", sx_m.min(), sx_m.max())
        logger.info("  gx: %.1f → %.1f m", gx_m.min(), gx_m.max())
        logger.info("  offset: %.1f → %.1f m", offset_m.min(), offset_m.max())
        logger.info("  cmp_x: %.1f → %.1f m", cmp_x_m.min(), cmp_x_m.max())
        return self.geometry

    # ------------------------------------------------------------------
    #  Static corrections
    # ------------------------------------------------------------------
    def compute_static_corrections(self) -> np.ndarray:
        """Compute per-trace static shifts (seconds) for datum correction.

        Uses replacement velocity to account for source/receiver depth below datum.
          Δt = (|h_s - datum| + |h_r - datum|) / v_rep
        """
        if self.geometry is None:
            raise RuntimeError("Call load_geometry() first.")

        ds = abs(self.src_depth - self.datum_elev)
        dr = abs(self.rec_depth - self.datum_elev)
        dt_static = (ds + dr) / self.replacement_vel
        self.static_shifts = np.full(self.geometry.n_traces, dt_static, dtype=np.float32)
        logger.info("Static shift: %.4f ms per trace (src=%.1fm, rec=%.1fm, v_rep=%.1fm/s)",
                     dt_static * 1000, self.src_depth, self.rec_depth, self.replacement_vel)
        return self.static_shifts

    # ------------------------------------------------------------------
    #  CMP grid
    # ------------------------------------------------------------------
    def build_cmp_grid(self) -> CmpGrid:
        """Build the output CMP grid covering data extent at cmp_spacing."""
        if self.geometry is None:
            raise RuntimeError("Call load_geometry() first.")

        cmp_min = self.geometry.cmp_x.min()
        cmp_max = self.geometry.cmp_x.max()
        n_cmp = int(round((cmp_max - cmp_min) / self.cmp_spacing)) + 1

        self.cmp_grid = CmpGrid(
            x_min=cmp_min,
            x_max=cmp_max,
            spacing=self.cmp_spacing,
            n_cmp=n_cmp,
            n_t=self.n_t,
            dt=self.dt,
        )
        logger.info("CMP grid: %.1f → %.1f m, %d points @ %.0fm spacing",
                     cmp_min, cmp_max, n_cmp, self.cmp_spacing)
        return self.cmp_grid

    # ------------------------------------------------------------------
    #  Velocity field
    # ------------------------------------------------------------------
    def load_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load RMS velocity traces from SEGY velocity file.

        Returns:
            vel_traces: [N_vel_traces, n_t] float32 array of v_rms (m/s)
            vel_x:      [N_vel_traces] float64 array of CMP positions (m)
        """
        logger.info("Loading velocity from %s ...", self.segy_vel_path)
        import segyio

        with self._open_segy(self.segy_vel_path) as f:
            n_vel = f.tracecount
            n_samp = len(f.samples)
            vel_dt = float(f.samples[1] - f.samples[0]) / 1000.0
            logger.info("  %d velocity traces, nt=%d, dt=%.4fs", n_vel, n_samp, vel_dt)

            vel_data = np.zeros((n_vel, n_samp), dtype=np.float32)
            for i in range(n_vel):
                vel_data[i] = f.trace[i].astype(np.float32)

        # Infer CMP positions: assume uniform spacing, first trace at x=0
        with self._open_segy(self.segy_vel_path) as f:
            h0 = f.header[0]
            import segyio
            TF = segyio.TraceField
            cdp_x0 = self._coord_to_m(h0[TF.CDP_X], self.coord_scale)
            sx0 = self._coord_to_m(h0[TF.SourceX], self.coord_scale)

        # Use SourceX if CDP_X is zero (common in 2D lines)
        first_x = cdp_x0 if cdp_x0 > 0 else sx0

        # If both are 0, assume start from 0
        if first_x == 0 and n_vel > 1:
            with self._open_segy(self.segy_vel_path) as f:
                hL = f.header[n_vel - 1]
                import segyio
                TF = segyio.TraceField
                last_x = self._coord_to_m(hL[TF.SourceX], self.coord_scale)
                if last_x == 0:
                    last_x = self._coord_to_m(hL[TF.CDP_X], self.coord_scale)

            if last_x > 0:
                spacing = last_x / (n_vel - 1)
            else:
                # Fallback: 12.5m spacing over 37275m line
                spacing = 12.5
        else:
            spacing = self.cmp_spacing  # nominal

        vel_x = np.arange(n_vel, dtype=np.float64) * spacing + first_x

        # If dt differs from shot dt or n_samp differs, handle
        if abs(vel_dt - self.dt) > 1e-6 or n_samp != self.n_t:
            logger.warning("Velocity time axis mismatch (dt=%.4f vs %.4f, nt=%d vs %d)",
                           vel_dt, self.dt, n_samp, self.n_t)
            # Resample to match shot time axis
            t_old = np.arange(n_samp) * vel_dt
            t_new = np.arange(self.n_t) * self.dt
            from scipy.interpolate import interp1d
            vel_resampled = np.zeros((n_vel, self.n_t), dtype=np.float32)
            for i in range(n_vel):
                vel_resampled[i] = interp1d(
                    t_old, vel_data[i], kind="linear",
                    bounds_error=False, fill_value="extrapolate"
                )(t_new).astype(np.float32)
            vel_data = vel_resampled

        self.vel_traces = vel_data
        self.vel_x_array = vel_x
        logger.info("  Velocity spacing: %.2f m, range: %.1f → %.1f m",
                     spacing, vel_x[0], vel_x[-1])
        return vel_data, vel_x

    # ------------------------------------------------------------------
    #  Chunking
    # ------------------------------------------------------------------
    def compute_chunking(
        self, n_gpus: int = 4, chunk_overlap: int = 10
    ) -> List[ChunkSpec]:
        """Divide 601 shots into n_gpus chunks with halo overlap."""
        if self.geometry is None:
            raise RuntimeError("Call load_geometry() first.")

        n_shots = self.geometry.n_shots
        if n_shots < n_gpus:
            n_gpus = max(1, n_shots)
            chunk_overlap = 0

        base = n_shots // n_gpus
        rem = n_shots % n_gpus
        chunks = []
        start = 1  # 1-based shot numbering

        for rank in range(n_gpus):
            size = base + (1 if rank < rem else 0)
            end = start + size - 1

            hl = chunk_overlap if rank > 0 else 0
            hr = chunk_overlap if rank < n_gpus - 1 else 0

            shot_start = start - hl
            shot_end = end + hr

            chunks.append(ChunkSpec(
                rank=rank,
                shot_start=max(1, shot_start),
                shot_end=min(n_shots, shot_end),
                halo_left=hl,
                halo_right=hr,
            ))
            start = end + 1

        for c in chunks:
            logger.info("  Rank %d: shots %d→%d (halo L=%d R=%d)",
                         c.rank, c.shot_start, c.shot_end, c.halo_left, c.halo_right)

        return chunks

    # ------------------------------------------------------------------
    #  Full pipeline step
    # ------------------------------------------------------------------
    def run(self) -> Tuple[Geometry, CmpGrid, np.ndarray, np.ndarray, np.ndarray]:
        """Run all preprocessing steps and return essential outputs."""
        geom = self.load_geometry()
        statics = self.compute_static_corrections()
        cmp_grid = self.build_cmp_grid()
        vel, vel_x = self.load_velocity()
        return geom, cmp_grid, statics, vel, vel_x
