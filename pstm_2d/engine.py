"""Top-level 2D PSTM engine orchestrating preprocessing, scheduling, and output."""

import gc
import logging
import os
import sys
import time
import numpy as np
import torch

from .geometry import GeomPreProcessor, Geometry, CmpGrid
from .scheduler import ResourceSafeScheduler
from .aggregator import ResultAggregator
from .safety import EmergencyCleanup, kill_all_children

logger = logging.getLogger(__name__)


class PSTM2DEngine:
    """Enterprise 2D PSTM migration engine.

    Full pipeline:
        1. Parse SEGY headers → Geometry, CmpGrid
        2. Compute statics, load velocity
        3. Schedule shots across 4 GPUs
        4. Aggregate and write SEGY output

    Parameters
    ----------
    segy_shot_path : str
        Path to input SEGY shot gathers.
    segy_vel_path : str
        Path to input SEGY RMS velocity field.
    segy_output_path : str
        Destination path for the migrated SEGY section.
    src_depth : float
        Source depth in metres (9.0).
    rec_depth : float
        Receiver depth in metres (10.0).
    datum_elev : float
        Datum elevation in metres (0.0).
    replacement_vel : float
        Replacement velocity for statics in m/s (1800.0).
    cmp_spacing_m : float
        Output CMP grid spacing in metres (25.0).
    max_aperture_m : float
        Maximum migration aperture in metres (3000.0).
    dt_ms : float
        Time sampling interval in milliseconds (4.0).  Used only if
        the SEGY binary header is unreliable.
    n_gpus : int
        Number of GPUs to use (4).
    chunk_overlap : int
        Number of overlapping shots between adjacent workers (10).
    coord_scale : float
        Divisor for SEGY coordinates (100 = cm→m).
    rec_batch_size : int
        Receiver batch size for GPU memory control (64).
    f_max : float
        Anti-aliasing maximum frequency in Hz (125.0).
    timeout_seconds : int
        Watchdog timeout in seconds (7200).
    log_level : str
        Logging level.
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
        self.max_aperture_m = 1500.0  # 60 CMP × 25m, matching C++ kongjing=60
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

    # ------------------------------------------------------------------
    #  GPU health check
    # ------------------------------------------------------------------
    def _check_gpus(self):
        available = torch.cuda.device_count()
        if available < self.n_gpus:
            logger.warning("Requested %d GPUs but only %d available; using %d",
                           self.n_gpus, available, available)
            self.n_gpus = max(1, available)
        logger.info("Using %d / %d GPUs available", self.n_gpus, available)

    # ------------------------------------------------------------------
    #  Full pipeline
    # ------------------------------------------------------------------
    def run(self) -> str:
        """Execute full PSTM pipeline and return output file path."""
        t_total = time.perf_counter()

        with EmergencyCleanup(on_cleanup=lambda: kill_all_children()):
            # === Phase 1: Preprocessing ===
            logger.info("=" * 60)
            logger.info("Phase 1: Geometry & Velocity Preprocessing")
            logger.info("=" * 60)

            geom, cmp_grid, statics, vel_traces, vel_x = self.preprocessor.run()

            # === Phase 2: Multi-GPU Migration ===
            logger.info("=" * 60)
            logger.info("Phase 2: Multi-GPU PSTM Migration")
            logger.info("=" * 60)

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

            # === Phase 3: Output ===
            logger.info("=" * 60)
            logger.info("Phase 3: SEGY Output")
            logger.info("=" * 60)

            aggregator = ResultAggregator(cmp_grid, self.segy_output_path)
            aggregator.accumulate(result)

            # Compute approximate fold map from CMP coordinates
            fold_map = self._compute_fold_map(geom, cmp_grid)
            aggregator.normalize_by_fold(fold_map)

            output_file = aggregator.write_segy(sample_interval_ms=self.dt_ms)

            elapsed = time.perf_counter() - t_total
            logger.info("=" * 60)
            logger.info("PSTM complete in %.0fs (%.1f min)", elapsed, elapsed / 60)
            logger.info("Output: %s", output_file)
            logger.info("=" * 60)

            return output_file

    # ------------------------------------------------------------------
    #  Fold map helper
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_fold_map(geom: Geometry, cmp_grid: CmpGrid) -> np.ndarray:
        """Estimate fold coverage at each CMP bin."""
        fold = np.zeros(cmp_grid.n_cmp, dtype=np.float32)
        cmp_x = geom.cmp_x
        edge = cmp_grid.spacing / 2.0
        for x in cmp_x:
            idx = int(round((x - cmp_grid.x_min) / cmp_grid.spacing))
            if 0 <= idx < cmp_grid.n_cmp:
                fold[idx] += 1.0
        fold = np.maximum(fold, 1.0)
        logger.info("Fold map: min=%.0f max=%.0f", fold.min(), fold.max())
        return fold

    # ------------------------------------------------------------------
    #  Emergency shutdown
    # ------------------------------------------------------------------
    def emergency_shutdown(self, reason: str = "unknown") -> None:
        logger.critical("EMERGENCY SHUTDOWN: %s", reason)
        kill_all_children()
        gc.collect()
        for i in range(self.n_gpus):
            try:
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            except Exception:
                pass
        sys.exit(1)
