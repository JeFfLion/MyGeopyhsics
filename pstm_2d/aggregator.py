# Copyright (c) LJF. All Rights Reserved. | Licensed under GPL-3.0.
# Unauthorized commercial use is strictly prohibited.
"""Result aggregation and SEGY output writer."""

import logging
import os
import numpy as np

from .geometry import CmpGrid

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Accumulates partial migration results and writes SEGY output.

    Parameters
    ----------
    cmp_grid : CmpGrid
        Output imaging grid metadata.
    output_path : str
        Destination path for the output SEGY file.
    """

    def __init__(self, cmp_grid: CmpGrid, output_path: str):
        self.cmp_grid = cmp_grid
        self.output_path = output_path
        self._global_image: np.ndarray = np.zeros(
            (cmp_grid.n_cmp, cmp_grid.n_t), dtype=np.float32
        )

    def accumulate(self, image: np.ndarray) -> None:
        """Accumulate a partial image onto the global buffer."""
        if image.shape != self._global_image.shape:
            raise ValueError(
                f"Shape mismatch: got {image.shape}, expected {self._global_image.shape}"
            )
        self._global_image += image.astype(np.float32)

    def normalize_by_fold(self, fold_map: np.ndarray) -> None:
        """Divide by fold coverage map (avoid division by zero)."""
        mask = fold_map > 0
        self._global_image[mask] /= fold_map[mask, np.newaxis]
        logger.info("Normalized by fold (max_fold=%.0f)", fold_map.max())

    def get_image(self) -> np.ndarray:
        return self._global_image.copy()

    def write_segy(self, sample_interval_ms: float = 4.0) -> str:
        """Write accumulated image to SEGY file.

        Uses the first SEGY file as a template for header metadata.
        """
        logger.info("Writing SEGY output to %s ...", self.output_path)
        import segyio

        spec = segyio.spec()
        spec.sorting = 2           # 2 = inline sort (2D line)
        spec.format = 1            # 1 = IBM float, 5 = IEEE float
        spec.iline = 189
        spec.xline = 193
        spec.samples = list(range(self.cmp_grid.n_t))
        spec.tracecount = self.cmp_grid.n_cmp

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        with segyio.create(self.output_path, spec) as dst:
            # Binary header
            dst.bin[segyio.BinField.Traces] = self.cmp_grid.n_cmp
            dst.bin[segyio.BinField.Samples] = self.cmp_grid.n_t
            dst.bin[segyio.BinField.Interval] = int(sample_interval_ms * 1000)  # µs

            for i in range(self.cmp_grid.n_cmp):
                x = self.cmp_grid.x_array[i]
                header = {
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                    segyio.TraceField.CDP: i + 1,
                    segyio.TraceField.CDP_X: int(x * 100),    # store as cm
                    segyio.TraceField.CDP_Y: 0,
                    segyio.TraceField.offset: 0,               # stacked section
                    segyio.TraceField.TRACE_SAMPLE_COUNT: self.cmp_grid.n_t,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(sample_interval_ms * 1000),
                }
                dst.header[i] = header
                dst.trace[i] = self._global_image[i]

        logger.info("SEGY written: %d traces, %d samples", self.cmp_grid.n_cmp, self.cmp_grid.n_t)
        return self.output_path
