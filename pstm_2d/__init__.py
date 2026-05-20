# Copyright (c) LJF. All Rights Reserved. | Licensed under GPL-3.0.
# Unauthorized commercial use is strictly prohibited.
"""PSTM2D - Enterprise-grade 2D Pre-Stack Time Migration on Multi-GPU."""

__version__ = "1.0.0"
__author__ = "LJF"
__copyright__ = "Copyright (c) LJF. All Rights Reserved."

from .engine import PSTM2DEngine
from .geometry import Geometry, CmpGrid, ChunkSpec, GeomPreProcessor
from .worker import KirchhoffCUDAWorker
from .scheduler import ResourceSafeScheduler
from .aggregator import ResultAggregator
from .safety import CircuitBreaker, EmergencyCleanup, memory_snapshot
