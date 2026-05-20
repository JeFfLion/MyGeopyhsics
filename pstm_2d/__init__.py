"""PSTM2D - Enterprise-grade 2D Pre-Stack Time Migration on Multi-GPU."""

__version__ = "1.0.0"

from .engine import PSTM2DEngine
from .geometry import Geometry, CmpGrid, ChunkSpec, GeomPreProcessor
from .worker import KirchhoffCUDAWorker
from .scheduler import ResourceSafeScheduler
from .aggregator import ResultAggregator
from .safety import CircuitBreaker, EmergencyCleanup, memory_snapshot
