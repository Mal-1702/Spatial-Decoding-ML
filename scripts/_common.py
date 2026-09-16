"""Shared script setup: limit BLAS threads per worker and make the package importable."""
import os
import sys
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
