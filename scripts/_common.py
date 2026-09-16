"""Shared script setup: limit BLAS threads per worker and make the package importable."""
import os
import sys
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parallel_map(fn, args_list, n_jobs=5):
    """Run fn(*args) for each args tuple in separate processes, keeping order."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(fn, *args) for args in args_list]
        return [f.result() for f in futures]
