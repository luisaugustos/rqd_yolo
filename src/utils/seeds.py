"""Global and per-module seed management (NFR-001, NFR-016).

Provides a single entry-point to set all stochastic elements from one seed,
and per-module seed helpers for isolated unit testing.
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """Set seeds for Python random, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value to apply everywhere.
        deterministic: When True, enable torch deterministic algorithms.
            This may slow training but guarantees bit-for-bit reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    _set_torch_seed(seed, deterministic=deterministic)
    logger.info("Global seed set to %d (deterministic=%s)", seed, deterministic)


def _set_torch_seed(seed: int, *, deterministic: bool) -> None:
    """Set PyTorch seeds, guarding the import so CPU-only environments work."""
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            import os

            # Required environment variable for full CUDA determinism
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            logger.warning(
                "Deterministic CUDA mode enabled. This may reduce training throughput."
            )
    except ImportError:
        logger.debug("torch not available; skipping torch seed setup")


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seed function (pass to DataLoader worker_init_fn).

    Ensures that each dataloader worker uses a different but deterministic seed
    derived from the worker's ID and the base numpy seed.

    Args:
        worker_id: Worker index provided by PyTorch DataLoader.
    """
    worker_seed = int(np.random.get_state()[1][0]) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
