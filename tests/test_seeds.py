"""Tests for seed management (NFR-001, NFR-016)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.utils.seeds import set_global_seed


class TestSetGlobalSeed:
    """Tests for set_global_seed."""

    def test_python_random_reproducible(self) -> None:
        set_global_seed(42)
        val1 = random.random()
        set_global_seed(42)
        val2 = random.random()
        assert val1 == val2

    def test_numpy_random_reproducible(self) -> None:
        set_global_seed(123)
        arr1 = np.random.rand(10)
        set_global_seed(123)
        arr2 = np.random.rand(10)
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_seeds_give_different_values(self) -> None:
        set_global_seed(1)
        val1 = random.random()
        set_global_seed(2)
        val2 = random.random()
        assert val1 != val2

    def test_torch_seed_when_available(self) -> None:
        """Verify torch seed is set without error (torch may not be installed)."""
        try:
            import torch

            set_global_seed(99)
            t1 = torch.rand(5)
            set_global_seed(99)
            t2 = torch.rand(5)
            assert torch.allclose(t1, t2)
        except ImportError:
            pytest.skip("torch not installed")
