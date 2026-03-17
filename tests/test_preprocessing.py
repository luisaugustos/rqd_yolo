"""Tests for the Preprocessing module (FR-002)."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.preprocessor import Preprocessor
from src.utils.contracts import BBox, ImageSample


def _make_sample(h: int = 480, w: int = 640) -> ImageSample:
    image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return ImageSample(image_id="test", file_path="/fake.jpg", image=image, width=w, height=h)


class TestPreprocessor:
    """Tests for Preprocessor.process and coordinate inversion."""

    def test_output_tensor_shape(self) -> None:
        pp = Preprocessor({"target_size": 640})
        sample = _make_sample(480, 640)
        processed = pp.process(sample)
        assert processed.tensor.shape == (640, 640, 3)
        assert processed.tensor.dtype == np.float32

    def test_output_tensor_shape_non_square_input(self) -> None:
        pp = Preprocessor({"target_size": 640})
        sample = _make_sample(300, 800)
        processed = pp.process(sample)
        assert processed.tensor.shape == (640, 640, 3)

    def test_metadata_original_dimensions_preserved(self) -> None:
        pp = Preprocessor()
        sample = _make_sample(480, 640)
        processed = pp.process(sample)
        assert processed.metadata.original_width == 640
        assert processed.metadata.original_height == 480

    def test_invert_coords_identity_for_non_padded_image(self) -> None:
        """A square image at target size should invert to near-original coords."""
        pp = Preprocessor({"target_size": 640})
        sample = _make_sample(640, 640)
        processed = pp.process(sample)
        coords = np.array([[0.0, 0.0, 640.0, 640.0]])
        inv = pp.invert_coords(coords, processed.metadata)
        assert inv[0, 0] == pytest.approx(0.0, abs=1.0)
        assert inv[0, 2] == pytest.approx(640.0, abs=1.0)

    def test_invert_bbox_roundtrip(self) -> None:
        pp = Preprocessor({"target_size": 320})
        sample = _make_sample(480, 640)
        processed = pp.process(sample)
        # A box in original image space
        original_box = BBox(x1=100.0, y1=50.0, x2=300.0, y2=150.0)
        # Scale to preprocessed space
        meta = processed.metadata
        scaled_box = BBox(
            x1=original_box.x1 * meta.scale_x + meta.pad_left,
            y1=original_box.y1 * meta.scale_y + meta.pad_top,
            x2=original_box.x2 * meta.scale_x + meta.pad_left,
            y2=original_box.y2 * meta.scale_y + meta.pad_top,
        )
        # Invert back
        restored = pp.invert_bbox(scaled_box, meta)
        assert restored.x1 == pytest.approx(original_box.x1, abs=1.0)
        assert restored.y1 == pytest.approx(original_box.y1, abs=1.0)

    def test_deterministic_output(self) -> None:
        pp = Preprocessor()
        sample = _make_sample()
        t1 = pp.process(sample).tensor
        t2 = pp.process(sample).tensor
        np.testing.assert_array_equal(t1, t2)

    def test_imagenet_normalization_range(self) -> None:
        pp = Preprocessor({"normalization": "imagenet"})
        sample = _make_sample()
        t = pp.process(sample).tensor
        # ImageNet-normalized values can be negative; check dtype and finite
        assert t.dtype == np.float32
        assert np.all(np.isfinite(t))

    def test_zero_one_normalization_range(self) -> None:
        pp = Preprocessor({"normalization": "zero_one"})
        sample = _make_sample()
        t = pp.process(sample).tensor
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.01  # small tolerance for padding grey
