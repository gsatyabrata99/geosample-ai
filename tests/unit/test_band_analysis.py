"""Unit tests for spectral band analysis."""

import numpy as np
import pytest

from src.processing.spectral.band_analysis import (
    compute_alteration_score,
    compute_clay_index,
    compute_iron_oxide_index,
    normalise,
)


def test_clay_index_basic():
    b11 = np.array([1.0, 2.0, 3.0])
    b12 = np.array([1.0, 1.0, 1.0])
    result = compute_clay_index(b11, b12)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_clay_index_zero_denominator():
    b11 = np.array([1.0, 2.0])
    b12 = np.array([0.0, 1.0])
    result = compute_clay_index(b11, b12)
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(2.0)


def test_iron_oxide_index():
    b11 = np.array([2.0, 4.0])
    b8a = np.array([1.0, 2.0])
    result = compute_iron_oxide_index(b11, b8a)
    np.testing.assert_allclose(result, [2.0, 2.0])


def test_normalise_range():
    arr = np.array([0.0, 5.0, 10.0])
    result = normalise(arr)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_normalise_constant_array():
    arr = np.array([3.0, 3.0, 3.0])
    result = normalise(arr)
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])


def test_alteration_score_shape():
    shape = (10, 10)
    b11 = np.random.rand(*shape)
    b12 = np.random.rand(*shape) + 0.1
    b8a = np.random.rand(*shape) + 0.1
    score = compute_alteration_score(b11, b12, b8a)
    assert score.shape == shape
    assert score.min() >= 0.0
    assert score.max() <= 1.0
