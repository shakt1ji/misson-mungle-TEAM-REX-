"""Tests for terrain.slope (flat terrain, 45-degree terrain, NoData, invalid inputs)."""

import numpy as np
import pytest

from terrain.slope import compute_slope_degrees


def test_flat_dem_zero_slope() -> None:
    elev = np.full((6, 6), 42.5, dtype=np.float64)
    slope = compute_slope_degrees(elev, resolution_m=(2.0, 2.0))

    assert slope.shape == (6, 6)
    assert np.allclose(slope, 0.0)


def test_45_degree_inclined_dem() -> None:
    # Elevation rises 1 meter per column; dx = 1.0 m -> dz/dx = 1.0 -> arctan(1) = 45 deg
    _, cols = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    elev = cols.astype(np.float64) * 1.0

    slope = compute_slope_degrees(elev, resolution_m=(1.0, 1.0))
    assert np.allclose(slope, 45.0)


def test_anisotropic_pixel_spacing_slope() -> None:
    # Elevation rises 2 meters per row; dy = 2.0 m -> dz/dy = 1.0 -> 45 deg
    rows, _ = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    elev = rows.astype(np.float64) * 2.0

    slope = compute_slope_degrees(elev, resolution_m=(2.0, 5.0))
    assert np.allclose(slope, 45.0)


def test_nodata_handling_does_not_fabricate_measurements() -> None:
    # Planar 45-deg slope with an interior NoData cell
    _, cols = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    elev = cols.astype(np.float64)
    elev[2, 2] = np.nan

    slope = compute_slope_degrees(elev, resolution_m=(1.0, 1.0))

    # NoData cell itself must stay NaN
    assert np.isnan(slope[2, 2])
    # Valid neighbors still compute exact 45 deg from real adjacent valid elevations
    assert slope[2, 1] == pytest.approx(45.0)
    assert slope[2, 3] == pytest.approx(45.0)
    assert slope[1, 2] == pytest.approx(45.0)


def test_isolated_valid_pixel_becomes_nan_slope() -> None:
    elev = np.full((3, 3), np.nan, dtype=np.float64)
    elev[1, 1] = 10.0  # Surrounded completely by NoData

    slope = compute_slope_degrees(elev, resolution_m=(1.0, 1.0))
    assert np.all(np.isnan(slope))


def test_invalid_dem_dimensions_and_inputs() -> None:
    with pytest.raises(ValueError, match="2D"):
        compute_slope_degrees(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="at least 2 rows and 2 columns"):
        compute_slope_degrees(np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError, match="positive and finite"):
        compute_slope_degrees(np.ones((3, 3)), resolution_m=(1.0, -2.0))

    with pytest.raises(ValueError, match="does not match"):
        compute_slope_degrees(
            np.ones((3, 3)),
            resolution_m=(1.0, 1.0),
            valid_mask=np.ones((2, 2), dtype=bool),
        )
