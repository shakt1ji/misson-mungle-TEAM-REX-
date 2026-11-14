"""Tests for terrain.dem_loader (GeoTIFF loading, CRS validation, NoData, fixtures)."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from terrain.dem_loader import (
    DEMLoadError,
    InvalidDEMError,
    UnsupportedCRSUnitError,
    create_demo_dem_fixture,
    create_synthetic_dem,
    load_dem_geotiff,
)


def _write_geotiff(
    path: Path,
    array: np.ndarray,
    crs: CRS | None,
    pixel_size: float = 2.0,
    nodata: float | None = None,
) -> Path:
    transform = from_origin(0.0, 100.0, pixel_size, pixel_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)
    return path


def test_load_projected_geotiff_with_nodata(tmp_path: Path) -> None:
    arr = np.array(
        [
            [10.0, 12.0, -9999.0],
            [14.0, 16.0, 18.0],
            [20.0, 22.0, 24.0],
        ],
        dtype=np.float32,
    )
    tif_path = _write_geotiff(
        tmp_path / "projected_mars.tif",
        arr,
        crs=CRS.from_epsg(32615),  # Projected UTM meters
        pixel_size=2.5,
        nodata=-9999.0,
    )

    dem = load_dem_geotiff(tif_path)

    assert dem.elevation.shape == (3, 3)
    assert dem.resolution_m == (2.5, 2.5)
    assert dem.valid_mask[0, 2] is np.bool_(False)
    assert np.isnan(dem.elevation[0, 2])
    assert dem.valid_mask[1, 1] is np.bool_(True)
    assert dem.elevation[1, 1] == pytest.approx(16.0)
    assert dem.is_synthetic is False


def test_geographic_crs_rejected_never_assumed_meters(tmp_path: Path) -> None:
    arr = np.ones((4, 4), dtype=np.float32) * 100.0
    tif_path = _write_geotiff(
        tmp_path / "geographic_deg.tif",
        arr,
        crs=CRS.from_epsg(4326),  # WGS84 degrees
        pixel_size=0.0001,
    )

    with pytest.raises(UnsupportedCRSUnitError, match="geographic"):
        load_dem_geotiff(tif_path)


def test_missing_crs_rejected_by_default(tmp_path: Path) -> None:
    arr = np.ones((4, 4), dtype=np.float32)
    tif_path = _write_geotiff(tmp_path / "no_crs.tif", arr, crs=None)

    with pytest.raises(UnsupportedCRSUnitError, match="no Coordinate Reference System"):
        load_dem_geotiff(tif_path)


def test_incompatible_elevation_unit_rejected(tmp_path: Path) -> None:
    arr = np.ones((4, 4), dtype=np.float32)
    tif_path = _write_geotiff(tmp_path / "proj.tif", arr, crs=CRS.from_epsg(32615))

    with pytest.raises(UnsupportedCRSUnitError, match="Unsupported elevation_unit"):
        load_dem_geotiff(tif_path, elevation_unit="degrees")


def test_missing_file_raises_dem_load_error(tmp_path: Path) -> None:
    with pytest.raises(DEMLoadError, match="not found"):
        load_dem_geotiff(tmp_path / "does_not_exist.tif")


def test_synthetic_dem_nodata_and_invalid_dimensions() -> None:
    elev = np.array([[1.0, -9999.0], [3.0, np.nan]], dtype=np.float64)
    dem = create_synthetic_dem(elev, resolution_m=(2.0, 3.0), nodata_value=-9999.0)

    assert dem.is_synthetic is True
    assert dem.resolution_m == (2.0, 3.0)
    assert dem.valid_mask.tolist() == [[True, False], [True, False]]
    assert np.isnan(dem.elevation[0, 1])
    assert np.isnan(dem.elevation[1, 1])

    with pytest.raises(InvalidDEMError, match="2D"):
        create_synthetic_dem(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(InvalidDEMError, match="at least 2 rows and 2 columns"):
        create_synthetic_dem(np.array([[1.0]]))

    with pytest.raises(InvalidDEMError, match="positive and finite"):
        create_synthetic_dem(np.ones((3, 3)), resolution_m=(0.0, 1.0))


def test_demo_dem_fixture_deterministic() -> None:
    dem = create_demo_dem_fixture(rows=20, cols=20, resolution_m=(2.0, 2.0))
    assert dem.elevation.shape == (20, 20)
    assert dem.is_synthetic is True
    assert not np.all(dem.valid_mask)


def test_float32_extreme_nodata_sentinel_masked(tmp_path: Path) -> None:
    nodata_val = -3.4028235e38
    arr = np.array([[10.0, nodata_val], [12.0, 14.0]], dtype=np.float32)
    tif_path = _write_geotiff(
        tmp_path / "float32_extreme_nodata.tif",
        arr,
        crs=CRS.from_epsg(32615),
        pixel_size=2.0,
        nodata=nodata_val,
    )

    dem = load_dem_geotiff(tif_path)
    assert dem.valid_mask[0, 1] is np.bool_(False)
    assert np.isnan(dem.elevation[0, 1])
    assert dem.valid_mask[0, 0] is np.bool_(True)

