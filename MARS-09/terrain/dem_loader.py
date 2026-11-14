"""DEM loading, CRS/unit validation, NoData handling, and synthetic test fixtures.

Conventions:
- Array indexing is [row, column].
- Elevation is returned in meters as float64, with invalid/NoData cells set to np.nan.
- Pixel spacing resolution_m is (row_spacing_m, col_spacing_m) in meters.
- Invalid DEM cells are tracked explicitly via valid_mask (True = valid, False = invalid).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class DEMLoadError(Exception):
    """Base exception raised when loading or validating a DEM fails."""


class UnsupportedCRSUnitError(DEMLoadError):
    """Raised when CRS or elevation units cannot be verified as compatible metric units."""


class InvalidDEMError(DEMLoadError):
    """Raised when DEM dimensions, band data, or resolution values are invalid."""


_METRIC_ELEVATION_FACTORS: dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "kilometre": 1000.0,
    "kilometres": 1000.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "millimetre": 0.001,
    "millimetres": 0.001,
}


@dataclass(frozen=True)
class DEMData:
    """Container for validated Digital Elevation Model (DEM) data."""

    elevation: np.ndarray
    valid_mask: np.ndarray
    resolution_m: tuple[float, float]
    crs: str | None
    transform: Any | None
    nodata_value: float | None
    source_name: str
    is_synthetic: bool = False


def _validate_elevation_unit(elevation_unit: str) -> float:
    """Validate elevation unit string and return scale factor to meters."""
    if not isinstance(elevation_unit, str) or not elevation_unit.strip():
        raise UnsupportedCRSUnitError(
            "elevation_unit must be a non-empty string (for example, 'meter')."
        )
    normalized = elevation_unit.strip().lower()
    if normalized not in _METRIC_ELEVATION_FACTORS:
        raise UnsupportedCRSUnitError(
            f"Unsupported elevation_unit '{elevation_unit}'. "
            "Elevation must be in metric units (e.g., 'meter', 'm', 'km', 'cm', 'mm') "
            "and cannot be angular units like degrees."
        )
    return _METRIC_ELEVATION_FACTORS[normalized]


def validate_crs_and_units(crs: Any, elevation_unit: str = "meter") -> tuple[float, str]:
    """Validate that a rasterio CRS uses projected linear units compatible with meters.

    Never silently assumes geographic coordinates (degrees) are in meters.
    """
    _validate_elevation_unit(elevation_unit)

    if crs is None:
        raise UnsupportedCRSUnitError(
            "DEM GeoTIFF has no Coordinate Reference System (CRS). "
            "Cannot verify that horizontal pixel spacing is in meters. "
            "Reproject or assign a projected metric CRS before loading."
        )

    is_geographic = getattr(crs, "is_geographic", False)
    if is_geographic:
        raise UnsupportedCRSUnitError(
            f"DEM uses a geographic CRS ({crs}) with angular units (degrees). "
            "Geographic coordinates must never be treated as meters. "
            "Reproject the DEM to a projected metric CRS (e.g., Mars Equirectangular "
            "or Polar Stereographic in meters) before slope and distance calculations."
        )

    is_projected = getattr(crs, "is_projected", False)
    if not is_projected:
        raise UnsupportedCRSUnitError(
            f"DEM CRS ({crs}) is not a projected coordinate system. "
            "A projected CRS with linear units convertible to meters is required."
        )

    try:
        unit_name, factor_to_meters = crs.linear_units_factor
    except Exception as exc:
        raise UnsupportedCRSUnitError(
            f"Could not determine linear units for CRS ({crs}): {exc}"
        ) from exc

    if factor_to_meters is None or not np.isfinite(factor_to_meters) or factor_to_meters <= 0:
        raise UnsupportedCRSUnitError(
            f"CRS ({crs}) reported invalid linear unit factor '{factor_to_meters}' "
            f"for unit '{unit_name}'."
        )

    return float(factor_to_meters), str(crs)


def _validate_resolution_tuple(resolution_m: tuple[float, float]) -> tuple[float, float]:
    """Validate (row_spacing_m, col_spacing_m) resolution tuple."""
    if not isinstance(resolution_m, (tuple, list)) or len(resolution_m) != 2:
        raise InvalidDEMError(
            "resolution_m must be a 2-tuple of (row_spacing_m, col_spacing_m)."
        )
    dy_m = float(resolution_m[0])
    dx_m = float(resolution_m[1])
    if not (np.isfinite(dy_m) and np.isfinite(dx_m)) or dy_m <= 0.0 or dx_m <= 0.0:
        raise InvalidDEMError(
            f"Pixel spacing in meters must be positive and finite; got {resolution_m}."
        )
    return (dy_m, dx_m)


def load_dem_geotiff(
    file_path: str | Path,
    band: int = 1,
    elevation_unit: str = "meter",
    allow_missing_crs: bool = False,
) -> DEMData:
    """Load a GeoTIFF DEM using Rasterio, validating CRS, units, and NoData masks."""
    try:
        import rasterio
    except ImportError as exc:
        raise DEMLoadError(
            "The 'rasterio' package is required to load GeoTIFF DEMs. "
            "Install it via 'pip install rasterio'."
        ) from exc

    path = Path(file_path)
    if not path.is_file():
        raise DEMLoadError(f"DEM file not found: '{path}'.")

    elev_scale = _validate_elevation_unit(elevation_unit)

    try:
        with rasterio.open(path) as dataset:
            if band < 1 or band > dataset.count:
                raise InvalidDEMError(
                    f"Requested band {band} is out of range for '{path}' "
                    f"(available bands: 1..{dataset.count})."
                )

            if dataset.crs is None:
                if not allow_missing_crs:
                    validate_crs_and_units(None, elevation_unit=elevation_unit)
                horizontal_factor = 1.0
                crs_str: str | None = None
            else:
                horizontal_factor, crs_str = validate_crs_and_units(
                    dataset.crs, elevation_unit=elevation_unit
                )

            x_res_raw, y_res_raw = dataset.res
            row_spacing_m = abs(float(y_res_raw)) * horizontal_factor
            col_spacing_m = abs(float(x_res_raw)) * horizontal_factor
            resolution_m = _validate_resolution_tuple((row_spacing_m, col_spacing_m))

            raw_band = dataset.read(band, masked=False)
            if raw_band.ndim != 2 or raw_band.shape[0] < 2 or raw_band.shape[1] < 2:
                raise InvalidDEMError(
                    f"DEM band must be a 2D array with at least 2x2 pixels; "
                    f"got shape {raw_band.shape}."
                )

            elevation = np.asarray(raw_band, dtype=np.float64)
            nodata = dataset.nodata

            valid_mask = np.isfinite(elevation)
            if nodata is not None and not np.isnan(nodata):
                valid_mask &= ~np.isclose(elevation, float(nodata), rtol=1e-7, atol=1e-12)

            elevation_meters = np.where(valid_mask, elevation * elev_scale, np.nan)

            return DEMData(
                elevation=elevation_meters,
                valid_mask=valid_mask,
                resolution_m=resolution_m,
                crs=crs_str,
                transform=dataset.transform,
                nodata_value=float(nodata) if nodata is not None else None,
                source_name=str(path),
                is_synthetic=False,
            )
    except DEMLoadError:
        raise
    except Exception as exc:
        raise DEMLoadError(f"Failed to read GeoTIFF DEM '{path}': {exc}") from exc


def create_synthetic_dem(
    elevation: np.ndarray,
    resolution_m: tuple[float, float] = (1.0, 1.0),
    nodata_value: float | None = None,
    source_name: str = "synthetic_dem",
) -> DEMData:
    """Create a validated DEMData object from an in-memory NumPy array."""
    try:
        arr = np.asarray(elevation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidDEMError(f"Elevation data must be numeric: {exc}") from exc

    if arr.ndim != 2:
        raise InvalidDEMError(
            f"DEM elevation array must be 2D (rows, cols); got {arr.ndim}D with shape {arr.shape}."
        )
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise InvalidDEMError(
            f"DEM elevation array must have at least 2 rows and 2 columns; got shape {arr.shape}."
        )

    validated_res = _validate_resolution_tuple(resolution_m)

    valid_mask = np.isfinite(arr)
    if nodata_value is not None and not np.isnan(nodata_value):
        valid_mask &= ~np.isclose(arr, float(nodata_value), rtol=1e-7, atol=1e-12)

    cleaned_elevation = np.where(valid_mask, arr, np.nan)

    return DEMData(
        elevation=cleaned_elevation,
        valid_mask=valid_mask,
        resolution_m=validated_res,
        crs="SYNTHETIC_LOCAL_METERS",
        transform=None,
        nodata_value=float(nodata_value) if nodata_value is not None else None,
        source_name=source_name,
        is_synthetic=True,
    )


def create_demo_dem_fixture(
    rows: int = 40,
    cols: int = 40,
    resolution_m: tuple[float, float] = (2.0, 2.0),
) -> DEMData:
    """Generate a deterministic synthetic Mars-like terrain DEM fixture for demos/tests."""
    if rows < 10 or cols < 10:
        raise InvalidDEMError("Demo DEM fixture requires at least 10x10 dimensions.")

    r_idx, c_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    dy_m, dx_m = _validate_resolution_tuple(resolution_m)
    y_m = r_idx * dy_m
    x_m = c_idx * dx_m

    center_y = (rows * dy_m) * 0.5
    center_x = (cols * dx_m) * 0.5

    mound = 12.0 * np.exp(-(((y_m - center_y) ** 2) / (2.0 * (8.0 ** 2)) + ((x_m - center_x) ** 2) / (2.0 * (5.0 ** 2))))
    depression = -4.0 * np.exp(-(((y_m - center_y * 0.45) ** 2) + ((x_m - center_x * 1.45) ** 2)) / (2.0 * (6.0 ** 2)))
    regional_tilt = 0.05 * x_m + 0.03 * y_m

    elevation = 100.0 + regional_tilt + mound + depression

    nodata_sentinel = -9999.0
    elevation[0:2, -2:] = nodata_sentinel

    return create_synthetic_dem(
        elevation=elevation,
        resolution_m=(dy_m, dx_m),
        nodata_value=nodata_sentinel,
        source_name="synthetic_demo_fixture",
    )
