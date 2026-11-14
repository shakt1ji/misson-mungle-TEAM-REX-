"""Terrain slope calculation from 2D Digital Elevation Models."""

from __future__ import annotations

import numpy as np


def _axis_gradient_with_mask(
    values: np.ndarray,
    valid: np.ndarray,
    spacing: float,
    axis: int,
) -> np.ndarray:
    """Compute 1D numerical gradient along `axis` without fabricating NoData values."""
    length = values.shape[axis]
    grad = np.full(values.shape, np.nan, dtype=np.float64)

    if length < 2:
        return grad

    v = np.moveaxis(values, axis, 0)
    m = np.moveaxis(valid, axis, 0)
    g = np.moveaxis(grad, axis, 0)

    if length >= 3:
        both_neighbors = m[1:-1] & m[:-2] & m[2:]
        g_int = g[1:-1]
        g_int[both_neighbors] = (v[2:][both_neighbors] - v[:-2][both_neighbors]) / (2.0 * spacing)

        forward_only = m[1:-1] & (~m[:-2]) & m[2:]
        g_int[forward_only] = (v[2:][forward_only] - v[1:-1][forward_only]) / spacing

        backward_only = m[1:-1] & m[:-2] & (~m[2:])
        g_int[backward_only] = (v[1:-1][backward_only] - v[:-2][backward_only]) / spacing

    first_valid = m[0] & m[1]
    g[0][first_valid] = (v[1][first_valid] - v[0][first_valid]) / spacing

    last_valid = m[-1] & m[-2]
    g[-1][last_valid] = (v[-1][last_valid] - v[-2][last_valid]) / spacing

    return grad


def compute_slope_degrees(
    elevation: np.ndarray,
    resolution_m: tuple[float, float] = (1.0, 1.0),
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute terrain slope in degrees using numerical finite-difference gradients."""
    try:
        elev = np.asarray(elevation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"elevation must be a numeric 2D array: {exc}") from exc

    if elev.ndim != 2:
        raise ValueError(
            f"elevation must be a 2D array (rows, cols); got {elev.ndim}D with shape {elev.shape}."
        )
    if elev.shape[0] < 2 or elev.shape[1] < 2:
        raise ValueError(
            f"elevation must have at least 2 rows and 2 columns to compute gradients; "
            f"got shape {elev.shape}."
        )

    if not isinstance(resolution_m, (tuple, list)) or len(resolution_m) != 2:
        raise ValueError("resolution_m must be a 2-tuple of (row_spacing_m, col_spacing_m).")
    dy_m = float(resolution_m[0])
    dx_m = float(resolution_m[1])
    if not (np.isfinite(dy_m) and np.isfinite(dx_m)) or dy_m <= 0.0 or dx_m <= 0.0:
        raise ValueError(
            f"resolution_m values must be positive and finite; got {resolution_m}."
        )

    effective_valid = np.isfinite(elev)
    if valid_mask is not None:
        mask_arr = np.asarray(valid_mask, dtype=bool)
        if mask_arr.shape != elev.shape:
            raise ValueError(
                f"valid_mask shape {mask_arr.shape} does not match elevation shape {elev.shape}."
            )
        effective_valid &= mask_arr

    dz_dy = _axis_gradient_with_mask(elev, effective_valid, dy_m, axis=0)
    dz_dx = _axis_gradient_with_mask(elev, effective_valid, dx_m, axis=1)

    slope = np.full(elev.shape, np.nan, dtype=np.float64)
    computable = effective_valid & np.isfinite(dz_dy) & np.isfinite(dz_dx)

    gradient_magnitude = np.hypot(dz_dx[computable], dz_dy[computable])
    slope[computable] = np.degrees(np.arctan(gradient_magnitude))

    return slope
