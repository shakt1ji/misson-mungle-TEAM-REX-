"""Martian rover terramechanics, slope-grade dynamics, and slip physics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SoilTerramechanics:
    """Empirical terramechanics parameters for a Martian terrain class."""

    name: str
    rolling_resistance_crr: float
    base_slip: float
    slope_slip_factor: float
    sinkage_cm: float
    cohesion_kpa: float


TERRAIN_PHYSICS_MODEL: dict[int, SoilTerramechanics] = {
    0: SoilTerramechanics(
        name="soil",
        rolling_resistance_crr=0.08,
        base_slip=0.07,
        slope_slip_factor=0.008,
        sinkage_cm=2.2,
        cohesion_kpa=4.5,
    ),
    1: SoilTerramechanics(
        name="bedrock",
        rolling_resistance_crr=0.04,
        base_slip=0.02,
        slope_slip_factor=0.003,
        sinkage_cm=0.3,
        cohesion_kpa=18.0,
    ),
    2: SoilTerramechanics(
        name="sand",
        rolling_resistance_crr=0.18,
        base_slip=0.24,
        slope_slip_factor=0.018,
        sinkage_cm=7.8,
        cohesion_kpa=1.1,
    ),
    3: SoilTerramechanics(
        name="big_rock",
        rolling_resistance_crr=0.30,
        base_slip=0.45,
        slope_slip_factor=0.025,
        sinkage_cm=0.8,
        cohesion_kpa=25.0,
    ),
    255: SoilTerramechanics(
        name="unknown",
        rolling_resistance_crr=0.14,
        base_slip=0.16,
        slope_slip_factor=0.012,
        sinkage_cm=3.5,
        cohesion_kpa=3.0,
    ),
}


@dataclass(frozen=True)
class RoverPhysicsParams:
    """Physical constants for a Curiosity/Perseverance-class 6-wheel rover on Mars."""

    mass_kg: float = 1025.0
    mars_gravity_mps2: float = 3.721
    nominal_speed_mps: float = 1.4
    max_safe_slope_deg: float = 28.0
    wheel_radius_m: float = 0.26
    drivetrain_efficiency: float = 0.78


@dataclass(frozen=True)
class RoverPhysicsTelemetry:
    """Instantaneous terramechanics and drivetrain state for the rover."""

    grade_slope_deg: float
    pitch_deg: float
    roll_deg: float
    slip_ratio: float
    sinkage_cm: float
    rolling_resistance_n: float
    grade_resistance_n: float
    tractive_force_n: float
    effective_speed_mps: float
    motor_power_w: float
    energy_wh: float
    wheel_spin_rad: float
    tip_over_margin_deg: float


def _sample_bilinear_elevation(
    elevation: np.ndarray,
    valid_mask: np.ndarray,
    row_f: float,
    col_f: float,
) -> float:
    """Safely sample elevation at fractional (row_f, col_f) ignoring NaN neighbors."""
    rows, cols = elevation.shape
    r = float(np.clip(row_f, 0.0, rows - 1.0))
    c = float(np.clip(col_f, 0.0, cols - 1.0))
    r0 = int(math.floor(r))
    c0 = int(math.floor(c))
    r1 = min(r0 + 1, rows - 1)
    c1 = min(c0 + 1, cols - 1)
    fr = r - r0
    fc = c - c0

    weights = [
        ((r0, c0), (1.0 - fr) * (1.0 - fc)),
        ((r0, c1), (1.0 - fr) * fc),
        ((r1, c0), fr * (1.0 - fc)),
        ((r1, c1), fr * fc),
    ]
    num = 0.0
    den = 0.0
    for (rr, cc), w in weights:
        if valid_mask[rr, cc] and np.isfinite(elevation[rr, cc]):
            num += float(elevation[rr, cc]) * w
            den += w
    if den > 1e-9:
        return num / den
    nearest_r = int(round(r))
    nearest_c = int(round(c))
    if valid_mask[nearest_r, nearest_c] and np.isfinite(elevation[nearest_r, nearest_c]):
        return float(elevation[nearest_r, nearest_c])
    finite_vals = elevation[valid_mask & np.isfinite(elevation)]
    return float(np.mean(finite_vals)) if finite_vals.size > 0 else 0.0


def compute_directional_pitch_roll(
    elevation: np.ndarray,
    valid_mask: np.ndarray,
    resolution_m: tuple[float, float],
    row_f: float,
    col_f: float,
    heading_rad: float,
) -> tuple[float, float]:
    """Compute signed longitudinal pitch (deg) and lateral roll (deg) along rover heading."""
    dy_m, dx_m = float(resolution_m[0]), float(resolution_m[1])
    step_cells = 0.5
    dr_fwd = math.sin(heading_rad) * step_cells
    dc_fwd = math.cos(heading_rad) * step_cells
    dr_right = math.cos(heading_rad) * step_cells
    dc_right = -math.sin(heading_rad) * step_cells

    z_fwd = _sample_bilinear_elevation(elevation, valid_mask, row_f + dr_fwd, col_f + dc_fwd)
    z_aft = _sample_bilinear_elevation(elevation, valid_mask, row_f - dr_fwd, col_f - dc_fwd)
    z_right = _sample_bilinear_elevation(elevation, valid_mask, row_f + dr_right, col_f + dc_right)
    z_left = _sample_bilinear_elevation(elevation, valid_mask, row_f - dr_right, col_f - dc_right)

    ds_fwd_m = 2.0 * math.hypot(dr_fwd * dy_m, dc_fwd * dx_m)
    ds_lat_m = 2.0 * math.hypot(dr_right * dy_m, dc_right * dx_m)

    pitch_deg = math.degrees(math.atan2(z_fwd - z_aft, max(ds_fwd_m, 1e-6)))
    roll_deg = math.degrees(math.atan2(z_right - z_left, max(ds_lat_m, 1e-6)))
    return pitch_deg, roll_deg


def step_rover_physics(
    elevation: np.ndarray,
    slope_deg: np.ndarray,
    valid_mask: np.ndarray,
    terrain_class_ids: np.ndarray,
    resolution_m: tuple[float, float],
    row_f: float,
    col_f: float,
    heading_rad: float,
    speed_multiplier: float,
    dt_sec: float,
    is_moving: bool,
    prev_energy_wh: float,
    prev_wheel_spin_rad: float,
    params: RoverPhysicsParams | None = None,
) -> RoverPhysicsTelemetry:
    """Evaluate Martian rover terramechanics, forces, slip, speed, and energy draw."""
    p = params or RoverPhysicsParams()
    rows, cols = elevation.shape
    cr = int(np.clip(round(row_f), 0, rows - 1))
    cc = int(np.clip(round(col_f), 0, cols - 1))

    class_id = int(terrain_class_ids[cr, cc])
    soil = TERRAIN_PHYSICS_MODEL.get(class_id, TERRAIN_PHYSICS_MODEL[255])

    local_slope = (
        float(slope_deg[cr, cc])
        if valid_mask[cr, cc] and np.isfinite(slope_deg[cr, cc])
        else 0.0
    )
    pitch_deg, roll_deg = compute_directional_pitch_roll(
        elevation=elevation,
        valid_mask=valid_mask,
        resolution_m=resolution_m,
        row_f=row_f,
        col_f=col_f,
        heading_rad=heading_rad,
    )

    pitch_rad = math.radians(pitch_deg)
    slope_abs_deg = abs(local_slope)

    # Uphill climbing increases slip; downhill reduces slip slightly but never negative
    uphill_factor = max(0.0, pitch_deg)
    slip_ratio = float(
        np.clip(soil.base_slip + soil.slope_slip_factor * uphill_factor, 0.01, 0.85)
    )

    weight_n = p.mass_kg * p.mars_gravity_mps2
    rolling_res_n = weight_n * soil.rolling_resistance_crr * math.cos(pitch_rad)
    grade_res_n = weight_n * math.sin(pitch_rad)
    tractive_force_n = max(0.0, rolling_res_n + grade_res_n)

    if is_moving:
        grade_factor = max(0.35, math.cos(math.radians(min(slope_abs_deg, 65.0))))
        effective_speed_mps = max(
            0.12,
            p.nominal_speed_mps * float(speed_multiplier) * (1.0 - slip_ratio) * grade_factor,
        )
        # Mechanical + avionics hotel load (~45 W idle + traction power)
        mech_power_w = (tractive_force_n * effective_speed_mps) / max(p.drivetrain_efficiency, 0.1)
        motor_power_w = 45.0 + mech_power_w
        energy_wh = prev_energy_wh + (motor_power_w * max(dt_sec, 0.0)) / 3600.0
        wheel_angular_vel = effective_speed_mps / max((1.0 - slip_ratio) * p.wheel_radius_m, 0.05)
        wheel_spin_rad = (prev_wheel_spin_rad + wheel_angular_vel * max(dt_sec, 0.0)) % (2.0 * math.pi * 1000.0)
    else:
        effective_speed_mps = 0.0
        motor_power_w = 35.0  # Idle avionics & hazard-cam standby
        energy_wh = prev_energy_wh
        wheel_spin_rad = prev_wheel_spin_rad

    max_tilt = max(abs(pitch_deg), abs(roll_deg))
    tip_over_margin_deg = p.max_safe_slope_deg - max_tilt

    return RoverPhysicsTelemetry(
        grade_slope_deg=round(local_slope, 3),
        pitch_deg=round(pitch_deg, 3),
        roll_deg=round(roll_deg, 3),
        slip_ratio=round(slip_ratio, 4),
        sinkage_cm=round(soil.sinkage_cm * (1.0 + 0.015 * slope_abs_deg), 3),
        rolling_resistance_n=round(rolling_res_n, 2),
        grade_resistance_n=round(grade_res_n, 2),
        tractive_force_n=round(tractive_force_n, 2),
        effective_speed_mps=round(effective_speed_mps, 3),
        motor_power_w=round(motor_power_w, 2),
        energy_wh=round(energy_wh, 4),
        wheel_spin_rad=round(wheel_spin_rad, 4),
        tip_over_margin_deg=round(tip_over_margin_deg, 3),
    )
