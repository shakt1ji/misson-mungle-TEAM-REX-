"""Real-time Mars rover simulation, terramechanics physics, and dynamic UARA* replanning."""

from simulation.dynamic_obstacles import (
    DynamicObstacleManager,
    DynamicRockObstacle,
    InvalidObstaclePlacementError,
    canvas_to_grid,
)
from simulation.environment_editor import (
    SCENARIO_DESCRIPTIONS,
    TERRAIN_TYPE_TO_CLASS_ID,
    EnvironmentBuildResult,
    apply_terrain_edit,
    build_scenario_environment,
    recompute_cost_map,
)
from simulation.physics import (
    RoverPhysicsParams,
    RoverPhysicsTelemetry,
    SoilTerramechanics,
    TERRAIN_PHYSICS_MODEL,
    compute_directional_pitch_roll,
    step_rover_physics,
)
from simulation.simulation_controller import RoverSimulationController
from simulation.simulation_state import (
    MissionStatus,
    ReplanEvent,
    RoverSimulationState,
)

__all__ = [
    "DynamicObstacleManager",
    "DynamicRockObstacle",
    "EnvironmentBuildResult",
    "InvalidObstaclePlacementError",
    "MissionStatus",
    "ReplanEvent",
    "RoverPhysicsParams",
    "RoverPhysicsTelemetry",
    "RoverSimulationController",
    "RoverSimulationState",
    "SCENARIO_DESCRIPTIONS",
    "SoilTerramechanics",
    "TERRAIN_PHYSICS_MODEL",
    "TERRAIN_TYPE_TO_CLASS_ID",
    "apply_terrain_edit",
    "build_scenario_environment",
    "canvas_to_grid",
    "compute_directional_pitch_roll",
    "recompute_cost_map",
    "step_rover_physics",
]
