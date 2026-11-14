export type MissionStatus =
  | "IDLE"
  | "MOVING"
  | "OBSTACLE_DETECTED"
  | "REPLANNING"
  | "ROUTE_UPDATED"
  | "GOAL_REACHED"
  | "NO_SAFE_PATH";

export type InteractionMode =
  | "drag"
  | "add"
  | "remove"
  | "paint"
  | "start"
  | "goal";

export type TerrainBrushType =
  | "soil"
  | "bedrock"
  | "sand"
  | "big_rock"
  | "unknown"
  | "blocked_region";

export type ViewLayout = "split" | "3d" | "2d";

export type CameraPreset = "chase" | "orbit" | "top";

export interface ObstacleItem {
  obstacle_id: string;
  row: number;
  col: number;
  radius_cells: number;
  height_m: number;
  label: string;
  is_draggable: boolean;
  footprint: [number, number][];
}

export interface RoverPosition {
  row: number;
  col: number;
  row_f: number;
  col_f: number;
  heading_rad: number;
  elevation_m: number;
  terrain_class: string;
  terrain_class_id: number;
}

export interface PhysicsTelemetry {
  grade_slope_deg: number;
  pitch_deg: number;
  roll_deg: number;
  slip_ratio: number;
  sinkage_cm: number;
  rolling_resistance_n: number;
  grade_resistance_n: number;
  tractive_force_n: number;
  effective_speed_mps: number;
  motor_power_w: number;
  energy_wh: number;
  wheel_spin_rad: number;
  tip_over_margin_deg: number;
}

export interface ReplanEvent {
  event_id: number;
  timestamp_sec: number;
  trigger: string;
  rover_position: [number, number];
  old_remaining_steps: number;
  new_steps: number;
  new_cost: number | null;
  nodes_expanded: number;
  success: boolean;
  message: string;
}

export interface SimulationGrids {
  elevation: number[][];
  slope_deg: number[][];
  terrain_class_ids: number[][];
  blocked_mask: number[][];
  total_cost: number[][];
  uncertainty: number[][];
}

export interface SimulationStatePayload {
  scenario_name: string;
  grid_shape: [number, number];
  resolution_m: [number, number];
  start_cell: [number, number];
  goal_cell: [number, number];
  rover: RoverPosition;
  mission_status: MissionStatus;
  is_running: boolean;
  speed_multiplier: number;
  auto_replan: boolean;
  show_cost_overlay: boolean;
  show_uncertainty_overlay: boolean;
  show_previous_route: boolean;
  sim_time_sec: number;
  distance_travelled_m: number;
  replan_count: number;
  alert_message: string;
  alert_seq: number;
  obstacles: ObstacleItem[];
  routes: {
    original_route: [number, number][];
    previous_route: [number, number][];
    active_route: [number, number][];
    expanded_route: [number, number][];
    travelled_cells: [number, number][];
    travelled_points: [number, number][];
    remaining_steps: number;
  };
  planner_metrics: {
    success: boolean;
    total_cost: number | null;
    path_distance_m: number;
    nodes_expanded: number;
    termination_reason: string;
  };
  physics: PhysicsTelemetry;
  replan_events: ReplanEvent[];
  grids?: SimulationGrids;
}
