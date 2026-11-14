import type { SimulationStatePayload, TerrainBrushType } from "../types/simulation";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = (await res.json()) as { ok?: boolean; error?: string } & T;
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

export interface StateEnvelope {
  ok: true;
  state: SimulationStatePayload;
}

export const simClient = {
  getState: (includeGrids = true) =>
    request<StateEnvelope>(`/api/sim/state?grids=${includeGrids ? "1" : "0"}`),

  getScenarios: () =>
    request<{ ok: true; scenarios: Record<string, string> }>("/api/sim/scenarios"),

  loadScenario: (scenario_name: string) =>
    request<StateEnvelope>("/api/sim/scenario", {
      method: "POST",
      body: JSON.stringify({ scenario_name }),
    }),

  start: () => request<StateEnvelope>("/api/sim/start", { method: "POST" }),
  pause: () => request<StateEnvelope>("/api/sim/pause", { method: "POST" }),
  resume: () => request<StateEnvelope>("/api/sim/resume", { method: "POST" }),
  reset: () => request<StateEnvelope>("/api/sim/reset", { method: "POST" }),
  replan: () => request<StateEnvelope>("/api/sim/replan", { method: "POST" }),

  step: (dt_sec = 0.2) =>
    request<StateEnvelope>("/api/sim/step", {
      method: "POST",
      body: JSON.stringify({ dt_sec, include_grids: false }),
    }),

  setControls: (controls: {
    speed_multiplier?: number;
    auto_replan?: boolean;
    show_cost_overlay?: boolean;
    show_uncertainty_overlay?: boolean;
    show_previous_route?: boolean;
  }) =>
    request<StateEnvelope>("/api/sim/controls", {
      method: "POST",
      body: JSON.stringify(controls),
    }),

  addObstacle: (row: number, col: number, radius_cells = 0) =>
    request<StateEnvelope>("/api/sim/obstacle/add", {
      method: "POST",
      body: JSON.stringify({
        row,
        col,
        radius_cells,
        height_m: radius_cells > 0 ? 1.55 : 1.15,
        label: radius_cells > 0 ? "Boulder" : "Rock",
      }),
    }),

  moveObstacle: (obstacle_id: string, row: number, col: number) =>
    request<StateEnvelope>("/api/sim/obstacle/move", {
      method: "POST",
      body: JSON.stringify({ obstacle_id, row, col }),
    }),

  removeObstacle: (args: { obstacle_id?: string; row?: number; col?: number }) =>
    request<StateEnvelope>("/api/sim/obstacle/remove", {
      method: "POST",
      body: JSON.stringify(args),
    }),

  paintTerrain: (cells: [number, number][], terrain_type: TerrainBrushType) =>
    request<StateEnvelope>("/api/sim/terrain/paint", {
      method: "POST",
      body: JSON.stringify({ cells, terrain_type }),
    }),

  setEndpoints: (endpoints: { start_cell?: [number, number]; goal_cell?: [number, number] }) =>
    request<StateEnvelope>("/api/sim/endpoints", {
      method: "POST",
      body: JSON.stringify(endpoints),
    }),
};
