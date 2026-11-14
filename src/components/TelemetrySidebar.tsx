import React from "react";
import {
  Flag,
  Layers,
  MapPin,
  MousePointer,
  Paintbrush,
  PlusCircle,
  Trash2,
} from "lucide-react";
import type {
  InteractionMode,
  SimulationStatePayload,
  TerrainBrushType,
} from "../types/simulation";

interface TelemetrySidebarProps {
  simState: SimulationStatePayload;
  interactionMode: InteractionMode;
  terrainBrush: TerrainBrushType;
  rockRadiusCells: number;
  onChangeMode: (mode: InteractionMode) => void;
  onChangeBrush: (brush: TerrainBrushType) => void;
  onChangeRockRadius: (radius: number) => void;
  onToggleControl: (
    patch: Partial<{
      auto_replan: boolean;
      show_cost_overlay: boolean;
      show_uncertainty_overlay: boolean;
      show_previous_route: boolean;
    }>
  ) => void;
}

const TOOL_BUTTONS: { id: InteractionMode; label: string; icon: React.ReactNode }[] = [
  { id: "drag", label: "Drag Rock", icon: <MousePointer className="w-3.5 h-3.5" /> },
  { id: "add", label: "Add Rock", icon: <PlusCircle className="w-3.5 h-3.5" /> },
  { id: "remove", label: "Remove Rock", icon: <Trash2 className="w-3.5 h-3.5" /> },
  { id: "paint", label: "Paint Terrain", icon: <Paintbrush className="w-3.5 h-3.5" /> },
  { id: "start", label: "Set Start", icon: <MapPin className="w-3.5 h-3.5" /> },
  { id: "goal", label: "Set Goal", icon: <Flag className="w-3.5 h-3.5" /> },
];

const BRUSH_OPTIONS: { id: TerrainBrushType; label: string; color: string }[] = [
  { id: "soil", label: "Soil", color: "#a4724a" },
  { id: "bedrock", label: "Bedrock", color: "#6e6c70" },
  { id: "sand", label: "Loose Sand", color: "#d4a662" },
  { id: "big_rock", label: "Big Rock", color: "#984230" },
  { id: "unknown", label: "Unknown", color: "#3d3d42" },
  { id: "blocked_region", label: "Blocked Zone", color: "#dc2626" },
];

export const TelemetrySidebar: React.FC<TelemetrySidebarProps> = ({
  simState,
  interactionMode,
  terrainBrush,
  rockRadiusCells,
  onChangeMode,
  onChangeBrush,
  onChangeRockRadius,
  onToggleControl,
}) => {
  const { rover, physics, routes, planner_metrics } = simState;

  return (
    <aside className="w-80 shrink-0 border-l border-[#2b221c] bg-[#120e0c] flex flex-col overflow-y-auto text-xs">
      {/* Interactive Tool Palette */}
      <section className="p-3 border-b border-[#2b221c]">
        <div className="text-[11px] font-semibold tracking-wider text-[#9e8d7f] uppercase mb-2">
          Environment Editor Tool
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          {TOOL_BUTTONS.map((tool) => (
            <button
              key={tool.id}
              type="button"
              onClick={() => onChangeMode(tool.id)}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded border text-left transition-colors cursor-pointer ${
                interactionMode === tool.id
                  ? "bg-[#e05a2b] border-[#f06b3c] text-[#0c0a09] font-semibold"
                  : "bg-[#1a1411] border-[#30251e] text-[#d6c7b8] hover:bg-[#231b17]"
              }`}
            >
              {tool.icon}
              <span>{tool.label}</span>
            </button>
          ))}
        </div>

        {interactionMode === "add" && (
          <div className="mt-2.5 flex items-center gap-1.5">
            <span className="text-[#9e8d7f]">Size:</span>
            {[
              [0, "1x1 Rock"],
              [1, "3x3 Boulder"],
            ].map(([rad, lbl]) => (
              <button
                key={rad}
                type="button"
                onClick={() => onChangeRockRadius(Number(rad))}
                className={`flex-1 py-1 rounded border text-[11px] font-medium ${
                  rockRadiusCells === rad
                    ? "bg-[#38bdf8] border-[#38bdf8] text-[#0c0a09] font-semibold"
                    : "bg-[#1a1411] border-[#30251e] text-[#c7b7a8]"
                }`}
              >
                {lbl}
              </button>
            ))}
          </div>
        )}

        {interactionMode === "paint" && (
          <div className="mt-2.5 grid grid-cols-3 gap-1">
            {BRUSH_OPTIONS.map((b) => (
              <button
                key={b.id}
                type="button"
                onClick={() => onChangeBrush(b.id)}
                className={`flex items-center gap-1.5 px-1.5 py-1 rounded border text-[11px] ${
                  terrainBrush === b.id
                    ? "border-[#38bdf8] bg-[#082f49] text-white font-semibold"
                    : "border-[#30251e] bg-[#1a1411] text-[#c7b7a8]"
                }`}
              >
                <span
                  className="w-2.5 h-2.5 rounded-xs shrink-0"
                  style={{ backgroundColor: b.color }}
                />
                <span className="truncate">{b.label}</span>
              </button>
            ))}
          </div>
        )}
      </section>

      {/* Overlays & Planner Options */}
      <section className="p-3 border-b border-[#2b221c]">
        <div className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-[#9e8d7f] uppercase mb-2">
          <Layers className="w-3.5 h-3.5" />
          Overlays &amp; Planner
        </div>
        <div className="space-y-1.5">
          {[
            ["auto_replan", "Auto-Replan on Obstacles", simState.auto_replan],
            ["show_cost_overlay", "Terrain Cost Heatmap", simState.show_cost_overlay],
            ["show_uncertainty_overlay", "AI4MARS Uncertainty", simState.show_uncertainty_overlay],
            ["show_previous_route", "Show Previous Route", simState.show_previous_route],
          ].map(([key, label, checked]) => (
            <label
              key={String(key)}
              className="flex items-center justify-between px-2 py-1 rounded bg-[#1a1411] border border-[#2b221c] cursor-pointer"
            >
              <span className="text-[#dfd1c3]">{label}</span>
              <input
                type="checkbox"
                checked={Boolean(checked)}
                onChange={(e) =>
                  onToggleControl({ [String(key)]: e.target.checked })
                }
                className="accent-[#e05a2b]"
              />
            </label>
          ))}
        </div>
      </section>

      {/* Live Rover & Martian Physics Readout */}
      <section className="p-3 border-b border-[#2b221c]">
        <div className="text-[11px] font-semibold tracking-wider text-[#9e8d7f] uppercase mb-2">
          Martian Terramechanics (g = 3.721 m/s²)
        </div>
        <div className="grid grid-cols-2 gap-1.5 font-mono tabular-nums text-[11px]">
          <div className="p-2 rounded bg-[#1a1411] border border-[#2b221c]">
            <div className="text-[#9e8d7f]">ROVER GRID</div>
            <div className="text-sm font-semibold text-[#f3ece4]">
              ({rover.row}, {rover.col})
            </div>
            <div className="text-[#38bdf8] uppercase">{rover.terrain_class}</div>
          </div>
          <div className="p-2 rounded bg-[#1a1411] border border-[#2b221c]">
            <div className="text-[#9e8d7f]">ELEV / GRADE</div>
            <div className="text-sm font-semibold text-[#f3ece4]">
              {rover.elevation_m.toFixed(1)} m
            </div>
            <div className="text-[#fbbf24]">{physics.grade_slope_deg.toFixed(1)}° slope</div>
          </div>
          <div className="p-2 rounded bg-[#1a1411] border border-[#2b221c]">
            <div className="text-[#9e8d7f]">PITCH / ROLL</div>
            <div className="text-sm font-semibold text-[#f3ece4]">
              {physics.pitch_deg >= 0 ? "+" : ""}
              {physics.pitch_deg.toFixed(1)}°
            </div>
            <div className="text-[#b4a494]">
              Roll {physics.roll_deg >= 0 ? "+" : ""}
              {physics.roll_deg.toFixed(1)}°
            </div>
          </div>
          <div className="p-2 rounded bg-[#1a1411] border border-[#2b221c]">
            <div className="text-[#9e8d7f]">WHEEL SLIP</div>
            <div className="text-sm font-semibold text-[#f3ece4]">
              {(physics.slip_ratio * 100).toFixed(1)}%
            </div>
            <div className="text-[#b4a494]">Sink {physics.sinkage_cm.toFixed(1)} cm</div>
          </div>
          <div className="p-2 rounded bg-[#1a1411] border border-[#2b221c]">
            <div className="text-[#9e8d7f]">SPEED / POWER</div>
            <div className="text-sm font-semibold text-[#f3ece4]">
              {physics.effective_speed_mps.toFixed(2)} m/s
            </div>
            <div className="text-[#b4a494]">
              {physics.motor_power_w.toFixed(0)} W · {physics.energy_wh.toFixed(2)} Wh
            </div>
          </div>
          <div className="p-2 rounded bg-[#1a1411] border border-[#2b221c]">
            <div className="text-[#9e8d7f]">PATH / REPLANS</div>
            <div className="text-sm font-semibold text-[#00e5ff]">
              {simState.distance_travelled_m.toFixed(1)} m
            </div>
            <div className="text-[#b4a494]">
              {routes.remaining_steps} left · {simState.replan_count} replans
            </div>
          </div>
        </div>
        <div className="mt-1.5 flex justify-between text-[11px] font-mono tabular-nums text-[#9e8d7f] px-1">
          <span>Traction {physics.tractive_force_n.toFixed(0)} N</span>
          <span>
            UARA* Cost{" "}
            {planner_metrics.total_cost !== null
              ? planner_metrics.total_cost.toFixed(2)
              : "INF"}
          </span>
        </div>
      </section>

      {/* Replan Event Log */}
      <section className="p-3 flex-1 min-h-36 flex flex-col">
        <div className="text-[11px] font-semibold tracking-wider text-[#9e8d7f] uppercase mb-2">
          Dynamic Replan Log ({simState.replan_events.length})
        </div>
        {simState.replan_events.length === 0 ? (
          <p className="text-[11px] text-[#857568] leading-relaxed">
            Press Start and drag a boulder onto the active route to observe real-time UARA* avoidance.
          </p>
        ) : (
          <div className="space-y-1.5 overflow-y-auto pr-1">
            {simState.replan_events.map((ev) => (
              <div
                key={ev.event_id}
                className={`p-2 rounded border text-[11px] font-mono ${
                  ev.success
                    ? "bg-[#161c22] border-[#1e3a5f] text-[#bae6fd]"
                    : "bg-[#2a1212] border-[#7f1d1d] text-[#fecaca]"
                }`}
              >
                <div className="flex justify-between text-[10px] opacity-80">
                  <span>t={ev.timestamp_sec.toFixed(1)}s</span>
                  <span>{ev.trigger}</span>
                </div>
                <div className="mt-0.5 leading-snug">{ev.message}</div>
              </div>
            ))}
          </div>
        )}
      </section>
    </aside>
  );
};
