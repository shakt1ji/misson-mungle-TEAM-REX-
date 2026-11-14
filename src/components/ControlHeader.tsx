import React from "react";
import {
  AlertTriangle,
  Compass,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  StepForward,
} from "lucide-react";
import type {
  CameraPreset,
  MissionStatus,
  SimulationStatePayload,
  ViewLayout,
} from "../types/simulation";

interface ControlHeaderProps {
  simState: SimulationStatePayload;
  scenarios: Record<string, string>;
  viewLayout: ViewLayout;
  cameraPreset: CameraPreset;
  errorMsg: string | null;
  onChangeScenario: (name: string) => void;
  onChangeLayout: (layout: ViewLayout) => void;
  onChangeCamera: (preset: CameraPreset) => void;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStep: () => void;
  onReplan: () => void;
  onReset: () => void;
  onChangeSpeed: (speed: number) => void;
}

const STATUS_STYLE: Record<MissionStatus, string> = {
  IDLE: "border-[#3c3028] bg-[#1b1511] text-[#d6c7b8]",
  MOVING: "border-[#0284c7] bg-[#082f49] text-[#7dd3fc]",
  OBSTACLE_DETECTED: "border-[#d97706] bg-[#451a03] text-[#fcd34d]",
  REPLANNING: "border-[#d97706] bg-[#451a03] text-[#fde68a]",
  ROUTE_UPDATED: "border-[#0891b2] bg-[#083344] text-[#67e8f9]",
  GOAL_REACHED: "border-[#16a34a] bg-[#052e16] text-[#86efac]",
  NO_SAFE_PATH: "border-[#dc2626] bg-[#450a0a] text-[#fca5a5]",
};

export const ControlHeader: React.FC<ControlHeaderProps> = ({
  simState,
  scenarios,
  viewLayout,
  cameraPreset,
  errorMsg,
  onChangeScenario,
  onChangeLayout,
  onChangeCamera,
  onStart,
  onPause,
  onResume,
  onStep,
  onReplan,
  onReset,
  onChangeSpeed,
}) => {
  const status = simState.mission_status;
  const showReplanBanner =
    status === "OBSTACLE_DETECTED" ||
    status === "REPLANNING" ||
    status === "ROUTE_UPDATED" ||
    status === "NO_SAFE_PATH";

  return (
    <header className="flex flex-col border-b border-[#2b221c] bg-[#120e0c]">
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Compass className="w-5 h-5 text-[#e05a2b]" />
            <div>
              <h1 className="text-sm font-bold tracking-wider text-[#f5ede4] leading-none">
                MARS-09 AUTONOMOUS ROVER SIMULATOR
              </h1>
              <p className="text-[11px] text-[#9e8d7f] mt-0.5">
                UARA* Risk-Adaptive Planner · Martian Terramechanics Physics
              </p>
            </div>
          </div>

          <div
            className={`px-2.5 py-1 border rounded text-xs font-mono font-semibold tracking-wide ${STATUS_STYLE[status]}`}
          >
            {status}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            aria-label="Mission Scenario"
            value={simState.scenario_name}
            onChange={(e) => onChangeScenario(e.target.value)}
            className="bg-[#1b1512] border border-[#382c24] text-[#f3ece4] text-xs rounded px-2.5 py-1.5 focus:outline-none focus:border-[#e05a2b]"
          >
            {Object.keys(scenarios).map((key) => (
              <option key={key} value={key}>
                {key.replaceAll("_", " ").toUpperCase()}
              </option>
            ))}
          </select>

          <div className="flex border border-[#382c24] rounded overflow-hidden text-xs">
            {(
              [
                ["split", "Split 3D+2D"],
                ["3d", "3D View"],
                ["2d", "2D Grid"],
              ] as [ViewLayout, string][]
            ).map(([k, label]) => (
              <button
                key={k}
                type="button"
                onClick={() => onChangeLayout(k)}
                className={`px-2.5 py-1.5 font-medium transition-colors ${
                  viewLayout === k
                    ? "bg-[#e05a2b] text-[#0c0a09]"
                    : "bg-[#1b1512] text-[#c7b7a8] hover:bg-[#251d18]"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {viewLayout !== "2d" && (
            <div className="flex border border-[#382c24] rounded overflow-hidden text-xs">
              {(
                [
                  ["chase", "Chase Cam"],
                  ["orbit", "Orbit 3D"],
                  ["top", "Top 3D"],
                ] as [CameraPreset, string][]
              ).map(([k, label]) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => onChangeCamera(k)}
                  className={`px-2 py-1.5 font-medium transition-colors ${
                    cameraPreset === k
                      ? "bg-[#0284c7] text-white"
                      : "bg-[#1b1512] text-[#c7b7a8] hover:bg-[#251d18]"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          )}

          <div className="flex items-center gap-1.5 ml-1">
            {!simState.is_running ? (
              <button
                type="button"
                onClick={simState.sim_time_sec > 0 ? onResume : onStart}
                disabled={status === "GOAL_REACHED" || status === "NO_SAFE_PATH"}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold bg-[#e05a2b] text-[#0c0a09] hover:bg-[#f06b3c] disabled:opacity-40 cursor-pointer"
              >
                <Play className="w-3.5 h-3.5" />
                {simState.sim_time_sec > 0 ? "Resume" : "Start"}
              </button>
            ) : (
              <button
                type="button"
                onClick={onPause}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold bg-[#d97706] text-[#0c0a09] hover:bg-[#f59e0b] cursor-pointer"
              >
                <Pause className="w-3.5 h-3.5" />
                Pause
              </button>
            )}

            <button
              type="button"
              onClick={onStep}
              disabled={simState.is_running || status === "GOAL_REACHED" || status === "NO_SAFE_PATH"}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded text-xs font-medium bg-[#1b1512] border border-[#382c24] text-[#e7d8c9] hover:bg-[#251d18] disabled:opacity-40 cursor-pointer"
            >
              <StepForward className="w-3.5 h-3.5" />
              Step
            </button>

            <button
              type="button"
              onClick={onReplan}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded text-xs font-medium bg-[#1b1512] border border-[#382c24] text-[#67e8f9] hover:bg-[#251d18] cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Replan
            </button>

            <button
              type="button"
              onClick={onReset}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded text-xs font-medium bg-[#1b1512] border border-[#382c24] text-[#e7d8c9] hover:bg-[#251d18] cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset
            </button>
          </div>

          <div className="flex items-center gap-1 border border-[#382c24] rounded bg-[#1b1512] px-1.5 py-1 text-xs">
            <span className="text-[#9e8d7f] px-1">Speed</span>
            {[0.5, 1, 2, 4].map((spd) => (
              <button
                key={spd}
                type="button"
                onClick={() => onChangeSpeed(spd)}
                className={`px-1.5 py-0.5 rounded font-mono tabular-nums text-[11px] ${
                  Math.abs(simState.speed_multiplier - spd) < 0.05
                    ? "bg-[#38bdf8] text-[#0c0a09] font-bold"
                    : "text-[#c7b7a8] hover:bg-[#251d18]"
                }`}
              >
                {spd}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {(showReplanBanner || errorMsg) && (
        <div
          className={`flex items-center gap-2 px-4 py-1.5 text-xs font-medium border-t ${
            errorMsg || status === "NO_SAFE_PATH"
              ? "bg-[#3b0b0b] border-[#7f1d1d] text-[#fecaca]"
              : "bg-[#341d07] border-[#92400e] text-[#fde68a]"
          }`}
        >
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span>{errorMsg || simState.alert_message}</span>
        </div>
      )}
    </header>
  );
};
